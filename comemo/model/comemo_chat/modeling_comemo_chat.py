import os
import random
import warnings
from typing import Any, List, Optional, Tuple, Union

from einops import rearrange,repeat
import torch.distributed as dist
import torch.utils.checkpoint
import transformers

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from comemo.conversation import get_conv_template
from comemo.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from .configuration_comemo_chat import CoMemoChatConfig
from .modeling_intern_vit import InternVisionModel
from .mixin_lm import LMMixin
from .utils import _infer_decoder_layers_attr_name, extend_instance
from .helpers import *

import numpy as np

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

ORIGINAL_SIZE = 16
THUMBNAIL_TOKEN_LENGTH = 256
def calculate_subimage_indices(X, Y, position_bias):
    """
    Calculate the index mapping for X×Y sub-images, which maps tokens from a 16×16 sub-image into the thumbnail.

    Args:
        X (int): The number of columns the subimage is divided into.
        Y (int): The number of rows the subimage is divided into.
        position_bias (int): Offset added to the indices.

    Returns:
        list: A list containing indices for all subimage tokens combined with thumbnail image indices.
    """
    result = []
    
    if X > 1 or Y > 1:
        # Use RoPE-DHR
        subimage_width = ORIGINAL_SIZE / X - 1e-6
        subimage_height = ORIGINAL_SIZE / Y - 1e-6
        for i in range(X):
            for j in range(Y):
                # The indices of the top-left and bottom-right corners of the current subimage.
                start_x = i * subimage_width
                end_x = (i + 1) * subimage_width
                start_y = j * subimage_height
                end_y = (j + 1) * subimage_height
                
                # Generate the index list for the current subimage.
                indices = [
                    (int(row) * ORIGINAL_SIZE + int(col) + position_bias)
                    for row in np.linspace(start_y, end_y, ORIGINAL_SIZE)
                    for col in np.linspace(start_x, end_x, ORIGINAL_SIZE)
                ]

                result.extend(indices)

    thumnail_position_ids = (np.arange(0, THUMBNAIL_TOKEN_LENGTH) + position_bias).tolist()
    result.extend(thumnail_position_ids)

    return result

class CoMemoChatModel(PreTrainedModel):
    config_class = CoMemoChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: CoMemoChatConfig, vision_model=None, language_model=None, delay_init_new_param=False):
        super().__init__(config)
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.use_temporal:
                self.vision_model = InternVisionTemporalModel(config.vision_config, delay_init_new_param=delay_init_new_param)
            else:
                self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.num_samples = 0

        ## Init Mixin Layers
        self.mixin_every_n_layers = config.mixin_config.mixin_every_n_layers
        extend_instance(self.language_model, LMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(self.language_model)
        self.language_model.set_decoder_layers_attr_name(decoder_layers_attr_name)
        self.language_model.init_mixin(
            config=config.mixin_config,
            gradient_checkpointing=True,
        )

    def _condition_attn_mask_and_pos_ids(self, attn_mask, cross_attn_media_position_ids, cross_attn_text_position_ids, text_time, cu_seqlens_q=None, cu_seqlens_k=None):
        for layer in self.language_model._get_decoder_layers():
            if layer.gated_cross_attn_layer is not None:
                layer.gated_cross_attn_layer.cross_attn.media_attn_mask = attn_mask
                layer.gated_cross_attn_layer.cross_attn_media_position_ids = cross_attn_media_position_ids
                layer.gated_cross_attn_layer.cross_attn_text_position_ids = cross_attn_text_position_ids
                layer.gated_cross_attn_layer.text_time = text_time
                layer.gated_cross_attn_layer.cross_attn.cu_seqlens_q = cu_seqlens_q
                layer.gated_cross_attn_layer.cross_attn.cu_seqlens_k = cu_seqlens_k

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            seq_imgs: Optional[torch.LongTensor] = None,
            cross_attention_media_position_ids: Optional[torch.LongTensor] = None,
            text_time: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert (
            self.language_model.initialized_mixin
        ), "Mixin layers are not initialized. Please call `init_mixin` first."

        assert (
            self.language_model._use_cached_vision_x or pixel_values is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        # During the training process, the forward method is called. 
        # Since training only performs a single-step inference at a time, there is no need to use cached media.
        for layer in self.language_model._get_decoder_layers():
            layer.condition_use_cached_media(False)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_batch_size = pixel_values.shape[0]

        vision_x = vit_embeds.unsqueeze(0)
        _, patch_n, patch_token_n, _ = vision_x.shape
        vision_x = rearrange(vision_x, "b t n d -> b (t n) d")

        if self.language_model._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                pixel_values is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.language_model.is_conditioned()
        else:
            for i, layer in enumerate(self.language_model._get_decoder_layers()):
                if (i+1) % self.mixin_every_n_layers == 0:
                    layer.condition_vis_x(vision_x)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        vit_embeds = vit_embeds[image_flags == 1]
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        ## To support the Training Data Packing strategy in cross attention
        ## Note: Currently, only flash attention and torch.SDPA implementations are supported.
        ## Example:
        ## 1 1 1 0 0 0
        ## 1 1 1 0 0 0
        ## 1 1 1 0 0 0
        ## 0 0 0 1 1 1
        ## 0 0 0 1 1 1
        ## 0 0 0 1 1 1
        media_attn_mask = None
        cu_seqlens_q = None
        cu_seqlens_k = None
        if self.config.llm_config.attn_implementation == 'flash_attention_2':
            cu_seqlens_q = attention_mask
            cu_seqlens_k = torch.cat((torch.tensor([[0]], device=seq_imgs.device, dtype=seq_imgs.dtype), (seq_imgs * patch_token_n).cumsum(dim=-1)), dim=-1).to(attention_mask.dtype)
        else:
            seq_cnt = seq_imgs[0].size(0)
            seq_to_media_mask_1d = torch.zeros((seq_cnt, patch_n * patch_token_n))

            cum_sum = 0
            for i in range(seq_cnt):
                current_lens = seq_imgs[0, i].item() * patch_token_n
                seq_to_media_mask_1d[i, cum_sum: cum_sum + current_lens] = 1
                cum_sum += current_lens

            seq_to_media_mask = torch.cat([repeat(cu, 'i -> b i', b=(attention_mask[0,i+1] - attention_mask[0,i])) for i, cu in enumerate(seq_to_media_mask_1d)], dim=0)
            seq_to_media_mask = seq_to_media_mask.to(input_ids.device).unsqueeze(0).unsqueeze(0)

            media_attn_mask = seq_to_media_mask.bool() 
        
        if not self.language_model._use_cached_vision_x:
            self._condition_attn_mask_and_pos_ids(media_attn_mask, cross_attention_media_position_ids, position_ids, text_time, cu_seqlens_q, cu_seqlens_k)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        kwargs = {}
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
                **kwargs
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def chat(self, tokenizer, pixel_values, question, generation_config, target_aspect_ratio=None, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             IMAGE_START_TOKEN_ID = 92544, IMAGE_END_TOKEN_ID = 92545, verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id

        position_ids=None
        cross_attention_media_position_ids=None
        
        position_bias = torch.where(input_ids[0] == IMAGE_START_TOKEN_ID)[0] + 1
        img_start_idx = torch.where(input_ids[0] == IMAGE_START_TOKEN_ID)[0].tolist()
        img_end_idx = torch.where(input_ids[0] == IMAGE_END_TOKEN_ID)[0]
        if target_aspect_ratio is not None:
            # Use RoPE-DHR
            position_ids = torch.tensor([])
            seq_lens = input_ids[0].shape[0]

            cross_attention_media_position_ids = []
            cum_tile_lens = 0
            for i in range(len(position_bias)):
                cur_position_bias = position_bias[i].item()
                cur_aspect_ratio = target_aspect_ratio[i]
                cur_cross_attention_media_position_ids = calculate_subimage_indices(cur_aspect_ratio[0], cur_aspect_ratio[1], (cur_position_bias - cum_tile_lens))
                cross_attention_media_position_ids.extend(cur_cross_attention_media_position_ids)

                if i == 0:
                    position_ids = torch.concat((torch.arange(cur_position_bias), torch.tensor(cur_cross_attention_media_position_ids)))
                else:
                    position_ids = torch.concat((position_ids, torch.arange(img_end_idx[i-1], cur_position_bias) - cum_tile_lens, torch.tensor(cur_cross_attention_media_position_ids)))
                
                cum_tile_lens += THUMBNAIL_TOKEN_LENGTH * (num_patches_list[i] - 1)

            position_ids = torch.concat((position_ids, torch.arange(img_end_idx[-1], seq_lens) - cum_tile_lens)).unsqueeze(0)

            cross_attention_media_position_ids = torch.tensor(cross_attention_media_position_ids).unsqueeze(0)
        else:
            # Use Original RoPE
            position_ids = torch.arange(
                input_ids.shape[1]
            )

            cross_attention_media_position_ids = []
            for i in range(len(img_start_idx)):
                cross_attention_media_position_ids.append(position_ids[img_start_idx[i]+1:img_end_idx[i]])
            cross_attention_media_position_ids = torch.cat(cross_attention_media_position_ids, dim=0).unsqueeze(0)

            position_ids = position_ids.unsqueeze(0)

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_patches_list=num_patches_list,
            cross_attention_media_position_ids=cross_attention_media_position_ids,
            **generation_config
        )
        
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            num_patches_list: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.FloatTensor] = None,
            cross_attention_media_position_ids: Optional[torch.FloatTensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        assert self.img_context_token_id is not None
        for layer in self.language_model._get_decoder_layers():
            layer.condition_use_cached_media(True)
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            self.language_model._use_cached_vision_x = True
            
            vision_x = rearrange(vit_embeds, "t n d -> (t n) d").unsqueeze(0)

            for i, layer in enumerate(self.language_model._get_decoder_layers()):
                if (i+1) % self.mixin_every_n_layers == 0:
                    layer.condition_vis_x(vision_x)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        self._condition_attn_mask_and_pos_ids(None, cross_attention_media_position_ids, position_ids, None)

        kwargs = {}
        kwargs['position_ids'] = position_ids
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
            **kwargs,
        )

        self.language_model.clear_conditioned_layers()
        self.language_model._use_cached_vision_x = False
        for layer in self.language_model._get_decoder_layers():
            layer.condition_use_cached_media(False)
        return outputs
