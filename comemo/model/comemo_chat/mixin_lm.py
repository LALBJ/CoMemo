"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch.nn as nn
from .helpers import GatedCrossAttentionBlock
from .utils import getattr_recursive, setattr_recursive

from typing import List, Optional, Tuple, Union
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.utils import ModelOutput

import torch
class MixinLayer(nn.Module):
    """
    MixinLayer is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(
        self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False
    ):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = (
                gradient_checkpointing
            )
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x
    
    def condition_media(self, media, text_position_ids):
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer.media = media
            self.gated_cross_attn_layer.cross_attn.text_position_ids = text_position_ids
    
    def condition_use_cached_media(self, use_cached_media):
        self.use_cached_media = use_cached_media

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        # Cross attention
        if self.gated_cross_attn_layer is not None and self.vis_x is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            hidden_states = self.gated_cross_attn_layer(
                hidden_states,
                self.vis_x,
                use_cached_media=self.use_cached_media,
            )

        # Normal decoder layer
        hidden_states = self.decoder_layer(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_value=past_key_value, 
            output_attentions=output_attentions, 
            use_cache=use_cache,
            **kwargs
        )
        return hidden_states


class LMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_mixin(
        self,
        config,
        gradient_checkpointing,
    ):
        """
        Initialize Mixin by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.old_decoder_blocks = self._get_decoder_layers()
        mixin_every_n_layers = config.mixin_every_n_layers
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(config)
                if (layer_idx + 1) % mixin_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )

        self.init_mixin_layers(gradient_checkpointing)
        self.old_decoder_blocks = None
        self.gated_cross_attn_layers = None
        self.initialized_mixin = True
        self._use_cached_vision_x = False

    def init_mixin_layers(self, gradient_checkpointing):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    MixinLayer(
                        gated_cross_attn_layer, decoder_layer, gradient_checkpointing
                    )
                    for gated_cross_attn_layer, decoder_layer in zip(
                        self.gated_cross_attn_layers, self.old_decoder_blocks
                    )
                ]
            )
        )

    def forward(self, position_ids=None,**kwargs
        ):
        if not self.initialized_mixin:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        kwargs["position_ids"] = position_ids
        return super().forward(**kwargs)  # Call the other parent's forward method


    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        # To support RoPE-DHR's position_ids calculation method
        if model_kwargs['past_key_values'] and 'position_ids' in model_kwargs:
            new_pos_ids = model_kwargs['position_ids'][:, -1:] + 1
            model_kwargs['position_ids'] = new_pos_ids

        return model_kwargs


    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_use_cached_media(False)
            layer.condition_media(None, None)
