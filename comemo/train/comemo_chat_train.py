import gc
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import orjson as json
import torch
import torch.distributed as dist
import transformers
from comemo.dist_utils import init_dist
from comemo.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from comemo.model.comemo_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          MixinConfig,
                                          CoMemoChatConfig,
                                          CoMemoChatModel,)
from comemo.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_train_dataloader, replace_train_sampler)
from comemo.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from comemo.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    find_closest_aspect_ratio, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
from comemo.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments, 
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mixin_config_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained comemo model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    unfreeze_mixin: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the mixin layers. Default is False.'},
    )
    freeze_gate: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the gates in mixin layers. Default is False.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT model. Default is 0.'},
    )
    ps_version: str = field(
        default='v1',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is `v1`.'
                          'Please use `v2` to fix the bug of transposed image.'}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=224,
        metadata={'help': 'Set the desired size for the image. Default is 224.'},
    )
    down_sample_ratio: Optional[float] = field(
        default=1.0,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 1.0.'},
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internvl_zh', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic image size.'},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image.'},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    max_num_frame: Optional[int] = field(
        default=24,
        metadata={'help': 'Max number of frames.'}
    )
    neftune_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'The noise_alpha value for NEFTune. Default is None.'},
    )
    normalize_type: Optional[str] = field(
        default='imagenet',
        metadata={'help': 'The normalize type for the image. Default is imagenet.'},
    )
    use_packed_ds: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for training. Default is False.'},
    )
    num_images_expected: Optional[int] = field(
        default=12,
        metadata={'help': 'The maximum number of images per packed sample. Default is 12.'},
    )
    max_packed_tokens: Optional[int] = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: Optional[int] = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: Optional[int] = field(
        default=1000,
        metadata={'help': 'The log frequence of the packed dataset. Default is 1000.'},
    )
    strict_mode: Optional[bool] = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: Optional[str] = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is `token`'},
    )
    loss_reduction_all_gather: Optional[bool] = field(
        default=False,
        metadata={'help': 'Whether to all gahter when loss reduction. Default is False'},
    )

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio

ORIGINAL_SIZE = 16
THUMBNAIL_TOKEN_LENGTH = 256
IMAGE_START_TOKEN_ID = 92544
IMAGE_END_TOKEN_ID = 92545
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

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        max_num_frame=24,
        min_num_frame=4,
        sampling_method='rand',
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # used for quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        total_ranks = torch.distributed.get_world_size()
        self.total_ranks = total_ranks
        current_rank = torch.distributed.get_rank()
        basename = os.path.basename(meta['annotation']).replace('.jsonl', '')
        data_dir = os.path.join(os.path.dirname(meta['annotation']), basename)
        data_dir = data_dir.replace('metas/', 'metas/cache/')
        # os.makedirs(data_dir, exist_ok=True)

        if os.environ.get('DEBUG_FLAG', False):
            repeat_time = 0.1

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
        if repeat_time < 1:
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        else:
            self.raw_data = self.raw_data * int(repeat_time)
        total_lines = len(self.raw_data)
        logger.info(f'total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}')
        lines_per_rank = total_lines // total_ranks  # 每个rank分得的行数
        lines_per_rank = max(1, lines_per_rank)
        start_line = lines_per_rank * current_rank  # 当前rank开始的行数
        end_line = start_line + lines_per_rank  # 当前rank结束的行数
        self.raw_data = self.raw_data[start_line:end_line]  # 读取当前rank对应的行

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        if self.group_by_length:
            self.conv2length = {}  # using dict to speedup the calculation of token length
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # use precomputed length if exists
                else:
                    # compute token length using tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    def __len__(self):
        # if not self.use_packed_ds:
        #     return len(self.raw_data) * self.total_ranks
        # else:
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        if data_item['image'].startswith('s3://'):
            image_path = self.root + data_item['image']
        else:
            image_path = os.path.join(self.root, data_item['image'])
        if self.tcs_loader is not None:
            image = self.tcs_loader(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        if self.dynamic_image_size:
            images, aspect_ratio = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Construct position IDs for cross-attention
        input_ids = ret['input_ids'][0]
        img_start_idx = torch.where(input_ids == IMAGE_START_TOKEN_ID)[0].item()
        position_bias = img_start_idx + 1
        cross_attention_media_position_ids = calculate_subimage_indices(aspect_ratio[0], aspect_ratio[1], position_bias)
        cross_attention_media_position_ids = torch.tensor(cross_attention_media_position_ids)
        text_time = torch.zeros_like(ret['input_ids'][0])

        # Construct updated position IDs for input sequence
        img_end_idx = torch.where(input_ids == IMAGE_END_TOKEN_ID)[0].item()
        tile_lens = img_end_idx - position_bias - 256
        seq_lens = input_ids.shape[0]
        position_ids = torch.concat((torch.arange(position_bias), cross_attention_media_position_ids, torch.arange(img_end_idx, seq_lens) - tile_lens), dim=0).unsqueeze(0)

        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        ####

        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            seq_imgs=torch.tensor([pixel_values.size(0)], dtype=torch.long),
            cross_attention_media_position_ids=cross_attention_media_position_ids,
            text_time=text_time,
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        images, num_tiles, aspect_ratios = [], [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            if image_path.startswith('s3://'):
                image_path = self.root + image_path
            else:
                image_path = os.path.join(self.root, image_path)
            if self.tcs_loader is not None:
                image = self.tcs_loader(image_path)
            else:
                image = Image.open(image_path).convert('RGB')
            if self.dynamic_image_size:
                image, aspect_ratio = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch // num_image,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                aspect_ratios.append(aspect_ratio)
                num_tiles.append(len(image))
            else:
                images.append(image)
                num_tiles.append(1)

        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
            
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess

        if self.template_name == 'internlm2-flamingo-chat':
            num_image_tokens = [self.num_image_token * 1 for num_tile in num_tiles]
        else:
            num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        ### Construct position IDs for cross-attention
        input_ids = ret['input_ids'][0]
        img_start_idx = torch.where(input_ids == IMAGE_START_TOKEN_ID)[0]
        position_bias = img_start_idx + 1
        imgs_end_idx = torch.where(input_ids == IMAGE_END_TOKEN_ID)[0]
        position_ids = torch.tensor([])
        seq_lens = input_ids.shape[0]

        cross_attention_media_position_ids = []
        cum_tile_lens = 0
        for i in range(len(position_bias)):
            cur_position_bias = position_bias[i].item()
            cur_aspect_ratio = aspect_ratios[i]
            cur_cross_attention_media_position_ids = calculate_subimage_indices(cur_aspect_ratio[0], cur_aspect_ratio[1], (cur_position_bias - cum_tile_lens))
            cross_attention_media_position_ids.extend(cur_cross_attention_media_position_ids)

            if i == 0:
                position_ids = torch.concat((torch.arange(cur_position_bias), torch.tensor(cur_cross_attention_media_position_ids)))
            else:
                position_ids = torch.concat((position_ids, torch.arange(imgs_end_idx[i-1], cur_position_bias) - cum_tile_lens, torch.tensor(cur_cross_attention_media_position_ids)))

            cum_tile_lens += (imgs_end_idx[i] - cur_position_bias - 256).item()
            
        position_ids = torch.concat((position_ids, torch.arange(imgs_end_idx[-1], seq_lens) - cum_tile_lens)).unsqueeze(0)
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        cross_attention_media_position_ids = torch.tensor(cross_attention_media_position_ids)
        text_time = torch.zeros_like(ret['input_ids'][0])
        ####

        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            seq_imgs=torch.tensor([sum(num_tiles)], dtype=torch.long),
            cross_attention_media_position_ids=cross_attention_media_position_ids, 
            text_time=text_time
        )
        return ret

    def video_get_item(self, data_item):
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)

        clip = data_item.get('clip', None)
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=clip)

        special_tokens = '\n'.join(['Frame{}:<image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens)

        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess

        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        ### Construct position IDs for cross-attention
        input_ids = ret['input_ids'][0]
        position_bias = torch.where(input_ids == IMAGE_START_TOKEN_ID)[0] + 1
        imgs_end_idx = torch.where(input_ids == IMAGE_END_TOKEN_ID)[0]
        position_ids = torch.tensor([])
        seq_lens = input_ids.shape[0]

        cross_attention_media_position_ids = []
        for i in range(len(position_bias)):
            cur_position_bias = position_bias[i].item()
            cur_cross_attention_media_position_ids = calculate_subimage_indices(0, 0, cur_position_bias)
            cross_attention_media_position_ids.extend(cur_cross_attention_media_position_ids)

            if i == 0:
                position_ids = torch.concat((torch.arange(cur_position_bias), torch.tensor(cur_cross_attention_media_position_ids)))
            else:
                position_ids = torch.concat((position_ids, torch.arange(imgs_end_idx[i-1], cur_position_bias), torch.tensor(cur_cross_attention_media_position_ids)))

        position_ids = torch.concat((position_ids, torch.arange(imgs_end_idx[-1], seq_lens))).unsqueeze(0)
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        cross_attention_media_position_ids = torch.tensor(cross_attention_media_position_ids)
        text_time = torch.zeros_like(ret['input_ids'][0])
        ####

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            seq_imgs=torch.tensor([num_patches], dtype=torch.long),
            cross_attention_media_position_ids=cross_attention_media_position_ids,
            text_time=text_time
        )
        return ret

    def pure_text_get_item(self, data_item):
        image = Image.new('RGB', (224, 224), (255, 255, 255))
        images, aspect_ratio = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-flamingo-chat':
            preprocess_function = preprocess_internlm
            num_patches = 1
        elif self.template_name.startswith('internlm2'):
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        else:
            preprocess_function = preprocess
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        ### Construct position IDs for cross-attention
        cross_attention_media_position_ids = calculate_subimage_indices(0, 0, 0)
        cross_attention_media_position_ids = torch.tensor(cross_attention_media_position_ids)
        text_time = torch.ones_like(ret['input_ids'][0])
        ####
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            seq_imgs=torch.tensor([1], dtype=torch.long),
            cross_attention_media_position_ids=cross_attention_media_position_ids, 
            text_time=text_time,
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            num_worker_per_rank = self.num_workers // self.total_ranks
            self.raw_data = self.raw_data[self.worker_id % num_worker_per_rank::num_worker_per_rank]
            gc.collect()
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            raise NotImplementedError

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name)
                sys.stdout.flush()
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]

def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=6,
    max_num_frame=24,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []

    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    if os.environ.get('DEBUG_FLAG', False):
        log_level = transformers.logging.DEBUG
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if os.environ.get('no_tcs', False):
        tcs_loader = None
    else:
        tcs_loader = TCSLoader('/mnt/petrelfs/suweijie/workspace_ls/petreloss.conf') if has_tcs_loader else None

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_llama_attention_class()

    if model_args.model_name_or_path is not None:
        logger.info('Loading CoMemoChatModel...')
        config = CoMemoChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        if os.environ.get('DEBUG_FLAG', False):
            logger.info('DEBUG MODE turn on')
            config.llm_config.num_hidden_layers = 8
            config.vision_config.num_hidden_layers = 8

        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch

        # NOTE: do not initialize new params in `from_pretrained`
        model = CoMemoChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    else:
        mixin_config = MixinConfig.from_pretrained(model_args.mixin_config_path)
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate

        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        if os.environ.get('DEBUG_FLAG', False):
            llm_config.num_hidden_layers = 8
            llm = model_type(llm_config)
        else:
            llm = model_type.from_pretrained(
                model_args.llm_path, torch_dtype=torch.bfloat16,
                config=llm_config, trust_remote_code=True)
        logger.info('Building CoMemoChatConfig...')
        internvl_chat_config = CoMemoChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), 
            mixin_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size

        logger.info('Building CoMemoChatModel...')
        model = CoMemoChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id
    model.neftune_alpha = data_args.neftune_alpha

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    train_dataset = build_datasets(
        data_args, tokenizer, tcs_loader, model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch, max_num_frame=data_args.max_num_frame,
        normalize_type=data_args.normalize_type)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    if model_args.unfreeze_mixin:
        for layer in model.language_model.model.layers:
            if hasattr(layer, 'gated_cross_attn_layer') and layer.gated_cross_attn_layer is not None:
                for param in layer.gated_cross_attn_layer.parameters():
                    param.requires_grad = True

                if model_args.freeze_gate:
                    layer.gated_cross_attn_layer.attn_gate.requires_grad = False
                    layer.gated_cross_attn_layer.ff_gate.requires_grad = False

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator

    model.tokenizer = tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
