import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MixinConfig(PretrainedConfig):
    def __init__(
        self,
        mixin_every_n_layers=4,
        language_dim=4096,
        vision_dim=4096,
        head_dim=128,
        num_heads=16,
        intermediate_size=16384,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mixin_every_n_layers = mixin_every_n_layers
        self.language_dim=language_dim
        self.vision_dim=vision_dim
        self.head_dim=head_dim
        self.num_heads=num_heads
        self.intermediate_size=intermediate_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'mixin_config' in config_dict:
            config_dict = config_dict['mixin_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)