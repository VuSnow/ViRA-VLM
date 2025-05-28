from transformers import PretrainedConfig
from easydict import EasyDict

class GeneratingCaptionConfig(PretrainedConfig):
    model_type = "generating_caption"
    def __init__(
        self,
        vision_encoder_config: EasyDict = None,
        cross_attention_config: EasyDict = None,
        llm_config: EasyDict = None,
        lora: EasyDict = None,
        train_config: EasyDict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_encoder_config = vision_encoder_config or {}
        self.cross_attention_config = cross_attention_config or {}
        self.llm_config = llm_config or {}
        self.lora = lora or {}
        self.train_config = train_config or {}
