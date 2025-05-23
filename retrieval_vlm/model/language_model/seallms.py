import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from easydict import EasyDict
logger = logging.getLogger(__name__)

class SeaLLMs(nn.Module):
    def __init__ (self, config: EasyDict):
        super(SeaLLMs, self).__init__()
        self._validate_config(config)
        self.config = config

        # Decide the device map
        self.device_map = self._decide_device_map()

        # Set compute dtype
        try:
            self.compute_dtype = torch.bfloat16
            _ = torch.zeros(1, dtype=torch.bfloat16)
            logger.info("Using bfloat16 for LLM")
        except RuntimeError:
            self.compute_dtype = torch.float32
            logger.warning("bfloat16 not supported, falling back to float32. This might be less stable.")

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            device_map=self.device_map,
            torch_dtype=self.compute_dtype,
            output_hidden_states=True,
            output_attentions=True,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """
            Forward pass for the SeaLLMs model.

            Args:
                input_ids (torch.Tensor): The input ids of the model.
                shape: (batch_size, sequence_length)

                attention_mask (torch.Tensor): The attention mask of the model.
                shape: (batch_size, sequence_length)

            Returns:
                torch.Tensor: The output tensor.
                shape: (batch_size, sequence_length, hidden_size)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            return_dict_in_generate=True,
            **kwargs,
        )

        return outputs.hidden_states[-1]
    
    def _decide_device_map(self):
        """Decide device map based on available hardware"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 2:
                logger.info(
                    f"Found {gpu_count} GPUs. Using 'balanced' device map.")
                return "balanced"
            else:
                logger.info("Found 1 GPU. Using 'auto' device map.")
                return "auto"
        else:
            logger.info("No CUDA GPUs found. Using 'cpu' device map.")
            return "cpu"

    def _validate_config(self, config):
        required_fields = ['name']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"SeaLLMs config missing required field: {field}")

    @property
    def dim(self) -> int:
        """
            Get the dimension of the output tensor.
        """
        return self.model.config.hidden_size
    
    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype
    
    @property
    def device(self) -> torch.device:
        return self.model.device
        
