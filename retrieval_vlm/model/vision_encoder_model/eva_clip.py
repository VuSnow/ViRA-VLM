import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms
from easydict import EasyDict


class EvaClip(nn.Module):
    def __init__(self, config: EasyDict):
        super(EvaClip, self).__init__()
        self._validate_config(config)
        self.config = config
        self.model = create_model(
            model_name="timm/eva02_small_patch14_336.mim_in22k_ft_in1k",
            pretrained=self.config.pretrained,
            out_indices=(self.config.select_layer,),
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the EvaClip model.
            Args:
                image_tensor (torch.Tensor): The input image tensor. 
                shape: (batch_size, 3, height, width) ~ (batch_size, 3, height, width)

            Returns:
                torch.Tensor: The output tensor.
                shape: (batch_size, embed_dim)
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"Expected input to be a Tensor of shape [B, 3, H, W]")
        image_tensor = image_tensor.to(next(self.model.parameters()).device)

        features = self.model.forward_features(image_tensor)
        if self.config.select_feature == 'patch':
            return features[:, 1:, :]
        elif self.config.select_feature == 'cls_patch':
            return features
        else:
            raise ValueError(f"Invalid select_feature: {self.config.select_feature}")

    def _validate_config(self, config: EasyDict) -> None:
        """Validate the configuration"""
        required_fields = ['name', 'pretrained', 'select_layer', 'select_feature']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config must contain {field}")
        
        if config.select_feature not in ['patch', 'cls_patch']:
            raise ValueError(f"select_feature must be either 'patch' or 'cls_patch'")

    @property
    def dim(self) -> int:
        """
            Get the dimension of the output tensor.
        """
        if hasattr(self.model, "embed_dim"):
            return self.model.embed_dim
        elif hasattr(self.model, "num_features"):
            return self.model.num_features
        else:
            raise AttributeError(
                "Cannot determine embed_dims from vision_tower.")

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype