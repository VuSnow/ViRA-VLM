import torch
import torch.nn as nn
from timm import create_model
from easydict import EasyDict


class EvaClip(nn.Module):
    def __init__(self, config: EasyDict):
        super(EvaClip, self).__init__()
        self.config = config
        self.model = create_model(
            model_name=self.config['name'],
            pretrained=self.config['pretrained'],
            out_indices=(self.config['select_layer'],)
        )

    def forward(self, image_tensor: torch.Tensor):
        """
            Forward pass for the EvaClip model.
            Args:
                image_tensor (torch.Tensor): The input image tensor. 
                shape: (batch_size, 3, height, width) ~ (batch_size, 3, height, width)

            Returns:
                torch.Tensor: The output tensor.
                shape: (batch_size, patch_num, embed_dim)
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(
                f"Expected input to be a Tensor of shape [B, 3, H, W]")

        image_tensor = image_tensor.to(next(self.model.parameters()).device)
        features = self.model.forward_features(image_tensor)

        # SỬA LỖI: Dùng ['key'] thay vì .key
        if self.config['select_feature'] == 'patch':
            return features[:, 1:, :]
        elif self.config['select_feature'] == 'cls_patch':
            return features
        else:
            raise ValueError(
                f"Invalid select_feature: {self.config['select_feature']}")

    def _validate_config(self, config: EasyDict) -> None:
        """Validate the configuration"""
        required_fields = ['name', 'pretrained',
                           'select_layer', 'select_feature']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config must contain {field}")

        if config.select_feature not in ['patch', 'cls_patch']:
            raise ValueError(
                f"select_feature must be either 'patch' or 'cls_patch'")

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def dim(self):
        if hasattr(self.model, "embed_dim"):
            return self.model.embed_dim
        elif hasattr(self.model, "num_features"):
            return self.model.num_features
        else:
            raise AttributeError(
                "Cannot determine embed_dims from vision_tower.")
