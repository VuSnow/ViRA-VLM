import torch
import torch.nn as nn
from timm import create_model


class EVA02VisionTower(nn.Module):
    def __init__(
        self,
        vision_tower_name='timm/eva02_enormous_patch14_clip_224',
        pretrained=True,
        select_feature='patch',
        select_layer=-2,
        image_size=336,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        delay_load=False,
        unfreeze_vision_tower=False
    ):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.pretrained = pretrained
        self.unfreeze_vision_tower = unfreeze_vision_tower

        if not delay_load or self.unfreeze_vision_tower:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        self.vision_tower = create_model(
            model_name=self.vision_tower_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=(self.select_layer,)
        )
