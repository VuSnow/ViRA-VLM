import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms
import requests


class EVA02VisionTower(nn.Module):
    def __init__(
        self,
        vision_tower_name='timm/eva02_enormous_patch14_clip_224',
        pretrained=True,
        select_feature='patch',
        select_layer=-2,
        image_size=224,
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
        # self.blocks = self.vision_tower.blocks
        # self.path_embed = self.vision_tower.patch_embed
        # self.norm = self.vision_tower.norm
        # self.pos_embed = self.vision_tower.pos_embed
        # self.cls_token = self.vision_tower.cls_token
        # self.pos_drop = self.vision_tower.pos_drop
        self.image_processor = transforms.Compose([
            transforms.Resize(
                self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        if not delay_load or self.unfreeze_vision_tower:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        self.vision_tower = create_model(
            model_name=self.vision_tower_name,
            pretrained=self.pretrained,
            out_indices=(self.select_layer,),
            # features_only=True,
        )

        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def preprocess_image(self, image):
        return self.image_processor(image).unsqueeze(0)

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            raise RuntimeError(
                "Vision tower is not loaded. Call load_model() first.")

        image_tensor = self.vision_tower(images)
        x = self.vision_tower.patch_embed(image_tensor)
        x = x.flatten(2).transpose(1, 2)

        return x
