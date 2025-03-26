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
        unfreeze_vision_tower=False,
        device=None
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
        self.image_processor = transforms.Compose([
            transforms.Resize(
                self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        if device is not None:
            self._device = device
        else:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not delay_load or self.unfreeze_vision_tower:
            self.load_model()

        self.blocks = None
        self.patch_embed = None
        self.norm = None
        self.pos_embed = None
        self.cls_token = None
        self.pos_drop = None
        if self.is_loaded:
            self.blocks = self.vision_tower.blocks
            self.path_embed = self.vision_tower.patch_embed
            self.norm = self.vision_tower.norm
            self.pos_embed = self.vision_tower.pos_embed
            self.cls_token = self.vision_tower.cls_token
            self.pos_drop = self.vision_tower.pos_drop

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
        self.vision_tower = self.vision_tower.to(self._device)
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def preprocess_image(self, image):
        return self.image_processor(image)

    def select_features(self, patch_tokens):
        if self.select_feature == 'patch':
            image_features = patch_tokens
        elif self.select_feature == 'cls_patch':
            if self.cls_token is None:
                raise ValueError(
                    "cls_token is not defined. Model might not be loaded.")
            batch_size = patch_tokens.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            image_features = torch.cat((cls_tokens, patch_tokens), dim=1)
        else:
            raise ValueError(
                f"Unexpected select feature: {self.select_feature}")

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            raise RuntimeError(
                "Vision tower is not loaded. Call load_model() first.")

        if isinstance(images, list):
            images = torch.stack([self.preprocess_image(img)
                                 for img in images], dim=0)

        images = images.to(self._device)

        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)

        x = self.select_feature(x)
        return x

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower._device
