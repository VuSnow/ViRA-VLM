import torch
import torch.nn as nn
from timm import create_model
from torchvision import transforms


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
        device=None,
        use_data_parallel=False,
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
        self.use_data_parallel = use_data_parallel
        # Decide device
        self._device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.image_processor = transforms.Compose([
            transforms.Resize(
                self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        if not delay_load or self.unfreeze_vision_tower:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, skipping.')
            return

        model = create_model(
            model_name=self.vision_tower_name,
            pretrained=self.pretrained,
            out_indices=(self.select_layer,),
        )
        if self.use_data_parallel and torch.cuda.device_count() > 1:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
            model = nn.DataParallel(model)

        model = model.to(self._device)
        model.eval()
        model.requires_grad_(self.unfreeze_vision_tower)

        self.vision_tower = model
        self.is_loaded = True

    # def preprocess_image(self, images):
    #     # Nếu ảnh đầu vào là list, chuyển thành tensor
    #     if isinstance(images, list):
    #         images = torch.stack([self.image_processor(
    #             image.convert('RGB')) for image in images])
    #     else:
    #         # Nếu ảnh đã là tensor thì tiến hành preprocessing cho toàn bộ batch
    #         images = self.image_processor(images)  # [B, C, H, W]

    #     return images.to(self._device)

    def select_features(self, feature_forward):  # [B, 257, 1792]
        if self.select_feature == 'patch':
            # [B, 256, 1792] - delete cls_token
            image_features = feature_forward[:, 1:, :]
        elif self.select_feature == 'cls_patch':
            image_features = feature_forward        # [B, 257 , 1792]
        else:
            raise ValueError(
                f"Unknown select_feature: {self.select_feature}.")
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            raise RuntimeError(
                "Vision tower is not loaded. Call load_model() first.")

        if not isinstance(images, torch.Tensor):
            raise TypeError(
                "Expected input to be a Tensor of shape [B, 3, H, W]")

        model_device = next(self.vision_tower.parameters()).device
        images = images.to(device=model_device, dtype=next(
            self.vision_tower.parameters()).dtype)

        features = self.vision_tower.forward_features(images)  # [B, 257, 1792]
        return self.select_features(features)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def embed_dims(self):
        if not self.is_loaded:
            self.load_model()
        return self.vision_tower.num_features

    @property
    def transform(self):
        return self.image_processor
