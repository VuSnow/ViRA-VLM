from PIL import Image, UnidentifiedImageError
import io


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        all_params = sum(p.numel() for p in model.parameters())
        if param.requires_grad:
            trainable_params = sum(p.numel()
                                   for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(
        f"Trainable parameters percentage: {100 * trainable_params / all_params}")


def check_image_validity(example, index):
    try:
        image_bytes = example['image']['bytes']
        if image_bytes is None:
            return False
        image = Image.open(io.BytesIO(image_bytes))
        _ = image.getexif()
        return True
    except (UnicodeDecodeError, UnidentifiedImageError, OSError, TypeError, KeyError):
        return False
