from PIL import Image, UnidentifiedImageError
import io
import re
import os


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


def find_best_checkpoint(output_dir):
    checkpoints = []
    pattern = re.compile(r"checkpoint-(\d+)")
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if os.path.isdir(full_path):
            match = pattern.match(name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, full_path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {output_dir}")
    best_ckpt = max(checkpoints, key=lambda x: x[0])[1]
    return best_ckpt
