import shutil
import torch
from torchvision import transforms
from PIL import Image
import requests
from easydict import EasyDict
import yaml
from datasets import load_from_disk
from huggingface_hub import login
from transformers import AutoTokenizer
import os

# Import các class model của bạn
# Đảm bảo file description_model.py nằm trong đường dẫn có thể import
from stage1_description.model.description_model import DescriptionModel, DescriptionModelConfig


def find_best_checkpoint(output_dir, best_step):
    """Tìm checkpoint tốt nhất dựa trên step đã biết."""
    ckpt_name = f"checkpoint-{best_step}"
    ckpt_path = os.path.join(output_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint '{ckpt_path}' không tồn tại. Vui lòng kiểm tra lại step.")
    return ckpt_path


# --- CONFIG ---
CONFIG_PATH = "/home/user05/dungvm/configs/configs.yaml"
OUTPUT_DIR = "/home/user05/dungvm/stage1_description/outputs/stage1"
BEST_STEP = 145
CHECKPOINT_PATH = find_best_checkpoint(OUTPUT_DIR, BEST_STEP)
print(f"Sử dụng checkpoint TỐT NHẤT được khuyến nghị: {CHECKPOINT_PATH}")

# Ảnh mèo và máy tính
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DATASET_PATH = "/home/user05/dungvm/stage1_description/dataset/viet_laion_gemini_vqa_cleaned_v2"

# !!! QUAN TRỌNG: Sử dụng chính xác PROMPT đã dùng trong training
PROMPT = "Mô tả chi tiết bức ảnh này từ hành động, vật thể cho đến cảm giác hoặc bối cảnh. Bao gồm mọi yếu tố như màu sắc, ánh sáng, bố cục, các đối tượng và sự tương tác giữa chúng, cảm xúc hoặc thông điệp mà bức ảnh truyền tải. Hãy chú ý đến cả các yếu tố nhỏ mà có thể mang lại chiều sâu cho bức ảnh. Chỉ miêu tả những gì bạn nhận diện được, không được bịa đặt, ảo tưởng. <|im_sep|>"


def load_image_from_url(url: str) -> Image.Image:
    """Tải ảnh từ URL."""
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None


def main():
    print("1. Loading configuration...")
    with open(CONFIG_PATH, "r") as f:
        configs = EasyDict(yaml.safe_load(f))

    # Load tokenizer từ checkpoint (đã được resize đúng vocab size)
    print(f"2. Loading tokenizer from checkpoint: {CHECKPOINT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Define image transform (phải giống hệt lúc training)
    image_transform = transforms.Compose([
        transforms.Resize(
            (configs.eva_vision_model.image_size,
             configs.eva_vision_model.image_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=configs.eva_vision_model.mean,
                             std=configs.eva_vision_model.std)
    ])

    print(f"3. Loading model from checkpoint: {CHECKPOINT_PATH}...")
    # Tạo config với vocab_size đúng từ tokenizer
    model_config = DescriptionModelConfig(**configs)
    # Đặt vocab_size đúng từ tokenizer đã resize
    model_config.vocab_size = len(tokenizer)
    # Khi load từ checkpoint, không cần load lại pretrained weights cho vision model
    model_config.eva_vision_model['pretrained'] = False

    model = DescriptionModel.from_pretrained(
        CHECKPOINT_PATH,
        config=model_config
    )
    print(f"Model loaded from checkpoint: {CHECKPOINT_PATH}")

    # Chuyển model sang chế độ eval và đưa lên GPU nếu có
    print("4. Moving model to device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    dataset = load_from_disk(DATASET_PATH)
    sample = dataset[0]
    image = sample['image']

    # Chuẩn bị ảnh
    # Thêm unsqueeze(0) để tạo batch dimension
    image_tensor = image_transform(
        image.convert("RGB")).unsqueeze(0).to(device)
    saved_image_path = os.path.join("/home/user05/dungvm", f"test_image.png")
    image.save(saved_image_path)

    print(f"6. Preparing prompt...")
    # Tokenize prompt và đưa lên device
    prompt_encoding = tokenizer(PROMPT, return_tensors="pt").to(device)

    print("\n7. Generating description...")
    try:
        # === THÊM DÒNG NÀY ĐỂ DEBUG ===
        # print(f"DEBUG: Shape của image_tensor trước khi generate: {image_tensor.shape}")
        # print(f"DEBUG: Kiểu dữ liệu của image_tensor: {image_tensor.dtype}")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_encoding.input_ids,
                attention_mask=prompt_encoding.attention_mask,
                image_tensor=image_tensor,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_token_len = prompt_encoding.input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print("\n" + "="*50)
        print("      GENERATED DESCRIPTION")
        print("="*50)
        print(text.strip())
        print("="*50)

    except Exception as e:
        print("\nEXCEPTION OCCURRED:", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
