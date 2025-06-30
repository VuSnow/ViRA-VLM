import os
import io
import sys
from PIL import Image, UnidentifiedImageError
from datasets import load_dataset, load_from_disk
from datasets.features import Image as DatasetImage

CLEANED_DATASET_PATH = "/home/user05/dungvm/stage1_description/dataset/viet_laion_gemini_vqa_cleaned_v2"


def check_image_validity(example, index):
    """
    This function checks the validity of an image and returns a dictionary containing an 'is_valid' flag.
    It is designed to be used with `dataset.map()`.
    
    Args:
        example (dict): A sample from the dataset.
        index (int): Index of the sample (provided by `with_indices=True`).
    
    Returns:
        dict: {'is_valid': True} if the image is valid, {'is_valid': False} if there's an error.
    """
    try:
        image_data = example.get('image')
        if image_data is None or image_data.get('bytes') is None:
            # print(f"Index {index}: Image data or bytes is None.") # Uncomment for debugging
            return {'is_valid': False}

        image_bytes = image_data['bytes']

        # Open image from bytes data to check
        with io.BytesIO(image_bytes) as img_byte_stream:
            img = Image.open(img_byte_stream)
            img.verify()  # Quick check for corrupted file

        # For more thorough checking, can reopen and read metadata
        with io.BytesIO(image_bytes) as img_byte_stream:
            img = Image.open(img_byte_stream)
            _ = img.getexif()

        # If no errors occur, the image is valid
        return {'is_valid': True}

    except Exception as e:
        # Catch ALL possible errors that can occur when processing a row.
        # Including UnicodeDecodeError, PIL.UnidentifiedImageError, OSError, etc.
        # This makes the mapping process extremely robust.
        # print(f"Index {index}: Caught an exception -> {type(e).__name__}: {e}", file=sys.stderr)
        return {'is_valid': False}


if os.path.exists(CLEANED_DATASET_PATH):
    print(f"Dataset already exists at {CLEANED_DATASET_PATH}")
else:
    print("Dataset đã làm sạch không tồn tại. Bắt đầu quá trình tải và làm sạch từ đầu.")

    print("Đang tải dataset gốc '5CD-AI/Viet-LAION-Gemini-VQA'...")
    dataset = load_dataset("5CD-AI/Viet-LAION-Gemini-VQA", split='train')

    # Rất quan trọng: Chuyển cột 'image' để không tự động decode, giữ lại dạng bytes
    dataset = dataset.cast_column("image", DatasetImage(decode=False))

    original_size = len(dataset)
    print(f"Tải xong dataset gốc. Số lượng mẫu ban đầu: {original_size}")

    # BƯỚC 2: MAP ĐỂ ĐÁNH DẤU CÁC HÀNG HỢP LỆ/LỖI
    print("\nBắt đầu giai đoạn MAP: Thêm cột 'is_valid' để đánh dấu ảnh lỗi...")
    # Sử dụng đa tiến trình để tăng tốc
    num_procs = os.cpu_count()
    print(f"Sử dụng {num_procs} tiến trình để map.")

    mapped_dataset = dataset.map(
        check_image_validity,
        with_indices=True,  # Cung cấp chỉ số cho hàm map
        num_proc=num_procs,
        desc="Đánh dấu ảnh hợp lệ"
    )

    print("Giai đoạn MAP hoàn tất.")

    # BƯỚC 3: FILTER DỰA TRÊN CỘT ĐÃ ĐÁNH DẤU
    print("\nBắt đầu giai đoạn FILTER: Loại bỏ các hàng có 'is_valid' là False...")

    # Lọc rất nhanh vì chỉ kiểm tra boolean
    cleaned_dataset = mapped_dataset.filter(
        lambda example: example['is_valid'],
        num_proc=num_procs,
        desc="Lọc bỏ ảnh lỗi"
    )

    print("Giai đoạn FILTER hoàn tất.")

    # BƯỚC 4: DỌN DẸP VÀ LƯU
    print("\nDọn dẹp dataset cuối cùng...")

    # Xóa cột tạm 'is_valid'
    cleaned_dataset = cleaned_dataset.remove_columns(['is_valid'])

    # Chuyển cột 'image' trở lại chế độ tự động decode để tiện sử dụng
    cleaned_dataset = cleaned_dataset.cast_column(
        "image", DatasetImage(decode=True))

    cleaned_size = len(cleaned_dataset)
    removed_count = original_size - cleaned_size
    print("\nQuá trình làm sạch hoàn tất.")
    print(f"Số lượng mẫu ban đầu: {original_size}")
    print(f"Số lượng mẫu sau khi làm sạch: {cleaned_size}")
    print(f"Số lượng mẫu bị lỗi đã loại bỏ: {removed_count}")

    print(f"\nĐang lưu dataset đã làm sạch vào '{CLEANED_DATASET_PATH}'...")
    cleaned_dataset.save_to_disk(CLEANED_DATASET_PATH)
    print("Lưu thành công!")

print("Dọn dẹp các file cache trung gian không còn cần thiết...")

# mapped_dataset và cleaned_dataset là các đối tượng dataset trung gian
# Chúng ta có thể xóa các file cache liên quan đến chúng
# Lưu ý: Điều này yêu cầu phiên bản datasets>=2.0.0
try:
    if 'mapped_dataset' in locals():
        mapped_dataset.cleanup_cache_files()
    if 'cleaned_dataset' in locals() and 'mapped_dataset' in locals() and cleaned_dataset is not mapped_dataset:
        # cleaned_dataset có thể trỏ đến cùng một đối tượng nếu không có gì bị lọc
        cleaned_dataset.cleanup_cache_files()

    # Dọn dẹp cache của dataset gốc ban đầu nếu bạn không cần nó nữa
    dataset.cleanup_cache_files()
    print("Dọn dẹp cache thành công.")

except AttributeError:
    print("Phiên bản 'datasets' của bạn có thể không hỗ trợ `cleanup_cache_files()`.")
    print("Hãy cân nhắc nâng cấp 'pip install -U datasets' hoặc dọn dẹp thủ công thư mục `~/.cache/huggingface/datasets`.")

print("\nDataset đã sẵn sàng để sử dụng.")
print(cleaned_dataset)
