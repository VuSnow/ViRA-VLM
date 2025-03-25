import json
import os
import torch
import transformers
import subprocess
import zipfile
import requests
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


def unzip_file(abs_path):
    if not zipfile.is_zipfile(abs_path):
        print(f"Not a valid zip file: {abs_path}")
        return False

    try:
        print(f"📦 Extracting {abs_path} ...")
        with zipfile.ZipFile(abs_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(abs_path))
        print(f"Extracted to: {os.path.dirname(abs_path)}")
        os.remove(abs_path)
        print(f"Deleted zip file: {abs_path}")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def get_destination_path(destination):
    try:
        current_path = os.path.abspath(__file__)
    except NameError:
        current_path = os.getcwd()
    current_folder = os.path.dirname(current_path)
    des_folder = destination if os.path.isabs(
        destination) else os.path.join(current_folder, destination)
    return des_folder


def download_and_extract(url, destination="coco"):
    des_folder = get_destination_path(
        destination)

    if not os.path.isdir(des_folder):
        os.makedirs(des_folder)
    else:
        print(f"Folder exists. Downloading zip files in {des_folder}")

    file_name = url.split("/")[-1]
    zip_path = os.path.join(des_folder, file_name)
    print(f"full zip path: {zip_path}")

    if os.path.exists(zip_path):
        print(f"File already existed: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            print(f"Removing corrupted file and re-downloading: {file_name}")
            os.remove(zip_path)
        else:
            unzip_file(zip_path)

    if not os.path.exists(zip_path):
        print(f"Downloading {file_name} from {url}...")
        try:
            subprocess.run(["wget", url, "-O", zip_path], check=True)
            print(f"Downloaded {file_name} in {zip_path}")
        except subprocess.CalledProcessError:
            print(f"Failed to download {file_name} from {url}")
        unzip_file(zip_path)

# download_and_extract(
#     "http://images.cocodataset.org/annotations/annotations_trainval2017.zip")


def load_data(destination="coco", size="partial", output_path="training_data.json"):
    des_folder = get_destination_path(destination)

    anno_path = os.path.join(des_folder, "annotations",
                             "captions_train2017.json")
    output_path = get_destination_path(output_path)

    # print(f"output: {output_path}, anno: {anno_path}")
    with open(anno_path, "r") as f:
        data = json.load(f)
    if size == "partial":
        images = data["images"][0:10000]
    else:
        images = data["images"]
    print(images[0].keys())

    outputs = []
    for idx, item in enumerate(tqdm(images, desc="Loading image data in json file")):
        image_id = item["id"]
        image_filename = item["file_name"]
        image_url = item["coco_url"]
        image_flickr = item["flickr_url"]
        question = "Describe this image thoroughly, including all visible objects, colors, actions, people, background elements, and their relationships. Be as descriptive and exhaustive as possible."
        answer = ""

        outputs.append({
            "id": image_id,
            "filename": image_filename,
            "coco_url": image_url,
            "flickr_url": image_flickr,
            "question": question,
            "answer": answer
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Save {len(outputs)} image information to {output_path}")
    return outputs


def generate_description(image_dict, model, processor, device):
    image_url = image_dict["coco_url"]
    question = image_dict["question"]
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        raw_image = Image.open(requests.get(
            image_url, stream=True).raw).convert("RGB")
    except Exception as e:
        print(f"Error fetching or opening image from {image_url}: {e}")
        return ""

    inputs = processor(images=raw_image, text=question,
                       return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=1024,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generating detailed image descriptions using InstructBLIP")
    parser.add_argument("--folder_path", type=str,
                        default="coco", help="Path to save coco dataset")
    parser.add_argument("--size", type=str, choices=[
                        "partial", "all"], default="partial", help="Process all or just 10000 images")
    parser.add_argument("--output", type=str,
                        default="training_data.json", help="Output file name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl")
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl")
    model.to(device)
    print("InstructBLIP model running on:", device)

    annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    download_and_extract(annotation_url, args.folder_path)

    images = load_data(destination=args.folder_path,
                       size=args.size, output_path=args.output)

    output_path = get_destination_path(args.output)

    for idx, image in enumerate(tqdm(images, desc="Generating detailed image description")):
        description = generate_description(image, model, processor, device)
        image["answer"] = description

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(images, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(images)} image descriptions to {output_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
