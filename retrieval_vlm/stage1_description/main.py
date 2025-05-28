from retrieval_vlm.stage1_description.trainer.train import Train
from retrieval_vlm.stage1_description.config.model_config import GeneratingCaptionConfig
from datasets import load_dataset
import yaml
import os


def main():
    dataset = load_dataset("lukebarousse/coco_caption", split="train")
    config = yaml.load(open("config.yaml", "r"))

if __name__ == "__main__":
    main()
