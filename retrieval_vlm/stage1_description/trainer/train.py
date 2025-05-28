from typing import Optional
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from retrieval_vlm.model.metrics import MetricComputer
from retrieval_vlm.stage1_description.dataset.image_caption_dataset import ImageCaptionDataset
from retrieval_vlm.stage1_description.dataset.vqa_collator import VQADataCollator
from retrieval_vlm.stage1_description.model.generating_caption import GeneratingCaption
from retrieval_vlm.stage1_description.config.model_config import GeneratingCaptionConfig
import logging
logger = logging.getLogger(__name__)

class Train:
    def __init__(
        self,
        model: GeneratingCaption,
        train_dataset: ImageCaptionDataset,
        tokenizer: AutoTokenizer,
        transform: transforms.Compose,
        config: GeneratingCaptionConfig,
        eval_dataset: Optional[ImageCaptionDataset] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.config = config
        self.eval_dataset = eval_dataset if eval_dataset else None
        self.vqa_collator = VQADataCollator()
        self.train_args = TrainingArguments(
            output_dir=self.config.train_config.get("output_dir", "./checkpoints"),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=self.config.train_config.get("epochs", 3),
            learning_rate=1e-5,
            weight_decay=self.config.train_config.get("weight_decay", 0.01),
            logging_dir=self.config.train_config.get("logging_dir", "logs"),
            logging_steps=self.config.train_config.get("logging_steps", 100),
            save_steps=self.config.train_config.get("save_steps", 100),
            save_total_limit=self.config.train_config.get("save_total_limit", 1),
            save_strategy=self.config.train_config.get("save_strategy", "epoch"),
            eval_strategy=self.config.train_config.get("evaluation_strategy", "epoch"),
            metric_for_best_model="bleu",
            greater_is_better=True,
            load_best_model_at_end=True,
            gradient_accumulation_steps=8,
            bf16=True,
            disable_tqdm=False,
            dataloader_num_workers=0,
            optim="adamw_torch",
            report_to=None,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.vqa_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
            compute_metrics=MetricComputer(self.tokenizer),
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()