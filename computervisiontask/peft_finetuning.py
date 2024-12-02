import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from datasets import Dataset, load_dataset
from matplotlib import pyplot as plt
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
from peft import LoraConfig, PeftModel, get_peft_model,PeftConfig
import requests
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset_name: str = "pcuenq/oxford-pets"
    model_name: str = "vit-base-patch16-224"
    train_size: float = 0.8
    batch_size: int = 128
    learning_rate: float = 5e-3
    num_epochs: int = 5
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_total_limit: int = 2
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    metric_for_best_model: str = "accuracy"
    report_to: str = "tensorboard"
    fp16: bool = True
    push_to_hub: bool = True
    remove_unused_columns: bool = False
    load_best_model_at_end: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "value"]
    )
    lora_bias: str = "none"
    lora_modules_to_save: List[str] = field(default_factory=lambda: ["classifier"])
    inference_image_url: str = (
        "https://huggingface.co/datasets/alanahmet/LoRA-pets-dataset/resolve/main/shiba_inu_136.jpg"
    )
    repository_name: str = "alanahmet/vit-finetuned-lora-oxford-pets"


class OxfordPetsClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.dataset: Optional[Dict[str, Dataset]] = None
        self.classes: Optional[List[str]] = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.processor: Optional[AutoImageProcessor] = None
        self.model: Optional[AutoModelForImageClassification] = None
        self.trainer: Optional[Trainer] = None
        self.accuracy = evaluate.load("accuracy")
        logger.info("OxfordPetsClassifier initialized with config: %s", config)

    def load_data(self) -> None:
        try:
            logger.info("Loading dataset: %s", self.config.dataset_name)
            self.dataset = load_dataset(self.config.dataset_name)
            if self.dataset is None:
                raise ValueError("Dataset loading returned None.")
            self.classes = self.dataset["train"].unique("label")
            if not self.classes:
                raise ValueError("No classes found in the dataset.")
            self.label2id = {label: idx for idx, label in enumerate(self.classes)}
            self.id2label = {idx: label for idx, label in enumerate(self.classes)}
            logger.info("Dataset loaded successfully with %d classes.", len(self.classes))
        except Exception as e:
            logger.error("Failed to load dataset: %s", e)
            raise

    def show_samples(self, ds: Dataset, rows: int = 2, cols: int = 4) -> None:
        try:
            samples = ds.shuffle(seed=42).select(np.arange(rows * cols))
            fig = plt.figure(figsize=(cols * 4, rows * 4))
            for i in range(rows * cols):
                img = samples[i]["image"]
                label = samples[i]["label"]
                fig.add_subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(self.id2label.get(label, "Unknown"))
                plt.axis("off")
            plt.tight_layout()
            plt.show()
            logger.info("Displayed sample images.")
        except Exception as e:
            logger.error("Failed to display samples: %s", e)
            raise

    def preprocess_data(self) -> None:
        try:
            if not self.dataset:
                raise ValueError("Dataset not loaded.")
            logger.info("Splitting dataset into train and test sets.")
            self.dataset = self.dataset["train"].train_test_split(
                train_size=self.config.train_size, seed=42
            )
            logger.info(
                "Dataset split into train (%d) and test (%d).",
                len(self.dataset["train"]),
                len(self.dataset["test"]),
            )

            model_checkpoint = f"google/{self.config.model_name}"
            self.processor = AutoImageProcessor.from_pretrained(model_checkpoint)
            logger.info("Image processor loaded from checkpoint: %s", model_checkpoint)

            def transform(batch: Dict[str, Any]) -> Dict[str, Any]:
                batch["image"] = [x.convert("RGB") for x in batch["image"]]
                inputs = self.processor(
                    [x for x in batch["image"]],
                    return_tensors="pt",
                )
                inputs["labels"] = [self.label2id[y] for y in batch["label"]]
                return inputs

            self.dataset = self.dataset.with_transform(transform)
            logger.info("Data transformations applied.")

        except Exception as e:
            logger.error("Failed to preprocess data: %s", e)
            raise

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            pixel_values = torch.stack([x["pixel_values"] for x in batch])
            labels = torch.tensor([x["labels"] for x in batch])
            return {"pixel_values": pixel_values, "labels": labels}
        except Exception as e:
            logger.error("Collate function failed: %s", e)
            raise

    @staticmethod
    def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        try:
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=1)
            acc = evaluate.load("accuracy").compute(predictions=predictions, references=labels)
            return acc
        except Exception as e:
            logger.error("Metric computation failed: %s", e)
            raise

    def print_trainable_parameters(self, model: nn.Module) -> None:
        try:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            trainable_percent = 100 * trainable_params / all_params if all_params else 0
            logger.info(
                "Trainable params: %d || All params: %d || Trainable%%: %.2f",
                trainable_params,
                all_params,
                trainable_percent,
            )
        except Exception as e:
            logger.error("Failed to print trainable parameters: %s", e)
            raise

    def setup_model(self) -> None:
        try:
            model_checkpoint = f"google/{self.config.model_name}"
            self.model = AutoModelForImageClassification.from_pretrained(
                model_checkpoint,
                label2id=self.label2id,
                id2label=self.id2label,
                ignore_mismatched_sizes=True,
            )
            logger.info("Base model loaded from checkpoint: %s", model_checkpoint)
            self.print_trainable_parameters(self.model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.lora_bias,
                modules_to_save=self.config.lora_modules_to_save,
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA configuration applied to the model.")
            self.print_trainable_parameters(self.model)
        except Exception as e:
            logger.error("Failed to set up the model: %s", e)
            raise

    def configure_trainer(self) -> None:
        try:
            training_args = TrainingArguments(
                output_dir=f"{self.config.model_name}-finetuned-lora-oxford-pets",
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                num_train_epochs=self.config.num_epochs,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_steps=self.config.logging_steps,
                save_total_limit=self.config.save_total_limit,
                evaluation_strategy=self.config.evaluation_strategy,
                save_strategy=self.config.save_strategy,
                metric_for_best_model=self.config.metric_for_best_model,
                report_to=self.config.report_to,
                fp16=self.config.fp16,
                push_to_hub=self.config.push_to_hub,
                remove_unused_columns=self.config.remove_unused_columns,
                load_best_model_at_end=self.config.load_best_model_at_end,
            )
            logger.info("TrainingArguments configured.")

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metrics,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                tokenizer=self.processor,
            )
            logger.info("Trainer initialized.")
        except Exception as e:
            logger.error("Failed to configure trainer: %s", e)
            raise

    def train(self) -> None:
        if not self.trainer:
            raise ValueError("Trainer is not configured.")
        try:
            logger.info("Starting training.")
            self.trainer.train()
            logger.info("Training completed.")
        except Exception as e:
            logger.error("Training failed: %s", e)
            raise

    def push_model_to_hub(self) -> None:
        try:
            if not self.model:
                raise ValueError("Model is not initialized.")
            logger.info("Pushing model to hub: %s", self.config.repository_name)
            self.model.push_to_hub(self.config.repository_name)
            logger.info("Model successfully pushed to hub.")
        except Exception as e:
            logger.error("Failed to push model to hub: %s", e)
            raise

    def load_inference_model(self) -> PeftModel:
        try:
            logger.info("Loading inference model from hub: %s", self.config.repository_name)
            config = PeftConfig.from_pretrained(self.config.repository_name)
            base_model = AutoModelForImageClassification.from_pretrained(
                config.base_model_name_or_path,
                label2id=self.label2id,
                id2label=self.id2label,
                ignore_mismatched_sizes=True,
            )
            inference_model = PeftModel.from_pretrained(base_model, self.config.repository_name)
            logger.info("Inference model loaded successfully.")
            return inference_model
        except Exception as e:
            logger.error("Failed to load inference model: %s", e)
            raise

    def infer(self, model: PeftModel) -> None:
        try:
            logger.info("Performing inference on the sample image.")
            image = Image.open(requests.get(self.config.inference_image_url, stream=True).raw)
            encoding = self.processor(image.convert("RGB"), return_tensors="pt")
            logger.info("Image encoding shape: %s", encoding.pixel_values.shape)

            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits

            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = self.id2label.get(predicted_class_idx, "Unknown")
            logger.info("Predicted class: %s", predicted_class)
            print(f"Predicted class: {predicted_class}")
        except Exception as e:
            logger.error("Inference failed: %s", e)
            raise

    def run(self) -> None:
        try:
            self.load_data()
            self.show_samples(self.dataset["train"], rows=3, cols=5)
            self.preprocess_data()
            self.setup_model()
            self.configure_trainer()
            self.train()
            self.push_model_to_hub()
            inference_model = self.load_inference_model()
            self.infer(inference_model)
        except Exception as e:
            logger.error("An error occurred during execution: %s", e)
            raise


def main() -> None:
    config = Config()
    classifier = OxfordPetsClassifier(config)
    classifier.run()


if __name__ == "__main__":
    main()