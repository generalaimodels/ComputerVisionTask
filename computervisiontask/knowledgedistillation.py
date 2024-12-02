import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from matplotlib.axes import Axes
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset_name: str = "pcuenq/oxford-pets"
    teacher_model_name: str = "asusevski/vit-base-patch16-224-oxford-pets"
    image_processor_name: str = "google/vit-base-patch16-224"
    student_model_name: str = "WinKawaks/vit-tiny-patch16-224"
    output_dir: str = "./oxford-pets-vit"
    batch_size: int = 48
    eval_batch_size: int = 48
    num_train_epochs: int = 10
    learning_rate: float = 3e-4
    temperature: float = 5.0
    lambda_param: float = 0.9
    push_to_hub: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class OxfordPetsTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(self.config.device)
        self.dataset = self.load_dataset()
        self.id2label, self.label2id = self.create_label_mappings()
        self.processor = self.load_processor()
        self.teacher_model = self.load_teacher_model()
        self.transformed_dataset = self.transform_dataset()
        self.train_valid_test = self.split_dataset()
        self.base_model = self.initialize_student_model()
        self.training_args = self.configure_training_arguments()
        self.accuracy = evaluate.load("accuracy")
        self.collate_fn = self.get_collate_fn()

    def load_dataset(self) -> DatasetDict:
        try:
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name)
            logger.info("Dataset loaded successfully.")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.config.dataset_name}': {e}")
            sys.exit(1)

    def create_label_mappings(self) -> Tuple[Dict[int, str], Dict[str, int]]:
        try:
            logger.info("Creating label mappings.")
            labels = sorted(set(self.dataset["train"]["label"]))
            id2label = {idx: label for idx, label in enumerate(labels)}
            label2id = {label: idx for idx, label in enumerate(labels)}
            logger.info("Label mappings created successfully.")
            return id2label, label2id
        except Exception as e:
            logger.error(f"Error creating label mappings: {e}")
            sys.exit(1)

    def load_processor(self) -> ViTImageProcessor:
        try:
            logger.info(f"Loading image processor: {self.config.image_processor_name}")
            processor = ViTImageProcessor.from_pretrained(self.config.image_processor_name)
            logger.info("Image processor loaded successfully.")
            return processor
        except Exception as e:
            logger.error(f"Failed to load processor '{self.config.image_processor_name}': {e}")
            sys.exit(1)

    def load_teacher_model(self) -> AutoModelForImageClassification:
        try:
            logger.info(f"Loading teacher model: {self.config.teacher_model_name}")
            model = AutoModelForImageClassification.from_pretrained(
                self.config.teacher_model_name
            ).to(self.device)
            model.eval()
            logger.info("Teacher model loaded and set to evaluation mode.")
            return model
        except Exception as e:
            logger.error(f"Failed to load teacher model '{self.config.teacher_model_name}': {e}")
            sys.exit(1)

    def transform_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Convert image to RGB
            image = example["image"].convert("RGB")
            # Process image
            inputs = self.processor(image, return_tensors="pt")
            example["pixel_values"] = inputs["pixel_values"].squeeze()
            # Map label
            example["label"] = self.label2id[example["label"]]
            return example
        except Exception as e:
            logger.error(f"Error transforming example: {e}")
            raise

    def transform_dataset(self) -> DatasetDict:
        try:
            logger.info("Transforming dataset.")
            transformed = self.dataset.map(self.transform_example, num_proc=4)
            transformed.set_format("torch", columns=["pixel_values", "label"], output_all_columns=True)
            logger.info("Dataset transformed successfully.")
            return transformed
        except Exception as e:
            logger.error(f"Failed to transform dataset: {e}")
            sys.exit(1)

    def split_dataset(self) -> DatasetDict:
        try:
            logger.info("Splitting dataset into train, validation, and test sets.")
            split_test = self.transformed_dataset["train"].train_test_split(test_size=0.2, seed=42)
            split_valid = split_test["train"].train_test_split(test_size=0.125, seed=42)  # 0.125 x 0.8 = 0.1
            dataset_dict = DatasetDict(
                {
                    "train": split_valid["train"],
                    "valid": split_valid["test"],
                    "test": split_test["test"],
                }
            )
            logger.info("Dataset split successfully.")
            return dataset_dict
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            sys.exit(1)

    def initialize_student_model(self) -> ViTForImageClassification:
        try:
            logger.info(f"Initializing student model: {self.config.student_model_name}")
            config = ViTConfig.from_pretrained(self.config.student_model_name)
            config.id2label = self.id2label
            config.label2id = self.label2id
            config.num_labels = len(self.id2label)
            model = ViTForImageClassification.from_pretrained(
                self.config.student_model_name, config=config
            ).to(self.device)
            logger.info("Student model initialized successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize student model '{self.config.student_model_name}': {e}")
            sys.exit(1)

    def configure_training_arguments(self) -> TrainingArguments:
        try:
            logger.info("Configuring training arguments.")
            args = TrainingArguments(
                output_dir=self.config.output_dir,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.eval_batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_steps=100,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                push_to_hub=self.config.push_to_hub,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
            )
            logger.info("Training arguments configured successfully.")
            return args
        except Exception as e:
            logger.error(f"Failed to configure training arguments: {e}")
            sys.exit(1)

    def get_collate_fn(self):
        def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
            try:
                pixel_values = torch.stack([item["pixel_values"] for item in batch])
                labels = torch.tensor([item["label"] for item in batch])
                return {"pixel_values": pixel_values, "labels": labels}
            except Exception as e:
                logger.error(f"Error in collate_fn: {e}")
                raise

        return collate_fn

    def plot_teacher_distributions(self, image_index: int = 0) -> None:
        try:
            logger.info(f"Plotting teacher distributions for image index: {image_index}")
            image = self.dataset["train"][image_index]["image"]
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.teacher_model(**inputs).logits

            prediction = torch.argmax(logits, dim=1).item()
            logger.info(f"Teacher model prediction: {self.id2label[prediction]}")

            temperatures = {
                "No Change (1)": 1.0,
                "Low Temp (2)": 2.0,
                "High Temp (8)": 8.0,
            }
            distributions = {}
            for label, temp in temperatures.items():
                softmax = F.softmax(logits / temp, dim=-1).cpu().numpy().flatten()
                distributions[label] = softmax

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            # Plotting temperature distributions
            axs[0].bar(self.id2label.keys(), distributions["No Change (1)"], width=0.4)
            axs[0].set_title("Teacher Probability Over All Classes")
            axs[0].set_ylabel("Probability")
            axs[0].set_xlabel("Class ID")

            bar_width = 0.35
            indices = np.array(list(self.id2label.keys()))
            axs[1].bar(
                indices - bar_width / 2,
                distributions["Low Temp (2)"],
                width=bar_width,
                label="Temperature=2",
            )
            axs[1].bar(
                indices + bar_width / 2,
                distributions["High Temp (8)"],
                width=bar_width,
                label="Temperature=8",
            )
            axs[1].set_title("Teacher Probability with Different Temperatures")
            axs[1].set_ylabel("Probability")
            axs[1].set_xlabel("Class ID")
            axs[1].legend()

            plt.tight_layout()
            plt.show()
            logger.info("Teacher distributions plotted successfully.")
        except Exception as e:
            logger.error(f"Failed to plot teacher distributions: {e}")

    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        try:
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            acc = self.accuracy.compute(references=labels, predictions=preds)
            return {"accuracy": acc["accuracy"]}
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {"accuracy": 0.0}

    def train_student_model(self) -> Trainer:
        try:
            logger.info("Initializing Trainer for student model.")
            trainer = Trainer(
                model=self.base_model,
                args=self.training_args,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metrics,
                train_dataset=self.train_valid_test["train"],
                eval_dataset=self.train_valid_test["valid"],
                tokenizer=self.processor,
            )
            logger.info("Trainer initialized successfully.")
            return trainer
        except Exception as e:
            logger.error(f"Failed to initialize Trainer: {e}")
            sys.exit(1)

    def evaluate_model(self, trainer: Trainer, dataset_split: str = "test") -> None:
        try:
            logger.info(f"Evaluating model on the '{dataset_split}' dataset.")
            results = trainer.evaluate(self.train_valid_test[dataset_split])
            logger.info(f"Evaluation results: {results}")
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")

    def initialize_distillation_trainer(
        self, trainer: Trainer
    ) -> "ImageDistilTrainer":
        try:
            logger.info("Initializing ImageDistilTrainer for knowledge distillation.")
            distil_trainer = ImageDistilTrainer(
                teacher_model=self.teacher_model,
                student_model=self.base_model,
                temperature=self.config.temperature,
                lambda_param=self.config.lambda_param,
                args=self.training_args,
                data_collator=self.collate_fn,
                compute_metrics=self.compute_metrics,
                train_dataset=self.train_valid_test["train"],
                eval_dataset=self.train_valid_test["valid"],
                tokenizer=self.processor,
            )
            logger.info("ImageDistilTrainer initialized successfully.")
            return distil_trainer
        except Exception as e:
            logger.error(f"Failed to initialize ImageDistilTrainer: {e}")
            sys.exit(1)

    def run(self) -> None:
        try:
            # Plot teacher distributions for the first image
            self.plot_teacher_distributions(image_index=0)

            # Train student model
            student_trainer = self.train_student_model()
            logger.info("Starting training of the student model.")
            student_trainer.train()
            logger.info("Student model training completed.")
            self.evaluate_model(student_trainer, dataset_split="test")

            # Initialize and train with knowledge distillation
            distil_trainer = self.initialize_distillation_trainer(student_trainer)
            logger.info("Starting training with knowledge distillation.")
            distil_trainer.train()
            logger.info("Knowledge distillation training completed.")
            self.evaluate_model(distil_trainer, dataset_split="test")
        except Exception as e:
            logger.error(f"An error occurred during training and evaluation: {e}")
            sys.exit(1)


class ImageDistilTrainer(Trainer):
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float,
        lambda_param: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.device = next(self.model.parameters()).device
        self.teacher.to(self.device)
        self.teacher.eval()

    def compute_loss(
        self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False
    ) -> torch.Tensor:
        try:
            outputs = model(**inputs)
            student_logits = outputs.logits

            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
                teacher_logits = teacher_outputs.logits

            # Compute soft targets
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

            # Distillation loss
            distillation_loss = self.loss_function(soft_student, soft_teacher) * (
                self.temperature ** 2
            )

            # Classification loss
            labels = inputs.get("labels")
            classification_loss = F.cross_entropy(student_logits, labels)

            # Combined loss
            loss = (1.0 - self.lambda_param) * classification_loss + self.lambda_param * distillation_loss

            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            logger.error(f"Error computing loss in ImageDistilTrainer: {e}")
            raise


def main() -> None:
    config = Config()
    trainer = OxfordPetsTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()