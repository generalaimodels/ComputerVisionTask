import os
import random
from functools import partial
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizerFast,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict, load_dataset
from datasets.formatting.formatting import LazyBatch
from transformers.tokenization_utils_base import BatchEncoding
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment Variables
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_all(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"All seeds set to {seed}")

class Config:
    """Configuration for the training pipeline."""
    SEED: int = 69
    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    DATASET_NAME: str = "pcuenq/oxford-pets"
    TEST_SIZE: float = 0.3
    VAL_SIZE: float = 0.2
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 5
    LEARNING_RATE: float = 3e-5
    NUM_EPOCHS: int = 2
    SAVE_DIR: str = "model_weights"
    PLOTS_DIR: str = "plots"
    LOGGING_STEPS: int = 10
    GRADIENT_ACCUMULATION_STEPS: int = 1
    FP16: bool = True
    REMOVE_UNUSED_COLUMNS: bool = False
    LOAD_BEST_MODEL_AT_END: bool = True

    @staticmethod
    def create_directories() -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        logger.info(f"Directories '{Config.SAVE_DIR}' and '{Config.PLOTS_DIR}' are ready.")

class CLIPClassifier(nn.Module):
    """Custom CLIP-based classifier."""

    def __init__(self, clip_model: CLIPModel, tokenizer: CLIPTokenizerFast, labels: List[str]) -> None:
        super().__init__()
        self.model = clip_model
        self.tokenizer = tokenizer
        self.logit_scale = self.model.logit_scale.exp()
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.labels_embeddings = nn.Parameter(self.generate_labels_embeddings(labels))
        logger.info("CLIPClassifier initialized.")

    def generate_labels_embeddings(self, labels: List[str]) -> torch.Tensor:
        """
        Generate embeddings for each label using the CLIP text encoder.

        Args:
            labels (List[str]): List of label names.

        Returns:
            torch.Tensor: Normalized label embeddings.
        """
        try:
            label_prompts = [f"a photo of {label}" for label in labels]
            inputs = self.tokenizer(
                label_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            logger.info("Label embeddings generated.")
            return embeddings
        except Exception as e:
            logger.error(f"Error in generating label embeddings: {e}")
            raise

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute similarity between image features and label embeddings.

        Args:
            images (torch.Tensor): Batch of images.

        Returns:
            torch.Tensor: Similarity scores.
        """
        image_features = self.model.get_image_features(images)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        similarity = torch.matmul(image_features, self.labels_embeddings.T) * self.logit_scale
        return similarity

def freeze_params(module: nn.Module, freeze_top_percent: float = 1.0) -> None:
    """
    Freeze a percentage of the top parameters in a module.

    Args:
        module (nn.Module): The module to freeze.
        freeze_top_percent (float, optional): Portion to freeze. Defaults to 1.0.
    """
    try:
        total_params = list(module.parameters())
        freeze_until = int(len(total_params) * freeze_top_percent)
        for idx, param in enumerate(total_params):
            if idx < freeze_until:
                param.requires_grad = False
        logger.info(f"Frozen top {freeze_top_percent * 100}% of parameters in {module.__class__.__name__}.")
    except Exception as e:
        logger.error(f"Error in freezing parameters: {e}")
        raise

def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.

    Args:
        model (nn.Module): The model to inspect.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100.0 * trainable_params / total_params if total_params > 0 else 0
    logger.info(
        f"Trainable params: {trainable_params / 1e6:.4f}M | "
        f"Total params: {total_params / 1e6:.4f}M | "
        f"Trainable%: {trainable_percent:.2f}%"
    )

def load_and_prepare_dataset(config: Config) -> DatasetDict:
    """
    Load and split the dataset into train, validation, and test sets.

    Args:
        config (Config): Configuration object.

    Returns:
        DatasetDict: Dictionary containing train, val, and test datasets.
    """
    try:
        dataset = load_dataset(config.DATASET_NAME)
        dataset_train_val = dataset['train'].train_test_split(test_size=config.TEST_SIZE, seed=config.SEED)
        dataset_val_test = dataset_train_val['test'].train_test_split(test_size=config.VAL_SIZE, seed=config.SEED)

        dataset = DatasetDict({
            "train": dataset_train_val['train'],
            "val": dataset_val_test['test'],
            "test": dataset_val_test['train']
        })
        logger.info("Dataset loaded and split into train, val, test.")
        return dataset
    except Exception as e:
        logger.error(f"Error in loading/preparing dataset: {e}")
        raise

def visualize_sample_images(dataset: Dataset, num_images: int = 9) -> None:
    """
    Display a grid of sample images from the dataset.

    Args:
        dataset (Dataset): The dataset to visualize.
        num_images (int, optional): Number of images to display. Defaults to 9.
    """
    try:
        fig = go.Figure()
        for idx in range(num_images):
            img = dataset[idx]['image']
            label = dataset[idx]['label']
            fig.add_trace(go.Image(z=np.array(img)))
            fig.layout.annotations = fig.layout.annotations or []
            fig.layout.annotations += [dict(
                text=str(label),
                x=0.1 + (idx % 3) * 0.33,
                y=0.9 - (idx // 3) * 0.33,
                showarrow=False,
                xref="paper",
                yref="paper",
                font=dict(color="white", size=12)
            )]
        fig.update_layout(height=600, width=600, title="Sample Images from Training Set")
        plot_path = os.path.join(Config.PLOTS_DIR, "sample_images.html")
        fig.write_html(plot_path)
        logger.info(f"Sample images saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error in visualizing sample images: {e}")
        raise

def transform_class_labels(
    items: LazyBatch, tokenizer: CLIPTokenizerFast, label2id: Dict[str, int]
) -> Dict[str, Any]:
    """
    Encode class labels using the tokenizer.

    Args:
        items (LazyBatch): Batch of dataset items.
        tokenizer (CLIPTokenizerFast): Tokenizer for CLIP.
        label2id (Dict[str, int]): Mapping from labels to IDs.

    Returns:
        Dict[str, Any]: Transformed batch with encoded labels.
    """
    try:
        label_prompts = [f"a photo of {label}" for label in items["label"]]
        encoded = tokenizer(label_prompts, padding=True, return_tensors="pt")
        items["input_ids"] = encoded["input_ids"]
        items["attention_mask"] = encoded["attention_mask"]
        items["label_id"] = [label2id[label] for label in items["label"]]
        return items
    except Exception as e:
        logger.error(f"Error in transforming class labels: {e}")
        raise

def transform_image(
    items: LazyBatch, image_processor: CLIPImageProcessor
) -> Dict[str, Any]:
    """
    Preprocess images using the image processor.

    Args:
        items (LazyBatch): Batch of dataset items.
        image_processor (CLIPImageProcessor): Image processor for CLIP.

    Returns:
        Dict[str, Any]: Transformed batch with preprocessed images.
    """
    try:
        processed = image_processor(items["image"], return_tensors="pt")
        items["pixel_values"] = processed["pixel_values"]
        return items
    except Exception as e:
        logger.error(f"Error in transforming images: {e}")
        raise

def collate_fn(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    Args:
        items (List[Dict[str, Any]]): List of dataset items.

    Returns:
        Dict[str, Any]: Collated batch.
    """
    try:
        batch = {
            "pixel_values": torch.stack([item["pixel_values"] for item in items]),
            "input_ids": torch.stack([item["input_ids"] for item in items]),
            "attention_mask": torch.stack([item["attention_mask"] for item in items]),
            "label_id": torch.tensor([item["label_id"] for item in items], dtype=torch.long),
        }
        return batch
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        raise

def get_default_training_args(
    experiment_name: str,
    learning_rate: float,
    config: Config,
) -> TrainingArguments:
    """
    Generate default training arguments.

    Args:
        experiment_name (str): Name of the experiment.
        learning_rate (float): Learning rate.
        config (Config): Configuration object.

    Returns:
        TrainingArguments: Configured training arguments.
    """
    try:
        training_args = TrainingArguments(
            output_dir=os.path.join(Config.SAVE_DIR, experiment_name),
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            learning_rate=learning_rate,
            num_train_epochs=config.NUM_EPOCHS,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            logging_steps=config.LOGGING_STEPS,
            save_total_limit=2,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            fp16=config.FP16,
            remove_unused_columns=config.REMOVE_UNUSED_COLUMNS,
            load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
            dataloader_num_workers=config.NUM_WORKERS,
            seed=config.SEED,
        )
        logger.info(f"Training arguments for '{experiment_name}' prepared.")
        return training_args
    except Exception as e:
        logger.error(f"Error in getting training arguments: {e}")
        raise

def calculate_accuracy(model: CLIPClassifier, dataloader: DataLoader) -> float:
    """
    Calculate the accuracy of the model on the provided dataloader.

    Args:
        model (CLIPClassifier): The classifier model.
        dataloader (DataLoader): DataLoader for evaluation.

    Returns:
        float: Accuracy score.
    """
    try:
        metric = evaluate.load("accuracy")
        model.eval()
        device = next(model.parameters()).device
        predictions_list: List[torch.Tensor] = []
        references_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label_id"].to(device)
                outputs = model(pixel_values)
                preds = torch.argmax(outputs, dim=1)
                predictions_list.append(preds)
                references_list.append(labels)

        predictions = torch.cat(predictions_list).cpu()
        references = torch.cat(references_list).cpu()
        accuracy = metric.compute(predictions=predictions, references=references)["accuracy"]
        logger.info(f"Model accuracy: {accuracy * 100:.2f}%")
        return accuracy
    except Exception as e:
        logger.error(f"Error in calculating accuracy: {e}")
        raise

def plot_metrics(history: Dict[str, List[Any]], experiment_name: str, config: Config) -> None:
    """
    Plot training and evaluation metrics.

    Args:
        history (Dict[str, List[Any]]): Dictionary containing metrics history.
        experiment_name (str): Name of the experiment.
        config (Config): Configuration object.
    """
    try:
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig = go.Figure()

        # Plot Training Loss
        fig.add_trace(go.Scatter(
            x=epochs, y=history["train_loss"],
            mode='lines+markers',
            name='Train Loss'
        ))

        # Plot Evaluation Loss
        if "eval_loss" in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history["eval_loss"],
                mode='lines+markers',
                name='Eval Loss'
            ))

        # Plot Accuracy
        if "eval_accuracy" in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=[acc * 100 for acc in history["eval_accuracy"]],
                mode='lines+markers',
                name='Accuracy (%)',
                yaxis="y2"
            ))

        # Configure layout
        fig.update_layout(
            title=f"Training Metrics for {experiment_name}",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2=dict(
                title="Accuracy (%)",
                overlaying="y",
                side="right"
            ),
            legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
        )

        plot_path = os.path.join(Config.PLOTS_DIR, f"{experiment_name}_metrics.html")
        fig.write_html(plot_path)
        logger.info(f"Training metrics plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error in plotting metrics: {e}")
        raise

def main():
    """Main function to execute the training pipeline."""
    try:
        # Initialize configuration and setup
        config = Config()
        Config.create_directories()
        seed_all(config.SEED)

        # Load and prepare dataset
        dataset = load_and_prepare_dataset(config)
        labels = sorted(set(dataset["train"]["label"]))
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        # Visualize sample images
        visualize_sample_images(dataset["train"])

        # Initialize tokenizer and image processor
        tokenizer = CLIPTokenizerFast.from_pretrained(config.MODEL_NAME)
        image_processor = CLIPImageProcessor.from_pretrained(config.MODEL_NAME)
        logger.info("Tokenizer and Image Processor loaded.")

        # Transform dataset
        transform_labels = partial(transform_class_labels, tokenizer=tokenizer, label2id=label2id)
        dataset = dataset.map(transform_labels, batched=True)
        dataset.set_transform(partial(transform_image, image_processor=image_processor))
        logger.info("Dataset transformed with encoded labels and preprocessed images.")

        # Initialize CLIP model
        clip_model = CLIPModel.from_pretrained(config.MODEL_NAME)
        logger.info("CLIP model loaded.")

        # Initialize CLIPClassifier
        classifier = CLIPClassifier(clip_model, tokenizer, labels)
        classifier.to('cuda' if torch.cuda.is_available() else 'cpu')
        print_trainable_parameters(classifier)

        # Evaluate baseline
        test_dataloader = DataLoader(
            dataset["test"],
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn
        )
        accuracy = calculate_accuracy(classifier, test_dataloader)
        logger.info(f"Baseline model accuracy: {accuracy * 100:.2f}%")

        # Define training experiments
        experiments = [
            {
                "name": "clip-all-layers-tuning",
                "freeze_top_percent": 0.0,
                "learning_rate": 3e-6,
                "lora": False
            },
            {
                "name": "clip-text-model-tuning",
                "freeze_modules": ["vision_model"],
                "learning_rate": 3e-5,
                "lora": False
            },
            {
                "name": "clip-vision-model-tuning",
                "freeze_modules": ["text_model"],
                "learning_rate": 3e-5,
                "lora": False
            },
            {
                "name": "clip-partial-model-tuning",
                "freeze_top_percent_text": 0.7,
                "freeze_top_percent_vision": 0.7,
                "learning_rate": 3e-5,
                "lora": False
            },
            {
                "name": "clip-lora-model-tuning",
                "freeze_modules": [],
                "learning_rate": 3e-4,
                "lora": True
            }
        ]

        for exp in experiments:
            logger.info(f"Starting experiment: {exp['name']}")
            # Reload model for each experiment
            model = CLIPModel.from_pretrained(config.MODEL_NAME)
            if exp.get("freeze_modules"):
                for module_name in exp["freeze_modules"]:
                    module = getattr(model, module_name, None)
                    if module:
                        freeze_params(module)
            if exp.get("freeze_top_percent_text") and exp.get("freeze_top_percent_vision"):
                freeze_params(model.text_model, freeze_top_percent=exp["freeze_top_percent_text"])
                freeze_params(model.vision_model, freeze_top_percent=exp["freeze_top_percent_vision"])
            if exp.get("lora"):
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=64,
                    target_modules=['q_proj', 'k_proj', 'v_proj'],
                    lora_dropout=0.05,
                    bias="none"
                )
                model = get_peft_model(model, lora_config)
                logger.info("LoRA configuration applied.")

            # Initialize Trainer
            training_args = get_default_training_args(
                experiment_name=exp["name"],
                learning_rate=exp["learning_rate"],
                config=config
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                train_dataset=dataset["train"],
                eval_dataset=dataset["val"]
            )

            # Train the model
            trainer.train()

            # Save the model
            save_path = os.path.join(config.SAVE_DIR, exp["name"])
            trainer.save_model(save_path)
            logger.info(f"Model saved to {save_path}")

            # Evaluate the model
            model.to('cuda' if torch.cuda.is_available() else 'cpu')
            classifier = CLIPClassifier(model, tokenizer, labels).to(model.device)
            print_trainable_parameters(classifier)
            accuracy = calculate_accuracy(classifier, test_dataloader)

            # Optionally, collect metrics for plotting
            history = {
                "train_loss": trainer.state.log_history.get("loss", []),
                "eval_loss": trainer.state.log_history.get("eval_loss", []),
                "eval_accuracy": [accuracy]
            }
            plot_metrics(history, exp["name"], config)

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")
        raise

if __name__ == "__main__":
    main()