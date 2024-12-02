import os
import random
from functools import partial
from typing import Any, Dict, List, Optional

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPModel,
    CLIPTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding
from datasets.formatting.formatting import LazyBatch

# Set environment variables at the beginning
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def seed_all(seed: int) -> None:
    """
    Seed all relevant libraries for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def show_images(ds: Dataset, num_images: int = 9) -> None:
    """
    Display a grid of images from the dataset.

    Args:
        ds (Dataset): The dataset containing images and labels.
        num_images (int, optional): Number of images to display. Defaults to 9.
    """
    fig, axs = plt.subplots(3, 3, figsize=(13, 13))
    for i in range(num_images):
        img = ds[i]["image"]
        label = ds[i]["label"]
        if isinstance(img, Image.Image):
            img = np.array(img)
        axs[i // 3][i % 3].imshow(img)
        axs[i // 3][i % 3].set_title(label)
        axs[i // 3][i % 3].axis("off")
    plt.tight_layout()
    plt.show()


def transform_class_labels(
    items: LazyBatch, tokenizer: CLIPTokenizerFast, label2id: Dict[str, int]
) -> Dict[str, Any]:
    """
    Encode item's label prompt with tokenizer.

    Args:
        items (LazyBatch): Input dataset items.
        tokenizer (CLIPTokenizerFast): CLIP's tokenizer.
        label2id (Dict[str, int]): Mapping from label to id.

    Returns:
        Dict[str, Any]: Transformed items with encoded labels.
    """
    try:
        label_prompt = [f"a photo of {label}" for label in items["label"]]
        output = tokenizer(
            label_prompt, padding=True, return_tensors="pt", truncation=True
        )
        items["input_ids"] = output["input_ids"]
        items["attention_mask"] = output["attention_mask"]
        items["label_id"] = [label2id[label] for label in items["label"]]
    except Exception as e:
        print(f"Error in transform_class_labels: {e}")
    return items


def transform_image(
    items: LazyBatch, image_processor: CLIPImageProcessor
) -> Dict[str, Any]:
    """
    Preprocess input image with image processor.

    Args:
        items (LazyBatch): Input dataset items.
        image_processor (CLIPImageProcessor): CLIP's image processor.

    Returns:
        Dict[str, Any]: Transformed items with processed images.
    """
    try:
        output = image_processor(
            images=items["image"], return_tensors="pt", padding=True
        )
        items["pixel_values"] = output["pixel_values"]
    except Exception as e:
        print(f"Error in transform_image: {e}")
    return items


def get_module_device(module: nn.Module) -> torch.device:
    """
    Get the device of the provided module.

    Args:
        module (nn.Module): PyTorch module.

    Returns:
        torch.device: The device where the module is located.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def freeze_params(module: nn.Module, freeze_top_percent: float = 1.0) -> None:
    """
    Freeze a percentage of the top layers of a module.

    Args:
        module (nn.Module): PyTorch module.
        freeze_top_percent (float, optional): Percentage of layers to freeze. Defaults to 1.0.
    """
    try:
        all_params = list(module.parameters())
        freeze_until = int(len(all_params) * freeze_top_percent)
        for param in all_params[:freeze_until]:
            param.requires_grad = False
    except Exception as e:
        print(f"Error in freeze_params: {e}")


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print statistics about trainable parameters.

    Args:
        model (nn.Module): PyTorch model.
    """
    try:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / total_params if total_params else 0
        print(
            f"Trainable params: {trainable_params / 1e6:.4f}M || "
            f"All params: {total_params / 1e6:.4f}M || "
            f"Trainable%: {trainable_percent:.2f}%"
        )
    except Exception as e:
        print(f"Error in print_trainable_parameters: {e}")


class CLIPClassifier(nn.Module):
    """
    Custom CLIP-based classifier.
    """

    def __init__(
        self, clip_model: CLIPModel, tokenizer: CLIPTokenizerFast, labels: List[str]
    ):
        """
        Initialize the CLIPClassifier.

        Args:
            clip_model (CLIPModel): Pretrained CLIP model.
            tokenizer (CLIPTokenizerFast): CLIP tokenizer.
            labels (List[str]): List of class labels.
        """
        super().__init__()
        self.model = clip_model
        self.tokenizer = tokenizer
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.labels_embeddings = nn.Parameter(
            self.generate_labels_embeddings(labels)
        )

    def generate_labels_embeddings(self, labels: List[str]) -> torch.Tensor:
        """
        Generate embeddings for class labels.

        Args:
            labels (List[str]): List of class labels.

        Returns:
            torch.Tensor: Normalized label embeddings.
        """
        try:
            label_prompts = [f"a photo of {label}" for label in labels]
            inputs = self.tokenizer(
                label_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(get_module_device(self.model))
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features
        except Exception as e:
            print(f"Error in generate_labels_embeddings: {e}")
            return torch.empty(0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute similarity between image and label embeddings.

        Args:
            images (torch.Tensor): Batch of images.

        Returns:
            torch.Tensor: Similarity scores.
        """
        try:
            image_features = self.model.get_image_features(images)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            logits = torch.matmul(image_features, self.labels_embeddings.T)
            return logits * self.logit_scale.exp()
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.empty(0)


def calculate_accuracy(
    model: CLIPClassifier, dataloader: DataLoader
) -> float:
    """
    Calculate the accuracy of the model on the provided dataloader.

    Args:
        model (CLIPClassifier): The classifier model.
        dataloader (DataLoader): DataLoader for evaluation.

    Returns:
        float: Accuracy of the model.
    """
    metric = evaluate.load("accuracy")
    predictions_list = []
    references_list = []
    device = get_module_device(model)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(
            dataloader, total=len(dataloader), desc="Evaluating model"
        ):
            try:
                pixel_values = batch["pixel_values"].to(device)
                logits = model(pixel_values)
                preds = torch.argmax(logits, dim=1)
                predictions_list.append(preds.cpu())
                references_list.append(batch["label_id"])
            except Exception as e:
                print(f"Error during evaluation batch: {e}")

    if predictions_list and references_list:
        predictions = torch.cat(predictions_list)
        references = torch.cat(references_list)
        accuracy = metric.compute(predictions=predictions, references=references)
        return accuracy.get("accuracy", 0.0)
    else:
        return 0.0


def collate_fn(items: LazyBatch) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    Args:
        items (LazyBatch): Batch of items.

    Returns:
        Dict[str, Any]: Collated batch.
    """
    try:
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in items]),
            "input_ids": torch.stack([item["input_ids"] for item in items]),
            "attention_mask": torch.stack([item["attention_mask"] for item in items]),
            "label_id": torch.tensor([item["label_id"] for item in items]),
            "return_loss": True,
        }
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return {}


def collate_train_fn(items: LazyBatch) -> Dict[str, Any]:
    """
    Collate function for training DataLoader.

    Args:
        items (LazyBatch): Batch of items.

    Returns:
        Dict[str, Any]: Collated batch without label_id.
    """
    try:
        batch = collate_fn(items)
        batch.pop("label_id", None)
        return batch
    except Exception as e:
        print(f"Error in collate_train_fn: {e}")
        return {}


def get_default_training_args(
    experiment_name: str,
    lr: float,
    batch_size: int = 256,
    num_epochs: int = 2,
    num_workers: int = 15,
) -> TrainingArguments:
    """
    Get default training arguments for Hugging Face Trainer.

    Args:
        experiment_name (str): Name of the experiment.
        lr (float): Learning rate.
        batch_size (int, optional): Batch size. Defaults to 256.
        num_epochs (int, optional): Number of epochs. Defaults to 2.
        num_workers (int, optional): Number of worker threads. Defaults to 15.

    Returns:
        TrainingArguments: Configured training arguments.
    """
    try:
        return TrainingArguments(
            output_dir=f"./results/{experiment_name}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_steps=10,
            save_total_limit=2,
            fp16=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            dataloader_num_workers=num_workers,
        )
    except Exception as e:
        print(f"Error in get_default_training_args: {e}")
        return TrainingArguments()


@torch.no_grad()
def evaluate_clip_classifier(
    model: nn.Module,
    dataset: Dataset,
    tokenizer: CLIPTokenizerFast,
    labels: List[str],
    batch_size: int = 64,
    num_workers: int = 5,
    device: Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Evaluate the CLIP classifier on the given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): Evaluation dataset.
        tokenizer (CLIPTokenizerFast): Tokenizer for labels.
        labels (List[str]): List of label names.
        batch_size (int, optional): Batch size. Defaults to 64.
        num_workers (int, optional): Number of workers. Defaults to 5.
        device (Optional[str], optional): Device to run evaluation on. Defaults to CUDA if available.
    """
    try:
        classifier = CLIPClassifier(model, tokenizer, labels).to(device)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        accuracy = calculate_accuracy(classifier, dataloader)
        print(f"Model accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error in evaluate_clip_classifier: {e}")


def main() -> None:
    """
    Main function to execute the training and evaluation pipeline.
    """
    try:
        # Seed for reproducibility
        SEED = 69
        seed_all(SEED)

        # Load dataset
        dataset = load_dataset("pcuenq/oxford-pets")
        dataset_train_val = dataset["train"].train_test_split(test_size=0.3, seed=SEED)
        dataset_val_test = dataset_train_val["test"].train_test_split(
            test_size=0.2, seed=SEED
        )

        dataset = DatasetDict(
            {
                "train": dataset_train_val["train"],
                "val": dataset_val_test["test"],
                "test": dataset_val_test["train"],
            }
        )

        # Display sample images
        show_images(dataset["train"])

        # Create label mappings
        labels = sorted(set(dataset["train"]["label"]))
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        # Load models and processors
        MODEL_NAME = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_NAME)
        image_processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)

        # Transform dataset
        transform_partial = partial(
            transform_class_labels, tokenizer=tokenizer, label2id=label2id
        )
        dataset = dataset.map(transform_partial, batched=True, remove_columns=dataset["train"].column_names)
        dataset.set_transform(partial(transform_image, image_processor=image_processor))

        # Initialize baseline model
        clip_baseline = CLIPModel.from_pretrained(MODEL_NAME)
        freeze_params(clip_baseline)
        print_trainable_parameters(clip_baseline)
        evaluate_clip_classifier(
            clip_baseline, dataset["test"], tokenizer, labels
        )

        # Finetune all layers
        clip_full_finetuned = CLIPModel.from_pretrained(MODEL_NAME)
        trainer_full = Trainer(
            model=clip_full_finetuned,
            args=get_default_training_args(
                "clip-all-layers-tuning-oxford-pets", lr=3e-6
            ),
            data_collator=collate_train_fn,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
        )
        trainer_full.train()
        print_trainable_parameters(clip_full_finetuned)
        evaluate_clip_classifier(
            clip_full_finetuned, dataset["test"], tokenizer, labels
        )

        # Finetune only text model
        clip_text_model_tuning = CLIPModel.from_pretrained(MODEL_NAME)
        freeze_params(clip_text_model_tuning.vision_model)
        trainer_text = Trainer(
            model=clip_text_model_tuning,
            args=get_default_training_args(
                "clip-text-model-tuning-oxford-pets", lr=3e-5
            ),
            data_collator=collate_train_fn,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
        )
        trainer_text.train()

        # Finetune partially
        clip_partial_tuning = CLIPModel.from_pretrained(MODEL_NAME)
        freeze_params(clip_partial_tuning.text_model, freeze_top_percent=0.7)
        freeze_params(clip_partial_tuning.vision_model, freeze_top_percent=0.7)
        trainer_partial = Trainer(
            model=clip_partial_tuning,
            args=get_default_training_args(
                "clip-partial-model-tuning-oxford-pets", lr=3e-5
            ),
            data_collator=collate_train_fn,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
        )
        trainer_partial.train()

        # LoRA tuning
        clip_lora_tuning = CLIPModel.from_pretrained(MODEL_NAME)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias="none",
            task_type="IMAGE_CLASSIFICATION",
        )
        lora_model = get_peft_model(clip_lora_tuning, lora_config)
        trainer_lora = Trainer(
            model=lora_model,
            args=get_default_training_args(
                "clip-lora-model-tuning-oxford-pets", lr=3e-4
            ),
            data_collator=collate_train_fn,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
        )
        trainer_lora.train()
        print_trainable_parameters(lora_model)
        evaluate_clip_classifier(lora_model, dataset["test"], tokenizer, labels)

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()