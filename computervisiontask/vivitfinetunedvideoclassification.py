import os
import pathlib
import tarfile
from typing import Dict, List, Tuple

import torch
import evaluate
import numpy as np
import imageio
import plotly.graph_objs as go
from IPython.display import Image
from huggingface_hub import hf_hub_download
from pytorchvideo.data import Ucf101, make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomHorizontalFlip,
    RandomShortSideScale,
    Resize,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda, RandomCrop
from transformers import (
    TrainingArguments,
    Trainer,
    VivitForVideoClassification,
    VivitImageProcessor,
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def download_and_extract_dataset(
    hf_dataset_identifier: str,
    filename: str,
    extract_path: str
) -> pathlib.Path:
    """
    Downloads and extracts a dataset from Hugging Face Hub.

    Args:
        hf_dataset_identifier (str): Hugging Face dataset repository identifier.
        filename (str): Name of the file to download.
        extract_path (str): Path to extract the dataset.

    Returns:
        pathlib.Path: Path to the extracted dataset.
    """
    try:
        logger.info("Downloading dataset from Hugging Face Hub...")
        file_path = hf_hub_download(
            repo_id=hf_dataset_identifier,
            filename=filename,
            repo_type="dataset"
        )
        logger.info(f"Downloaded file to {file_path}")

        logger.info("Extracting dataset...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        logger.info(f"Extracted dataset to {extract_path}")

        dataset_root = pathlib.Path(extract_path)
        return dataset_root
    except Exception as e:
        logger.error(f"Failed to download or extract dataset: {e}")
        raise


def count_videos(dataset_root: pathlib.Path) -> Tuple[int, int, int, int]:
    """
    Counts the number of videos in train, validation, and test splits.

    Args:
        dataset_root (pathlib.Path): Root path of the dataset.

    Returns:
        Tuple[int, int, int, int]: Counts for train, validation, test, and total videos.
    """
    try:
        train_count = len(list(dataset_root.glob("train/*/*.avi")))
        val_count = len(list(dataset_root.glob("val/*/*.avi")))
        test_count = len(list(dataset_root.glob("test/*/*.avi")))
        total_videos = train_count + val_count + test_count
        logger.info(f"Train videos: {train_count}")
        logger.info(f"Validation videos: {val_count}")
        logger.info(f"Test videos: {test_count}")
        logger.info(f"Total videos: {total_videos}")
        return train_count, val_count, test_count, total_videos
    except Exception as e:
        logger.error(f"Error counting videos: {e}")
        raise


def get_all_video_file_paths(dataset_root: pathlib.Path) -> List[pathlib.Path]:
    """
    Retrieves all video file paths from train, validation, and test splits.

    Args:
        dataset_root (pathlib.Path): Root path of the dataset.

    Returns:
        List[pathlib.Path]: List of all video file paths.
    """
    try:
        train_videos = list(dataset_root.glob("train/*/*.avi"))
        val_videos = list(dataset_root.glob("val/*/*.avi"))
        test_videos = list(dataset_root.glob("test/*/*.avi"))
        all_videos = train_videos + val_videos + test_videos
        logger.info(f"Retrieved {len(all_videos)} video file paths.")
        return all_videos
    except Exception as e:
        logger.error(f"Error retrieving video file paths: {e}")
        raise


def get_label_mappings(all_video_paths: List[pathlib.Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates label to ID and ID to label mappings.

    Args:
        all_video_paths (List[pathlib.Path]): List of all video file paths.

    Returns:
        Tuple[Dict[str, int], Dict[int, str]]: Label to ID and ID to label mappings.
    """
    try:
        class_labels = sorted({path.parts[2] for path in all_video_paths})
        label_to_id = {label: idx for idx, label in enumerate(class_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        logger.info(f"Unique classes: {list(label_to_id.keys())}.")
        return label_to_id, id_to_label
    except Exception as e:
        logger.error(f"Error creating label mappings: {e}")
        raise


def define_transforms(
    image_processor: VivitImageProcessor,
    num_frames: int,
    sample_rate: int,
    clip_duration: float,
    split: str
) -> Compose:
    """
    Defines data transformations for training or evaluation.

    Args:
        image_processor (VivitImageProcessor): Image processor for normalization.
        num_frames (int): Number of frames to sample.
        sample_rate (int): Sampling rate for frames.
        clip_duration (float): Duration of the clip in seconds.
        split (str): Data split - 'train', 'val', or 'test'.

    Returns:
        Compose: Composed transformations.
    """
    try:
        mean = image_processor.image_mean
        std = image_processor.image_std

        size = image_processor.size
        if "shortest_edge" in size:
            height = width = size["shortest_edge"]
        else:
            height = size["height"]
            width = size["width"]
        resize_to = (height, width)

        if split == "train":
            transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]),
                ),
            ])
        else:
            transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose([
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]),
                ),
            ])
        logger.info(f"Defined transforms for split: {split}")
        return transform
    except Exception as e:
        logger.error(f"Error defining transforms: {e}")
        raise


def create_datasets(
    dataset_root: pathlib.Path,
    label_to_id: Dict[str, int],
    image_processor: VivitImageProcessor,
    num_frames: int,
    sample_rate: int,
    fps: int
) -> Tuple[Ucf101, Ucf101, Ucf101]:
    """
    Creates training, validation, and test datasets.

    Args:
        dataset_root (pathlib.Path): Root path of the dataset.
        label_to_id (Dict[str, int]): Label to ID mapping.
        image_processor (VivitImageProcessor): Image processor for normalization.
        num_frames (int): Number of frames to sample.
        sample_rate (int): Sampling rate for frames.
        fps (int): Frames per second.

    Returns:
        Tuple[Ucf101, Ucf101, Ucf101]: Training, validation, and test datasets.
    """
    try:
        resize_to = (
            image_processor.size["height"],
            image_processor.size["width"]
        ) if "height" in image_processor.size else (
            image_processor.size["shortest_edge"],
            image_processor.size["shortest_edge"]
        )

        clip_duration = num_frames * sample_rate / fps

        train_transform = define_transforms(
            image_processor, num_frames, sample_rate, clip_duration, split="train"
        )
        val_test_transform = define_transforms(
            image_processor, num_frames, sample_rate, clip_duration, split="val"
        )

        train_dataset = Ucf101(
            data_path=dataset_root / "train",
            clip_sampler=make_clip_sampler("random", clip_duration),
            decode_audio=False,
            transform=train_transform,
        )

        val_dataset = Ucf101(
            data_path=dataset_root / "val",
            clip_sampler=make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_test_transform,
        )

        test_dataset = Ucf101(
            data_path=dataset_root / "test",
            clip_sampler=make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_test_transform,
        )

        logger.info("Created training, validation, and test datasets.")
        return train_dataset, val_dataset, test_dataset
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise


def visualize_sample_video(sample_video: Dict, id_to_label: Dict[int, str]) -> None:
    """
    Creates and displays a GIF from a sample video tensor.

    Args:
        sample_video (Dict): Sample video data.
        id_to_label (Dict[int, str]): ID to label mapping.
    """
    try:
        video_tensor = sample_video["video"]
        label_id = sample_video["label"].item()
        predicted_label = id_to_label.get(label_id, "Unknown")

        gif_filename = create_gif(video_tensor, "sample.gif")
        logger.info(f"Displaying sample video GIF: {gif_filename}")
        display_gif(gif_filename)

        logger.info(f"Video label: {predicted_label}")
    except Exception as e:
        logger.error(f"Error visualizing sample video: {e}")
        raise


def unnormalize_image(img: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """
    Un-normalizes image pixels.

    Args:
        img (torch.Tensor): Normalized image tensor.
        mean (List[float]): Mean values for normalization.
        std (List[float]): Standard deviation values for normalization.

    Returns:
        np.ndarray: Un-normalized image as a NumPy array.
    """
    img = (img * torch.tensor(std).view(-1, 1, 1)) + torch.tensor(mean).view(-1, 1, 1)
    img = (img * 255).clamp(0, 255).byte().numpy()
    return img


def create_gif(video_tensor: torch.Tensor, filename: str = "sample.gif") -> str:
    """
    Creates a GIF from a video tensor.

    Args:
        video_tensor (torch.Tensor): Video tensor.
        filename (str, optional): Filename for the GIF. Defaults to "sample.gif".

    Returns:
        str: Path to the created GIF.
    """
    try:
        frames = [
            unnormalize_image(frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            for frame in video_tensor.permute(1, 0, 2, 3)
        ]
        imageio.mimsave(filename, frames, format="GIF", duration=0.4)
        logger.info(f"Created GIF: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error creating GIF: {e}")
        raise


def display_gif(gif_path: str) -> Image:
    """
    Displays a GIF from a given path.

    Args:
        gif_path (str): Path to the GIF file.

    Returns:
        Image: IPython Image object.
    """
    try:
        return Image(filename=gif_path)
    except Exception as e:
        logger.error(f"Error displaying GIF: {e}")
        raise


def plot_metrics(
    history: Dict[str, List[float]],
    metric_name: str,
    save_path: str
) -> None:
    """
    Plots training and evaluation metrics using Plotly and saves as HTML.

    Args:
        history (Dict[str, List[float]]): Dictionary containing metric history.
        metric_name (str): Name of the metric to plot.
        save_path (str): Path to save the HTML plot.
    """
    try:
        epochs = list(range(1, len(history['train']) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history['train'], mode='lines+markers', name='Train'))
        fig.add_trace(go.Scatter(x=epochs, y=history['eval'], mode='lines+markers', name='Validation'))
        fig.update_layout(
            title=f'{metric_name.capitalize()} over Epochs',
            xaxis_title='Epoch',
            yaxis_title=metric_name.capitalize(),
            template='plotly_dark'
        )
        fig.write_html(save_path)
        logger.info(f"Saved {metric_name} plot to {save_path}")
    except Exception as e:
        logger.error(f"Error plotting metrics: {e}")
        raise


def initialize_trainer(
    model: VivitForVideoClassification,
    training_args: TrainingArguments,
    train_dataset: Ucf101,
    val_dataset: Ucf101,
    compute_metrics_func,
    data_collator
) -> Trainer:
    """
    Initializes the HuggingFace Trainer.

    Args:
        model (VivitForVideoClassification): The video classification model.
        training_args (TrainingArguments): Training arguments.
        train_dataset (Ucf101): Training dataset.
        val_dataset (Ucf101): Validation dataset.
        compute_metrics_func: Function to compute metrics.
        data_collator: Data collator function.

    Returns:
        Trainer: Initialized Trainer object.
    """
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_func,
            data_collator=data_collator,
        )
        logger.info("Initialized Trainer.")
        return trainer
    except Exception as e:
        logger.error(f"Error initializing Trainer: {e}")
        raise


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Computes accuracy metric.

    Args:
        eval_pred: Evaluation predictions.

    Returns:
        Dict[str, float]: Computed metrics.
    """
    try:
        predictions = np.argmax(eval_pred.predictions, axis=1)
        metric = evaluate.load("accuracy")
        result = metric.compute(predictions=predictions, references=eval_pred.label_ids)
        return result
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise


def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collates a list of examples into a batch.

    Args:
        examples (List[Dict]): List of examples.

    Returns:
        Dict[str, torch.Tensor]: Batch of data.
    """
    try:
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        raise


def run_inference(
    model: VivitForVideoClassification,
    video: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Runs inference on a given video tensor.

    Args:
        model (VivitForVideoClassification): Trained model.
        video (torch.Tensor): Video tensor.
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Logits from the model.
    """
    try:
        model.to(device)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=video)
            logits = outputs.logits
        return logits
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


def main():
    """Main function to execute the training, evaluation, and inference pipeline."""
    try:
        # Configuration
        HF_DATASET_IDENTIFIER = "sayakpaul/ucf101-subset"
        FILENAME = "UCF101_subset.tar.gz"
        EXTRACT_PATH = "./data"
        DATASET_ROOT_PATH = "UCF101_subset"
        MODEL_CHECKPOINT = "google/vivit-b-16x2-kinetics400"
        BATCH_SIZE = 4
        NUM_EPOCHS = 5
        LEARNING_RATE = 5e-5
        LOGGING_STEPS = 10
        SAVE_DIR = "trained_model"
        PLOTS_DIR = "plots"

        os.makedirs(EXTRACT_PATH, exist_ok=True)
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # Download and extract dataset
        dataset_root = download_and_extract_dataset(
            HF_DATASET_IDENTIFIER, FILENAME, EXTRACT_PATH
        ) / DATASET_ROOT_PATH

        # Count videos
        train_count, val_count, test_count, total_videos = count_videos(dataset_root)

        # Get all video file paths
        all_videos = get_all_video_file_paths(dataset_root)

        # Create label mappings
        label_to_id, id_to_label = get_label_mappings(all_videos)

        # Load image processor and model
        image_processor = VivitImageProcessor.from_pretrained(MODEL_CHECKPOINT)
        model = VivitForVideoClassification.from_pretrained(
            MODEL_CHECKPOINT,
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
        )
        logger.info("Loaded image processor and model.")

        # Define parameters
        num_frames = model.config.num_frames
        sample_rate = 4
        fps = 30

        # Create datasets
        train_dataset, val_dataset, test_dataset = create_datasets(
            dataset_root, label_to_id, image_processor, num_frames, sample_rate, fps
        )

        # Visualize a sample video
        sample_video = next(iter(train_dataset))
        visualize_sample_video(sample_video, id_to_label)

        # Define Training Arguments
        training_args = TrainingArguments(
            output_dir=SAVE_DIR,
            auto_find_batch_size=True,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_ratio=0.1,
            logging_steps=LOGGING_STEPS,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            num_train_epochs=NUM_EPOCHS,
            max_steps=(len(train_dataset) // BATCH_SIZE) * NUM_EPOCHS,
            report_to="none",  # Disable reporting to avoid unnecessary outputs
        )
        logger.info("Defined training arguments.")

        # Initialize Trainer
        trainer = initialize_trainer(
            model,
            training_args,
            train_dataset,
            val_dataset,
            compute_metrics,
            collate_fn
        )

        # Train the model
        logger.info("Starting training...")
        train_results = trainer.train()
        trainer.save_model(SAVE_DIR)
        logger.info("Training completed and model saved.")

        # Evaluate on test dataset
        logger.info("Evaluating on test dataset...")
        test_results = trainer.evaluate(test_dataset)
        trainer.log_metrics("test", test_results)
        trainer.save_metrics("test", test_results)
        trainer.save_state()

        # Plot metrics
        history = {
            'train': train_results.metrics.get('train_accuracy', []),
            'eval': train_results.metrics.get('eval_accuracy', [])
        }
        plot_metrics(history, "accuracy", os.path.join(PLOTS_DIR, "accuracy.html"))

        # Load the trained model for inference
        trained_model = VivitForVideoClassification.from_pretrained(SAVE_DIR)
        logger.info("Loaded the trained model for inference.")

        # Perform inference on a sample test video
        sample_test_video = next(iter(test_dataset))
        logits = run_inference(trained_model, sample_test_video["video"], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = id_to_label.get(predicted_class_idx, "Unknown")
        logger.info(f"Predicted class: {predicted_class}")

        # Visualize the test video
        visualize_sample_video(sample_test_video, id_to_label)

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}")
        raise


if __name__ == "__main__":
    main()