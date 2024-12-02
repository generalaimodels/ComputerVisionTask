import os
import warnings
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError
from transformers import (
    CLIPProcessor,
    CLIPModel,
    YolosImageProcessor,
    YolosForObjectDetection,
)

# Suppress warnings
warnings.simplefilter("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Constants for model names and URLs
DETECTION_MODEL_NAME = "hustvl/yolos-tiny"
MULTIMODAL_MODEL_NAME = "openai/clip-vit-base-patch16"
IMAGE_URL = "https://i.pinimg.com/736x/1b/51/42/1b5142c269f2e9a356202af3f5569a87.jpg"
TEXT_DESCRIPTIONS = [
    "Man carrying a green bag",
    "man riding a bicycle",
    "yellow colored taxi car",
    "red colored bus",
]


class ImageProcessorPipeline:
    """
    A pipeline for object detection and multimodal processing using pre-trained models.
    """

    def __init__(
        self,
        detection_model_name: str = DETECTION_MODEL_NAME,
        multimodal_model_name: str = MULTIMODAL_MODEL_NAME,
        detection_threshold: float = 0.85,
    ) -> None:
        """
        Initialize the pipeline with specified models and threshold.

        :param detection_model_name: Pre-trained detection model name.
        :param multimodal_model_name: Pre-trained multimodal model name.
        :param detection_threshold: Probability threshold for object detection.
        """
        self.detection_threshold = detection_threshold
        try:
            self.det_image_processor = YolosImageProcessor.from_pretrained(
                detection_model_name
            )
            self.det_model = YolosForObjectDetection.from_pretrained(
                detection_model_name
            )
            self.mm_model = CLIPModel.from_pretrained(multimodal_model_name)
            self.mm_processor = CLIPProcessor.from_pretrained(multimodal_model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")

    def load_image_from_url(self, url: str) -> Image.Image:
        """
        Load an image from a given URL.

        :param url: URL of the image to load.
        :return: PIL Image object.
        :raises ValueError: If the image cannot be loaded.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
            return image
        except (requests.HTTPError, UnidentifiedImageError) as e:
            raise ValueError(f"Failed to load image from URL: {e}")

    def extract_detected_objects(
        self, image: Image.Image
    ) -> Tuple[List[Image.Image], np.ndarray]:
        """
        Perform object detection on the image and extract cropped objects with scores.

        :param image: PIL Image object.
        :return: Tuple of list of cropped images and their confidence scores.
        """
        try:
            inputs = self.det_image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.det_model(**inputs)

            logits = outputs.logits
            probas = logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > self.detection_threshold

            outs = self.det_image_processor.post_process(
                outputs, torch.tensor(image.size[::-1]).unsqueeze(0)
            )
            bboxes_scaled = outs[0]["boxes"][keep].detach().cpu().numpy()
            scores = outs[0]["scores"][keep].detach().cpu().numpy()

            images_list = self._crop_images(image, bboxes_scaled)
            return images_list, scores
        except Exception as e:
            raise RuntimeError(f"Object detection failed: {e}")

    @staticmethod
    def _crop_images(
        image: Image.Image, bboxes: np.ndarray
    ) -> List[Image.Image]:
        """
        Crop regions of interest from the image based on bounding boxes.

        :param image: PIL Image object.
        :param bboxes: Numpy array of bounding boxes.
        :return: List of cropped PIL Image objects.
        """
        image_np = np.array(image)
        cropped_images = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = map(int, bbox)
            roi = image_np[ymin:ymax, xmin:xmax]
            cropped_image = Image.fromarray(roi)
            cropped_images.append(cropped_image)
        return cropped_images

    def get_best_matching_image(
        self, images: List[Image.Image], text: str
    ) -> Tuple[Image.Image, float]:
        """
        Find the image that best matches the given text description.

        :param images: List of PIL Image objects.
        :param text: Text description to match.
        :return: Tuple of best matching image and its score.
        """
        try:
            inputs = self.mm_processor(
                text=[text], images=images, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = self.mm_model(**inputs)
            logits_per_text = outputs.logits_per_text
            probs = logits_per_text.softmax(dim=-1).cpu().numpy()[0]

            if not probs.size:
                raise ValueError("No probabilities returned from the model.")

            best_idx = np.argmax(probs)
            best_image = images[best_idx]
            best_score = float(probs[best_idx])
            return best_image, best_score
        except Exception as e:
            raise RuntimeError(f"Multimodal processing failed: {e}")

    def process(
        self, url: str, text_descriptions: List[str]
    ) -> List[Dict[str, object]]:
        """
        Process the image from URL and extract images matching the text descriptions.

        :param url: URL of the image to process.
        :param text_descriptions: List of text descriptions to match.
        :return: List of dictionaries containing matched images, descriptions, and scores.
        """
        try:
            image = self.load_image_from_url(url)
            detected_images, scores = self.extract_detected_objects(image)

            if not detected_images:
                raise ValueError("No objects detected with the specified threshold.")

            matched_images = []
            for description in text_descriptions:
                img, score = self.get_best_matching_image(detected_images, description)
                matched_images.append(
                    {
                        "image": img,
                        "description": description,
                        "confidence_score": score,
                    }
                )
            return matched_images, image
        except Exception as e:
            raise RuntimeError(f"Processing failed: {e}")

    @staticmethod
    def display_results(
        original_image: Image.Image, matched_images: List[Dict[str, object]]
    ) -> None:
        """
        Display the original image and the matched images with descriptions and scores.

        :param original_image: The original PIL Image object.
        :param matched_images: List of dictionaries with matched images and metadata.
        """
        try:
            plt.figure(figsize=(15, 7))
            plt.subplot(1, len(matched_images) + 1, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")

            for idx, img_dict in enumerate(matched_images, start=2):
                plt.subplot(1, len(matched_images) + 1, idx)
                plt.imshow(img_dict["image"])
                plt.title(img_dict["description"])
                plt.xlabel(f"Confidence: {img_dict['confidence_score']:.3f}")
                plt.axis("off")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Displaying results failed: {e}")


def main() -> None:
    """
    Main function to execute the image processing pipeline.
    """
    pipeline = ImageProcessorPipeline(
        detection_model_name=DETECTION_MODEL_NAME,
        multimodal_model_name=MULTIMODAL_MODEL_NAME,
        detection_threshold=0.85,
    )

    try:
        matched_images, original_image = pipeline.process(
            url=IMAGE_URL, text_descriptions=TEXT_DESCRIPTIONS
        )
        pipeline.display_results(original_image, matched_images)
    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()