import logging
import time
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
from datasets import Dataset, load_dataset
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from transformers import CLIPModel, AutoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_device() -> torch.device:
    """Determine the available device (GPU if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def load_clip_model(device: torch.device) -> Tuple[AutoProcessor, CLIPModel]:
    """Load the CLIP model and processor."""
    try:
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        logging.info("CLIP model and processor loaded successfully.")
        return processor, model
    except Exception as e:
        logging.error(f"Error loading CLIP model: {e}")
        raise


def load_cifar10_dataset() -> Dataset:
    """Load the CIFAR-10 dataset."""
    try:
        dataset = load_dataset("cifar10")
        logging.info("CIFAR-10 dataset loaded successfully.")
        return dataset
    except Exception as e:
        logging.error(f"Error loading CIFAR-10 dataset: {e}")
        raise


def extract_features(
    images: List[Any],
    processor: AutoProcessor,
    model: CLIPModel,
    device: torch.device,
) -> np.ndarray:
    """
    Extract image features using CLIP model.

    Args:
        images (List[Any]): List of images.
        processor (AutoProcessor): CLIP processor.
        model (CLIPModel): CLIP model.
        device (torch.device): Device to perform computation.

    Returns:
        np.ndarray: Extracted features.
    """
    try:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        features = image_features.cpu().numpy().astype("float32")
        faiss.normalize_L2(features)
        return features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise


def build_faiss_index(
    data: np.ndarray, dimension: int = 512
) -> faiss.IndexFlatL2:
    """
    Build a FAISS index.

    Args:
        data (np.ndarray): Data to index.
        dimension (int, optional): Dimension of the vectors. Defaults to 512.

    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    try:
        index = faiss.IndexFlatL2(dimension)
        index.add(data)
        logging.info(f"FAISS index built with {index.ntotal} vectors.")
        return index
    except Exception as e:
        logging.error(f"Error building FAISS index: {e}")
        raise


def perform_clustering(
    vectors: np.ndarray,
    ncentroids: int = 10,
    niter: int = 50,
    verbose: bool = True,
) -> faiss.Clustering:
    """
    Perform K-Means clustering using FAISS.

    Args:
        vectors (np.ndarray): Vectors to cluster.
        ncentroids (int, optional): Number of clusters. Defaults to 10.
        niter (int, optional): Number of iterations. Defaults to 50.
        verbose (bool, optional): Verbosity flag. Defaults to True.

    Returns:
        faiss.Kmeans: Trained K-Means model.
    """
    try:
        d = vectors.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        start_time = time.time()
        kmeans.train(vectors)
        elapsed_time = time.time() - start_time
        logging.info(f"Clustering completed in {elapsed_time:.2f} seconds.")
        return kmeans
    except Exception as e:
        logging.error(f"Error during clustering: {e}")
        raise


def assign_clusters(
    kmeans: faiss.Kmeans, vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign vectors to clusters.

    Args:
        kmeans (faiss.Kmeans): Trained K-Means model.
        vectors (np.ndarray): Vectors to assign.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cluster assignments and distances.
    """
    try:
        distances, assignments = kmeans.index.search(vectors, 1)
        logging.info("Cluster assignments completed.")
        return assignments.flatten(), distances.flatten()
    except Exception as e:
        logging.error(f"Error assigning clusters: {e}")
        raise


def compute_distribution(
    assignments: np.ndarray, labels: List[int], n_clusters: int = 10
) -> np.ndarray:
    """
    Compute the distribution of labels per cluster.

    Args:
        assignments (np.ndarray): Cluster assignments.
        labels (List[int]): True labels of the data.
        n_clusters (int, optional): Number of clusters. Defaults to 10.

    Returns:
        np.ndarray: Distribution matrix.
    """
    try:
        distribution = np.zeros((n_clusters, 10), dtype=int)
        for cluster_id, label in zip(assignments, labels):
            distribution[cluster_id][label] += 1
        logging.info("Distribution matrix computed.")
        return distribution
    except Exception as e:
        logging.error(f"Error computing distribution: {e}")
        raise


def create_interactive_confusion_matrix(
    distribution: np.ndarray,
    class_labels: List[str],
    output_file: str = "confusion_matrix.html",
) -> None:
    """
    Create and save an interactive confusion matrix using Plotly.

    Args:
        distribution (np.ndarray): Distribution matrix.
        class_labels (List[str]): List of class names.
        output_file (str, optional): Output HTML file. Defaults to "confusion_matrix.html".
    """
    try:
        fig = go.Figure(
            data=go.Heatmap(
                z=distribution,
                x=class_labels,
                y=[f"Cluster {i}" for i in range(distribution.shape[0])],
                colorscale='Viridis'
            )
        )
        fig.update_layout(
            title="Distribution of CIFAR-10 Classes Across Clusters",
            xaxis_title="CIFAR-10 Classes",
            yaxis_title="Cluster ID",
            autosize=False,
            width=800,
            height=600,
        )
        fig.write_html(output_file)
        logging.info(f"Confusion matrix saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error creating confusion matrix: {e}")
        raise


def create_cluster_distribution_bar_chart(
    distribution: np.ndarray,
    class_labels: List[str],
    output_file: str = "cluster_distribution.html",
) -> None:
    """
    Create and save an interactive bar chart showing the number of images per cluster.

    Args:
        distribution (np.ndarray): Distribution matrix.
        class_labels (List[str]): List of class names.
        output_file (str, optional): Output HTML file. Defaults to "cluster_distribution.html".
    """
    try:
        cluster_sums = distribution.sum(axis=1)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[f"Cluster {i}" for i in range(distribution.shape[0])],
                    y=cluster_sums,
                    text=cluster_sums,
                    textposition='auto',
                )
            ]
        )
        fig.update_layout(
            title="Number of Images per Cluster",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Images",
            autosize=False,
            width=800,
            height=600,
        )
        fig.write_html(output_file)
        logging.info(f"Cluster distribution bar chart saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error creating cluster distribution bar chart: {e}")
        raise


def main() -> None:
    """Main function to execute the workflow."""
    try:
        device = get_device()
        processor, model = load_clip_model(device)
        dataset = load_cifar10_dataset()

        # Parameters
        batch_size = 64
        n_clusters = 10
        n_iterations = 50

        # Extract features in batches for efficiency
        test_dataset = dataset["test"]
        labels = test_dataset["label"]
        num_samples = len(test_dataset)
        all_features = np.empty((num_samples, 512), dtype="float32")

        logging.info("Starting feature extraction...")
        for i in range(0, num_samples, batch_size):
            batch_images = test_dataset[i : i + batch_size]["img"]
            features = extract_features(batch_images, processor, model, device)
            all_features[i : i + batch_size] = features
            if (i // batch_size) % 10 == 0:
                logging.info(f"Processed {i} / {num_samples} images.")

        logging.info("Feature extraction completed.")

        # Build FAISS index
        index = build_faiss_index(all_features)

        # Perform clustering
        kmeans = perform_clustering(
            vectors=all_features, ncentroids=n_clusters, niter=n_iterations, verbose=True
        )

        # Assign clusters
        assignments, _ = assign_clusters(kmeans, all_features)

        # Compute distribution
        distribution = compute_distribution(assignments, labels, n_clusters=n_clusters)

        # Get class labels
        class_labels = test_dataset.features["label"].names

        # Create and save visualizations
        create_interactive_confusion_matrix(distribution, class_labels)
        create_cluster_distribution_bar_chart(distribution, class_labels)

        # Optionally, save the FAISS index for future use
        faiss.write_index(index, "clip_faiss_index.index")
        logging.info("FAISS index saved as 'clip_faiss_index.index'.")

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()