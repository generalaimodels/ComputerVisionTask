# utils/save_utils.py

import os
from typing import Any


def save_model_weights(model: Any, path: str) -> None:
    """
    Saves the model weights to the specified path.

    Args:
        model (Any): The PyTorch model to save.
        path (str): Path where the model weights will be saved.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    except Exception as e:
        raise IOError(f"Error saving model to {path}: {e}")


def save_plot(html_content: str, filename: str, folder: str = 'plots') -> None:
    """
    Saves the Plotly HTML plot to the specified folder.

    Args:
        html_content (str): HTML string of the Plotly plot.
        filename (str): Name of the HTML file.
        folder (str, optional): Directory to save the plot. Defaults to 'plots'.
    """
    try:
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)
        with open(file_path, 'w') as f:
            f.write(html_content)
        print(f"Plot saved to {file_path}")
    except Exception as e:
        raise IOError(f"Error saving plot to {filename}: {e}")