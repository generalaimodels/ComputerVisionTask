# utils/plot_utils.py

from typing import Optional
import plotly.graph_objects as go
import plotly.io as pio
import torch
from torchvision.utils import make_grid
import io


def show_tensor_images(
    image_tensor: torch.Tensor,
    num_images: int = 25,
    size: Optional[tuple] = (3, 256, 256),
    title: Optional[str] = "Image Grid"
) -> str:
    """
    Visualizes images from a tensor using Plotly.

    Args:
        image_tensor (torch.Tensor): Tensor containing images.
        num_images (int, optional): Number of images to display. Defaults to 25.
        size (tuple, optional): Size of each image (C, H, W). Defaults to (3, 256, 256).
        title (str, optional): Title of the plot. Defaults to "Image Grid".

    Returns:
        str: HTML string of the Plotly plot.
    """
    try:
        image_tensor = (image_tensor + 1) / 2  # Normalize to [0,1]
        image_unflat = image_tensor.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        image_np = image_grid.permute(1, 2, 0).numpy()

        fig = go.Figure(
            data=[
                go.Image(
                    source=image_np
                )
            ]
        )
        fig.update_layout(title=title)
        html_str = pio.to_html(fig, full_html=False)
        return html_str
    except Exception as e:
        raise RuntimeError(f"Error in show_tensor_images: {e}")