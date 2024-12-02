# main.py

import os
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.generator import Generator
from models.discriminator import Discriminator
from datasets.image_dataset import ImageDataset
from utils.plot_utils import show_tensor_images
from utils.save_utils import save_model_weights, save_plot
from utils.train_utils import (
    weights_init,
    get_disc_loss,
    get_gen_loss
)

import datasets  # Ensure you have the `datasets` library installed


def load_dataset(split: str = "train") -> datasets.Dataset:
    """
    Loads the horse2zebra dataset.

    Args:
        split (str, optional): Dataset split. Defaults to "train".

    Returns:
        datasets.Dataset: Loaded dataset.
    """
    try:
        dataset = datasets.load_dataset("johko/horse2zebra", split=split)
        return dataset
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


def get_transform(load_shape: int, target_shape: int) -> transforms.Compose:
    """
    Returns the transformation pipeline for the images.

    Args:
        load_shape (int): Size to resize the shorter side of the image.
        target_shape (int): Size to crop the image.

    Returns:
        transforms.Compose: Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize(load_shape),
        transforms.RandomCrop(target_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def initialize_models(
    dim_A: int,
    dim_B: int,
    device: torch.device
) -> tuple:
    """
    Initializes the Generator and Discriminator models.

    Args:
        dim_A (int): Number of channels in domain A images.
        dim_B (int): Number of channels in domain B images.
        device (torch.device): Device to move models to.

    Returns:
        tuple: Initialized generators and discriminators.
    """
    gen_AB = Generator(dim_A, dim_B).to(device)
    gen_BA = Generator(dim_B, dim_A).to(device)
    disc_A = Discriminator(dim_A).to(device)
    disc_B = Discriminator(dim_B).to(device)

    # Initialize weights
    gen_AB.apply(weights_init)
    gen_BA.apply(weights_init)
    disc_A.apply(weights_init)
    disc_B.apply(weights_init)

    return gen_AB, gen_BA, disc_A, disc_B


def save_all_models(gen_AB: nn.Module, gen_BA: nn.Module, disc_A: nn.Module, disc_B: nn.Module, folder: str = 'pth') -> None:
    """
    Saves all model weights.

    Args:
        gen_AB (nn.Module): Generator A to B.
        gen_BA (nn.Module): Generator B to A.
        disc_A (nn.Module): Discriminator A.
        disc_B (nn.Module): Discriminator B.
        folder (str, optional): Directory to save the models. Defaults to 'pth'.
    """
    save_model_weights(gen_AB, os.path.join(folder, 'gen_AB.pth'))
    save_model_weights(gen_BA, os.path.join(folder, 'gen_BA.pth'))
    save_model_weights(disc_A, os.path.join(folder, 'disc_A.pth'))
    save_model_weights(disc_B, os.path.join(folder, 'disc_B.pth'))


def train(
    device: torch.device,
    n_epochs: int = 20,
    batch_size: int = 1,
    lr: float = 0.0002,
    display_step: int = 200,
    save_model: bool = False
) -> None:
    """
    Trains the CycleGAN model.

    Args:
        device (torch.device): Device to perform training on.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size. Defaults to 1.
        lr (float, optional): Learning rate. Defaults to 0.0002.
        display_step (int, optional): Steps between displays. Defaults to 200.
        save_model (bool, optional): Whether to save the model after training. Defaults to False.
    """
    try:
        # Load dataset
        raw_dataset = load_dataset()
        transform = get_transform(load_shape=286, target_shape=256)
        dataset = ImageDataset(raw_dataset, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize models
        gen_AB, gen_BA, disc_A, disc_B = initialize_models(dim_A=3, dim_B=3, device=device)

        # Define optimizers
        gen_opt = Adam(
            list(gen_AB.parameters()) + list(gen_BA.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        disc_A_opt = Adam(disc_A.parameters(), lr=lr, betas=(0.5, 0.999))
        disc_B_opt = Adam(disc_B.parameters(), lr=lr, betas=(0.5, 0.999))

        # Define loss functions
        adv_criterion = nn.MSELoss().to(device)
        recon_criterion = nn.L1Loss().to(device)

        mean_generator_loss = 0.0
        mean_discriminator_loss = 0.0
        cur_step = 0

        for epoch in range(n_epochs):
            print(f"Starting Epoch {epoch + 1}/{n_epochs}")
            for real_A, real_B in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                ### Update Discriminator A ###
                disc_A_opt.zero_grad()
                with torch.no_grad():
                    fake_A = gen_BA(real_B)
                disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)
                disc_A_loss.backward()
                disc_A_opt.step()

                ### Update Discriminator B ###
                disc_B_opt.zero_grad()
                with torch.no_grad():
                    fake_B = gen_AB(real_A)
                disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
                disc_B_loss.backward()
                disc_B_opt.step()

                ### Update Generators ###
                gen_opt.zero_grad()
                gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A=real_A,
                    real_B=real_B,
                    gen_AB=gen_AB,
                    gen_BA=gen_BA,
                    disc_A=disc_A,
                    disc_B=disc_B,
                    adv_criterion=adv_criterion,
                    identity_criterion=recon_criterion,
                    cycle_criterion=recon_criterion
                )
                gen_loss.backward()
                gen_opt.step()

                # Accumulate losses
                mean_discriminator_loss += (disc_A_loss.item() + disc_B_loss.item()) / 2 / display_step
                mean_generator_loss += gen_loss.item() / display_step

                ### Visualization and Logging ###
                if cur_step % display_step == 0:
                    print(
                        f"Epoch [{epoch + 1}/{n_epochs}] Step [{cur_step}] "
                        f"Generator Loss: {mean_generator_loss:.4f}, "
                        f"Discriminator Loss: {mean_discriminator_loss:.4f}"
                    )
                    plot_html_A = show_tensor_images(torch.cat([real_A, fake_B]), size=(3, 256, 256), title="Domain A: Real and Fake B")
                    plot_html_B = show_tensor_images(torch.cat([real_B, fake_A]), size=(3, 256, 256), title="Domain B: Real and Fake A")
                    save_plot(plot_html_A, f"epoch_{epoch + 1}_step_{cur_step}_A.html")
                    save_plot(plot_html_B, f"epoch_{epoch + 1}_step_{cur_step}_B.html")
                    mean_generator_loss = 0.0
                    mean_discriminator_loss = 0.0

                cur_step += 1

            ### Save Models After Each Epoch ###
            if save_model:
                save_all_models(gen_AB, gen_BA, disc_A, disc_B)

        ### Final Model Save ###
        if save_model:
            save_all_models(gen_AB, gen_BA, disc_A, disc_B)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")


def main() -> None:
    """
    Main function to execute the training pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train(
        device=device,
        n_epochs=20,
        batch_size=1,
        lr=0.0002,
        display_step=200,
        save_model=True
    )


if __name__ == "__main__":
    main()