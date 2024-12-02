import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_split: str = "train"
    dataset_name: str = "johko/horse2zebra"
    load_shape: int = 286
    target_shape: int = 256
    batch_size: int = 1
    n_epochs: int = 20
    learning_rate: float = 0.0002
    betas: Tuple[float, float] = (0.5, 0.999)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    display_step: int = 200
    model_save_path: str = "pth"
    repo_name: str = "johko/cyclegan-horse2zebra"


class ImageDataset(Dataset):
    """Custom Dataset for CycleGAN."""

    def __init__(
        self,
        dataset: datasets.Dataset,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train'
    ):
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError("dataset must be an instance of datasets.Dataset")
        self.transform = transform
        self.files_A = dataset.filter(lambda x: x['label'] == 2)["image"]
        self.files_B = dataset.filter(lambda x: x['label'] == 3)["image"]

        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise ValueError("Ensure the dataset contains labels 2 and 3 for Horse and Zebra respectively.")

        min_len = min(len(self.files_A), len(self.files_B))
        self.files_A = self.files_A[:min_len]
        self.files_B = self.files_B[:min_len]
        self.new_perm()

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transform is None:
            raise ValueError("Transformations must be provided.")

        item_A = self.transform(self.files_A[index % len(self.files_A)])
        item_B = self.transform(self.files_B[self.randperm[index]])

        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3:
            item_B = item_B.repeat(3, 1, 1)

        if index == len(self) - 1:
            self.new_perm()

        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self) -> int:
        return min(len(self.files_A), len(self.files_B))


def show_tensor_images(
    image_tensor: torch.Tensor,
    num_images: int = 25,
    size: Tuple[int, int, int] = (3, 256, 256)
) -> None:
    """
    Visualize images from a tensor.

    Args:
        image_tensor (torch.Tensor): Tensor containing images.
        num_images (int): Number of images to display.
        size (Tuple[int, int, int]): Size of each image.
    """
    try:
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.axis('off')
        plt.show()
    except Exception as e:
        logger.error(f"Error in show_tensor_images: {e}")


class ResidualBlock(nn.Module):
    """Residual Block with two convolutional layers."""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Encoder Block for downsampling."""

    def __init__(
        self,
        input_channels: int,
        use_bn: bool = True,
        kernel_size: int = 3,
        activation: str = 'relu'
    ):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(
                input_channels,
                input_channels * 2,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                padding_mode='reflect'
            )
        ]

        if use_bn:
            layers.append(nn.InstanceNorm2d(input_channels * 2))

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DecoderBlock(nn.Module):
    """Decoder Block for upsampling."""

    def __init__(self, input_channels: int, use_bn: bool = True):
        super(DecoderBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                input_channels,
                input_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        ]

        if use_bn:
            layers.append(nn.InstanceNorm2d(input_channels // 2))

        layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class FeatureMapBlock(nn.Module):
    """Feature Mapping Block to adjust channel dimensions."""

    def __init__(self, input_channels: int, output_channels: int):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Generator(nn.Module):
    """Generator Network."""

    def __init__(self, input_channels: int, output_channels: int, hidden_channels: int = 64):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            FeatureMapBlock(input_channels, hidden_channels),
            EncoderBlock(hidden_channels),
            EncoderBlock(hidden_channels * 2),
            ResidualBlock(hidden_channels * 2),
            ResidualBlock(hidden_channels * 2),
            ResidualBlock(hidden_channels * 2),
            ResidualBlock(hidden_channels * 2),
            DecoderBlock(hidden_channels * 2),
            DecoderBlock(hidden_channels),
            FeatureMapBlock(hidden_channels, output_channels),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator Network."""

    def __init__(self, input_channels: int, hidden_channels: int = 64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            FeatureMapBlock(input_channels, hidden_channels),
            EncoderBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu'),
            EncoderBlock(hidden_channels * 2, kernel_size=4, activation='lrelu'),
            EncoderBlock(hidden_channels * 4, kernel_size=4, activation='lrelu'),
            nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def weights_init(m: nn.Module) -> None:
    """Initialize model weights."""
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class CycleGANTrainer:
    """Trainer class for CycleGAN."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        # Load dataset
        try:
            dataset = datasets.load_dataset(config.dataset_name, split=config.data_split)
            transform = transforms.Compose([
                transforms.Resize(config.load_shape),
                transforms.RandomCrop(config.target_shape),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.dataset = ImageDataset(dataset, transform=transform)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True if config.device == 'cuda' else False
            )
            logger.info("Dataset loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        # Initialize models
        try:
            self.gen_AB = Generator(3, 3).to(self.device)
            self.gen_BA = Generator(3, 3).to(self.device)
            self.disc_A = Discriminator(3).to(self.device)
            self.disc_B = Discriminator(3).to(self.device)

            self.gen_AB.apply(weights_init)
            self.gen_BA.apply(weights_init)
            self.disc_A.apply(weights_init)
            self.disc_B.apply(weights_init)
            logger.info("Models initialized and weights applied.")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

        # Define loss functions
        self.adv_criterion = nn.MSELoss()
        self.recon_criterion = nn.L1Loss()

        # Optimizers
        self.gen_opt = torch.optim.Adam(
            list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()),
            lr=config.learning_rate,
            betas=config.betas
        )
        self.disc_A_opt = torch.optim.Adam(
            self.disc_A.parameters(),
            lr=config.learning_rate,
            betas=config.betas
        )
        self.disc_B_opt = torch.optim.Adam(
            self.disc_B.parameters(),
            lr=config.learning_rate,
            betas=config.betas
        )

    def get_disc_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: nn.Module
    ) -> torch.Tensor:
        """Calculate discriminator loss."""
        # Real loss
        real_pred = discriminator(real)
        real_label = torch.ones_like(real_pred, device=self.device)
        real_loss = self.adv_criterion(real_pred, real_label)

        # Fake loss
        fake_pred = discriminator(fake.detach())
        fake_label = torch.zeros_like(fake_pred, device=self.device)
        fake_loss = self.adv_criterion(fake_pred, fake_label)

        # Total loss
        total_loss = (real_loss + fake_loss) / 2
        return total_loss

    def get_gen_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate generator loss."""
        # Adversarial loss
        fake_B = self.gen_AB(real_A)
        pred_fake = self.disc_B(fake_B)
        target_real = torch.ones_like(pred_fake, device=self.device)
        adv_loss_AB = self.adv_criterion(pred_fake, target_real)

        fake_A = self.gen_BA(real_B)
        pred_fake_A = self.disc_A(fake_A)
        adv_loss_BA = self.adv_criterion(pred_fake_A, target_real)
        adversarial_loss = adv_loss_AB + adv_loss_BA

        # Cycle consistency loss
        cycle_A = self.gen_BA(fake_B)
        cycle_B = self.gen_AB(fake_A)
        cycle_loss_A = self.recon_criterion(cycle_A, real_A)
        cycle_loss_B = self.recon_criterion(cycle_B, real_B)
        cycle_loss = cycle_loss_A + cycle_loss_B

        # Identity loss
        identity_A = self.gen_BA(real_A)
        identity_B = self.gen_AB(real_B)
        identity_loss_A = self.recon_criterion(identity_A, real_A)
        identity_loss_B = self.recon_criterion(identity_B, real_B)
        identity_loss = identity_loss_A + identity_loss_B

        # Total generator loss
        lambda_identity = 0.5
        lambda_cycle = 10.0
        total_gen_loss = (
            adversarial_loss +
            lambda_identity * identity_loss +
            lambda_cycle * cycle_loss
        )

        return total_gen_loss, fake_A, fake_B

    def train(self, save_model: bool = False) -> None:
        """Training loop for CycleGAN."""
        mean_generator_loss = 0.0
        mean_discriminator_loss = 0.0
        global_step = 0

        for epoch in range(1, self.config.n_epochs + 1):
            logger.info(f"Starting Epoch {epoch}/{self.config.n_epochs}")
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False)

            for real_A, real_B in progress_bar:
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                # ---------------------
                #  Train Discriminators
                # ---------------------
                self.disc_A_opt.zero_grad()
                self.disc_B_opt.zero_grad()

                # Generate fake images
                fake_A = self.gen_BA(real_B)
                fake_B = self.gen_AB(real_A)

                # Calculate discriminator losses
                disc_A_loss = self.get_disc_loss(real_A, fake_A, self.disc_A)
                disc_B_loss = self.get_disc_loss(real_B, fake_B, self.disc_B)
                disc_loss = disc_A_loss + disc_B_loss

                # Backpropagate and optimize
                disc_loss.backward()
                self.disc_A_opt.step()
                self.disc_B_opt.step()

                # -----------------
                #  Train Generators
                # -----------------
                self.gen_opt.zero_grad()

                gen_loss, fake_A, fake_B = self.get_gen_loss(real_A, real_B)
                gen_loss.backward()
                self.gen_opt.step()

                # Logging losses
                mean_generator_loss += gen_loss.item()
                mean_discriminator_loss += disc_loss.item()
                global_step += 1

                if global_step % self.config.display_step == 0:
                    avg_gen_loss = mean_generator_loss / self.config.display_step
                    avg_disc_loss = mean_discriminator_loss / self.config.display_step
                    logger.info(
                        f"Epoch [{epoch}/{self.config.n_epochs}] "
                        f"Step [{global_step}] "
                        f"Generator Loss: {avg_gen_loss:.4f} "
                        f"Discriminator Loss: {avg_disc_loss:.4f}"
                    )
                    show_tensor_images(torch.cat([real_A, real_B]), size=(3, self.config.target_shape, self.config.target_shape))
                    show_tensor_images(torch.cat([fake_B, fake_A]), size=(3, self.config.target_shape, self.config.target_shape))
                    mean_generator_loss = 0.0
                    mean_discriminator_loss = 0.0

            # Save models at the end of each epoch
            if save_model:
                os.makedirs(self.config.model_save_path, exist_ok=True)
                torch.save(self.gen_AB.state_dict(), os.path.join(self.config.model_save_path, 'gen_AB.pth'))
                torch.save(self.gen_BA.state_dict(), os.path.join(self.config.model_save_path, 'gen_BA.pth'))
                torch.save(self.disc_A.state_dict(), os.path.join(self.config.model_save_path, 'disc_A.pth'))
                torch.save(self.disc_B.state_dict(), os.path.join(self.config.model_save_path, 'disc_B.pth'))
                logger.info(f"Models saved at epoch {epoch}.")

        logger.info("Training completed.")


def main():
    """Main function to initiate training."""
    try:
        config = Config()
        trainer = CycleGANTrainer(config)
        trainer.train(save_model=True)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")


if __name__ == "__main__":
    main()