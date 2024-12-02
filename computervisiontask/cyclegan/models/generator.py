# models/generator.py

from torch import nn
from .blocks import (
    FeatureMapBlock,
    EncoderBlock,
    DecoderBlock,
    ResidualBlock
)


class Generator(nn.Module):
    """
    Generator model for CycleGAN.

    Attributes:
        upfeature (FeatureMapBlock): Initial feature mapping layer.
        encoder1 (EncoderBlock): First encoder block.
        encoder2 (EncoderBlock): Second encoder block.
        residual_blocks (nn.Sequential): Sequence of residual blocks.
        decoder1 (DecoderBlock): First decoder block.
        decoder2 (DecoderBlock): Second decoder block.
        downfeature (FeatureMapBlock): Final feature mapping layer.
        tanh (nn.Tanh): Tanh activation for output.
    """

    def __init__(self, input_channels: int, output_channels: int, hidden_channels: int = 64, num_residuals: int = 9) -> None:
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.encoder1 = EncoderBlock(hidden_channels)
        self.encoder2 = EncoderBlock(hidden_channels * 2)

        residual_blocks = []
        for _ in range(num_residuals):
            residual_blocks.append(ResidualBlock(hidden_channels * 4))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.decoder1 = DecoderBlock(hidden_channels * 4)
        self.decoder2 = DecoderBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.tanh = nn.Tanh()

    def forward(self, x: nn.Module) -> nn.Module:
        """
        Forward pass of the generator.

        Args:
            x (nn.Module): Input tensor.

        Returns:
            nn.Module: Output tensor after passing through the generator.
        """
        x = self.upfeature(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.residual_blocks(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.downfeature(x)
        return self.tanh(x)