# models/discriminator.py

from torch import nn
from .blocks import FeatureMapBlock, EncoderBlock


class Discriminator(nn.Module):
    """
    Discriminator model for CycleGAN.

    Attributes:
        upfeature (FeatureMapBlock): Initial feature mapping layer.
        contract1 (EncoderBlock): First contracting block.
        contract2 (EncoderBlock): Second contracting block.
        contract3 (EncoderBlock): Third contracting block.
        final_conv (nn.Conv2d): Final convolutional layer to produce a single-channel output.
    """

    def __init__(self, input_channels: int, hidden_channels: int = 64) -> None:
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = EncoderBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = EncoderBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = EncoderBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final_conv = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x: nn.Module) -> nn.Module:
        """
        Forward pass of the discriminator.

        Args:
            x (nn.Module): Input tensor.

        Returns:
            nn.Module: Output tensor after passing through the discriminator.
        """
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.final_conv(x)
        return x