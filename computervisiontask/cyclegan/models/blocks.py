# models/blocks.py

from torch import nn


class ResidualBlock(nn.Module):
    """
    Defines a residual block for the Generator.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        instancenorm (nn.InstanceNorm2d): Instance normalization layer.
        activation (nn.ReLU): Activation function.
    """

    def __init__(self, input_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, input_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.conv2 = nn.Conv2d(
            input_channels, input_channels, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: nn.Module) -> nn.Module:
        """
        Forward pass of the residual block.

        Args:
            x (nn.Module): Input tensor.

        Returns:
            nn.Module: Output tensor after residual connection.
        """
        residual = x
        out = self.conv1(x)
        out = self.instancenorm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.instancenorm(out)
        return residual + out


class EncoderBlock(nn.Module):
    """
    Encoder block for the Generator and Discriminator.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer for downsampling.
        instancenorm (nn.InstanceNorm2d): Instance normalization layer.
        activation (nn.Module): Activation function.
    """

    def __init__(
        self,
        input_channels: int,
        use_bn: bool = True,
        kernel_size: int = 3,
        activation: str = 'relu',
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            input_channels * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            padding_mode='reflect'
        )
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.2, inplace=True)
        self.use_bn = use_bn
        if self.use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)

    def forward(self, x: nn.Module) -> nn.Module:
        """
        Forward pass of the encoder block.

        Args:
            x (nn.Module): Input tensor.

        Returns:
            nn.Module: Output tensor after convolution, normalization, and activation.
        """
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block for the Generator.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolutional layer for upsampling.
        instancenorm (nn.InstanceNorm2d): Instance normalization layer.
        activation (nn.ReLU): Activation function.
    """

    def __init__(self, input_channels: int, use_bn: bool = True) -> None:
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            input_channels,
            input_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.use_bn = use_bn
        if self.use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: nn.Module) -> nn.Module:
        """
        Forward pass of the decoder block.

        Args:
            x (nn.Module): Input tensor.

        Returns:
            nn.Module: Output tensor after transposed convolution, normalization, and activation.
        """
        x = self.conv_transpose(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    """
    Feature mapping block to adjust the number of feature maps.

    Attributes:
        conv (nn.Conv2d): Convolutional layer with a large kernel size.
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect'
        )

    def forward(self, x: nn.Module) -> nn.Module:
        """
        Forward pass of the feature map block.

        Args:
            x (nn.Module): Input tensor.

        Returns:
            nn.Module: Output tensor after convolution.
        """
        return self.conv(x)