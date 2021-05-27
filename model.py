import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Block consisting of a convolutional layer, batch-norm, relu activation and max-pooling (if needed).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool=False, pool_kernel_size=2):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv_bn = nn.BatchNorm2d(out_channels)

        if pool:
            self.pooling = nn.MaxPool2d(pool_kernel_size)
        else:
            self.pooling = None

    def forward(self, x):
        out = F.relu(self.conv_bn(self.conv(x)))
        if self.pooling is not None:
            out = self.pooling(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block consisting of 2 convolutional blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()

        self.conv_block1 = ConvBlock(in_channels, out_channels, kernel_size, padding)
        self.conv_block2 = ConvBlock(in_channels, out_channels, kernel_size, padding)

    def forward(self, x):
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        return out


class ResNet9(nn.Module):
    """
    ResNet consisting of 8 convolutional layers, 1 fully-connected layer and some forward paths for residuals.
    """
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()

        # 1st and 2nd convolutional layer
        self.conv_block1 = ConvBlock(in_channels, 64)
        self.conv_block2 = ConvBlock(64, 128, pool=True)

        # residual block consisting of the 3rd and 4th convolutional layer
        self.res_block1 = ResidualBlock(128, 128)

        # 5th and 6th convolutional layers
        self.conv_block3 = ConvBlock(128, 256, pool=True)
        self.conv_block4 = ConvBlock(256, 512, pool=True)

        # residual block consisting of the 7th and 8th convolutional layer
        self.res_block2 = ResidualBlock(512, 512)

        # final fully-connected layer
        self.classifier = nn.Sequential(nn.MaxPool2d(3),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.res_block1(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.res_block2(out)
        out = self.classifier(out)
        return out
