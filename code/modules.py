# modules.py
import torch
import torch.nn as nn
import torch.nn.init as init


class ConvReluBlock(nn.Module):
    """卷积+ReLU模块，轻量化"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvReluBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class DownSample(nn.Module):
    """下采样模块，轻量化"""

    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # self.in = nn.InstanceNorm2d(num_channels)

    def forward(self, x):
        return self.relu(self.conv(x))


class UpSampleAdd(nn.Module):
    """上采样模块，使用可分离卷积以减少计算量"""

    def __init__(self, in_channels):
        super(UpSampleAdd, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.cru_half = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        return self.cru_half(torch.cat([self.up(x1), x2], dim=1))  # 上采样后跳跃连接，再经过卷积层恢复通道数(注意顺序)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道注意力的两层卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化，得到 g_c
        g_c = self.avg_pool(x)
        # 通过两层卷积和激活函数
        attention = self.conv1(g_c)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        # 加权特征图 F_c^* = CA_c * F_c
        out = x * attention
        return out


class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        # 像素注意力的两层卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels//4, 1, kernel_size=3, padding=1)
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过两层卷积和激活函数
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        # 加权特征图 F = F_c^* * PA
        out = x * attention
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.pixel_attention = PixelAttention(in_channels)

    def forward(self, x):
        # 先应用通道注意力模块
        x = self.channel_attention(x)
        # 再应用像素注意力模块
        x = self.pixel_attention(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(num_channels)

        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += identity
        out = self.relu(out)
        return out


class FeatureFusionBlock(nn.Module):
    def __init__(self, num_blocks=18, num_channels=64, in_channels=256):
        super(FeatureFusionBlock, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv2d(num_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 创建残差块
        self.residual_blocks = self.create_residual_blocks(num_blocks, num_channels)

    def create_residual_blocks(self, num_blocks, num_channels):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(num_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.input_conv(x)
        x = self.residual_blocks(x)
        x = self.output_conv(x)
        x += identity
        return x