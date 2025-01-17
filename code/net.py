# net.py
import torch.nn as nn
from modules import ConvReluBlock, DownSample, UpSampleAdd, AttentionBlock, FeatureFusionBlock


class DeHazing(nn.Module):
    def __init__(self):
        super(DeHazing, self).__init__()
        # 编码器模块
        self.crd1 = ConvReluBlock(3, 64)  # conv+relu down
        self.down1 = DownSample(64)
        self.crd2 = ConvReluBlock(64, 128)
        self.down2 = DownSample(128)
        self.crd3 = ConvReluBlock(128, 256)
        self.down3 = DownSample(256)
        # 特征转换模块
        self.FF = FeatureFusionBlock()
        # 特征注意力模块
        self.att1 = AttentionBlock(256)
        self.att2 = AttentionBlock(128)
        self.att3 = AttentionBlock(64)
        # 解码器模块
        self.up1 = UpSampleAdd(256)
        self.cru1 = ConvReluBlock(256, 128)
        self.up2 = UpSampleAdd(128)
        self.cru2 = ConvReluBlock(128, 64)
        self.up3 = UpSampleAdd(64)
        self.cru3 = ConvReluBlock(64, 3)

    def forward(self, x):
        x1 = self.crd1(x)
        x2 = self.crd2(self.down1(x1))
        x3 = self.crd3(self.down2(x2))
        x4 = self.FF(self.down3(x3))
        x4 = self.up1(x4, x3)
        x4 = self.att1(x4)  # 先恢复通道数再注意力机制
        x4 = self.cru1(x4)
        # 合并写
        x4 = self.cru2(self.att2(self.up2(x4, x2)))
        x4 = self.cru3(self.att3(self.up3(x4, x1)))
        return x4


