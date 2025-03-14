import torch
import torch.nn as nn


class ECA3D(nn.Module):
    def __init__(self, channels, k_size=3):
        """
        3D 版 Efficient Channel Attention (ECA) 模块.

        Args:
            channels (int): 输入通道数。
            k_size (int): 1D 卷积核大小，用于捕获局部通道交互，默认为 3。
        """
        super(ECA3D, self).__init__()
        # 3D 全局平均池化，输出形状为 (B, C, 1, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 对通道描述进行 1D 卷积，输入形状为 (B, 1, C)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        # Squeeze: 得到每个通道的全局描述，形状为 (B, C)
        y = self.avg_pool(x).view(b, c)
        # 将描述转为 (B, 1, C) 以适配 1D 卷积
        y = y.unsqueeze(1)
        # 进行 1D 卷积操作捕获通道局部交互
        y = self.conv1d(y)
        # Sigmoid 激活获得通道权重，形状依然为 (B, 1, C)
        y = self.sigmoid(y)
        # 调整形状为 (B, C, 1, 1, 1) 以便与输入特征相乘
        y = y.view(b, c, 1, 1, 1)
        return x * y


class DMDS3D_ECA(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3):
        """
        3D 版混合下采样模块 (HDSB)，采用 Efficient Channel Attention 替代 SE 模块.

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数，通过 1x1x1 卷积进行转换。
            k_size (int): ECA 模块中 1D 卷积的卷积核大小。
        """
        super(DMDS3D_ECA, self).__init__()

        # 分支1：3D 最大池化，提取显著纹理特征
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        # 分支2：3D 平均池化，保留更多背景信息
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
        # 分支3：3D 卷积，使用 3x3x3 卷积进行自适应特征提取与下采样
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        # 拼接后通道数为 3 * in_channels，使用 ECA 模块进行通道注意力加权
        self.eca = ECA3D(channels=3 * in_channels, k_size=k_size)
        # 1x1x1 卷积转换通道数为 out_channels
        self.conv1 = nn.Conv3d(3 * in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 分别对输入进行下采样
        feat_max = self.maxpool(x)  # 输出尺寸: (B, in_channels, D/2, H/2, W/2)
        feat_avg = self.avgpool(x)  # 输出尺寸: (B, in_channels, D/2, H/2, W/2)
        feat_conv = self.conv3(x)  # 输出尺寸: (B, in_channels, D/2, H/2, W/2)

        # 在通道维度上拼接三个分支的输出，拼接后尺寸为 (B, 3*in_channels, D/2, H/2, W/2)
        concat_features = torch.cat([feat_max, feat_avg, feat_conv], dim=1)
        # 使用 ECA 模块进行通道注意力加权
        eca_features = self.eca(concat_features)
        # 最后通过 1x1x1 卷积转换通道数
        out = self.conv1(eca_features)
        return out


# 示例用法：
if __name__ == "__main__":
    # 构造一个 3D 输入张量，形状为 (batch, channels, depth, height, width)
    x = torch.randn(1, 64, 32, 128, 128)
    # 初始化 HDSB 模块，采用 ECA 模块，保持输入和输出通道数一致（可根据需求调整）
    model = DMDS3D_ECA(in_channels=64, out_channels=64, k_size=3)
    y = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
