import torch
import torch.nn as nn
"""
SeparableConv3D类：实现了3D版本的深度可分离卷积。它首先在空间维度（深度、高度、宽度）上应用深度卷积，然后在通道维度上应用逐点卷积。
spatial_conv：3D空间卷积，通过groups=in_channels参数实现深度卷积。
pointwise_conv：逐点卷积，它在每个通道上使用1x1x1卷积。
"""
class SeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv3D, self).__init__()
        # 3D空间维度上的卷积
        self.spatial_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        # 深度维度上的逐点卷积
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                        dilation=1, groups=1, bias=bias)

    def forward(self, x):
        x = self.spatial_conv(x)  # 空间卷积
        x = self.pointwise_conv(x)  # 逐点卷积
        return x

# 测试代码
if __name__ == "__main__":
    # 假设输入为 [batch_size, channels, depth, height, width]
    x = torch.randn(1, 16, 32, 32, 32)  # 一个随机的3D输入

    model = SeparableConv3D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    output = model(x)
    print(output.shape)  # 输出的形状应该是 [1, 32, 32, 32, 32]
