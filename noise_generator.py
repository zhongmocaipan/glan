

# import torch
# import torch.nn as nn

# class NoiseGenerator(nn.Module):
#     def __init__(self):
#         super(NoiseGenerator, self).__init__()
#         self.noise_layer = nn.Sequential(
#             nn.Conv2d(3, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 3, kernel_size=3, padding=1),
#             # nn.ReLU(),
#             # nn.Conv2d(64, 3, kernel_size=3, padding=1),
#             # nn.ReLU(),
#             # nn.Conv2d(64, 3, kernel_size=3, padding=1)
#         )
#         # 加载保存的噪声
#         self.noise = torch.load("noise1.pt")

#     def forward(self, shape):
#         # 确保加载的噪声与输入形状一致
#         if self.noise.shape != shape:
#             raise ValueError(f"Loaded noise shape {self.noise.shape} does not match input shape {shape}")
#         return self.noise_layer(self.noise)

# class Discriminator(nn.Module):
#     def __init__(self, input_channels=3):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 1, kernel_size=3, stride=2, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)
import torch
import torch.nn as nn


# 选择设备（如果有 GPU 则使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ 噪声生成器优化版
class ResidualBlock(nn.Module):
    """ 残差块: 提高网络的稳定性和噪声生成质量 """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # 残差连接


class NoiseGenerator(nn.Module):
    def __init__(self):
        super(NoiseGenerator, self).__init__()
        self.noise_layer = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, shape):
        # 生成一个新的噪声张量，并确保它在正确的设备上
        noise = torch.randn(shape, device=next(self.parameters()).device)  
        return self.noise_layer(noise)


# 2️⃣ 判别器优化版（对抗网络部分）
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        ).to(device)  # 直接移动到 GPU

    def forward(self, x):
        x = x.to(device)  # 确保输入数据在 GPU
        return self.model(x)
