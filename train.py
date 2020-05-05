import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __int__(self):
        super(UNet, self).__init__()


class Down(nn.Module):
    def __int__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, )
        )
