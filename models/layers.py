import torch
import torch.nn as nn
import torch.nn.functional as F


class MyDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3, padding=1,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()
        # self.upconv = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 skip_channel,
                 out_channel,
                 ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock(i, s, o,)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)

        last = d
        return last, decode