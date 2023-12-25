import torch
import torch.nn as nn
import torch.nn.functional as F


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(out_channels, 1, kernel_size=1,padding=0),
                                  nn.BatchNorm2d(1),
                                  nn.Sigmoid())
    def forward(self,x):
        x=self.conv(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(out_channels, int(out_channels/2), kernel_size=1,padding=0),
                                   nn.BatchNorm2d(int(out_channels/2)),
                                   nn.ReLU(inplace=True),
                                   )
                        
        self.conv2 = nn.Sequential(nn.Conv2d(int(out_channels/2), out_channels, kernel_size=1,padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Sigmoid(),
                                   )
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        x=self.conv1(x)
        x=self.conv2(x)
        return x
    
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

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3, padding=1,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.spatial_gate = sSE(out_channel)
        self.channel_gate = cSE(out_channel)


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x + g2*x
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
        print(i_channel, s_channel, o_channel)
        block = [
            MyDecoderBlock(i, s, o,)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)

        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            
        last = d
        return last