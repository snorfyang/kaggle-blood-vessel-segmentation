import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.resnet import *
from timm.models import create_model

# modify from segmentation_models_pytorch as smp
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

class Net(nn.Module):
    def __init__(self):
        super().__init__() 
        encoder_dim = [64, 256, 512, 1024, 2048]
        decoder_dim = [256, 128, 128, 64, 32 ]

        self.encoder = resnet50d(pretrained=True, in_chans=3)

        self.decoder = MyUnetDecoder(
            in_channel  = encoder_dim[-1],
            skip_channel= encoder_dim[:-1][::-1]+[0],
            out_channel = decoder_dim,
        )
        self.vessel = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)
        self.kidney = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)


    def forward(self, image):
        B, C, H, W = image.shape
        H_pad = (32 - H % 32) % 32
        W_pad = (32 - W % 32) % 32
        x = F.pad(image, (0, W_pad, 0, H_pad), 'constant', 0)
        x = x.expand(-1, 3, -1, -1)

        encode = [] #self.encoder.forward_features(x)
        e = self.encoder
        x = e.conv1(x)
        x = e.bn1(x)
        x = e.act1(x); encode.append(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = e.layer1(x); encode.append(x)
        x = e.layer2(x); encode.append(x)
        x = e.layer3(x); encode.append(x)
        x = e.layer4(x); encode.append(x)
        #[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None]
        )
        #[print(f'decode_{i}', e.shape) for i,e in enumerate(decode)]
        #print('last', last.shape)

        vessel = self.vessel(last).float()
        vessel = F.logsigmoid(vessel).exp()
        vessel = vessel[:, :, :H, :W].contiguous()
        
        kidney = self.kidney(last).float()
        kidney = F.logsigmoid(kidney).exp()
        kidney = kidney[:, :, :H, :W].contiguous()
        
        return vessel, kidney

class NewNet(nn.Module):
    def __init__(self):
        super().__init__() 
        encoder_dim = [32, 64, 96, 192, 384, 768]
        decoder_dim = [256, 128, 128, 64, 32]

        self.encoder = create_model('convnext_small.fb_in22k', pretrained=True, in_chans=3)

        self.decoder = MyUnetDecoder(
            in_channel  = encoder_dim[-1],
            skip_channel= encoder_dim[:-1][::-1]+[0],
            out_channel = decoder_dim,
        )
        self.vessel = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)
        self.kidney = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)
        self.stem0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True))
        self.stem1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True))

    def forward(self, image):
        B, C, H, W = image.shape
        H_pad = (32 - H % 32) % 32
        W_pad = (32 - W % 32) % 32
        x = F.pad(image, (0, W_pad, 0, H_pad), 'constant', 0)
        x = x.expand(-1, 3, -1, -1)

        encode = []
        xx = self.stem0(x); encode.append(xx)
        xx = F.avg_pool2d(xx,kernel_size=2,stride=2)
        xx = self.stem1(xx); encode.append(xx)

        e = self.encoder #convnext
        x = e.stem(x);

        x = e.stages[0](x); encode.append(x)
        x = e.stages[1](x); encode.append(x)
        x = e.stages[2](x); encode.append(x)
        x = e.stages[3](x); encode.append(x)
        # [print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]
        last, _ = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]
        )

        vessel = self.vessel(last).float()
        vessel = F.logsigmoid(vessel).exp()
        vessel = vessel[:, :, :H, :W].contiguous()
        
        kidney = self.kidney(last).float()
        kidney = F.logsigmoid(kidney).exp()
        kidney = kidney[:, :, :H, :W].contiguous()
        
        return vessel, kidney

def run_check_net():
    height, width = 260, 256
    batch_size = 2

    image = torch.from_numpy(np.random.uniform(0, 1, (batch_size, 1, height, width))).float().cuda()

    net = NewNet().cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            v, k = net(image)

    print('image', image.shape)
    print('vessel', v.shape)
    print('kidney', k.shape)

run_check_net()
torch.cuda.empty_cache()

def get_model():
    model = NewNet()
    model = model.cuda()
    return model

def get_old_model():
    model = Net()
    model = model.cuda()
    return model