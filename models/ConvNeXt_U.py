import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models import create_model
from .layers import *


class ConvNeXt_U(nn.Module):
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

        e = self.encoder
        x = e.stem(x);

        x = e.stages[0](x); encode.append(x)
        x = e.stages[1](x); encode.append(x)
        x = e.stages[2](x); encode.append(x)
        x = e.stages[3](x); encode.append(x)
        #[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]
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

    net = ConvNeXt_U().cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            v, k = net(image)

    print('image', image.shape)
    print('vessel', v.shape)
    print('kidney', k.shape)

if __name__ == '__main__':
    run_check_net()
    torch.cuda.empty_cache()