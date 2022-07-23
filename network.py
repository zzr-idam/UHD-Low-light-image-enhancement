# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision

from unet_model import UNet
from  torchvision import utils as vutils



class ConvBlock(nn.Module):
    def __init__(self, inc=3 , outc=3, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.PReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        # Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        
        return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        return torch.cat([R, G, B], dim=1)

class GuideNN(nn.Module):
    def __init__(self):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(3, 16, kernel_size=3, padding=1, batch_norm=True)
        self.conv2 = ConvBlock(16, 16, kernel_size=3, padding=1, batch_norm=True)
        self.conv3 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh) #nn.Tanh

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))

class Derivative(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_local = UNet(n_channels=3)
        self.u_g = UNet(n_channels=3)
        self.conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, stride=1,padding=1),
            nn.PReLU(),
            nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv3d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

    def forward(self, img):
        img_x = F.interpolate(img, (256,256), mode='bicubic', align_corners=True)
        f1 = self.u_local(img_x)
        f2 = torch.cat((img_x, f1), dim=1)
        f3 = self.conv(f2)
        output = self.u_g(f3) + f1
        output = output.view(-1,12,64,16,16)
        output = self.smooth(output.view(-1,12,64,16,16))
        return output


class HDRPointwiseNN(nn.Module):
    
    def __init__(self):
        super(HDRPointwiseNN, self).__init__()
        self.coeffs = Derivative()
        self.guide = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
       
        

    def forward(self, image):

        coeffs = self.coeffs(image)
        guide = self.guide(image)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, image)

        return out * image - out + 1





if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 1024
    width = 1024

    model = HDRPointwiseNN()


    x = torch.randn((2, 3, height, width))
    x = model(x)
    print(x.shape)


