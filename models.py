import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import math
from PIL import Image
import numpy as np

from utils import *


class SRPredictor:
    def __init__(self, device):
        self.device = device
        self.model_4x = Generator(num_rrdb_blocks=16, tanh_output=True, upscale_factor=4).to(device)
        self.model_4x_decompress = Generator(num_rrdb_blocks=23, tanh_output=True, upscale_factor=4).to(device)
        self.model_2x_decompress = Generator(num_rrdb_blocks=16, tanh_output=True, upscale_factor=2).to(device)
    
    def load_weights(self):
        #4x
        weights = load_inter_weights("weights/RRDBNet_PSNR_16blocks_dsV4_4x_checkpoint.pth",
                                     "weights/netG_16blocks_dsV4_4x_checkpoint.pth",
                                     alpha=0.8)
        self.model_4x.load_state_dict(weights)
        weights = load_inter_weights("weights/RRDBNet_PSNR_23blocks_dsv4_comprandom_4x_checkpoint.pth",
                                     "weights/netG_23blocks_dsv4_comprandom_4x_checkpoint.pth",
                                     alpha=0.8)
        self.model_4x_decompress.load_state_dict(weights)
        
        #2x
        weights = load_inter_weights("weights/RRDBNet_PSNR_16blocks_dsV4_comprandom_2x_checkpoint_epoch6.pth",
                                     "weights/netG_16blocks_dsV4_comprandom_2x_checkpoint.pth",
                                     alpha=0.85)
        self.model_2x_decompress.load_state_dict(weights)
    
    def _predict_by_patches(self, model, lr_img, batch_size=4, patches_size=192,
                       padding=10, scale=4):
        device = self.device

        lr_image = np.array(lr_img)
        patches, p_shape = split_image_into_overlapping_patches(lr_image, patch_size=patches_size, 
                                                                padding_size=padding)
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device)
        img = convert_image(img, source='[0, 1]', target='imagenet-norm')
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            res = model(img[0:batch_size]).detach().cpu()
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, model(img[i:i+batch_size]).detach().cpu()), 0)

        sr_image = convert_image(res, source='[-1, 1]', target='[0, 1]').permute((0,2,3,1))
        np_sr_image = np.array(sr_image.data)

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(np_sr_image, padded_image_shape=padded_size_scaled,
                                target_shape=scaled_image_shape, padding_size=padding * scale)
        sr_img = Image.fromarray((np_sr_image*255).astype(np.uint8))
        return sr_img
    
    def predict(self, img, scale=4, decompress=False):
        model = None
        sr_img = img
        if scale == 4:
            if decompress:
                model = self.model_4x_decompress
            else:
                model = self.model_4x
        elif scale == 2:
            model = self.model_2x_decompress
        
        if model is not None:
            sr_img = self._predict_by_patches(model, img, scale=scale)
        return sr_img
    

class Discriminator(nn.Module):
    r"""The main architecture of the discriminator. Similar to VGG structure."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input is (3) x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (64) x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (128) x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (256) x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (512) x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (512) x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.features(input)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def discriminator() -> Discriminator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1809.00219>`_ paper.
    """
    model = Discriminator()
    return model


model_urls = {
    "rrdbnet16": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/RRDBNet_4x4_16_DF2K-e31a1b2e.pth",
    "rrdbnet23": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/RRDBNet_4x4_23_DF2K-e31a1b2e.pth",
    "esrgan16": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/ESRGAN_4x4_16_DF2K-57e43f2f.pth",
    "esrgan23": "https://github.com/Lornatang/ESRGAN-PyTorch/releases/download/0.1.0/ESRGAN_4x4_23_DF2K-57e43f2f.pth"
}


class Generator(nn.Module):
    def __init__(self, num_rrdb_blocks=16, tanh_output=False, upscale_factor=4):
        r""" This is an esrgan model defined by the author himself.

        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.

        Args:
            num_rrdb_blocks (int): How many residual in residual blocks are combined. (Default: 16).

        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        super(Generator, self).__init__()
        
        self.num_upsample_blocks = int(math.log2(upscale_factor))
        
        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16/23 ResidualInResidualDenseBlock layer
        rrdb_blocks = []
        for _ in range(num_rrdb_blocks):
            rrdb_blocks += [ResidualInResidualDenseBlock(64, 32, 0.2)]
        self.Trunk_RRDB = nn.Sequential(*rrdb_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        #self.up1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #self.up2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        upsample_blocks = []
        for _ in range(self.num_upsample_blocks):
            upsample_blocks += [InterpolateUpsampleBlock(64,64,kernel_size=3, stride=1, padding=1)]
        self.upblocks = nn.Sequential(*upsample_blocks)
        
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # Final output layer
        if not tanh_output:
            self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(input)
        trunk = self.Trunk_RRDB(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out = self.upblocks(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class InterpolateUpsampleBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(InterpolateUpsampleBlock, self).__init__()
        
        self.up = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding)
   
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.up(F.interpolate(input, scale_factor=2, mode="nearest")), 0.2, True)
    
    
class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        """

        Args:
            channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))

        return conv5.mul(self.scale_ratio) + input


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        """

        Args:
            channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input
