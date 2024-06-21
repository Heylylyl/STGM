import torch
import torch.nn as nn
from kpn.network import KernelConv
import kpn.utils as kpn_utils
import numpy as np
from src.deform_conv_v2 import DeformConv2d
from PIL import Image

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.filter_type = config.FILTER_TYPE
        self.kernel_size = config.kernel_size

        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )
        self.d_firstconv = DeformConv2d(4, 32, kernel_size=3, padding=0)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        self.d_secondconv = DeformConv2d(64, 64, kernel_size=3, padding=1,stride=2)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True))

        self.channelconv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,padding=1),
            nn.ReLU(True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=0))
            # DeformConv2D(64,3, kernel_size=3, padding=0),
            # nn.ReLU(True),
        self.channelconv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=3),
            nn.ReLU(True)
        )

        self.d_lastconv = DeformConv2d(3,3, kernel_size=3, padding=0)
        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        self.kpn_model = kpn_utils.create_generator()

        if init_weights:
            self.init_weights()

    def save_featuremap(self,img):
        tensor = (img - img.min()) * (255 / (img.max() - img.min()))
        image = Image.fromarray(tensor.squeeze().cpu().numpy(), mode='L')

        image.save(f"D:\hyl\misfformal/result\kernel/0.jpg")
        return 0

    def forward(self, x):
        inputs = x.clone()

        x1 = self.encoder0(x)
        x2 = self.d_firstconv(x)
        x_cat1 = torch.cat((x1,x2),dim = 1)

        x = self.encoder1(x_cat1)
        x3 = self.d_secondconv(x_cat1)
        x_cat2 = torch.cat((x,x3),dim=1)

        kernels, kernels_img = self.kpn_model(inputs, x_cat2)

#        self.save_featuremap(kernels_img)

        x = self.encoder2(x_cat2)
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)

        x = self.middle(x)

        x = self.decoder1(x)
        x_cat3 = torch.cat((x,x_cat2),dim=1)
        x = self.channelconv1(x_cat3)

        x = self.decoder2(x)
        x_cat4 = torch.cat((x,x_cat1),dim=1)
        x = self.channelconv2(x_cat4)

        x = self.decoder3(x)

        x = self.d_lastconv(x)

        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        x = (torch.tanh(x) + 1) / 2

        return x

    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
