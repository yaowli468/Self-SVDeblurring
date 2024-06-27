import pdb

import torch
import torch.nn as nn
import functools
import numpy as np
import collections
import math
import random
import functools
import itertools
from collections import OrderedDict
from .base_model import Upscale2d, Downscale2d
from util.util import extract_image_patches
from sklearn.manifold import TSNE
import torch.nn.functional as F
from collections import namedtuple

from util.styleganOp import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Initial resolution.
_INIT_RES = 4


###############################################################################
# Functions
###############################################################################

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def define_G(phase, input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[],
             use_parallel=True,
             learn_residual=False, kernel_size=31):
    netG = None
    use_gpu = len(gpu_ids) > 0
    #norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    #if which_model_netG == 'my_resnet_9block':

    Encoder=GradualStyleEncoder(50, 'ir_se')
    Generator=StyleGAN(256, 512, 8)
    SD_Net = Spatial_Degradation_Net(kernel_size=kernel_size)


    if len(gpu_ids) > 0:
        Encoder.cuda(gpu_ids[0])
        SD_Net.cuda(gpu_ids[0])
        Generator.cuda(gpu_ids[0])
    Encoder.apply(weights_init)
    SD_Net.apply(weights_init)
    Generator.apply(weights_init)
    return Encoder, Generator, SD_Net


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class MAconv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super(MAconv, self).__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                nn.Conv2d(in_split_rest, int(in_split_rest // reduction), 1, 1, 0, True),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(in_split_rest // reduction), in_split * 2, 1, 1, 0, True),
            ]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_split, out_split, kernel_size, stride, padding, bias))

    def forward(self, input):
        input = torch.split(input, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            # a=torch.cat(input[:i] + input[i + 1:], 1)
            # b=getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1))
            scale, translation = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(input[:i] + input[i + 1:], 1)),
                                             (self.in_split[i], self.in_split[i]), dim=1)
            output.append(getattr(self, 'conv{}'.format(i))(input[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)


class MABlock(nn.Module):
    ''' Residual block based on MAConv '''

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 split=2, reduction=2):
        super(MABlock, self).__init__()

        self.res = nn.Sequential(*[
            MAconv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction),
            nn.LeakyReLU(inplace=True),
            MAconv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction),
        ])

    def forward(self, x):
        # return x + self.res(x)
        return self.res(x)


class Spatial_Degradation_Net(nn.Module):
    ''' Network of Spatial_Degradation_Net'''

    def __init__(self, in_nc=3, kernel_size=31, nc=[128, 256], nb=1, split=2):
        super(Spatial_Degradation_Net, self).__init__()
        self.kernel_size = kernel_size

        self.m_head = nn.Conv2d(in_channels=in_nc, out_channels=nc[0], kernel_size=3, padding=1, bias=True)
        self.m_down1 = sequential(*[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)],
                                  nn.Conv2d(in_channels=nc[0], out_channels=nc[1], kernel_size=2, stride=2, padding=0,
                                            bias=True))

        self.m_body = sequential(*[MABlock(nc[1], nc[1], bias=True, split=split) for _ in range(nb)])

        self.m_up1 = sequential(nn.ConvTranspose2d(in_channels=nc[1], out_channels=nc[0],
                                                   kernel_size=2, stride=2, padding=0, bias=True),
                                *[MABlock(nc[0], nc[0], bias=True, split=split) for _ in range(nb)])
        self.m_tail = nn.Conv2d(in_channels=nc[0], out_channels=kernel_size ** 2, kernel_size=3, padding=1, bias=True)

        self.softmax = nn.Softmax(1)

    def forward(self, x):
        h, w = x.size()[-2:]

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x = self.m_body(x2)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        x = self.softmax(x)

        return x

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride', 'is_max'])):
    """ A named tuple describing a ResNet block. """

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride, is_max=False)] + [Bottleneck(depth, depth, stride=1, is_max=True) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=128, depth=256, num_units=3),   #block_num=3
            get_block(in_channel=256, depth=512, num_units=4),   #block_num=4
            get_block(in_channel=512, depth=512, num_units=6),   #block_num=6
            get_block(in_channel=512, depth=512, num_units=3),   #block_num=3
            get_block(in_channel=512, depth=512, num_units=3),   #block_num=3
            #get_block(in_channel=512, depth=512, num_units=3)    #block_num=3
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride, is_max=True):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = torch.nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                torch.nn.BatchNorm2d(depth)
            )
        self.res_layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), torch.nn.PReLU(depth),
            torch.nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), torch.nn.BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride, is_max=True):
        super(bottleneck_IR_SE, self).__init__()
        if is_max == True:
            self.shortcut_layer = torch.nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                torch.nn.BatchNorm2d(depth)
            )
        self.res_layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channel),
            torch.nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            torch.nn.PReLU(depth),
            torch.nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            torch.nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                torch.nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class large_style_translation(nn.Module):
    def __init__(self, f_channel, f_size, f_num):
        super(large_style_translation, self).__init__()

        self.large_style_batch=torch.nn.BatchNorm2d(f_channel)
        self.large_style_pooling=torch.nn.AdaptiveAvgPool2d((f_size, f_size))
        self.large_style_linear=torch.nn.Linear(f_channel * f_size * f_size, f_channel * f_num)

    def forward(self, input):
        output_B=self.large_style_batch(input)
        output_P=self.large_style_pooling(output_B)
        output_P=output_P.view(input.size(0), -1)
        output_L=self.large_style_linear(output_P)

        return output_L

class medium_style_translation(nn.Module):
    def __init__(self, f_channel, f_size, f_num):
        super(medium_style_translation, self).__init__()

        self.medium_style_batch=torch.nn.BatchNorm2d(f_channel)
        self.medium_style_pooling=torch.nn.AdaptiveAvgPool2d((f_size, f_size))
        self.medium_style_linear=torch.nn.Linear(f_channel * f_size * f_size, f_channel * f_num)

    def forward(self, input):
        output_B=self.medium_style_batch(input)
        output_P=self.medium_style_pooling(output_B)
        output_P=output_P.view(input.size(0), -1)
        output_L=self.medium_style_linear(output_P)

        return output_L

class small_style_translation(nn.Module):
    def __init__(self, f_channel, f_size, f_num):
        super(small_style_translation, self).__init__()

        self.small_style_batch=torch.nn.BatchNorm2d(f_channel)
        self.small_style_pooling=torch.nn.AdaptiveAvgPool2d((f_size, f_size))
        self.small_style_linear=torch.nn.Linear(f_channel * f_size * f_size, f_channel * f_num)

    def forward(self, input):
        #pdb.set_trace()
        output_B=self.small_style_batch(input)
        output_P=self.small_style_pooling(output_B)
        output_P=output_P.view(input.size(0), -1)
        output_L=self.small_style_linear(output_P)

        return output_L

class style_translation(nn.Module):
    def __init__(self, f_channel, f_size, f_num):
        super(style_translation, self).__init__()

        self.style_batch=torch.nn.BatchNorm2d(f_channel)
        self.style_pooling=torch.nn.AdaptiveAvgPool2d((f_size, f_size))
        #self.style_linear=torch.nn.Linear(f_channel * f_size * f_size, f_channel * f_num)

    def forward(self, input):
        #pdb.set_trace()
        output_B=self.style_batch(input)
        output_P=self.style_pooling(output_B)
        output_P=output_P.view(input.size(0), -1)
        #output_L=self.style_linear(output_P)

        return output_P


class GradualStyleEncoder(nn.Module):
    def __init__(self, num_layers, mode='ir', input_nc=6, n_styles=14):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'

        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = torch.nn.Sequential(torch.nn.Conv2d(input_nc, 128, (3, 3), 1, 1, bias=False),
                                               torch.nn.BatchNorm2d(128),
                                               torch.nn.PReLU(128))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride,
                                           bottleneck.is_max))
        self.body = torch.nn.Sequential(*modules)

        # self.large_style_layer=large_style_translation(512, 7, 8)
        # self.medium_style_layer=medium_style_translation(512, 7, 4)
        # self.small_style_layer=small_style_translation(512, 7, 2)

        self.large_style_layer=style_translation(512, 3, 9)
        self.medium_style_layer=style_translation(512, 2, 4)
        self.small_style_layer=style_translation(512, 1, 1)

    def forward(self, x):
        noises=[]
        x = self.input_layer(x)
        noises.append(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, module_ir_se in enumerate(modulelist):
            x = module_ir_se(x)
            if i == 12:
                c1 = x
            elif i == 15:
                c2 = x
            elif i == 18:
                c3 = x
            if i==2 or i==6:   # or i==15: #or i==18 or i==21:
               noises.append(x)
            if i==12 or i==15 or i==18:   #or i==17:
                noises.append(None)
        noises.append(None)

        style_large=self.large_style_layer(c1).view(-1, 9, 512)
        style_medium=self.medium_style_layer(c2).view(-1, 4, 512)
        style_small=self.small_style_layer(c3).view(-1, 1, 512)
        out=torch.cat((style_large, style_medium, style_small), dim=1)

        noises = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noises))[::-1]

        return out, noises

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class StyleGAN(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size       #输出维度

        self.style_dim = style_dim     #style维度

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        #pdb.set_trace()
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            noise,
            return_latents=False,
            return_features=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            randomize_noise=True,
    ):

        # if not input_is_latent:
        #     styles = [self.style(s) for s in styles]   #z-->W

        #pdb.set_trace()
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers +[None]
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[2::2], noise[3::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        else:
            return image, None

