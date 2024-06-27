import pdb

import PIL.Image
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable

import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .networks import weights_init
from .losses import init_loss
import torchvision.transforms as transforms
from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib
import cv2
import random
from scipy import signal
from util.util import get_noise
from util.ranger import Ranger
from util.util import LRScheduler

matplotlib.use('Agg')

try:
    xrange  # Python2
except NameError:
    xrange = range  # Python 3

def plot_kernel(out_k_np, step, index, image_name):
    plt.clf()

    ax = plt.subplot(111)
    im = ax.imshow(out_k_np, vmin=out_k_np.min(), vmax=out_k_np.max())
    plt.colorbar(im, ax=ax)
    ax.set_title('Estimate Kernel')

    # plt.show()
    plt.savefig('./checkpoints/kernel/image_'+str(image_name) +'_step_' + str(step) +'_index_'+str(index) + '.png')

def spatial_batch_blur(image, kernel, kernel_size):
    if kernel_size % 2 == 1:
        image_pad = torch.nn.functional.pad(image,
                                            (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate') #mode='reflect'
    else:
        image_pad = torch.nn.functional.pad(image, (
            kernel_size // 2, kernel_size - kernel_size // 2, kernel_size // 2, kernel_size - kernel_size // 2), mode='replicate')

    H_p, W_p = image_pad.size()[-2:]
    B, C, H, W = image.size()
    image_pad = image_pad.view(B * C, 1, H_p, W_p)
    image_pad_unfold = torch.nn.functional.unfold(image_pad, kernel_size).transpose(1, 2)  # [B*C, HW, K*K]

    kernel_unfold = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)
    kernel_unfold = kernel_unfold.contiguous().view(-1, kernel_unfold.size(2), kernel_unfold.size(3))

    kernel_unfold = kernel_unfold.permute(0, 2, 1)  # [B*C, H*W, K*K]

    out = (image_pad_unfold * kernel_unfold).sum(2).unsqueeze(1)  #[B*C, H*W, K*K]×[B*C, H*W, K*K]

    out = torch.nn.functional.fold(out, (H, W), 1).view(B, C, H, W)

    return out

def compute_Lp_norm(kernels_tensor, p):
    N, K, S, S = kernels_tensor.shape
    output = 0
    for n in range(N):
        for k in range(K):
            kernel = kernels_tensor[n, k, :, :]
            p_norm = torch.pow(torch.sum(torch.pow(torch.abs(kernel), p)), 1. / p)
            output = output + p_norm
    return output/(N*K)

def compute_total_variation_loss(img):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).abs()).sum()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).abs()).sum()
    return  (tv_h + tv_w)/(img.shape[0]*img.shape[1])

def compute_kernels_regularization_loss(kernels, KERNELS_REGULARIZATION_TYPE):

    kernels_regularization_loss = 0.

    if KERNELS_REGULARIZATION_TYPE == 'L2':
        kernels_regularization_loss = torch.mean(kernels ** 2)
    if KERNELS_REGULARIZATION_TYPE == 'L1':
        kernels_regularization_loss = torch.mean(
            torch.abs(kernels))
    elif KERNELS_REGULARIZATION_TYPE == 'TV':
        kernels_regularization_loss = compute_total_variation_loss(kernels)
    elif KERNELS_REGULARIZATION_TYPE == 'Lp':
        kernels_regularization_loss = compute_Lp_norm(kernels, p=0.5)

    return kernels_regularization_loss

class Self_Spatial_Variant_Deblurring(BaseModel):
    def name(self):
        return 'TrainModel'

    def __init__(self, opt):
        super(Self_Spatial_Variant_Deblurring, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.kernel_size = opt.kernel_size
        self.batch_size = opt.batchSize
        self.save_root = opt.saveroot

        self.blur_dir=os.path.join(self.save_root, 'blur')
        self.reblur_dir=os.path.join(self.save_root, 'reblur')
        self.sharp_dir=os.path.join(self.save_root, 'sharp')

        self.n_styles = 14

        use_parallel = not opt.gan_type == 'wgan-gp'

        self.res_encoder, self.stylegan, self.SD_net = networks.define_G(
            opt.phase, opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
            not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual, opt.kernel_size
        )

        stylegan_ckpt = torch.load(opt.pretrainStyleGanModel)
        self.stylegan.load_state_dict(stylegan_ckpt['g_ema'], strict=False)
        loaded_latent_avg = stylegan_ckpt['latent_avg'].cuda()
        latent_avg = loaded_latent_avg.repeat(self.n_styles, 1)

        self.latent_avg=latent_avg.repeat(self.batch_size, 1, 1)
        latent_avg_image, _ =self.stylegan([self.latent_avg], noise=None, input_is_latent=True, randomize_noise=True,
                                                                return_latents=True)
        self.latent_avg_image=latent_avg_image.float().detach()

        self.optimizer = Ranger([{'params': self.res_encoder.parameters(), 'lr': 0.00025},       #previous
                                 {'params': self.stylegan.parameters(), 'lr': 0.00025},
                                 {'params': self.SD_net.parameters(), 'lr': 0.0001}])

        self.contentLoss, self.l1loss, self.ssimLoss = init_loss(opt, self.Tensor)

    def set_input(self, input):
        # pdb.set_trace()
        self.input_blur = input['A'].cuda()
        self.image_path = input['image_path'][0]


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_latent_image(self, step):
        if step==1:
            encoder_input=torch.cat((self.input_blur, self.input_blur), dim=1)
            latent_codes, noises = self.res_encoder(encoder_input)
            self.latent_codes = latent_codes + self.latent_avg
            self.latent_image, _ = self.stylegan([self.latent_codes], noise=noises, input_is_latent=True, randomize_noise=True,
                                                                   return_latents=True)
        else:
            latent_image_hat=self.latent_image.clone().detach()    #.requires_grad_(requires_grad=True)

            encoder_input=torch.cat((self.input_blur, latent_image_hat), dim=1)
            latent_codes, noises =self.res_encoder(encoder_input)
            self.latent_codes = latent_codes + self.latent_avg
            self.latent_image, _ = self.stylegan([self.latent_codes], noise=noises, input_is_latent=True, randomize_noise=True,
                                                                  return_latents=True)

        #self.image_out=latent_image
        self.image_out=torch.nn.functional.adaptive_avg_pool2d(self.latent_image, output_size=(256, 256))


    def optimize_parameters(self, step):

        self.optimizer.zero_grad()

        self.get_latent_image(step)
        #kernel_net_input=torch.cat((self.input_blur, self.latent_image), dim=1)
        kernel_net_input=self.input_blur
        self.kernel_out = self.SD_net(kernel_net_input)

        self.re_blur_out = spatial_batch_blur(self.image_out, self.kernel_out, self.kernel_size)
        self.main_loss = self.contentLoss.get_loss(self.re_blur_out, self.input_blur) * 10

        self.ssim_loss= (1 - self.ssimLoss.get_loss(self.re_blur_out, self.input_blur))*20

        self.regular_kernel_loss=compute_kernels_regularization_loss(self.kernel_out, 'TV')*0.005

        if step<500:
            self.loss=self.main_loss + self.regular_kernel_loss
        else:
            self.loss=self.main_loss + self.ssim_loss + self.regular_kernel_loss

        self.loss.backward()

        self.optimizer.step()

    def get_current_errors(self):
        return OrderedDict([  # ('G_GAN', self.loss_G_GAN.item()),
            ('Main_Loss', self.main_loss.item()),
            ('SSIM_Loss', self.ssim_loss.item()),
            ('Regular_Kernel_Loss', self.regular_kernel_loss.item())
        ])

    def get_current_visuals(self):
        self.input_blur_image = util.tensor2im_during01(self.input_blur.data)
        self.restored_sharp_image = util.tensor2im_during01(self.image_out.data)
        self.reblur_image = util.tensor2im_during01(self.re_blur_out.data)

    def save_image(self, step):
        image_name = os.path.basename(self.image_path)
        image_name = str(image_name).split('.')[-2]

        input_blur_image = Image.fromarray(self.input_blur_image)
        restored_sharp_image = Image.fromarray(self.restored_sharp_image)
        reblur_image = Image.fromarray(self.reblur_image)
        #reblur_image_shift_left = Image.fromarray(self.reblur_image_shift_left)

        input_blur_image.save(os.path.join(self.blur_dir, '%s_%s.png' % (image_name, step)))
        restored_sharp_image.save(os.path.join(self.sharp_dir, '%s_%s_sharp.png' % (image_name, step)))
        reblur_image.save(os.path.join(self.reblur_dir, '%s_%s_reblur.png' % (image_name, step)))

    def save(self, label):
        self.save_network(self.SD_net, 'kernel', label, self.gpu_ids)
        self.save_network(self.res_encoder, 'Image', label, self.gpu_ids)

    def save_kernel(self, step):
        detached_kernel_out=self.kernel_out.detach()
        all_kernel_out_flat=torch.flatten(detached_kernel_out, start_dim=2).permute(0, 2, 1)
        all_kernel=all_kernel_out_flat.view(self.batch_size, all_kernel_out_flat.size(1), self.kernel_size, self.kernel_size)
        one_kernel=all_kernel[:, 0, :, :][0].float().cpu()
        two_kernel=all_kernel[:, 2687, :, :][0].float().cpu()     #256*10+127=2687            #256*10=2560
        three_kernel=all_kernel[:, 32639, :, :][0].float().cpu()  #256*127+127=32639          #256*127=32512
        fourth_kernel=all_kernel[:, 63103, :, :][0].float().cpu()  #256*246+127=63103         #256*246=62976

        image_name = os.path.basename(self.image_path).split('.')[0]
        plot_kernel(one_kernel, step, 0, image_name)
        plot_kernel(two_kernel, step, 2687, image_name)
        plot_kernel(three_kernel, step, 32639, image_name)
        plot_kernel(fourth_kernel, step, 63103, image_name)


    def update_learning_rate(self, step):
        old_lr = self.optimizer.param_groups[0]['lr']
        #self.generator_scheduler.step()
        #self.kernel_net_scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        if step % 20 == 0:
            print('update learning rate: %8f -> %8f' % (old_lr, current_lr))