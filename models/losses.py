import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from util.image_pool import ImagePool
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
from pytorch_msssim import msssim, ssim
###############################################################################
# Functions
###############################################################################

class ContentLoss:
	def __init__(self, loss):
		self.criterion = loss
			
	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm, realIm)

class PerceptualLoss():
	
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss
		
class GANLoss(nn.Module):
	def __init__(
			self, use_l1=True, target_real_label=1.0,
			target_fake_label=0.0, tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_l1:
			self.loss = nn.L1Loss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)

class DiscLoss:
	def name(self):
		return 'DiscLoss'

	def __init__(self, opt, tensor):
		self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)
		self.fake_AB_pool = ImagePool(opt.pool_size)
		
	def get_g_loss(self,net, realA, fakeB):
		# First, G(A) should fake the discriminator
		pred_fake = net.forward(fakeB)
		return self.criterionGAN(pred_fake, 1)
		
	def get_loss(self, net, realA, fakeB, realB):
		# Fake
		# stop backprop to the generator by detaching fake_B
		# Generated Image Disc Output should be close to zero
		self.pred_fake = net.forward(fakeB.detach())
		self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

		# Real
		self.pred_real = net.forward(realB)
		self.loss_D_real = self.criterionGAN(self.pred_real, 1)

		# Combined loss
		self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
		return self.loss_D
		
class DiscLossLS(DiscLoss):
	def name(self):
		return 'DiscLossLS'

	def __init__(self, opt, tensor):
		super(DiscLoss, self).__init__(opt, tensor)
		# DiscLoss.initialize(self, opt, tensor)
		self.criterionGAN = GANLoss(use_l1=True, tensor=tensor)
		
	def get_g_loss(self,net, realA, fakeB):
		return DiscLoss.get_g_loss(self,net, realA, fakeB)
		
	def get_loss(self, net, realA, fakeB, realB):
		return DiscLoss.get_loss(self, net, realA, fakeB, realB)
		
class DiscLossWGANGP(DiscLossLS):
	def name(self):
		return 'DiscLossWGAN-GP'

	def __init__(self, opt, tensor):
		super(DiscLossWGANGP, self).__init__(opt, tensor)
		# DiscLossLS.initialize(self, opt, tensor)
		self.LAMBDA = 10
		
	def get_g_loss(self, net, realA, fakeB):
		# First, G(A) should fake the discriminator
		self.D_fake = net.forward(fakeB)
		return -self.D_fake.mean()
		
	def calc_gradient_penalty(self, netD, real_data, fake_data):
		alpha = torch.rand(1, 1)
		alpha = alpha.expand(real_data.size())
		alpha = alpha.cuda()

		interpolates = alpha * real_data + ((1 - alpha) * fake_data)

		interpolates = interpolates.cuda()
		interpolates = Variable(interpolates, requires_grad=True)
		
		disc_interpolates = netD.forward(interpolates)

		gradients = autograd.grad(
			outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
			create_graph=True, retain_graph=True, only_inputs=True
		)[0]

		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
		return gradient_penalty
		
	def get_loss(self, net, realA, fakeB, realB):
		self.D_fake = net.forward(fakeB.detach())
		self.D_fake = self.D_fake.mean()
		
		# Real
		self.D_real = net.forward(realB)
		self.D_real = self.D_real.mean()
		# Combined loss
		self.loss_D = self.D_fake - self.D_real
		gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
		return self.loss_D + gradient_penalty

class SSIMLoss():
	def __init__(self,metric):
		if metric == 'MSSSIM':
			self.criterion = msssim
		elif metric == 'SSIM':
			self.criterion = ssim

	def get_loss(self, fakeIm, realIm):
		return self.criterion(fakeIm,realIm)

class SSIM(torch.nn.Module):
	@staticmethod
	def gaussian(window_size, sigma):
		gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)])
		return gauss / gauss.sum()

	@staticmethod
	def create_window(window_size, channel):
		_1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
		_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
		window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
		return window

	@staticmethod
	def _ssim(img1, img2, window, window_size, channel, size_average=True):
		mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
		mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

		mu1_sq = mu1.pow(2)
		mu2_sq = mu2.pow(2)
		mu1_mu2 = mu1 * mu2

		sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
		sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
		sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

		C1 = 0.01 ** 2
		C2 = 0.03 ** 2

		ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

		if size_average:
			return ssim_map.mean()
		else:
			return ssim_map.mean(1).mean(1).mean(1)

	def __init__(self, window_size=11, size_average=True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = self.create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = self.create_window(self.window_size, channel)

			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)

			self.window = window
			self.channel = channel

		return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)



def init_loss(opt, tensor):
	# disc_loss = None
	# content_loss = None
	
	if opt.model == 'content_gan':
		content_loss = ContentLoss(nn.L1Loss())
		#ssim_loss=SSIM().cuda()
		msssim_loss=SSIMLoss('MSSSIM')

		L1_loss = ContentLoss(nn.L1Loss())
		#content_loss = PerceptualLoss(nn.MSELoss())
		# content_loss.initialize(nn.MSELoss())
	# elif opt.model == 'pix2pix':
	# 	content_loss = ContentLoss(nn.L1Loss())
		# content_loss.initialize(nn.L1Loss())
	else:
		raise ValueError("Model [%s] not recognized." % opt.model)

	# disc_loss.initialize(opt, tensor)
	return content_loss, L1_loss, msssim_loss