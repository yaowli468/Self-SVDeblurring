from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import random
import cv2

from scipy.ndimage import filters, measurements, interpolation
from math import pi
from skimage import color
import math

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor[0].cpu().float().numpy()
	image_numpy=np.clip(image_numpy, -1.0, 1.0)
	image_numpy = (np.transpose(image_numpy, (1, 2, 0))+1)/2.0 * 255.0
	return image_numpy.astype(imtype)

def tensor2im_during01(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor[0].cpu().float().numpy()
	image_numpy=np.clip(image_numpy, 0.0, 1.0)
	image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
	return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)


def save_image(image_numpy, image_path):
	image_pil = None
	if image_numpy.shape[2] == 1:
		image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
		image_pil = Image.fromarray(image_numpy, 'L')
	else:
		image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
	"""Print methods and doc strings.
	Takes module, class, list, dictionary, or string."""
	methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
	processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
	print( "\n".join(["%s %s" %
					 (method.ljust(spacing),
					  processFunc(str(getattr(object, method).__doc__)))
					 for method in methodList]) )

def varname(p):
	for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
		m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m:
			return m.group(1)

def print_numpy(x, val=True, shp=False):
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def random_anisotropic_gaussian_kernel(width=15, sig_min=0.2, sig_max=4.0):
	# width : kernel size of anisotropic gaussian filter
	# sig_min : minimum of standard deviation
	# sig_max : maximum of standard deviation
	sig_x = np.random.random() * (sig_max - sig_min) + sig_min
	sig_y = np.random.random() * (sig_max - sig_min) + sig_min
	theta = np.random.random() * 3.141/2.
	inv_cov = inv_covariance_matrix(sig_x, sig_y, theta)
	kernel = anisotropic_gaussian_kernel(width, inv_cov)
	kernel = kernel.astype(np.float32)
	return kernel

def inv_covariance_matrix(sig_x, sig_y, theta):
	# sig_x : x-direction standard deviation
	# sig_x : y-direction standard deviation
	# theta : rotation angle
	D_inv = np.array([[1/(sig_x ** 2), 0.], [0., 1/(sig_y ** 2)]])  # inverse of diagonal matrix D
	U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # eigenvector matrix
	inv_cov = np.dot(U, np.dot(D_inv, U.T))  # inverse of covariance matrix
	return inv_cov

def anisotropic_gaussian_kernel(width, inv_cov):
	# width : kernel size of anisotropic gaussian filter
	ax = np.arange(-width // 2 + 1., width // 2 + 1.)
	# avoid shift
	if width % 2 == 0:
		ax = ax - 0.5
	xx, yy = np.meshgrid(ax, ax)
	xy = np.stack([xx, yy], axis=2)
	# pdf of bivariate gaussian distribution with the covariance matrix
	kernel = np.exp(-0.5 * np.sum(np.dot(xy, inv_cov) * xy, 2))
	kernel = kernel / np.sum(kernel)
	return kernel

def extract_image_patches(x, kernel, stride=1, dilation=1):
	# Do TF 'SAME' Padding
	b, c, h, w = x.shape
	h2 = math.ceil(h / stride)
	w2 = math.ceil(w / stride)
	pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
	pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
	x = torch.nn.functional.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

	# Extract patches
	patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
	patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

	return patches.view(b, -1, patches.shape[-2], patches.shape[-1])

def np_to_torch(img_np):
	'''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
	return torch.from_numpy(img_np)[None, :]

def fill_noise(x, noise_type):
	"""Fills tensor `x` with noise of type `noise_type`."""
	torch.manual_seed(0)
	if noise_type == 'u':
		x.uniform_()
	elif noise_type == 'n':
		x.normal_()
	else:
		assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
	"""Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
	if isinstance(spatial_size, int):
		spatial_size = (spatial_size, spatial_size)
	if method == 'noise':
		shape = [1, input_depth, spatial_size[0], spatial_size[1]]
		net_input = torch.zeros(shape)

		fill_noise(net_input, noise_type)
		net_input *= var
	elif method == 'meshgrid':
		assert input_depth == 2
		X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
		a=X[None, :]
		b=Y[None, :]
		meshgrid = np.concatenate([X[None, :], Y[None, :]])
		net_input = np_to_torch(meshgrid)
	else:
		assert False

	return net_input

class LRScheduler(object):

	def __init__(self, optimizer):
		super(LRScheduler, self).__init__()
		self.optimizer = optimizer

	def update(self, step, learning_rate, ratio=1):

		for i, param_group in enumerate(self.optimizer.param_groups):
			param_group['lr'] = learning_rate * ratio**i


