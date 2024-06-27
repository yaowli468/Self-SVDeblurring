import os
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn


class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.phase=opt.phase
        self.checkpoints_dir=opt.checkpoints_dir
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.phase)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def delete(self):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        self.load_dir = os.path.join(self.checkpoints_dir, 'pre_train')
        save_path = os.path.join(self.load_dir, save_filename)

        network.load_state_dict(torch.load(save_path))
        #a=torch.load(save_path)

    def load_all_network(self, network, network_label, epoch_label, current_satge):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        self.load_dir = os.path.join(self.checkpoints_dir, current_satge)
        save_path = os.path.join(self.load_dir, save_filename)

        model_dict=network.state_dict()
        pretrained_dict=torch.load(save_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)

    def update_learning_rate():
        pass

    def generate_blur(self,tensor_image,degree,angle):
        image_numpy=tensor_image.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_pil=image_numpy.astype(np.uint8)
        image_cv2=cv2.cvtColor(image_pil,cv2.COLOR_RGB2BGR)
        #cv2.imwrite('./1.png',image_cv2)
        image=np.array(image_cv2)
        #A_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #A_img.save('./1.png')

        #A_path = self.AB_paths[index]
        #A_img = cv2.imread(A_path)
        #image = np.array(A_img)
        #degree = random.randrange(20, 50, 2)
        #angle = random.randint(40, 60)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        A_blur = np.array(blurred, dtype=np.uint8)

        #A_img = Image.fromarray(cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB))
        # A_img.save("./1.png")
        #A_img = Image.fromarray(image)
        #A_img.save("./1.png")
        A_blur = Image.fromarray(cv2.cvtColor(A_blur, cv2.COLOR_BGR2RGB))
        #A_blur=np.array(A_blur)
        #A_blur.save("./2.png")
        return A_blur

    def reset_grads(self, model,require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)
        return model

    def shuffle_pixel(self, img, p=0.3):
        if p == 0:
            return img.clone()
        *_, h, w = img.shape
        out = img.clone()
        original_idx = torch.arange(h * w)
        shuffle_idx = torch.randperm(h * w)
        shuffle_idx = torch.where(torch.rand(*original_idx.shape) > p, original_idx, shuffle_idx)
        out = out.view(4, 3, -1)[:, :, shuffle_idx].view(*out.shape)
        return out

class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)

class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        return F.avg_pool2d(x, self.factor)

class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x