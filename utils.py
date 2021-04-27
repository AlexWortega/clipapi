import logging
import math
import os
import json
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT

import io
import imageio
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img


def jpegBlur(im,q):
    buf = io.BytesIO()
    imageio.imwrite(buf,im,format='jpg',quality=q)
    s = buf.getbuffer()
    return imageio.imread(s,format='jpg')


def load_inter_weights(psnr_path, gan_path, alpha=0.75):
    psnr_w = torch.load(psnr_path)
    gan_w = torch.load(gan_path)

    inter_w = OrderedDict()

    for k, w_psnr in psnr_w.items():
        w_gan = gan_w[k]
        inter_w[k] = (1 - alpha) * w_psnr + alpha * w_gan

    #for old x4 model
    if "up1.weight" in psnr_w.keys():
        inter_w["upblocks.0.up.weight"] = inter_w.pop("up1.weight")
        inter_w["upblocks.0.up.bias"] = inter_w.pop("up1.bias")
        inter_w["upblocks.1.up.weight"] = inter_w.pop("up2.weight")
        inter_w["upblocks.1.up.bias"] = inter_w.pop("up2.bias")

    return inter_w


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a

    Args:
        seed (int): The desired seed.
    """
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Adam = torch.optim.Adam,
                    file: str = None) -> int:
    r""" Quick loading model functions

    Args:
        model (nn.Module): Neural network model.
        optimizer (torch.optim): Model optimizer. (Default: torch.optim.Adam)
        file (str): Model file.

    Returns:
        How much epoch to start training from.
    """
    if os.path.isfile(file):
        print(f"[*] Loading checkpoint `{file}`.")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[*] Loaded checkpoint `{file}` (epoch {checkpoint['epoch']})")
    else:
        print(f"[!] no checkpoint found at '{file}'")
        epoch = 0

    return epoch


def process_array(image_array, expand=True):
    """ Process a 3-dimensional array into a scaled, 4 dimensional batch of size 1. """
    
    image_batch = image_array / 255.0
    if expand:
        image_batch = np.expand_dims(image_batch, axis=0)
    return image_batch


def process_output(output_tensor):
    """ Transforms the 4-dimensional output tensor into a suitable image format. """
    
    sr_img = output_tensor.clip(0, 1) * 255
    sr_img = np.uint8(sr_img)
    return sr_img


def pad_patch(image_patch, padding_size, channel_last=True):
    """ Pads image_patch with with padding_size edge values. """
    
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Splits the image into partially overlapping patches.
    The patches overlap by padding_size pixels.
    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.
    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size
    
    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size
    
    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')
    
    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)
    
    xmax, ymax, _ = padded_image.shape
    patches = []
    
    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)
    
    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)
    
    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Reconstruct the image from overlapping patches.
    After scaling, shapes and padding should be scaled too.
    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """
    
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size
    
    complete_image = np.zeros((xmax, ymax, 3))
    
    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
        row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size,:
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]