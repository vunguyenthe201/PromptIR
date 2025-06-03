import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


import os
import random
import sys

from PIL import Image
import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from torch.utils.data import DataLoader
import torchvision 
from PIL import Image



def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution), 
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes

def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
            return paths, sizes
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
            return paths
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))


def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def augment(img, hflip=True, rot=True, mode=None):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    if mode in ['LQ','GT']:
        return _augment(img)
    elif mode in ['LQGT', 'MDGT', 'MD']:
        return [_augment(I) for I in img]

img = read_img(None, "../datasets/Universal_dataset/jpeg-compressed/LQ/0742x3.png", None)
print(img.shape)