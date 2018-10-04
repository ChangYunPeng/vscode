
from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from os import listdir
from os.path import join, split, splitext
from scipy import misc
import os
import scipy.io as scio
import numpy as np
from TEST_Patch import PatchTest
from turn_label2rgb import turn_label2rgb
import tifffile
from RS_images_utils import turn_mat_uint16_uint8
# ===========================================================
# Argument settings
# ===========================================================

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def load_nor_img(filepath):
    img_4 = tifffile.imread(filepath)
    img = img_4[:,:,0:3] 
    img = misc.imread(filepath)
    max_img = img.max()
    min_img = img.min()
    nor_img = (img - min_img) / (max_img - min_img)
    return nor_img

def preprocessing(data1):
    [r, w, b] = data1.shape
    w_size = 29
    data1_pad = np.pad(data1, ((14, 14), (14, 14), (0, 0)), 'symmetric')
    PatchImage = np.zeros([w_size, w_size, b, r*w])
    mark =0
    for i in range(r):
        for j in range(w):
            PatchImage[:, :, :, mark] = data1_pad[i: i + w_size, j: j + w_size, :]
            mark = mark + 1

    return PatchImage

def conver_images(file_name, model_path, save_file_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    GPU_IN_USE = torch.cuda.is_available()
    model = torch.load(model_path, map_location=lambda storage, loc: storage)

    split_num = 231
    patchCNN = np.zeros([])
    x = file_name
    image = load_nor_img(x)
    cropsize_r = 30
    row = image.shape[0]
    col = image.shape[1]
    patchCNN = np.zeros([row, col]).astype(np.int32)
    for i in range(split_num):
        print('image name: ',x ,'\n  strides -  ', i)
        if i+1 == split_num:
            image_crop = image[i* cropsize_r: row, :, :]
            patchCNN[i* cropsize_r: row, :] = PatchTest(model, GPU_IN_USE, image_crop, [image_crop.shape[0], col])
        else:
            image_crop = image[i* cropsize_r: (i+1)*cropsize_r, :, :]
            patchCNN[i* cropsize_r: (i+1)*cropsize_r, :] = PatchTest(model, GPU_IN_USE, image_crop, [image_crop.shape[0], col])

    rgb_result = turn_label2rgb(patchCNN)
    tifffile.imsave(save_file_name,rgb_result)
    return


if __name__=='__main__':
    conver_images(file_name = '',model_path='./DATA/train/SpatialNets_model_path.pth', save_file_name = '')


