import numpy as np
import os
from scipy.io import loadmat
import cv2

dataset_path = '/home/room304/TB/TB/DATASET/Avenue_Dataset/testing_vol/'
save_folder = '/home/room304/TB/TB/DATASET/Avenue_Dataset/testing_frame'

video_name_list = os.listdir(dataset_path)
video_name_list.sort()

for video_name_iter in video_name_list:
    path_iter = os.path.join(dataset_path, video_name_iter)
    save_path_iter = os.path.join(save_folder, video_name_iter.split('.')[0])
    if not os.path.exists(save_path_iter):
        os.makedirs(save_path_iter)
    mat = loadmat(path_iter)
    label_mask_list = mat['vol']
    frame_length = label_mask_list.shape[2]
    print(save_path_iter)
    print(frame_length)
    for frame_idx in range(frame_length):
        save_frame_path = os.path.join(save_path_iter,'%04d.jpg'%(frame_idx+1))
        im = cv2.resize(label_mask_list[:,:,frame_idx], (640, 360))
        cv2.imwrite(save_frame_path, im)







