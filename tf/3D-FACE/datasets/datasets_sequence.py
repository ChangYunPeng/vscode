import sys
import os
import numpy as np
import random
import cv2
from basic_dataset import basic_dataset, testing_dataset
from ucsd_t1 import TestVideoFile as ucsdt1
from ucsd_t2 import TestVideoFile as ucsdt2

class multi_train_datasets(object):
    def __init__(self, batch_size = 8, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_size = 4, img_size=256):
        self.multi_datasets = []
        self.multi_datasets.append(basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/' ,path='Train', vn_len=8, type_name='tif'))
        self.multi_datasets.append(basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/' ,path='Train', vn_len=8, type_name='tif'))
        self.multi_datasets.append(basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/Avenue_Dataset/' ,path='training_videos', vn_len=2,type_name='jpg'))
        self.multi_datasets.append(basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/ShanghaiTechCampus' ,path='training/videos',vn_len=6, type_name='jpg'))

        self.batch_size = batch_size
        self.video_num = video_num
        self.frame_interval = frame_interval
        self.is_frame = is_frame
        self.is_Optical = is_Optical
        self.crop_size = crop_size
        self.img_size = img_size
        return

    def get_batches(self):
        seletced_dataset_idx = random.randint(0, len(self.multi_datasets) - 1)
        batches = self.multi_datasets[seletced_dataset_idx].get_batches(batch_size = self.batch_size, video_num = self.video_num, frame_interval = self.frame_interval, is_frame = self.is_frame, is_Optical = self.is_Optical, crop_imgsize = self.crop_size, img_size = self.img_size)

        return batches

class multi_test_datasets(object):
    def __init__(self, batch_size = 8, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_size = 4, img_size=256):
        self.multi_datasets = []
        
        self.multi_datasets.append(testing_dataset(dataset_path='/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/' ,path='Test', vn_len=7, type_name='tif',label_list = ucsdt1))

        self.multi_datasets.append(testing_dataset(dataset_path='/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/' ,path='Test', vn_len=7, type_name='tif',label_list = ucsdt2))

        self.multi_datasets.append(basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/' ,path='Test', vn_len=7, type_name='tif'))
        # self.multi_datasets.append(basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/Avenue_Dataset/' ,path='testing_videos', vn_len=2,type_name='jpg'))
        # self.multi_datasets.append(basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/ShanghaiTechCampus' ,path='Testing/videos',vn_len=6, type_name='jpg'))

        self.batch_size = batch_size
        self.video_num = video_num
        self.frame_interval = frame_interval
        self.is_frame = is_frame
        self.is_Optical = is_Optical
        self.crop_size = crop_size
        self.img_size = img_size
        return
    
    def init_test_single_videos(self, seletced_dataset_idx= False):
        if not seletced_dataset_idx:
            self.seletced_dataset_idx = random.randint(0, len(self.multi_datasets) - 1)
        else:
            self.seletced_dataset_idx = seletced_dataset_idx
        video_label  =  self.multi_datasets[self.seletced_dataset_idx].init_video_sequence(selected_video_idx=False,video_num=self.video_num, frame_interval = self.frame_interval, is_frame = self.is_frame, is_Optical = self.is_Optical,crop_imgsize = self.crop_size ,img_size=self.img_size)
        return video_label

    def get_single_videos_batches(self):
        return self.multi_datasets[self.seletced_dataset_idx].get_targetd_video_batches()

if __name__ == '__main__':
    my_multi_test_datasets = multi_test_datasets(batch_size = 4, video_num = 4, frame_interval = 3, is_frame = True, is_Optical = True,crop_size=4, img_size=False)
    video_label  =  my_multi_test_datasets.init_test_single_videos()
    batches = True
    video_lenth = 0
    while 1:
        batches = my_multi_test_datasets.get_single_videos_batches()
        if not (batches == []):
            print(batches.shape)
            video_lenth += (batches.shape[0]*batches.shape[1])
        else:
            break
        # video_lenth += batches.shape
    print('test')
    print('video_length : ', video_lenth)
    print('video_label_length' ,len(video_label))
