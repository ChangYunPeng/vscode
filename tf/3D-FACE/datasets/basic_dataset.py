import sys
import os
import numpy as np
import random
import cv2
from ucsd_t1 import TestVideoFile as ucsdt1

def video_path_list(dataset_path,video_name_length=7,type_name=7):
    tmp_video_path_list = [ ]
    tmp_video_frame_path_list = [ ]
    for test_video_name in os.listdir(dataset_path):
        # print test_video_name
        if len(test_video_name) == video_name_length:
            # print test_video_name
            video_path_name = os.path.join(dataset_path,test_video_name)
            tmp_video_frame_path_list.append(frame_path_list(video_path_name,type_name))
            tmp_video_path_list.append(video_path_name)
    # tmp_video_path_list.sort()
    # for tmp_video_name in tmp_video_frame_path_list:
    #     print tmp_video_name
    return tmp_video_frame_path_list

def frame_path_list(video_path,type_name = ''):
    frame_fullpath_list = []
    frame_list = os.listdir(video_path)
    frame_list.sort()
    for tmp_frame_path_name in frame_list:
        if tmp_frame_path_name.split('.')[-1] == type_name:
            frame_fullpath_list.append(os.path.join(video_path , tmp_frame_path_name))
    frame_fullpath_list.sort()
    return frame_fullpath_list

def get_pointed_frame(video_list, img_num, img_interval, cur_start_idx=0, crop_imgsize = 8,img_size=256):
    frame_concate = []
    for i in range(img_num):
        cur_frame_path =  video_list[ cur_start_idx + i * img_interval]
        cur_frame = np.array(cv2.cvtColor(cv2.imread(cur_frame_path), cv2.COLOR_BGR2GRAY), dtype=np.float)
        if img_size:
            cur_frame = np.array(cv2.resize(cur_frame,(img_size, img_size)),dtype=np.float)
        else :
            cur_frame = np.array(cur_frame,dtype=np.float)
            w,h = cur_frame.shape
            w = w - w % crop_imgsize
            h = h - h % crop_imgsize
            cur_frame = cur_frame[0:w:,0:h]

        cur_frame_np = np.array(cur_frame, dtype=np.float) / np.float(255.0)
        cur_frame_np = cur_frame_np[np.newaxis, : , : , np.newaxis]
        frame_concate.append(cur_frame_np)
    frame_concate = np.concatenate(frame_concate, axis=0)
    frame_concate = frame_concate[np.newaxis, :, :, :, :]
    return frame_concate

def get_pointed_opticalflow(video_list, img_num, img_interval, cur_start_idx=0, crop_imgsize = 8,img_size=256):
    frame_concate = []
    for i in range(img_num):
        cur_frame_path =  video_list[ cur_start_idx + i * img_interval]
        next_frampe_path = video_list[ cur_start_idx + i * img_interval + 1]
        cur_frame = np.array(cv2.cvtColor(cv2.imread(cur_frame_path), cv2.COLOR_BGR2GRAY), dtype=np.float)
        next_frame = np.array(cv2.cvtColor(cv2.imread(next_frampe_path), cv2.COLOR_BGR2GRAY), dtype=np.float)
        cur_opticalflow = cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if img_size:
            cur_opticalflow = np.array(cv2.resize(cur_opticalflow,(img_size, img_size)),dtype=np.float)
        else :
            cur_opticalflow = np.array(cur_opticalflow,dtype=np.float)
            w,h = cur_frame.shape
            w = w - w % crop_imgsize
            h = h - h % crop_imgsize
            cur_opticalflow = cur_opticalflow[0:w,0:h,:]

        cur_opticalflow = cur_opticalflow[np.newaxis, : , : , :]
        frame_concate.append(cur_opticalflow)
    frame_concate = np.concatenate(frame_concate, axis=0)
    frame_concate = frame_concate[np.newaxis, :, :, :, :]
    return frame_concate

def get_pointed_frame_and_opticalflow(video_list, img_num, img_interval, cur_start_idx=0, crop_imgsize = 8,img_size=256):
    frame_concate = []
    for i in range(img_num):
        cur_frame_path =  video_list[ cur_start_idx + i * img_interval]
        next_frampe_path = video_list[ cur_start_idx + i * img_interval + 1]
        print(cur_frame_path)
        cur_frame = np.array(cv2.cvtColor(cv2.imread(cur_frame_path), cv2.COLOR_BGR2GRAY), dtype=np.float)
        next_frame = np.array(cv2.cvtColor(cv2.imread(next_frampe_path), cv2.COLOR_BGR2GRAY), dtype=np.float)
        cur_opticalflow = cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if img_size:
            cur_frame = np.array(cv2.resize(cur_frame,(img_size, img_size)),dtype=np.float)
            cur_opticalflow = np.array(cv2.resize(cur_opticalflow,(img_size, img_size)),dtype=np.float)
        else :
            cur_frame = np.array(cur_frame,dtype=np.float)
            cur_opticalflow = np.array(cur_opticalflow,dtype=np.float)
            w,h = cur_frame.shape
            w = w - w % crop_imgsize
            h = h - h % crop_imgsize
            cur_frame = cur_frame[0:w:,0:h]
            cur_opticalflow = cur_opticalflow[0:w,0:h,:]

        cur_frame_np = np.array(cur_frame, dtype=np.float) / np.float(255.0)
        cur_frame_np = cur_frame_np[np.newaxis, : , : , np.newaxis]
        cur_opticalflow = cur_opticalflow[np.newaxis, : , : , :]
        cur_frame_and_opticalflow = np.concatenate([cur_frame_np, cur_opticalflow], axis=3)
        frame_concate.append(cur_frame_and_opticalflow)
    frame_concate = np.concatenate(frame_concate, axis=0)
    frame_concate = frame_concate[np.newaxis, :, :, :, :]
    return frame_concate

class basic_dataset(object):
    def __init__(self, dataset_path ,path, vn_len, type_name):
        self.path = os.path.join( dataset_path, path)
        print(self.path)
        self.frame_path_list = video_path_list(self.path,vn_len, type_name)
        
    def get_batches(self, batch_size = 4, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_imgsize = 4 ,img_size=256):
        batches = []
        for idx in range(batch_size):
            seletced_video_idx = random.randint(0, len(self.frame_path_list) - 1)
            # print(seletced_video_idx)
            # print(len(self.frame_path_list[seletced_video_idx]))
            if is_Optical:
                selected_frame_idx = random.randint(0, len(self.frame_path_list[seletced_video_idx]) - (video_num+1) * frame_interval-1)
            else :
                selected_frame_idx = random.randint(0, len(self.frame_path_list[seletced_video_idx]) - (video_num) * frame_interval-1)
            
            if is_frame and is_Optical:
                batches.append(get_pointed_frame_and_opticalflow(video_list = self.frame_path_list[seletced_video_idx],img_num = video_num,img_interval = frame_interval, cur_start_idx=selected_frame_idx, crop_imgsize = crop_imgsize,img_size=img_size))
            elif is_frame :
                batches.append(get_pointed_frame(video_list = self.frame_path_list[seletced_video_idx],img_num = video_num,img_interval = frame_interval, cur_start_idx=selected_frame_idx, crop_imgsize = crop_imgsize,img_size=img_size))
            else :
                batches.append(get_pointed_opticalflow(video_list = self.frame_path_list[seletced_video_idx],img_num = video_num,img_interval = frame_interval, cur_start_idx=selected_frame_idx, crop_imgsize = crop_imgsize,img_size=img_size))
        batches = np.concatenate(batches, axis=0)
        # print(batches.shape)
        return batches
    
    def get_selected_batches(self, seletced_video_idx,selected_frame_idx, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_imgsize = 4 ,img_size=256):
        batches = []
        if is_frame and is_Optical:
            batches.append(get_pointed_frame_and_opticalflow(video_list = self.frame_path_list[seletced_video_idx],img_num = video_num,img_interval = frame_interval, cur_start_idx=selected_frame_idx, crop_imgsize = crop_imgsize,img_size=img_size))
        elif is_frame :
            batches.append(get_pointed_frame(video_list = self.frame_path_list[seletced_video_idx],img_num = video_num,img_interval = frame_interval, cur_start_idx=selected_frame_idx, crop_imgsize = crop_imgsize,img_size=img_size))
        else :
            batches.append(get_pointed_opticalflow(video_list = self.frame_path_list[seletced_video_idx],img_num = video_num,img_interval = frame_interval, cur_start_idx=selected_frame_idx, crop_imgsize = crop_imgsize,img_size=img_size))
        return batches[0]

class testing_dataset(basic_dataset):
    def __init__(self, dataset_path ,path, vn_len, type_name, label_list):
        super(testing_dataset, self).__init__(dataset_path ,path, vn_len, type_name)
        self.label_list = label_list
        
    def init_video_sequence(self, selected_video_idx = False, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_imgsize = 4 ,img_size=256):
        self.videos_end = False
        if not selected_video_idx:
            self.seletced_video_idx = random.randint(0, len(self.frame_path_list) - 1)
        else :
            self.seletced_video_idx =  selected_video_idx
        
        self.seletced_frame_idx = 0        
        target_frame_idx = []
        sample_num = np.int((len(self.frame_path_list[self.seletced_video_idx]) - 1)/(video_num* frame_interval))
        selected_label = []
        for sample_idx in range(sample_num):
            target_frame_idx.append(sample_idx*video_num*frame_interval)
            for img_idx in range(video_num):
                frame_idx = sample_idx*video_num*frame_interval + img_idx * frame_interval
                print(frame_idx)
                selected_label.append(self.label_list[self.seletced_video_idx][frame_idx])
        
        self.moving_idx = 0
        # print(selected_label)
        # print(target_frame_idx)
        self.target_frame_list = target_frame_idx


        self.video_num = video_num
        self.frame_interval = frame_interval
        self.is_frame = is_frame
        self.is_Optical = is_Optical
        self.crop_size = crop_imgsize
        self.img_size = img_size
        return selected_label
        
    def get_targetd_video_batches(self, batch_size = 4):
        if not self.videos_end:
            if self.moving_idx + batch_size <len(self.target_frame_list):
                range_0 = self.moving_idx
                range_1 = self.moving_idx + batch_size
                self.moving_idx += batch_size
            else:
                range_0 = self.moving_idx
                range_1 = len(self.target_frame_list)
                self.videos_end = True
                self.moving_idx = 0
            batches = []
            for sample_idx in range(range_0, range_1):
                target_idx = self.target_frame_list[sample_idx]
                print(target_idx)
                batches.append(self.get_selected_batches(self.seletced_video_idx,target_idx, video_num = self.video_num, frame_interval = self.frame_interval, is_frame = self.is_frame, is_Optical = self.is_Optical,crop_imgsize = self.crop_size ,img_size=self.img_size))
            batches = np.concatenate(batches, axis=0)
            print(batches.shape)
            return batches
        else:
            return []
        return 
    # def get_selected_batches(self, seletced_video_idx,selected_frame_idx, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_imgsize = 4 ,img_size=256):
    #     super(testing_dataset, self).get_selected_batches(seletced_video_idx,selected_frame_idx, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_imgsize = 4 ,img_size=256)

if __name__ == '__main__':
    # ave  =  basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/Avenue_Dataset/' ,path='training_videos', vn_len=2,type_name='jpg')
    # ave.get_batches(batch_size = 8, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True)
    # ucsd1  =  basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/' ,path='Train', test_path='Test' ,vn_len=8, test_vn_len=7, type_name='tif', test_type_name='tif')
    # ucsd1.get_batches(batch_size = 8, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = False)
   ucsd1_t = testing_dataset(dataset_path='/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/' ,path='Test', vn_len=7, type_name='tif',label_list = ucsdt1)
   ucsd1_t.init_video_sequence(selected_video_idx = False, video_num = 8, frame_interval = 1)
    # shtec = basic_dataset(dataset_path='/home/room304/TB/TB/DATASET/ShanghaiTechCampus' ,path='training/videos', vn_len=6,type_name='jpg')
    # shtec.get_batches(batch_size = 8, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True)

    
