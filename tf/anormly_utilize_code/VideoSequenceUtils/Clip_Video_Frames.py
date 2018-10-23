import sys
sys.path.append('/home/room304/storage/vscode/tf/anormly_utilize_code/VideoSequenceUtils/')
sys.path.append('/home/room304/storage/vscode/tf/anormly_utilize_code/ImageProcessUtils/')
import numpy as np
import cv2
import os
import random
from PIL import Image
from skimage.feature import hog
from skimage import io
from optical_flow import *
import time

def get_pointed_frame(video_list, img_num, img_interval, cur_start_idx=0, crop_imgsize = 16):
    frame_concate = []
    for i in range(img_num):
        cur_frame_path = video_list[cur_start_idx + i * img_interval]
        # print cur_frame_path

        im = cv2.imread(cur_frame_path)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cur_frame = np.array(gray_image, dtype=np.float)
        # im = cv2.imread(next_frampe_path)
        # gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # next_frame = np.array(gray_image, dtype=np.float)

        # cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float)
        w, h = cur_frame.shape
        w = w - w % crop_imgsize
        h = h - h % crop_imgsize

        cur_frame_np = np.array(cur_frame, dtype=np.float) / np.float(255.0)
        cur_frame_np = cur_frame_np[np.newaxis, 0:w, 0:h, np.newaxis]

        frame_concate.append(cur_frame_np)
    frame_concate = np.concatenate(frame_concate, axis=0)
    frame_concate = frame_concate[np.newaxis, :, :, :, :]
    return frame_concate

def get_pointed_opticalflow(video_list, img_num, img_interval, cur_start_idx=0, crop_imgsize = 16):
    frame_concate = []
    for i in range(img_num):
        cur_frame_path =  video_list[ cur_start_idx + i * img_interval]
        next_frampe_path = video_list[ cur_start_idx + i * img_interval + 1]
        # print cur_frame_path
        im = cv2.imread(cur_frame_path)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cur_frame = np.array(gray_image, dtype=np.float)
        print(cur_frame.shape)
        im = cv2.imread(next_frampe_path)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        next_frame = np.array(gray_image, dtype=np.float)
        print(next_frame.shape)

        # cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float)
        # next_frame = np.array(Image.open(next_frampe_path).convert('L'), dtype=np.float)
        cur_opticalflow = get_optical_flow_of_frames(cur_frame, next_frame)
        w,h = cur_frame.shape
        w = w - w % crop_imgsize
        h = h - h % crop_imgsize
        cur_opticalflow = cur_opticalflow[np.newaxis, 0:w, 0:h, :]
        frame_concate.append(cur_opticalflow)

    frame_concate = np.concatenate(frame_concate, axis=0)
    frame_concate = frame_concate[np.newaxis, :, :, :, :]
    return frame_concate

def get_pointed_frame_and_opticalflow(video_list, img_num, img_interval, cur_start_idx=0, crop_imgsize = 8,img_size=256):
    frame_concate = []
    for i in range(img_num):
        cur_frame_path =  video_list[ cur_start_idx + i * img_interval]
        next_frampe_path = video_list[ cur_start_idx + i * img_interval + 1]
        # print(cur_frame_path)

        im = cv2.imread(cur_frame_path)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cur_frame = np.array(gray_image, dtype=np.float)
        im = cv2.imread(next_frampe_path)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        next_frame = np.array(gray_image, dtype=np.float)

        # cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float)
        # next_frame = np.array(Image.open(next_frampe_path).convert('L'), dtype=np.float)

        cur_opticalflow = get_optical_flow_of_frames(cur_frame, next_frame)

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


        # cur_frame = np.resize(cur_frame,(img_size, img_size,1))
        # cur_opticalflow = np.resize(cur_opticalflow,(img_size, img_size,2))
        # print cur_frame.shape
        # print cur_opticalflow.shape
        # cur_frame = cur_frame.resize((img_size, img_size))
        # cur_opticalflow = cur_opticalflow.resize((img_size, img_size))
        # print cur_frame.shape
        # w,h = cur_frame.shape

        cur_frame_np = np.array(cur_frame, dtype=np.float) / np.float(255.0)
        cur_frame_np = cur_frame_np[np.newaxis, : , : , np.newaxis]
        cur_opticalflow = cur_opticalflow[np.newaxis, : , : , :]
        # print cur_frame_np.shape
        # print cur_opticalflow.shape
        cur_frame_and_opticalflow = np.concatenate([cur_frame_np, cur_opticalflow], axis=3)
        frame_concate.append(cur_frame_and_opticalflow)
    frame_concate = np.concatenate(frame_concate, axis=0)
    frame_concate = frame_concate[np.newaxis, :, :, :, :]
    return frame_concate

class Clip_Video_Frames_and_OpticalFlow_Randomly_Train:
    def __init__(self, video_frame_list,videopath = '../video_path',batch_size = 4,img_size = 112,img_interval = 4,img_num = 8, frame_tags = True, opticalflow_tags = True):

        self.img_size = img_size
        self.img_interval = img_interval
        self.img_num = img_num
        self.batch_size = batch_size

        self.frame_tags = frame_tags
        self.opticalflow_tags = opticalflow_tags

        self.start_idx = 0
        self.crop_imgsize = 4

        self.video_path = videopath
        
        self.video_list = video_frame_list

        self.selected_frame_range = [0, len(self.video_list) - (img_num + 1) * img_interval - 1]
        # print 'frame_num%d'%(len(self.video_list) )
        # print self.selected_frame_range

    def get_video_frame_batches(self):
        video_frame_sequence_batch = []

        if(self.selected_frame_range[1] - self.selected_frame_range[0] < self.batch_size):
            print ('sample_false')
            return video_frame_sequence_batch

        selected_list = np.array(random.sample(range(self.selected_frame_range[0], self.selected_frame_range[1]), self.batch_size), 'int')
        
        for i in selected_list:
            if self.frame_tags and self.opticalflow_tags:
                video_frame_sequence_batch.append(get_pointed_frame_and_opticalflow(self.video_list,self.img_num,self.img_interval,i,4,self.img_size))
                continue
            if self.frame_tags:
                video_frame_sequence_batch.append(get_pointed_frame(self.video_list,self.img_num,self.img_interval,i))
                continue
            if self.opticalflow_tags:
                video_frame_sequence_batch.append(get_pointed_opticalflow(self.video_list,self.img_num,self.img_interval,i))
                continue
            # video_frame_sequence_batch.append(get_pointed_frame(self.video_list,self.img_num,self.img_interval,i))
            # video_frame_sequence_batch.append(get_pointed_opticalflow(self.video_list,self.img_num,self.img_interval,i))
            # video_frame_sequence_batch.append(get_pointed_frame_and_opticalflow(self.video_list,self.img_num,self.img_interval,i))

        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)

        # print 'batch shape'
        # print video_frame_sequence_batch.shape
        return video_frame_sequence_batch




class Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test:
    def __init__(self, video_frame_list, videopath = './video_path',batch_size = 4,img_size = 112,img_interval = 4,img_num = 8, frame_tags = True, opticalflow_tags = True):
        self.img_size = img_size
        self.img_interval = img_interval
        self.test_interval = 1
        self.img_num = img_num
        self.batch_size = batch_size

        self.crop_imgsize = 8
        self.frame_tags = frame_tags
        self.opticalflow_tags = opticalflow_tags

        self.start_idx = 0
        self.video_path = videopath
        self.video_list = video_frame_list

        self.max_iteration = len(self.video_list)/(batch_size * img_num * img_interval)
        
        self.continued_tags = True
        print(self.max_iteration)

    def get_video_frame_batches(self):
        
        video_frame_sequence_batch = []

        for idx in range(self.batch_size):
            print (self.start_idx)
            print (self.video_list[self.start_idx])
            if self.frame_tags == True and self.opticalflow_tags == True:
                video_frame_sequence_batch.append(get_pointed_frame_and_opticalflow(self.video_list,self.img_num,self.img_interval,self.start_idx))
            if self.frame_tags == True and self.opticalflow_tags == False:
                video_frame_sequence_batch.append(get_pointed_frame(self.video_list,self.img_num,self.img_interval,self.start_idx))
            if self.frame_tags == False and self.opticalflow_tags == True:
                video_frame_sequence_batch.append(get_pointed_opticalflow(self.video_list,self.img_num,self.img_interval,self.start_idx))
            self.start_idx = self.start_idx + self.test_interval

        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)
        # video_frame_sequence_batch = get_pointed_frame(self.video_list,self.img_num,self.img_interval,cur_start_idx=self.start_idx)
        # self.start_idx = self.start_idx + self.test_interval

        if(self.start_idx + self.batch_size * self.img_num * self.img_interval + 1)>(len(self.video_list)):
            self.continued_tags = False

        return video_frame_sequence_batch

