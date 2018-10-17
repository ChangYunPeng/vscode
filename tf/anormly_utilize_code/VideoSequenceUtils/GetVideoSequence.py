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
from ImageProcessUtils.optical_flow import *
import time

class Clip_Video_Frames_and_OpticalFlow_Randomly_Train:
    def __init__(self, batch_size = 4,img_size = 112,img_interval = 4,video_name_length=9,img_num = 8,dataset_path = '../DATASET/0104009'):
        self.img_size = img_size
        self.img_interval = img_interval
        self.test_interval = 2
        self.img_num = img_num
        self.batch_size = batch_size

        self.dataset_path = dataset_path
        self.start_idx = 0
        self.crop_imgsize = 64
        tmp_list = os.listdir(self.dataset_path)
        tmp_list.sort()
        tmp_test_video_list = []
        for test_video_name in (tmp_list):
            if len(test_video_name) == video_name_length:
                tmp_test_video_list.append(test_video_name)
                if len(tmp_test_video_list)>=12000:
                    break
        self.video_list = tmp_test_video_list

        self.selected_frame_range = [0, len(self.video_list) - (img_num + 1) * img_interval - 1]
        # print 'frame_num%d'%(len(self.video_list) )
        # print self.selected_frame_range

    def get_video_frame_batches(self):

        selected_list = np.array(random.sample(range(self.selected_frame_range[0], self.selected_frame_range[1]), self.batch_size), 'int')
        video_frame_sequence_batch = []
        for i in selected_list:
            # video_frame_sequence_batch.append(self.get_pointed_frame_and_opticalflow(i))
            video_frame_sequence_batch.append(self.get_pointed_frame(i))
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)

        # print 'batch shape'
        # print video_frame_sequence_batch.shape
        return video_frame_sequence_batch

    def get_pointed_frame(self, cur_start_idx=0):
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path + '/' + self.video_list[cur_start_idx + i * self.img_interval]
            cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float32)
            w, h = cur_frame.shape
            w = w - w % self.crop_imgsize
            h = h - h % self.crop_imgsize

            cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, 0:w, 0:h, np.newaxis]

            # print cur_frame_and_opticalflow.shape
            # cur_opticalflow
            # cur_frame_and_opticalflow
            # cur_frame_np

            frame_concate.append(cur_frame_np)
        frame_concate = np.concatenate(frame_concate, axis=0)
        frame_concate = frame_concate[np.newaxis, :, :, :, :]
        # print 'single_input_shape'
        # print(frame_concate.shape)
        return frame_concate

    def get_pointed_frame_and_opticalflow(self,cur_start_idx=0):
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path + '/' + self.video_list[ cur_start_idx + i * self.img_interval]
            next_frampe_path = self.dataset_path + '/' + self.video_list[ cur_start_idx + i * self.img_interval + 1]
            # print('Current img-th%d' % i)
            # print(cur_frame_path)
            cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float32)
            next_frame = np.array(Image.open(next_frampe_path).convert('L'), dtype=np.float32)
            cur_opticalflow = get_optical_flow_of_frames(cur_frame, next_frame)
            w,h = cur_frame.shape
            w = w - w % self.crop_imgsize
            h = h - h % self.crop_imgsize
            # cur_frame = Image.open(cur_frame_path).convert('L')
            # cur_frame = cur_frame.resize((self.img_size, self.img_size))
            cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, 0:w, 0:h, np.newaxis]
            cur_opticalflow = cur_opticalflow[np.newaxis, 0:w, 0:h, :]
            cur_frame_and_opticalflow = np.concatenate([cur_frame_np, cur_opticalflow], axis=3)
            # print cur_frame_and_opticalflow.shape
            # cur_opticalflow
            # cur_frame_and_opticalflow
            # cur_frame_np

            frame_concate.append(cur_frame_and_opticalflow)
        frame_concate = np.concatenate(frame_concate, axis=0)
        frame_concate = frame_concate[np.newaxis, :, :, :, :]
        # print 'single_input_shape'
        # print(frame_concate.shape)
        return frame_concate

class Clip_Video_Frames_and_HOG_Randomly_Train:
    def __init__(self, batch_size = 4,img_size = 112,img_interval = 4,video_name_length=9,img_num = 8,dataset_path = '../DATASET/0104009'):
        self.img_size = img_size
        self.img_interval = img_interval
        self.test_interval = 2
        self.img_num = img_num
        self.batch_size = batch_size

        self.dataset_path = dataset_path
        self.start_idx = 0
        tmp_list = os.listdir(self.dataset_path)
        tmp_list.sort()
        tmp_test_video_list = []
        for test_video_name in (tmp_list):
            if len(test_video_name) == video_name_length:
                tmp_test_video_list.append(test_video_name)
                if len(tmp_test_video_list)>=12000:
                    break
        self.video_list = tmp_test_video_list

        self.selected_frame_range = [0, len(self.video_list) - (img_num + 1) * img_interval - 1]

        self.hog_descriptor = cv2.HOGDescriptor()
        # print 'frame_num%d'%(len(self.video_list) )
        # print self.selected_frame_range

    def get_video_frame_batches(self):

        selected_list = np.array(random.sample(range(self.selected_frame_range[0], self.selected_frame_range[1]), self.batch_size), 'int')
        video_frame_sequence_batch = []
        for i in selected_list:
            video_frame_sequence_batch.append(self.get_pointed_frame_and_opticalflow(i))
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)

        # print 'batch shape'
        # print video_frame_sequence_batch.shape
        return video_frame_sequence_batch

    def get_pointed_frame_and_opticalflow(self,cur_start_idx=0):
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path + '/' + self.video_list[ cur_start_idx + i * self.img_interval]
            next_frampe_path = self.dataset_path + '/' + self.video_list[ cur_start_idx + i * self.img_interval + 1]
            # print('Current img-th%d' % i)
            # print(cur_frame_path)
            cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float32)

            ski_image = io.imread(cur_frame_path,as_grey=True)
            # fg,hog_image = hog(ski_image,orientations=8,pixels_per_cell=())

            cur_frame_u = np.array(Image.open(cur_frame_path).convert('L'), dtype='uint8')
            cur_hog = self.hog_descriptor.compute(cur_frame_u)
            print cur_frame.shape
            print cur_hog.shape
            next_frame = np.array(Image.open(next_frampe_path).convert('L'), dtype=np.float32)
            cur_opticalflow = get_optical_flow_of_frames(cur_frame, next_frame)
            w,h = cur_frame.shape
            w = w - w % 8
            h = h - h % 8
            # cur_frame = Image.open(cur_frame_path).convert('L')
            # cur_frame = cur_frame.resize((self.img_size, self.img_size))
            cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, 0:w, 0:h, np.newaxis]
            cur_opticalflow = cur_opticalflow[np.newaxis, 0:w, 0:h, :]
            cur_frame_and_opticalflow = np.concatenate([cur_frame_np, cur_opticalflow], axis=3)
            # print cur_frame_and_opticalflow.shape
            # cur_opticalflow
            # cur_frame_and_opticalflow
            # cur_frame_np

            frame_concate.append(cur_frame_and_opticalflow)
        frame_concate = np.concatenate(frame_concate, axis=0)
        frame_concate = frame_concate[np.newaxis, :, :, :, :]
        # print 'single_input_shape'
        # print(frame_concate.shape)
        return frame_concate

class Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test:
    def __init__(self, batch_size = 4,img_size = 112,img_interval = 4,video_name_length=9,img_num = 8,dataset_path = '../DATASET/0104009'):
        self.img_size = img_size
        self.img_interval = img_interval
        self.test_interval = 2
        self.img_num = img_num
        self.batch_size = batch_size

        self.crop_imgsize = 64

        self.dataset_path = dataset_path
        self.start_idx = 0
        tmp_list = os.listdir(self.dataset_path)
        tmp_list.sort()
        tmp_test_video_list = []
        for test_video_name in (tmp_list):
            if len(test_video_name) == video_name_length:
                tmp_test_video_list.append(test_video_name)
                if len(tmp_test_video_list)>=12000:
                    break
        self.video_list = tmp_test_video_list

        self.max_iteration = len(self.video_list)/(batch_size * img_num * img_interval)
        self.continued_tags = True
        print(self.max_iteration)
        # print(len(self.video_list))
        # for test_name in self.video_list:
        #     print(test_name)

    def get_video_frame_batches(self):


        video_frame_sequence_batch = []
        for i in range(self.batch_size):
            video_frame_sequence_batch.append(self.get_orderly_only_frames())
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)

        if(self.start_idx + self.batch_size * self.img_num * self.img_interval + 1)>(len(self.video_list)):
            self.continued_tags = False
        print video_frame_sequence_batch.shape
        return video_frame_sequence_batch

    def get_orderly_only_frames(self):
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path + '/' + self.video_list[self.start_idx + i*self.img_interval]
            next_frampe_path = self.dataset_path + '/' + self.video_list[self.start_idx + (i + 1)*self.img_interval]
            print('Current img-th%d' % i)
            print(cur_frame_path)
            cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float32)
            # next_frame = np.array(Image.open(next_frampe_path).convert('L'), dtype=np.float32)
            # cur_opticalflow = get_optical_flow_of_frames(cur_frame,next_frame)

            w, h = cur_frame.shape
            w = w - w % self.crop_imgsize
            h = h - h % self.crop_imgsize
            cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, 0:w, 0:h, np.newaxis]
            # cur_opticalflow = cur_opticalflow[np.newaxis, 0:w, 0:h, :]

            # print cur_opticalflow.shape
            # cur_frame = Image.open(cur_frame_path).convert('L')
            # # cur_frame = cur_frame.resize((self.img_size, self.img_size))
            # cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            # cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
            # cur_opticalflow = cur_opticalflow[np.newaxis,:,:,:]
            # cur_frame_and_opticalflow = np.concatenate([cur_frame_np,cur_opticalflow],axis=3)
            # print cur_frame_and_opticalflow.shape
            frame_concate.append(cur_frame_np)
        frame_concate = np.concatenate(frame_concate, axis=0)
        frame_concate = frame_concate[np.newaxis, :, :, :, :]
        print(frame_concate.shape)
        self.start_idx = self.start_idx + self.test_interval
        return frame_concate

    def get_orderly_frames(self):
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path + '/' + self.video_list[self.start_idx + i*self.img_interval]
            next_frampe_path = self.dataset_path + '/' + self.video_list[self.start_idx + (i + 1)*self.img_interval]
            print('Current img-th%d' % i)
            print(cur_frame_path)
            cur_frame = np.array(Image.open(cur_frame_path).convert('L'), dtype=np.float32)
            next_frame = np.array(Image.open(next_frampe_path).convert('L'), dtype=np.float32)
            cur_opticalflow = get_optical_flow_of_frames(cur_frame,next_frame)

            w, h = cur_frame.shape
            w = w - w % 8
            h = h - h % 8
            cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, 0:w, 0:h, np.newaxis]
            cur_opticalflow = cur_opticalflow[np.newaxis, 0:w, 0:h, :]

            # print cur_opticalflow.shape
            # cur_frame = Image.open(cur_frame_path).convert('L')
            # # cur_frame = cur_frame.resize((self.img_size, self.img_size))
            # cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            # cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
            # cur_opticalflow = cur_opticalflow[np.newaxis,:,:,:]
            cur_frame_and_opticalflow = np.concatenate([cur_frame_np,cur_opticalflow],axis=3)
            print cur_frame_and_opticalflow.shape
            frame_concate.append(cur_frame_and_opticalflow)
        frame_concate = np.concatenate(frame_concate, axis=0)
        frame_concate = frame_concate[np.newaxis, :, :, :, :]
        print(frame_concate.shape)
        self.start_idx = self.start_idx + self.test_interval
        return frame_concate

# class Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test:
#     def __init__(self, batch_size = 4,img_size = 112,img_interval = 4,test_video_name_length=9,img_num = 8,dataset_path = '../DATASET/0104009'):
#         self.img_size = img_size
#         self.img_interval = img_interval
#         self.test_interval = 2
#         self.img_num = img_num
#         self.batch_size = batch_size
#
#         self.dataset_path = dataset_path
#         self.start_idx = 0
#         tmp_list = os.listdir(self.dataset_path)
#         tmp_list.sort()
#         tmp_test_video_list = []
#         for test_video_name in (tmp_list):
#             if len(test_video_name) == test_video_name_length:
#                 tmp_test_video_list.append(test_video_name)
#                 if len(tmp_test_video_list)>=12000:
#                     break
#         self.video_list = tmp_test_video_list
#
#         self.max_iteration = len(self.video_list)/(batch_size * img_num * img_interval)
#         self.continued_tags = True
#         print(self.max_iteration)
#         # print(len(self.video_list))
#         # for test_name in self.video_list:
#         #     print(test_name)
#
#     def get_video_frame_batches(self):
#
#
#         video_frame_sequence_batch = []
#         for i in range(self.batch_size):
#             video_frame_sequence_batch.append(self.get_orderly_frames())
#         video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)
#
#         if(self.start_idx + self.batch_size * self.img_num * self.img_interval)>(len(self.video_list)):
#             self.continued_tags = False
#         print video_frame_sequence_batch.shape
#         return video_frame_sequence_batch
#
#     def get_orderly_frames(self):
#         frame_concate = []
#         for i in range(self.img_num):
#             cur_frame_path = self.dataset_path + '/' + self.video_list[self.start_idx + i*self.img_interval]
#             next_frampe_path = self.dataset_path + '/' + self.video_list[self.start_idx + i * self.img_interval + 1]
#             print('Current img-th%d' % i)
#             print(cur_frame_path)
#             cur_frame = Image.open(cur_frame_path).convert('L')
#             # cur_frame = cur_frame.resize((self.img_size, self.img_size))
#             cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
#             cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
#             frame_concate.append(cur_frame_np)
#         frame_concate = np.concatenate(frame_concate, axis=0)
#         frame_concate = frame_concate[np.newaxis, :, :, :, :]
#         print(frame_concate.shape)
#         self.start_idx = self.start_idx + self.test_interval
#         return frame_concate

class Clip_Video_Frames_Orderly_Sequence_Test:
    def __init__(self, batch_size = 4,img_size = 112,img_interval = 4,test_video_name_length=9,img_num = 8,dataset_path = '../DATASET/0104009'):
        self.img_size = img_size
        self.img_interval = img_interval
        self.test_interval = 2
        self.img_num = img_num
        self.batch_size = batch_size

        self.dataset_path = dataset_path
        self.start_idx = 0
        tmp_list = os.listdir(self.dataset_path)
        tmp_list.sort()
        tmp_test_video_list = []
        for test_video_name in (tmp_list):
            if len(test_video_name) == test_video_name_length:
                tmp_test_video_list.append(test_video_name)
                if len(tmp_test_video_list)>=12000:
                    break
        self.video_list = tmp_test_video_list

        self.max_iteration = len(self.video_list)/(batch_size * img_num * img_interval)
        self.continued_tags = True
        print(self.max_iteration)
        # print(len(self.video_list))
        # for test_name in self.video_list:
        #     print(test_name)

    def get_video_frame_batches(self):


        video_frame_sequence_batch = []
        for i in range(self.batch_size):
            video_frame_sequence_batch.append(self.get_orderly_frames())
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)

        if(self.start_idx + self.batch_size * self.img_num * self.img_interval)>(len(self.video_list)):
            self.continued_tags = False
        print video_frame_sequence_batch.shape
        return video_frame_sequence_batch

    def get_orderly_frames(self):
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path + '/' + self.video_list[self.start_idx + i*self.img_interval]
            print('Current img-th%d' % i)
            print(cur_frame_path)
            cur_frame = Image.open(cur_frame_path).convert('L')
            cur_frame = cur_frame.resize((self.img_size, self.img_size))
            cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
            frame_concate.append(cur_frame_np)
        frame_concate = np.concatenate(frame_concate, axis=0)
        frame_concate = frame_concate[np.newaxis, :, :, :, :]
        print(frame_concate.shape)
        self.start_idx = self.start_idx + self.test_interval
        return frame_concate

class Clip_Video_Frames_Orderly_Sequence:
    def __init__(self, batch_size = 4,img_size = 112,img_interval = 4,test_video_name_length=9,img_num = 8,dataset_path = '../DATASET/0104009'):
        self.img_size = img_size
        self.img_interval = img_interval
        self.test_interval = 2
        self.img_num = img_num
        self.batch_size = batch_size

        self.dataset_path = dataset_path
        self.start_idx = 0
        tmp_list = os.listdir(self.dataset_path)
        tmp_list.sort()
        tmp_test_video_list = []
        for test_video_name in (tmp_list):
            if len(test_video_name) == test_video_name_length:
                tmp_test_video_list.append(test_video_name)
                if len(tmp_test_video_list)>=12000:
                    break
        self.video_list = tmp_test_video_list

        self.max_iteration = len(self.video_list)/(batch_size * img_num * img_interval)
        print(self.max_iteration)
        # print(len(self.video_list))
        # for test_name in self.video_list:
        #     print(test_name)

    def get_video_frame_batches(self):


        video_frame_sequence_batch = []
        for i in range(self.batch_size):
            video_frame_sequence_batch.append(self.get_orderly_frames())
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)
        print video_frame_sequence_batch.shape
        return video_frame_sequence_batch

    def get_orderly_frames(self):
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path + '/' + self.video_list[self.start_idx + i*self.img_interval]
            print('Current img-th%d' % i)
            print(cur_frame_path)
            cur_frame = Image.open(cur_frame_path).convert('L')
            cur_frame = cur_frame.resize((self.img_size, self.img_size))
            cur_frame_np = np.array(cur_frame, dtype=np.float32) / np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
            frame_concate.append(cur_frame_np)
        frame_concate = np.concatenate(frame_concate, axis=0)
        frame_concate = frame_concate[np.newaxis, :, :, :, :]
        print(frame_concate.shape)
        self.start_idx = self.start_idx + self.img_num * self.img_interval
        return frame_concate

class Clip_Video_Frames_Sequence:
    def __init__(self, Dataset_path = None,batch_size = 4,img_size = 112,max_img_interval = 4,min_img_interval = 1,img_num = 8,samesize_tags = False):
        self.img_size = img_size
        self.max_img_interval = max_img_interval
        self.min_img_interval = min_img_interval
        self.img_num = img_num
        self.batch_size = batch_size
        self.samesize_tags = samesize_tags
        self.train_dataset_path = []
        self.train_video_list = []
        self.test_video_list = []
        self.train_dataset_path.append('./UCSD/UCSDped1/Train')
        self.train_dataset_path.append('./UCSD/UCSDped2/Train')

        self.test_dataset_path = []
        self.test_dataset_path.append('./UCSD/UCSDped1/Test')
        self.test_dataset_path.append('./UCSD/UCSDped2/Test')

        # self.train_video_list.append(os.listdir(self.train_dataset_path[0]))
        # self.train_video_list.append(os.listdir(self.train_dataset_path[1]))

        # for test_video_name in self.train_video_list[0]:
        #     print test_video_name
        tmp_test_video_list = []
        for test_video_name in os.listdir(self.train_dataset_path[0]):
            if len(test_video_name) == 8:
                tmp_test_video_list.append(test_video_name)
        self.train_video_list.append(tmp_test_video_list)
        tmp_test_video_list = []
        for test_video_name in os.listdir(self.train_dataset_path[1]):
            if len(test_video_name) == 8:
                tmp_test_video_list.append(test_video_name)
        self.train_video_list.append(tmp_test_video_list)

        tmp_test_video_list = []
        for test_video_name in os.listdir(self.test_dataset_path[0]):
            if len(test_video_name) == 7:
                tmp_test_video_list.append(test_video_name)
        self.test_video_list.append(tmp_test_video_list)
        tmp_test_video_list = []
        for test_video_name in os.listdir(self.test_dataset_path[1]):
            if len(test_video_name) == 7:
                tmp_test_video_list.append(test_video_name)
        self.test_video_list.append(tmp_test_video_list)


        for test_name in self.train_video_list[0]:
            print(test_name)


    def get_video_frame_batches(self):
        video_frame_sequence_batch = []
        choosen_dataset = np.int(random.randint(0, len(self.train_dataset_path) - 1))
        for i in range(self.batch_size):
            if not (self.samesize_tags):
                choosen_dataset = np.int(random.randint(0, len(self.train_dataset_path) - 1))
            video_frame_sequence_batch.append(self.get_random_video_frams(choosen_dataset))
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)
        # print video_frame_sequence_batch.shape
        return video_frame_sequence_batch

    def get_test_video_frame_batches(self):
        video_frame_sequence_batch = []
        choosen_dataset = np.int(random.randint(0, len(self.test_dataset_path) - 1))
        for i in range(self.batch_size):
            if not (self.samesize_tags):
                choosen_dataset = np.int(random.randint(0, len(self.test_dataset_path) - 1))
            video_frame_sequence_batch.append(self.get_test_random_video_frams(choosen_dataset))
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)
        # print video_frame_sequence_batch.shape
        return video_frame_sequence_batch

    def get_random_video_frams(self,choosen_dataset = 0):

        choosen_video = np.int(random.randint(1,len(self.train_video_list[choosen_dataset]) - 1))
        img_interval = np.int(random.randint(self.min_img_interval, self.max_img_interval))

        train_imgs = os.listdir(self.train_dataset_path[choosen_dataset] + '/' + self.train_video_list[choosen_dataset][choosen_video] )
        train_imgs.sort()
        idx=0
        for imgs_name in train_imgs:
            if imgs_name == '.DS_Store':
                del train_imgs[idx]
                break
            idx+=1


        choosen_frames_num = np.int(random.randint(1, len(train_imgs) - img_interval * self.img_num))
        # print train_imgs[choosen_frames_num]

        frame_path = []
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.train_dataset_path[choosen_dataset] + '/' + self.train_video_list[choosen_dataset][choosen_video]   + '/' + \
                             train_imgs[choosen_frames_num + i * img_interval]
            # print('interval %d' % img_interval)
            # print('Current img-th%d' % i)
            frame_path.append(cur_frame_path)
            cur_frame = Image.open(cur_frame_path)
            if not (self.samesize_tags):
                cur_frame = cur_frame.resize((self.img_size, self.img_size))
            cur_frame_np = np.array(cur_frame, dtype=np.float32) /np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
            frame_concate.append(cur_frame_np)
            # print cur_frame_np.shape
        frame_concate = np.concatenate(frame_concate, axis=0)
        # print frame_concate.shape
        frame_concate = frame_concate[np.newaxis,:,:,:,:]

        return frame_concate

    def get_test_random_video_frams(self, choosen_dataset = 0):
        # choosen_dataset = np.int(random.randint(0,len(self.test_dataset_path) - 1))
        choosen_video = np.int(random.randint(0,len(self.test_video_list[choosen_dataset]) - 1))
        img_interval = np.int(random.randint(self.min_img_interval, self.max_img_interval))
        train_imgs = os.listdir(self.test_dataset_path[choosen_dataset] + '/' + self.test_video_list[choosen_dataset][choosen_video] )
        train_imgs.sort()
        idx = 0
        for imgs_name in train_imgs:
            if imgs_name == '.DS_Store':
                del train_imgs[idx]
                break
            idx += 1
        choosen_frames_num = np.int(random.randint(1, len(train_imgs) - img_interval * self.img_num))
        # print train_imgs[choosen_frames_num]

        frame_path = []
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.test_dataset_path[choosen_dataset] + '/' + self.test_video_list[choosen_dataset][choosen_video]   + '/' + \
                             train_imgs[choosen_frames_num + i * img_interval]
            # print('interval %d'%img_interval)
            # print('Current img-th%d'%i)
            # print(train_imgs[choosen_frames_num + i * img_interval])
            frame_path.append(cur_frame_path)
            cur_frame = Image.open(cur_frame_path)
            if not (self.samesize_tags):
                cur_frame = cur_frame.resize((self.img_size, self.img_size))
            cur_frame_np = np.array(cur_frame, dtype=np.float32) /np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
            frame_concate.append(cur_frame_np)
            # print cur_frame_np.shape
        frame_concate = np.concatenate(frame_concate, axis=0)
        # print frame_concate.shape
        frame_concate = frame_concate[np.newaxis,:,:,:,:]

        return frame_concate

class UCSD_Dataset_Video_List:
    def __init__(self):

        self.train_dataset_path = []
        self.train_video_list = []
        self.test_video_list = []
        self.train_dataset_path.append('/media/room304/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
        self.train_dataset_path.append('/media/room304/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train')

        self.test_dataset_path = []
        self.test_dataset_path.append('/media/room304/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test')
        self.test_dataset_path.append('/media/room304/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test')

        tmp_test_video_list = []
        for test_video_name in os.listdir(self.train_dataset_path[0]):
            if len(test_video_name) == 8:
                # print test_video_name
                tmp_test_video_list.append(test_video_name)
        tmp_test_video_list.sort()
        self.train_video_list.append(tmp_test_video_list)
        tmp_test_video_list = []
        for test_video_name in os.listdir(self.train_dataset_path[1]):
            if len(test_video_name) == 8:
                # print test_video_name
                tmp_test_video_list.append(test_video_name)
        tmp_test_video_list.sort()
        self.train_video_list.append(tmp_test_video_list)

        tmp_test_video_list = []
        for test_video_name in os.listdir(self.test_dataset_path[0]):
            if len(test_video_name) == 7:
                # print test_video_name
                tmp_test_video_list.append(test_video_name)
        tmp_test_video_list.sort()
        self.test_video_list.append(tmp_test_video_list)
        tmp_test_video_list = []
        for test_video_name in os.listdir(self.test_dataset_path[1]):
            if len(test_video_name) == 7:
                # print(test_video_name)
                tmp_test_video_list.append(test_video_name)
        tmp_test_video_list.sort()
        self.test_video_list.append(tmp_test_video_list)

class Video_Frames_Sequence:
    def __init__(self, Dataset_path = None,batch_size = 16,img_interval = 2,img_num = 8):
        self.img_size = 112
        self.img_interval = img_interval
        self.img_num = img_num
        self.batch_size = batch_size
        self.dataset_path = []
        self.train_video_list = []
        self.dataset_path.append('../../DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
        self.dataset_path.append('../../DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train')

        self.train_video_list.append(os.listdir(self.dataset_path[0]))
        self.train_video_list.append(os.listdir(self.dataset_path[1]))

    def get_video_frame_batches(self):
        choosen_dataset = np.int(random.randint(0, len(self.dataset_path) - 1))
        video_frame_sequence_batch = []
        for i in range(self.batch_size):
            video_frame_sequence_batch.append(self.get_random_video_frams(choosen_dataset))
        video_frame_sequence_batch = np.concatenate(video_frame_sequence_batch,axis=0)
        print video_frame_sequence_batch.shape
        return video_frame_sequence_batch


    def get_random_video_frams(self,choosen_dataset = 0):

        choosen_video = np.int(random.randint(1,len(self.train_video_list[choosen_dataset]) - 1))

        train_imgs = os.listdir(self.dataset_path[choosen_dataset] + '/' + self.train_video_list[choosen_dataset][choosen_video] )
        choosen_frames_num = np.int(random.randint(1, len(train_imgs) - self.img_interval * self.img_num))
        # print train_imgs[choosen_frames_num]

        frame_path = []
        frame_concate = []
        for i in range(self.img_num):
            cur_frame_path = self.dataset_path[choosen_dataset] + '/' + self.train_video_list[choosen_dataset][choosen_video]   + '/' + \
                             train_imgs[choosen_frames_num + i * self.img_interval]
            frame_path.append(cur_frame_path)
            cur_frame = Image.open(cur_frame_path)
            # cur_frame = cur_frame.resize((self.img_size, self.img_size))
            cur_frame_np = np.array(cur_frame, dtype=np.float32) /np.float(255.0)
            cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
            frame_concate.append(cur_frame_np)
            # print cur_frame_np.shape
        frame_concate = np.concatenate(frame_concate, axis=0)
        # print frame_concate.shape
        frame_concate = frame_concate[np.newaxis,:,:,:,:]



        return frame_concate


class UCSD_Dataset_Frame_and_OpticalFLow_Batches:
    def __init__(self):
        self.my_ucsd = UCSD_Dataset_Video_List()
        return

    def get_train_random_batches(self,batch_size = 4):
        selected_list_num = np.int(random.randint(0, len(self.my_ucsd.train_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_ucsd.train_video_list[selected_list_num]) - 1))
        # print selected_video_num
        random_interval = np.int(random.randint(2,4))
        my_tmp_ucsd_batch = Clip_Video_Frames_and_OpticalFlow_Randomly_Train(video_name_length=7,batch_size=batch_size, dataset_path=
        self.my_ucsd.train_dataset_path[selected_list_num] + '/' + self.my_ucsd.train_video_list[selected_list_num][
            selected_video_num],img_interval=random_interval)
        my_tmp_batch = my_tmp_ucsd_batch.get_video_frame_batches()
        return my_tmp_batch

    def get_test_random_batches(self,batch_size = 4):
        selected_list_num = np.int(random.randint(0, len(self.my_ucsd.test_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_ucsd.test_video_list[selected_list_num]) - 1))
        # print selected_video_num
        random_interval = np.int(random.randint(2, 4))
        my_tmp_ucsd_batch = Clip_Video_Frames_and_OpticalFlow_Randomly_Train(video_name_length=7,batch_size=batch_size, dataset_path=
        self.my_ucsd.test_dataset_path[selected_list_num] + '/' + self.my_ucsd.test_video_list[selected_list_num][
            selected_video_num],img_interval=random_interval)
        my_tmp_batch = my_tmp_ucsd_batch.get_video_frame_batches()
        return my_tmp_batch

    def get_random_test_video_name(self):
        selected_list_num = np.int(random.randint(0, len(self.my_ucsd.test_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_ucsd.test_video_list[selected_list_num]) - 1))
        selected_video_name = self.my_ucsd.test_dataset_path[selected_list_num] + '/' + self.my_ucsd.test_video_list[selected_list_num][
            selected_video_num]
        return selected_video_name

    def get_random_train_video_name(self):
        selected_list_num = np.int(random.randint(0, len(self.my_ucsd.train_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_ucsd.train_video_list[selected_list_num]) - 1))
        selected_video_name = self.my_ucsd.train_dataset_path[selected_list_num] + '/' + self.my_ucsd.train_video_list[selected_list_num][
            selected_video_num]
        return selected_video_name

class UCSD_Dataset_Frame_and_HOG_Batches:
    def __init__(self):
        self.my_ucsd = UCSD_Dataset_Video_List()
        return

    def get_train_random_batches(self, batch_size=4):
        selected_list_num = np.int(random.randint(0, len(self.my_ucsd.train_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_ucsd.train_video_list[selected_list_num]) - 1))
        # print selected_video_num
        random_interval = np.int(random.randint(2, 4))
        my_tmp_ucsd_batch = Clip_Video_Frames_and_HOG_Randomly_Train(video_name_length=7,
                                                                             batch_size=batch_size, dataset_path=
                                                                             self.my_ucsd.train_dataset_path[
                                                                                 selected_list_num] + '/' +
                                                                             self.my_ucsd.train_video_list[
                                                                                 selected_list_num][
                                                                                 selected_video_num],
                                                                             img_interval=random_interval)
        my_tmp_batch = my_tmp_ucsd_batch.get_video_frame_batches()
        return my_tmp_batch

    def get_test_random_batches(self, batch_size=4):
        selected_list_num = np.int(random.randint(0, len(self.my_ucsd.test_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_ucsd.test_video_list[selected_list_num]) - 1))
        # print selected_video_num
        random_interval = np.int(random.randint(2, 4))
        my_tmp_ucsd_batch = Clip_Video_Frames_and_HOG_Randomly_Train(video_name_length=7,
                                                                             batch_size=batch_size, dataset_path=
                                                                             self.my_ucsd.test_dataset_path[
                                                                                 selected_list_num] + '/' +
                                                                             self.my_ucsd.test_video_list[
                                                                                 selected_list_num][
                                                                                 selected_video_num],
                                                                             img_interval=random_interval)
        my_tmp_batch = my_tmp_ucsd_batch.get_video_frame_batches()
        return my_tmp_batch

# my_batches = UCSD_Dataset_Frame_and_OpticalFLow_Batches()
# my_batches.get_train_random_batches()
# my_ucsd =  UCSD_Dataset_Video_List()

# my_cvis =  Clip_Video_Framesand_OpticalFlow_Orderly_Sequence_Test(test_video_name_length=7,dataset_path=my_ucsd.test_dataset_path[0] + '/' + my_ucsd.test_video_list[0][0])
# my_cvis.get_video_frame_batches()
# my_cvis.get_random_video_frams()
# while(my_cvis.continued_tags):
#     t_start = time.time()
#     my_cvis.get_video_frame_batches()
#     t_cost = time.time() - t_start
#     print 'time used'
#     print t_cost
# for i in range(0,my_cvis.max_iteration):
#     t_start = time.time()
#     my_cvis.get_video_frame_batches()
#     t_cost = time.time() - t_start
#     print 'time used'
#     print t_cost

# cur_frame_path = '/media/room304/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif'
# ski_image = io.imread(cur_frame_path,as_grey=True)
# fg,hog_image = hog(ski_image,orientations=8,pixels_per_cell=(4,4),cells_per_block=(5,5),block_norm='L2',visualise=True)
# print fg.shape
# print hog_image.shape
# io.imsave('test.jpeg',hog_image)