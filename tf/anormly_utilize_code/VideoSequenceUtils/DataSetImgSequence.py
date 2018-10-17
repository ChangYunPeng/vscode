import sys
sys.path.append('/home/room304/storage/vscode/tf/anormly_utilize_code/VideoSequenceUtils/')
sys.path.append('/home/room304/storage/vscode/tf/anormly_utilize_code/ImageProcessUtils/')
import os
import numpy as np
import random
from Clip_Video_Frames import Clip_Video_Frames_and_OpticalFlow_Randomly_Train, Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test

def video_path_list(dataset_path,video_name_length=7,frame_name_length=7):
    tmp_video_path_list = [ ]
    tmp_video_frame_path_list = [ ]
    for test_video_name in os.listdir(dataset_path):
        # print test_video_name
        if len(test_video_name) == video_name_length:
            # print test_video_name
            video_path_name = dataset_path+'/'+test_video_name
            tmp_video_frame_path_list.append(frame_path_list(video_path_name,frame_name_length))
            tmp_video_path_list.append(video_path_name)
    # tmp_video_path_list.sort()
    # for tmp_video_name in tmp_video_frame_path_list:
    #     print tmp_video_name
    return tmp_video_path_list,tmp_video_frame_path_list

def frame_path_list(video_path,frame_name_length=7):
    tmp_frame_path_list = []
    for tmp_frame_path_name in os.listdir(video_path):
        if len(tmp_frame_path_name) == frame_name_length:
            frame_path_name = video_path + '/' + tmp_frame_path_name
            # print frame_path_name
            tmp_frame_path_list.append(frame_path_name)
    tmp_frame_path_list.sort()
    return tmp_frame_path_list

class UCSD_Dataset_Video_List:
    def __init__(self):

        self.train_dataset_path = []

        self.train_video_list = []
        self.train_video_frames_list = []
        
        self.test_video_list = []
        self.test_video_frames_list = []
        
        self.train_dataset_path = []
        self.train_dataset_path.append('/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train')
        self.train_dataset_path.append('/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train')

        self.test_dataset_path = []
        self.test_dataset_path.append('/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test')
        self.test_dataset_path.append('/home/room304/TB/TB/DATASET/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test')

        tmp_video_path_list,tmp_video_frame_path_list = video_path_list(self.train_dataset_path[0],8,7)
        self.train_video_list.append(tmp_video_path_list)
        self.train_video_frames_list.append(tmp_video_frame_path_list)
        tmp_video_path_list,tmp_video_frame_path_list = video_path_list(self.train_dataset_path[1],8,7)
        self.train_video_list.append(tmp_video_path_list)
        self.train_video_frames_list.append(tmp_video_frame_path_list)

        tmp_video_path_list,tmp_video_frame_path_list = video_path_list(self.test_dataset_path[0],7,7)
        self.test_video_list.append(tmp_video_path_list)
        self.test_video_frames_list.append(tmp_video_frame_path_list)
        tmp_video_path_list,tmp_video_frame_path_list = video_path_list(self.test_dataset_path[1],7,7)
        self.test_video_list.append(tmp_video_path_list)
        self.test_video_frames_list.append(tmp_video_frame_path_list)


class Avenue_Dataset_Video_List:
    def __init__(self):
        self.train_dataset_path = []

        self.train_video_list = []
        self.train_video_frames_list = []

        self.test_video_list = []
        self.test_video_frames_list = []

        self.train_dataset_path.append('/home/room304/TB/TB/DATASET/Avenue_Dataset/training_videos')

        self.test_dataset_path = []
        self.test_dataset_path.append('/home/room304/TB/TB/DATASET/Avenue_Dataset/testing_videos')

        tmp_video_path_list, tmp_video_frame_path_list = video_path_list(self.train_dataset_path[0], 2, 9)
        self.train_video_list.append(tmp_video_path_list)
        self.train_video_frames_list.append(tmp_video_frame_path_list)

        tmp_video_path_list, tmp_video_frame_path_list = video_path_list(self.test_dataset_path[0], 2, 9)
        self.test_video_list.append(tmp_video_path_list)
        self.test_video_frames_list.append(tmp_video_frame_path_list)


class ShanghaiTec_Dataset_Video_List:
    def __init__(self):
        self.train_dataset_path = []

        self.train_video_list = []
        self.train_video_frames_list = []

        self.test_video_list = []
        self.test_video_frames_list = []

        self.train_dataset_path.append('/home/room304/TB/TB/DATASET/ShanghaiTechCampus/training/videos')

        self.test_dataset_path = []
        self.test_dataset_path.append('/home/room304/TB/TB/DATASET/ShanghaiTechCampus/Testing/frames_part1')
        self.test_dataset_path.append('/home/room304/TB/TB/DATASET/ShanghaiTechCampus/Testing/frames_part2')



        tmp_video_path_list, tmp_video_frame_path_list = video_path_list(self.train_dataset_path[0], 6, 9)
        self.train_video_list.append(tmp_video_path_list)
        self.train_video_frames_list.append(tmp_video_frame_path_list)

        # for tmp_video_path_name in self.train_video_frames_list[0][0]:
        #     print tmp_video_path_name

        tmp_video_path_list, tmp_video_frame_path_list = video_path_list(self.test_dataset_path[0], 7, 7)
        self.test_video_list.append(tmp_video_path_list)
        self.test_video_frames_list.append(tmp_video_frame_path_list)
        tmp_video_path_list, tmp_video_frame_path_list = video_path_list(self.test_dataset_path[1], 7, 7)
        self.test_video_list.append(tmp_video_path_list)
        self.test_video_frames_list.append(tmp_video_frame_path_list)
        # print self.test_video_list[0][10]
        # for tmp_video_path_name in self.test_video_frames_list[0][10]:
        #     print tmp_video_path_name

        self.label_dataset_path = '/home/room304/TB/TB/DATASET/ShanghaiTechCampus/Testing/test_frame_mask/'
        self.label_path_list = os.listdir(self.label_dataset_path)
        self.label_path_list.sort()

class Dataset_Video_List:
    def __init__(self):
        my_ucsd = UCSD_Dataset_Video_List()
        my_avenue = Avenue_Dataset_Video_List()
        my_shtec = ShanghaiTec_Dataset_Video_List()

        # self.train_video_list = my_ucsd.train_video_list + my_avenue.train_video_list + my_shtec.train_video_list
        # self.train_video_frames_list = my_ucsd.train_video_frames_list + my_avenue.train_video_frames_list + my_shtec.train_video_frames_list
        #
        # self.test_video_list = my_ucsd.test_video_list + my_avenue.test_video_list + my_shtec.test_video_list
        # self.test_video_frames_list = my_ucsd.test_video_frames_list + my_avenue.test_video_frames_list  + my_shtec.test_video_frames_list

        # my_ucsd = UCSD_Dataset_Video_List()
        self.train_video_list = my_ucsd.train_video_list
        self.train_video_frames_list = my_ucsd.train_video_frames_list
        self.test_video_list = my_ucsd.test_video_list
        self.test_video_frames_list = my_ucsd.test_video_frames_list
        print(self.train_video_list)
        
class Dataset_Frame_and_OPTICALFLOW_Batches:
    def __init__(self,frame_tags = True, opticalflow_tags = True,img_num = 8):
        self.my_dataset = Dataset_Video_List()
        self.random_interval = [2,4]
        self.img_num = img_num
        self.frame_tags = frame_tags
        self.opticalflow_tags = opticalflow_tags

        return

    def get_train_random_batches(self, batch_size=4):
        selected_list_num = np.int(random.randint(0, len(self.my_dataset.train_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_dataset.train_video_list[selected_list_num]) - 1))
        # print self.my_dataset.train_video_list[selected_list_num][selected_video_num]
        random_interval = np.int(random.randint(self.random_interval[0], self.random_interval[1]))
        my_tmp_ucsd_batch = Clip_Video_Frames_and_OpticalFlow_Randomly_Train(self.my_dataset.train_video_frames_list[selected_list_num][selected_video_num],self.my_dataset.train_video_list[selected_list_num][selected_video_num],
                                                                             batch_size=batch_size,img_num=self.img_num,
                                                                             img_interval=random_interval,frame_tags=self.frame_tags,opticalflow_tags=self.opticalflow_tags)
        my_tmp_batch = my_tmp_ucsd_batch.get_video_frame_batches()
        # print len(my_tmp_batch)
        return my_tmp_batch

    def get_test_frames_objects(self, batch_size=4):
        selected_list_num = np.int(random.randint(0, len(self.my_dataset.test_video_list) - 1))
        selected_video_num = np.int(random.randint(0, len(self.my_dataset.test_video_list[selected_list_num]) - 1))
        # print selected_video_num
        random_interval = np.int(random.randint(2, 4))
        random_interval = 2
        my_tmp_ucsd_batch = Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test(self.my_dataset.test_video_frames_list[selected_list_num][selected_video_num],self.my_dataset.test_video_list[selected_list_num][selected_video_num],
                                                                             batch_size=batch_size,img_num=self.img_num,
                                                                             img_interval=random_interval,frame_tags=self.frame_tags,opticalflow_tags=self.opticalflow_tags)
        return my_tmp_ucsd_batch

class Sequence_Dataset_Frame_and_OPTICALFLOW_Batches:
    def __init__(self,frame_tags = True, opticalflow_tags = True):
        self.my_dataset = ShanghaiTec_Dataset_Video_List()
        # self.my_dataset = UCSD_Dataset_Video_List()
        # self.my_dataset = Dataset_Video_List()
        self.random_interval = [2,4]

        self.frame_tags = frame_tags
        self.opticalflow_tags = opticalflow_tags

        self.selected_list_num = 0
        self.selected_video_num = 0
        self.continue_tags = True

        return

    def get_test_frames_objects(self, batch_size=4):
        selected_list_num = self.selected_list_num
        selected_video_num = self.selected_video_num

        random_interval = 1
        my_tmp_ucsd_batch = Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test(
            self.my_dataset.test_video_frames_list[selected_list_num][selected_video_num],
            self.my_dataset.test_video_list[selected_list_num][selected_video_num],
            batch_size=batch_size,
            img_interval=random_interval, frame_tags=self.frame_tags, opticalflow_tags=self.opticalflow_tags)

        label_np = self.get_selected_test_frame_label()
        self.selected_video_num = self.selected_video_num + 1
        if self.selected_video_num == len(self.my_dataset.test_video_list[self.selected_list_num]):
            self.selected_video_num = 0
            self.selected_list_num = self.selected_list_num + 1
            if self.selected_list_num == len(self.my_dataset.test_video_list):
                self.continue_tags = False
        return my_tmp_ucsd_batch, label_np

    def get_selected_test_frame_label(self):
        selected_list_num = self.selected_list_num
        selected_video_num = self.selected_video_num
        # selected_list_num = np.int(random.randint(0, len(self.my_dataset.test_video_list) - 1))
        # selected_video_num = np.int(random.randint(0, len(self.my_dataset.test_video_list[selected_list_num]) - 1))

        video_path = (self.my_dataset.test_video_list[selected_list_num][selected_video_num])
        for name_idx in range(len(video_path)):
            if(video_path[len(video_path)-name_idx-1]) == '/':
                video_name = video_path[len(video_path)-name_idx:len(video_path)]
                break
        label_npy_path = self.my_dataset.label_dataset_path + video_name +'.npy'
        label_np = np.load(label_npy_path)


        return label_np

class Sequence_Shanghai_Dataset_Frame_and_OPTICALFLOW_Batches:
    def __init__(self,frame_tags = True, opticalflow_tags = True,img_num = 8):
        self.my_dataset = ShanghaiTec_Dataset_Video_List()
        # self.my_dataset = UCSD_Dataset_Video_List()
        # self.my_dataset = Dataset_Video_List()
        self.random_interval = [2,4]
        self.img_num = img_num

        self.frame_tags = frame_tags
        self.opticalflow_tags = opticalflow_tags

        self.selected_list_num = 0
        self.selected_video_num = 0
        self.continue_tags = True

        return

    def get_test_frames_objects(self, batch_size=4):
        selected_list_num = self.selected_list_num
        selected_video_num = self.selected_video_num

        random_interval = 1
        my_tmp_ucsd_batch = Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test(
            self.my_dataset.test_video_frames_list[selected_list_num][selected_video_num],
            self.my_dataset.test_video_list[selected_list_num][selected_video_num],
            batch_size=batch_size,img_num=self.img_num,
            img_interval=random_interval, frame_tags=self.frame_tags, opticalflow_tags=self.opticalflow_tags)

        label_np = self.get_selected_test_frame_label()
        self.selected_video_num = self.selected_video_num + 1
        if self.selected_video_num == len(self.my_dataset.test_video_list[self.selected_list_num]):
            self.selected_video_num = 0
            self.selected_list_num = self.selected_list_num + 1
            if self.selected_list_num == len(self.my_dataset.test_video_list):
                self.continue_tags = False
        return my_tmp_ucsd_batch, label_np

    def get_selected_test_frame_label(self):
        selected_list_num = self.selected_list_num
        selected_video_num = self.selected_video_num
        # selected_list_num = np.int(random.randint(0, len(self.my_dataset.test_video_list) - 1))
        # selected_video_num = np.int(random.randint(0, len(self.my_dataset.test_video_list[selected_list_num]) - 1))

        video_path = (self.my_dataset.test_video_list[selected_list_num][selected_video_num])
        for name_idx in range(len(video_path)):
            if(video_path[len(video_path)-name_idx-1]) == '/':
                video_name = video_path[len(video_path)-name_idx:len(video_path)]
                break
        label_npy_path = self.my_dataset.label_dataset_path + video_name +'.npy'
        label_np = np.load(label_npy_path)


        return label_np

class Sequence_UCSD_Dataset_Frame_and_OPTICALFLOW_Batches:
    def __init__(self,frame_tags = True, opticalflow_tags = True, img_num = 8):
        # self.my_dataset = ShanghaiTec_Dataset_Video_List()
        self.my_dataset = UCSD_Dataset_Video_List()
        # self.my_dataset = Dataset_Video_List()
        self.random_interval = [2,4]
        self.img_num = img_num
        self.frame_tags = frame_tags
        self.opticalflow_tags = opticalflow_tags

        self.selected_list_num = 0
        self.selected_video_num = 0
        self.continue_tags = True

        return

    def get_test_frames_objects(self, batch_size=4):
        selected_list_num = self.selected_list_num
        selected_video_num = self.selected_video_num

        random_interval = 1
        my_tmp_ucsd_batch = Clip_Video_Frames_and_OpticalFlow_Orderly_Sequence_Test(
            self.my_dataset.test_video_frames_list[selected_list_num][selected_video_num],
            self.my_dataset.test_video_list[selected_list_num][selected_video_num],
            batch_size=batch_size,img_num=self.img_num,
            img_interval=random_interval, frame_tags=self.frame_tags, opticalflow_tags=self.opticalflow_tags)

        self.selected_video_num = self.selected_video_num + 1
        if self.selected_video_num == len(self.my_dataset.test_video_list[self.selected_list_num]):
            self.selected_video_num = 0
            self.selected_list_num = self.selected_list_num + 1
            if self.selected_list_num == len(self.my_dataset.test_video_list):
                self.continue_tags = False
        return my_tmp_ucsd_batch
# my_dataset = Dataset_Frame_and_OPTICALFLOW_Batches(frame_tags = True, opticalflow_tags = True)
# batch_data = my_dataset.get_train_random_batches()
# batch_data_gray = batch_data[:,:,:,:,0:1]
# batch_data_optical = batch_data[:,:,:,:,1:3]
# my_shtec = ShanghaiTec_Dataset_Video_List()

# Data_Sequence = Dataset_Frame_and_OPTICALFLOW_Batches(frame_tags = True, opticalflow_tags = True)
# test_video_dataset = Data_Sequence.get_test_frames_objects()
# test_video_dataset.batch_size = 1
# print test_video_dataset.video_path
# while (test_video_dataset.continued_tags):
#     batch_data = test_video_dataset.get_video_frame_batches()

# test_sequence = Sequence_Dataset_Frame_and_OPTICALFLOW_Batches()
# test_sequence.get_select_test_frame_label()