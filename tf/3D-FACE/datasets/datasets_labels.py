import numpy as np
import os
from scipy.io import loadmat
# import scipy.io.loadmat as loadmat

def ShanghaiTechCampus_frames_labels():
    label_list_path = '/home/room304/TB/TB/DATASET/ShanghaiTechCampus/Testing/test_frame_mask'
    label_path_list = os.listdir(label_list_path)
    label_path_list.sort()
    label_list = []
    videos_label_path = []
    for label_path in label_path_list:
        # print(os.path.join(label_list_path, label_path))
        videos_label_path.append(os.path.join(label_list_path, label_path))
        label_iter = np.load(os.path.join(label_list_path, label_path))
        # print(label_iter.shape)
        # print(label_iter.dtype)
        # print('max value:', label_iter.max())
        # print('min value:', label_iter.min())
        label_list.append(label_iter)
    return label_list

def UCSD_v1_frames_labes():
    TestVideoFile = []

    video_size = np.zeros(shape=[200])
    # print(video_size)

    video_size[60:152]=1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[50:175]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[91:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[31:168]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[5:90]=1
    video_size[140:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:100]=1
    video_size[110:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:175]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:94]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:48]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:140]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[70:165]=1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[130:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:156]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[138:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[123:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:47]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[54:120]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[64:138]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[45:175]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[31:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[16:107]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[8:165]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[50:171]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[40:135]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[77:144]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[10:122]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[105:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:15]=1
    video_size[45:113]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[175:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:180]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:52]
    video_size[65:115]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[5:165]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[1:121]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[86:200]=1
    TestVideoFile.append(video_size)

    video_size = np.zeros(shape=[200])
    video_size[15:108]=1
    TestVideoFile.append(video_size)
    return TestVideoFile

def UCSD_v2_frames_labes():
    TestVideoFile = []
    video_size = np.zeros(shape=[200])
    video_size[61:180] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[95:180] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[1:146] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[31:180] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[1:129] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[1:159] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[46:180] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[1:180] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[1:120] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[1:150] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[1:180] = 1
    TestVideoFile.append(video_size)
    video_size = np.zeros(shape=[200])
    video_size[88:180] = 1
    TestVideoFile.append(video_size)
    return TestVideoFile

def Avenue_frames_labels():
    label_list_path = '/home/room304/TB/TB/DATASET/Avenue_Dataset/ground_truth_demo/testing_label_mask'
    label_path_list = os.listdir(label_list_path)
    label_path_list.sort()
    label_list = []
    videos_label_path = []

    for label_idx,label_path in enumerate(label_path_list) :
        # print(os.path.join(label_list_path, '%d_label'%(label_idx+1)))
        path_iter = os.path.join(label_list_path, '%d_label'%(label_idx+1))
        mat = loadmat(path_iter)
        label_mask_list = mat['volLabel']
        video_length = label_mask_list.shape[1]

        frame_label_list = []
        for video_idx in range(video_length):
            # print('video - idx :',video_idx,label_mask_list[0,video_idx].shape)
            # print('max value ',label_mask_list[0,video_idx].max())
            frame_label_list.append(np.asarray(label_mask_list[0,video_idx].max()))
        frame_label_list = np.hstack(frame_label_list)
        videos_label_path.append(path_iter)
        label_list.append(frame_label_list)
    return label_list

def Avenue_pixel_labels():
    label_list_path = '/home/room304/TB/TB/DATASET/Avenue_Dataset/ground_truth_demo/testing_label_mask'
    label_path_list = os.listdir(label_list_path)
    label_path_list.sort()
    label_list = []
    videos_label_path = []

    for label_idx,label_path in enumerate(label_path_list) :
        path_iter = os.path.join(label_list_path, '%d_label'%(label_idx+1))
        mat = loadmat(path_iter)
        label_mask_list = mat['volLabel']
        video_length = label_mask_list.shape[1]
        print(video_length)

        frame_label_list = []
        for video_idx in range(video_length):
            print(label_mask_list[0,video_idx].shape)
            # print('video - idx :',video_idx,label_mask_list[0,video_idx].shape)
            # print('max value ',label_mask_list[0,video_idx].max())
            # frame_label_list.append(np.asarray(label_mask_list[0,video_idx].max()))
        # frame_label_list = np.hstack(frame_label_list)
        # videos_label_path.append(path_iter)
        # label_list.append(frame_label_list)
    return label_list

if __name__ == '__main__':
    Avenue_pixel_labels()
    print('tesr')