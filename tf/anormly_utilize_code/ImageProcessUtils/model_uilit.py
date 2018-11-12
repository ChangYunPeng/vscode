import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.io as scio
from sklearn.metrics import roc_auc_score, roc_curve

from PIL import Image

def save_plot_img(save_path,losses_np):

    video_losses = min_max_np(losses_np)
    fig = plt.figure()
    plt.plot(video_losses)
    fig.savefig(save_path)
    plt.close()
    return

def compute_eer(far, frr):
    cords = zip(far, frr)
    min_dist = 999999
    for item in cords:
        item_far, item_frr = item
        dist = abs(item_far - item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_far + item_frr) / 2
    return eer

def save_double_plot_img(save_path,losses_np,losses_label):

    video_losses = min_max_np(losses_np)
    fig = plt.figure()
    plt.plot(video_losses,'b',losses_label,'r')
    fig.savefig(save_path)
    plt.close()
    return


def save_mat_file(save_path,losses_np):
    video_losses = min_max_np(losses_np)
    scio.savemat(save_path,{'result':video_losses})
    return

def min_max_np(video_loss):

    video_loss = np.asarray(video_loss)
    video_losses_max = np.max(video_loss, axis=0)
    video_losses_min = np.min(video_loss, axis=0)
    video_losses = np.ones(video_loss.shape) - (video_loss - video_losses_min * np.ones( video_loss.shape)) / (video_losses_max * np.ones(video_loss.shape)  - video_losses_min * np.ones( video_loss.shape))

    return video_losses

def max_min_np(video_loss):

    video_loss = np.asarray(video_loss)
    video_losses_max = np.max(video_loss, axis=0)
    video_losses_min = np.min(video_loss, axis=0)
    video_losses =  (video_loss - video_losses_min * np.ones( video_loss.shape)) / (video_losses_max * np.ones(video_loss.shape) - video_losses_min * np.ones( video_loss.shape))

    return video_losses

def mk_dirs(dir_list):    
    for dirs_idx in dir_list:
        if not os.path.exists(dirs_idx):
            os.makedirs(dirs_idx)
    return

def save_img_list(input_batch):
    for img_idx in range(input_batch.shape[2]):
        img_np = input_batch[:,:,img_idx]
        print (img_np.shape)
        img_pil = Image.fromarray(img_np, mode='L')
        img_pil.save('./tmp/saved_%d.bmp'%img_idx)

    return

def save_roc_auc_plot_img(save_path,y_score,y_true):
    # y_score = np.concatenate(y_score,axis=0)
    # y_true = np.concatenate(y_true,axis=0)
    frame_auc = roc_auc_score(y_true=y_true, y_score=y_score)
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=1)
    frame_eer = compute_eer(far=fpr, frr=1 - tpr)

    print('auc', frame_auc)
    print('eer', frame_eer)
    return frame_auc, frame_eer


def save_npy_as_mat(save_path,y_score,y_true):
    normalized_y_socre = np.ones(y_score.shape) - min_max_np(y_score)
    print (y_score.shape[0])
    print (y_true.shape[0])
    start_idx = int((y_true.shape[0] - y_score.shape[0])/2)
    y_true = y_true[start_idx:start_idx+y_score.shape[0]]
    scio.savemat(save_path,{'result':y_score,'label':y_true})
    return

def save_single_npy_as_mat(save_path,y_score):
    print (y_score.shape[0])
    scio.savemat(save_path,{'result':y_score})
    return

# video_label_path = '/media/room304/TB/DATASET/ShanghaiTechCampus/Testing/test_frame_mask//01_0130.npy'
# # video_label_path = '/media/room304/TB/DATASET/Avenue_Dataset/testing_vol/vol01.mat'
# # video_label_path = '/media/room304/TB/DATASET/Avenue_Dataset/training_vol/vol01.mat'
# np_file = np.load(video_label_path)
# save_plot_img('./tmp_label_loss.jpg',np_file)

# mat_file = scio.loadmat(video_label_path)
# # print type(mat_file)
# print mat_file['vol'].shape
# save_img_list(mat_file['vol'])
# save_plot_img('./tmp_mat_label_loss.jpg',mat_file['vol'])
#

