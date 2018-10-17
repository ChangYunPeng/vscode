from model_uilit import *
from PIL import Image
import os
import cv2
# ''' ShanghaiTec '''
# dataste_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/MID_LR_X3_VN4/RESULT/FULL_Data_Set/Shanghai/'
# mat_save_path = '/media/room304/TB/Matlab-Test/roc_auc_result/shanghai_tec_vn4/'
#
# dataset_path_list = os.listdir(dataste_path)
# dataset_path_list.sort()
# for dataset_idx in dataset_path_list:
#     tmp_dataset_path = dataste_path + dataset_idx
#     npy_file_1_path = tmp_dataset_path + '/total_loss.npy'
#     npy_file_2_path = tmp_dataset_path +  '/label_loss.npy'
#     y_score = np.load(npy_file_1_path)
#     y_true = np.load(npy_file_2_path)
#     save_npy_as_mat(mat_save_path+'/'+ dataset_idx + '_roc_auc.mat', y_score, y_true)


# '''  ucsd '''
# dataste_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/MID_LR_X3_VN4/RESULT/FULL_Data_Set/UCSD_Path/UCSD_Dataset_1/'
# mat_save_path = '/media/room304/TB/Matlab-Test/roc_auc_result/ucse_ped2_vn4/'
# dataset_path_list = os.listdir(dataste_path)
# dataset_path_list.sort()
# for dataset_idx in dataset_path_list:
#     tmp_dataset_path = dataste_path + dataset_idx
#     print tmp_dataset_path
#     npy_file_1_path = tmp_dataset_path + '/total_loss.npy'
#     y_score = np.load(npy_file_1_path)
#     save_single_npy_as_mat(mat_save_path+'/'+ dataset_idx + '_roc_auc.mat', y_score)

# ''' ShanghaiTec save mask '''
# dataste_path = '/media/room304/TB/DATASET/ShanghaiTechCampus/Testing/test_pixel_mask/'
# mask_save_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/PIXEL_MASK/'
#
# dataset_path_list = os.listdir(dataste_path)
# dataset_path_list.sort()
#
#
#
# for dataset_idx in dataset_path_list:
#     tmp_dataset_path = dataste_path + dataset_idx
#     tmp_mask_save_path = mask_save_path + dataset_idx
#
#     if not os.path.exists(tmp_mask_save_path):
#         os.makedirs(tmp_mask_save_path)
#
#     pixel_mask = np.load(tmp_dataset_path)
#     # print max(pixel_mask)
#     # print min(pixel_mask)
#
#     for mask_num in range(pixel_mask.shape[0]):
#         # print np.max(pixel_mask[:,:,mask_num])
#         # print np.min(pixel_mask[:,:,mask_num])
#         img_optical_concat = Image.fromarray(pixel_mask[mask_num,:,:]*255, mode='L')
#         img_optical_concat.save(tmp_mask_save_path + '/frame_%d.jpg'%mask_num)
#     # y_true = np.load(npy_file_2_path)
#     # save_npy_as_mat(mat_save_path+'/'+ dataset_idx + '_roc_auc.mat', y_score, y_true)

''' ShanghaiTec difference result '''
dataste_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/MID_LR_X3_VN4/RESULT/FULL_Data_Set/Shanghai/'
mask_save_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/PIXEL_MASK/'

dataset_path_list = os.listdir(dataste_path)
dataset_path_list.sort()



for dataset_idx in dataset_path_list:
    tmp_dataset_path = dataste_path + dataset_idx + '/GRAY/'

    tmp_dataset_save_path = dataste_path + dataset_idx + '/GRAY_DIFFERENCE/'
    if not os.path.exists(tmp_dataset_save_path):
        os.makedirs(tmp_dataset_save_path)

    img_path_list = os.listdir(tmp_dataset_path)
    img_path_list.sort()
    for img_idx in img_path_list:
        tmp_img_path = tmp_dataset_path + img_idx
        tmp_img_save_path = tmp_dataset_save_path + img_idx
        img = Image.open(tmp_img_path)
        img = np.array(img, dtype=float) / np.float(255.0)
        print (img.shape)
        weight = img.shape[0]

        weight = weight/2
        print (weight)
        img1 = img[0:weight,:]
        img2 = img[weight:2*weight,:]
        img_difference = img1 - img2
        # img_difference = img1*255.0
        img_difference = (img_difference - np.min(img_difference))/(np.max(img_difference) - np.min(img_difference))
        print (img_difference.shape)
        print (np.max(img_difference))
        print (np.min(img_difference))
        img_difference = img_difference*255.0
        # img_optical_concat = Image.fromarray(img_difference, mode='L')
        cv2.imwrite(tmp_img_save_path, img_difference)
        # img_optical_concat.save(tmp_img_save_path)

    # tmp_mask_save_path = mask_save_path + dataset_idx
    #
    # if not os.path.exists(tmp_mask_save_path):
    #     os.makedirs(tmp_mask_save_path)
    #
    # pixel_mask = Image.open(tmp_dataset_path)
    # # print max(pixel_mask)
    # # print min(pixel_mask)
    #
    # for mask_num in range(pixel_mask.shape[0]):
    #     # print np.max(pixel_mask[:,:,mask_num])
    #     # print np.min(pixel_mask[:,:,mask_num])
    #     img_optical_concat = Image.fromarray(pixel_mask[mask_num,:,:]*255, mode='L')
    #     img_optical_concat.save(tmp_mask_save_path + '/frame_%d.jpg'%mask_num)
    # # y_true = np.load(npy_file_2_path)
    # # save_npy_as_mat(mat_save_path+'/'+ dataset_idx + '_roc_auc.mat', y_score, y_true)