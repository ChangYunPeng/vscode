import sys
sys.path.append('/media/room304/TB/vscode/abnormly-code/anormly_utilize_code')
from model_uilit import *
from VideoSequenceUtils.DataSetImgSequence import UCSD_Dataset_Video_List
# from sklearn.metrics import roc_auc_score, roc_curve


my_dataset_ucsd = UCSD_Dataset_Video_List()
for path_list_iter in my_dataset_ucsd.test_video_list:
    for test_img_iter in path_list_iter:
        print (test_img_iter)
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
# dataste_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/COMPARED_VN4_L3/RESULT/FULL_Data_Set/UCSD_Path/UCSD_Dataset_2/'
# mat_save_path = '/media/room304/TB/Matlab-Test/roc_auc_result/ucse_ped2_compraed_vn4/'
# dataset_path_list = os.listdir(dataste_path)
# dataset_path_list.sort()
# for dataset_idx in dataset_path_list:
#     tmp_dataset_path = dataste_path + dataset_idx
#     print tmp_dataset_path
#     npy_file_1_path = tmp_dataset_path + '/total_loss.npy'
#     y_score = np.load(npy_file_1_path)
#     save_single_npy_as_mat(mat_save_path+'/'+ dataset_idx + '_roc_auc.mat', y_score)
