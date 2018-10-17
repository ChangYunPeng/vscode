from model_uilit import *
from sklearn.metrics import roc_auc_score, roc_curve

# ''' ShanghaiTec '''
# dataste_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/MID_LR_X3_VN4_NewTest111/RESULT/FULL_Data_Set/Shanghai/'
# mat_save_path = '/media/room304/TB/Matlab-Test/roc_auc_result/shanghai_tec_compared_vn4_NewTest111/'
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


'''  ucsd '''
dataste_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/MID_LR_X3_VN4_NewTest/RESULT/FULL_Data_Set/UCSD_Path/UCSD_Dataset_1/'
mat_save_path = '/media/room304/TB/Matlab-Test/roc_auc_result/ucse_ped1_vn4_NewTest/'
dataset_path_list = os.listdir(dataste_path)
dataset_path_list.sort()
for dataset_idx in dataset_path_list:
    tmp_dataset_path = dataste_path + dataset_idx
    print (tmp_dataset_path)
    npy_file_1_path = tmp_dataset_path + '/total_loss.npy'
    y_score = np.load(npy_file_1_path)
    save_single_npy_as_mat(mat_save_path+'/'+ dataset_idx + '_roc_auc.mat', y_score)
