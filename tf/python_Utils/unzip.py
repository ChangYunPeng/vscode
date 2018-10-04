import os
import tarfile

def unzip_singlefile(tar_path,target_path):
    tar = tarfile.open(tar_path,"r:gz")
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name,target_path)
    tar.close()

if __name__ == '__main__':
    # tar_path = '/home/room304/storage/GF2/GF2_PMS2_E113.1_N23.1_20180321_L1A0003074718.tar.gz'
    # target_path = '/home/room304/storage/unzip-GF2/GF2_PMS2_E113.1_N23.1_20180321_L1A0003074718'

    tar_dataset_path = '/home/room304/storage/GF2/'
    untar_dataset_path = '/home/room304/storage/unzip-GF2/'
    tar_path_list = os.listdir(tar_dataset_path)
    tar_fullpath_list = []
    untar_fullpath_list = []
    for tar_path_iter in tar_path_list:
        tar_fullpath_list.append(tar_dataset_path + tar_path_iter )
        tar_full_path = tar_dataset_path + tar_path_iter
        tar_fullpath_list.append(untar_dataset_path + tar_path_iter[:-6] )
        untar_full_path = untar_dataset_path + tar_path_iter[:-6]
        unzip_singlefile(tar_full_path,untar_full_path)
        
    # unzip_singlefile(tar_path,target_path)