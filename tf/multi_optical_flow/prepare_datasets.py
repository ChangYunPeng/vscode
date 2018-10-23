
import fileinput
import os
import numpy as np
import random
import cv2

def get_all_img_list(dataset_elder_path = '/home/room304/storage/datasets/MPI-Sintel/training', choosen_type = 'clean'):  
    datatsets_folder = os.path.join(dataset_elder_path, choosen_type)
    folder_list = os.listdir(datatsets_folder)
    folder_list.sort()
    all_vi_folder_list = []
    for folder in folder_list:
        folder_path = os.path.join(datatsets_folder, folder)
        img_list = os.listdir(folder_path)
        img_list.sort()
        per_vi_folder_list = []
        for img_name in img_list:
            img_path = os.path.join(folder_path, img_name)
            per_vi_folder_list.append(img_path)
            # print(img_path)
        print(len(per_vi_folder_list))
        all_vi_folder_list.append(per_vi_folder_list)    
    print('all datasets',len(all_vi_folder_list))
    return all_vi_folder_list

def load_flow(path):
    with open(path, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None

def get_random_frames_list(frame_l2_list,opt_l2_list,frames_num = 8, bacth_size=2,):
    
    try:
        for idx in range(bacth_size):
            folder_num = len(frame_l2_list)
            selected_folder = np.int(random.randint(0,folder_num-1))
            # print(selected_folder)
            # print('optical_num', len(opt_l2_list[selected_folder]))
            # print('img_num', len(frame_l2_list[selected_folder]))
            img_num = len(opt_l2_list[selected_folder])
            selected_img_iter = np.int(random.randint(0,img_num-frames_num-1))
            selected_img_list = frame_l2_list[selected_folder][selected_img_iter:selected_img_iter+frames_num]
            selected_flow_list = opt_l2_list[selected_folder][selected_img_iter:selected_img_iter+frames_num]
            for img_path_iter in selected_img_list:
                img_data = cv2.imread(img_path_iter)
                print('max', img_data.max())
                print('min', img_data.min())
                print('im',img_data.shape)

            for img_path_iter in selected_flow_list:
                img_data = load_flow(img_path_iter)
                print('max', img_data.max())
                print('min', img_data.min())
                print('of',img_data.shape)
            # print(selected_img_iter)
            # print(selected_img_list)
            # print(selected_flow_list)
        
        return 
        
    except :
        print('choose img and cor optical flow error')
        
    return




if __name__ == '__main__':
    frame_list = get_all_img_list()
    optical_flow_list = get_all_img_list(choosen_type='flow')
    get_random_frames_list(frame_list,optical_flow_list, frames_num=8,bacth_size=2)