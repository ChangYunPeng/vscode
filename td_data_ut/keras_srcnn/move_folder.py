from RS_images_utils import save_gray_tiff_png, save_tiff_png, block_png_imgs
import os
from PIL import Image
import shutil


if __name__=='__main__':
    folder_path = '/home/room304/storage/datasets/guangzhou_class_png'
    folder_list = os.listdir(folder_path)
    for i,folder_iter in enumerate(folder_list):
        if folder_iter[-8:] == 'MSS1.png':
            full_path = os.path.join(folder_path, folder_iter)
            new_path = os.path.join(folder_path,'m1')
            if not os.path.exists(new_path):
                    os.mkdir(new_path)
        else:
            full_path = os.path.join(folder_path, folder_iter)
            new_path = os.path.join(folder_path,'m2')
            if not os.path.exists(new_path):
                    os.mkdir(new_path)
        shutil.move(full_path, new_path)
        