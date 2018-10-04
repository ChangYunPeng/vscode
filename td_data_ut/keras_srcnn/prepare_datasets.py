from RS_images_utils import save_gray_tiff_png, save_tiff_png, block_png_imgs
import os
from PIL import Image
import random
import cv2

def turn_datasets_tiff_2_png():
    tiff_datasets_path = '/home/room304/moving_storage/OriginalIamge/BeijinCaptialPlane'
    tiff_rgb_save_path = '/home/room304/storage/TB/DATASET/GF/GF2/beijing_png/rgb'
    tiff_gray_save_path = '/home/room304/TB/DATASET/GF/GF2/beijing_png/gray'
    tiff_path_list = os.listdir(tiff_datasets_path)
   
    for tif_path_iter in tiff_path_list:
        if tif_path_iter[-9:] == 'MSS1.tiff':
            save_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter), os.path.join(tiff_rgb_save_path,tif_path_iter[:-4] + 'png'))
            save_gray_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter[:-9] + 'PAN1.tiff'), os.path.join(tiff_gray_save_path,tif_path_iter[:-9] + 'PAN1.png'))
            
    return

def turn_level2_datasets_tiff_2_png():
    tiff_datasets_path = '/home/room304/storage/unzip-GF2'
    tiff_rgb_save_path = '/home/room304/storage/datasets/guangzhou_mss2_png'
    tiff_folder_path_list = os.listdir(tiff_datasets_path)
   
    for tif_floder_path_iter in tiff_folder_path_list:
        for tiff_file_path_itert in os.listdir(os.path.join(tiff_datasets_path,tif_floder_path_iter)):
            if  tiff_file_path_itert[-9:] == 'MSS2.tiff' :
                print (os.path.join(tiff_datasets_path,tif_floder_path_iter,tiff_file_path_itert))
                save_tiff_png(os.path.join(tiff_datasets_path,tif_floder_path_iter,tiff_file_path_itert), os.path.join(tiff_rgb_save_path,tiff_file_path_itert[:-4] + 'png'))
                # save_gray_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter[:-9] + 'PAN1.tiff'), os.path.join(tiff_gray_save_path,tif_path_iter[:-9] + 'PAN1.png'))
            
    return

def resize_save_image(img_path,im_save_path,w_ratio=1.0,h_ratio=1.0,flipped = False):
    image = cv2.imread(img_path)
    image = cv2.resize(image, ( int(image.shape[1]*w_ratio),int(image.shape[0]*h_ratio)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if flipped:
        image = image[:, ::-1, :]
    cv2.imwrite(im_save_path, image)
    return

def resize_save_txt(txt_path,txt_save_path,w_ratio=1.0,h_ratio=1.0,flipped = False):
    file_open = open(txt_path)
    file_write = open(txt_save_path, 'w')
    

    for line in file_open:
        # line_p3 = map(float, line.split())
        line = list(map(float, line.split()))
        for line_ier in line:
            print(line_ier)
        out_line = ('%d %f %f %f %f')%(int(line[0]), float(line[1])*w_ratio, float(line[2])*h_ratio, float(line[3])*w_ratio, float(line[4])*h_ratio)
        file_write.write(out_line)
        file_write.write('\n')
    return

def enlarge_datasets(random_cycle = 8):
    png_folder_path = '/home/room304/storage/datasets/GF2_yolo/images'
    txt_folder_path = '/home/room304/storage/datasets/GF2_yolo/labelTxt'
    target_png_folder_path = '/home/room304/storage/datasets/GF2_yolo/images_enlarged'
    target_txt_folder_path = '/home/room304/storage/datasets/GF2_yolo/labelTxt_enlarged'
    png_path_list = os.listdir(png_folder_path)
   
    for png_floder_path_iter in png_path_list:
        if  png_floder_path_iter[-3:] == 'png' :
            print (os.path.join(png_folder_path,png_floder_path_iter))
            png_path = os.path.join(png_folder_path,png_floder_path_iter)
            txt_path = os.path.join(txt_folder_path,png_floder_path_iter[:-3] + 'txt')
            for ii in range(random_cycle):
                w_ratio =  float(random.randint(8,12))/10.0
                h_ratio =  float(random.randint(8,12))/10.0
                save_png_path = os.path.join(target_png_folder_path,png_floder_path_iter[:-4] + '%d_%d'%(int(w_ratio*10),int(h_ratio*10)) + '.png')
                save_txt_path = os.path.join(target_txt_folder_path,png_floder_path_iter[:-4] + '%d_%d'%(int(w_ratio*10),int(h_ratio*10)) + '.txt')
                
                if os.path.exists(save_png_path):
                    continue
                    # flipped = True
                    # save_png_path = save_png_path[:-4]+'flip' + '.png'
                    # save_txt_path = save_txt_path[:-4] + 'flip' + '.txt'
                    # if os.path.exists(save_png_path):
                    #     continue
                    # else:
                    #     resize_save_image(png_path, save_png_path,w_ratio,h_ratio,flipped)
                else:
                    resize_save_image(png_path, save_png_path,w_ratio,h_ratio)
                    resize_save_txt(txt_path, save_txt_path, w_ratio, h_ratio)
                    


                

            # save_tiff_png(os.path.join(tiff_datasets_path,tif_floder_path_iter,tiff_file_path_itert), os.path.join(tiff_rgb_save_path,tiff_file_path_itert[:-4] + 'png'))
            # save_gray_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter[:-9] + 'PAN1.tiff'), os.path.join(tiff_gray_save_path,tif_path_iter[:-9] + 'PAN1.png'))
            
    return

def block_datasets_png_images():
    
    tiff_rgb_save_path = '/home/room304/TB/DATASET/GF/GF2/png/rgb'
    tiff_gray_save_path = '/home/room304/TB/DATASET/GF/GF2/png/gray'

    gray_block_save_path = '/home/room304/TB/DATASET/GF/GF2/png/gray_block'
    rgb_block_save_path = '/home/room304/TB/DATASET/GF/GF2/png/rgb_block'

    tiff_path_list = os.listdir(tiff_gray_save_path)
   
    for tif_path_iter in tiff_path_list:
        if tif_path_iter[-8:] == 'PAN1.png':
            gray_png_path = os.path.join(tiff_gray_save_path,tif_path_iter)
            rgb_png_path = os.path.join(tiff_rgb_save_path,tif_path_iter[:-8] + 'MSS1.png')

            gray_block_png_path = os.path.join(gray_block_save_path,tif_path_iter[:-8])
            rgb_block_png_path = os.path.join(rgb_block_save_path,tif_path_iter[:-8])



            mat_list,_ = block_png_imgs(gray_png_path,1024)
            for mat_idex,mat_iter in enumerate(mat_list) :
                tmp_block_path = gray_block_png_path + '%d.png'%mat_idex
                img_save = Image.fromarray(mat_iter, mode='L')
                img_save.save(tmp_block_path)
                # print(mat_iter.shape)


            mat_list,_ = block_png_imgs(rgb_png_path,256)
            for mat_idex,mat_iter in enumerate(mat_list) :
                tmp_block_path = rgb_block_png_path + '%d.png'%mat_idex
                img_save = Image.fromarray(mat_iter, mode='RGB')
                img_save.save(tmp_block_path)
                
            # save_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter), os.path.join(tiff_rgb_save_path,tif_path_iter[:-4] + 'png'))
            # save_gray_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter[:-9] + 'PAN1.tiff'), os.path.join(tiff_gray_save_path,tif_path_iter[:-9] + 'PAN1.png'))
            
    return

def block_single_datasets_png_images():
    
    tiff_rgb_save_path = '/home/room304/storage/datasets/guangzhou_png'
    
    rgb_block_save_path = '/home/room304/storage/datasets/rgb_block'

    tiff_path_list = os.listdir(tiff_rgb_save_path)
   
    for tif_path_iter in tiff_path_list:
        if tif_path_iter[-8:] == 'MSS1.png' or tif_path_iter[-8:] == 'MSS2.png' :
            # gray_png_path = os.path.join(tiff_gray_save_path,tif_path_iter)
            rgb_png_path = os.path.join(tiff_rgb_save_path,tif_path_iter)

            # gray_block_png_path = os.path.join(gray_block_save_path,tif_path_iter[:-8])
            rgb_block_save_floder_path = os.path.join(rgb_block_save_path,tif_path_iter[:-8])
            if not os.path.exists(rgb_block_save_floder_path):
                os.mkdir(rgb_block_save_floder_path)

            mat_list,_ = block_png_imgs(rgb_png_path,1024)
            for mat_idex,mat_iter in enumerate(mat_list) :
                tmp_block_path = os.path.join(rgb_block_save_floder_path, tif_path_iter[:-8]+ '%d.png'%mat_idex)
                img_save = Image.fromarray(mat_iter, mode='RGB')
                img_save.save(tmp_block_path)
                
            # save_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter), os.path.join(tiff_rgb_save_path,tif_path_iter[:-4] + 'png'))
            # save_gray_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter[:-9] + 'PAN1.tiff'), os.path.join(tiff_gray_save_path,tif_path_iter[:-9] + 'PAN1.png'))
            
    return


if __name__=='__main__':
    # block_single_datasets_png_images()
    # turn_level2_datasets_tiff_2_png()
    enlarge_datasets()
    # block_datasets_png_images()
    # tiff_datasets_path = '/home/room304/TB/DATASET/GF/GF2/GF2'
    # tiff_rgb_save_path = '/home/room304/TB/DATASET/GF/GF2/png/rgb'
    # tiff_gray_save_path = '/home/room304/TB/DATASET/GF/GF2/png/gray'
    # tiff_path_list = os.listdir(tiff_datasets_path)
    # # print(tiff_path_list)
    # for tif_path_iter in tiff_path_list:
    #     if tif_path_iter[-9:] == 'MSS1.tiff':
    #         save_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter), os.path.join(tiff_rgb_save_path,tif_path_iter[:-4] + 'png'))
    #         save_gray_tiff_png(os.path.join(tiff_datasets_path,tif_path_iter[:-9] + 'PAN1.tiff'), os.path.join(tiff_gray_save_path,tif_path_iter[:-9] + 'PAN1.png'))
    #         # print(tif_path_iter[-9:])
    #         # print(tif_path_iter[:-9])