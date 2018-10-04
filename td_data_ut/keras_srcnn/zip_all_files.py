
import zipfile
import tarfile
import os


if __name__=='__main__':
    folder_path = '/Users/changyunpeng/Downloads/classed_GF2'
    target_path = '/Users/changyunpeng/Downloads/tar_gf2'
    folder_list = os.listdir(folder_path)
    for i,folder_iter in enumerate(folder_list):
        if folder_iter[-3:] == 'zip' :
            continue
        if folder_iter[0] == '.' :
            continue
        print os.path.join(target_path, folder_iter + '.tar.gz')
        tar = tarfile.open(os.path.join(target_path, folder_iter + '.tar.gz'), 'w:gz')
        for file in os.listdir(os.path.join(folder_path, folder_iter)):
            print file
            tmp_file_path_l1 =  os.path.join(folder_path, folder_iter)
            tmp_file_path_l2 =  os.path.join(tmp_file_path_l1, file)
            print tmp_file_path_l2
            tar.add(tmp_file_path_l2, file)
        tar.close()
    
         
        