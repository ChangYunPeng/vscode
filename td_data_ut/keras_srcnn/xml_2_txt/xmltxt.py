
import xml.dom.minidom
import numpy as np
import os
import shutil

CLASS_MAP = ['plane', 'bridge', 'football ground', 'lake']

def turn_xml_txt(xml_path,txt_path):
    
    DomTree = xml.dom.minidom.parse(xml_path)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    file = open(txt_path, 'w')
    i = 1
    for objects in objectlist:
        # print objects
        namelist = objects.getElementsByTagName('name')
        # print 'namelist:',namelist
        objectname = namelist[0].childNodes[0].data
        print objectname
        for cla_idx,cla_name in enumerate(CLASS_MAP):
            if objectname == cla_name:
                print(cla_idx)
                break
        bndbox = objects.getElementsByTagName('bndbox')
        cropboxes = []
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w/2.0
            cy = y1 + h/2.0
            print(x1, x2, y1, y2)
            print(w, h)
            print(cx,cy)

            norm_w = w/1024.0
            norm_h = h/1024.0
            norm_cx = cx/1024.0
            norm_cy = cy/1024.0
            out_line = ('%d %f %f %f %f')%(cla_idx, norm_cx, norm_cy, norm_w, norm_h)
            file.write(out_line)
            file.write('\n')
    file.close()

dataset_path_l2 = '/Users/changyunpeng/VSCODE/GF2_yolo/emmm'
# img_dataset_path_ls = '/Users/changyunpeng/VSCODE/'
target_txt_dataset_path = '/Users/changyunpeng/VSCODE/GF2_yolo/labelTxt'
target_img_dataset_path = '/Users/changyunpeng/VSCODE/GF2_yolo/img'
for labelfolder_dataset_path_iter in os.listdir(dataset_path_l2):
    if labelfolder_dataset_path_iter[-5:] == 'label':
        
        img_dataset_path_iter = labelfolder_dataset_path_iter[:-5]

        xml_dataset_path_l1 = os.path.join(dataset_path_l2, labelfolder_dataset_path_iter)
        img_dataset_path_l1 = os.path.join(dataset_path_l2, img_dataset_path_iter)

        print ('xml path l1 :', xml_dataset_path_l1)
        print ('png path l1 :', img_dataset_path_l1)
        if os.path.exists(xml_dataset_path_l1):
            xml_path_list = os.listdir(xml_dataset_path_l1)
            for xml_path_iter in xml_path_list:
                xml_full_path = os.path.join( xml_dataset_path_l1, xml_path_iter)
                
                print ('xml full path l1 :', xml_full_path)
                source_img_path = os.path.join(img_dataset_path_l1, xml_path_iter[:-3]+'png')
                print ('source path ', source_img_path)
                txt_full_path = os.path.join( target_txt_dataset_path, xml_path_iter)
                # img_full_path = os.path.join( target_img_dataset_path, xml_path_iter)
                turn_xml_txt(xml_full_path,txt_full_path)
                shutil.copy(source_img_path,target_img_dataset_path)
        else:
            print ('file not exits :', xml_dataset_path_l1)
