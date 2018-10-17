import os
import json
import xml.dom.minidom
from convert import calcu_trans, pixel_to_latlon, latlon_to_pixel

def get_lat_lon_from_xml(xml_path):
    lon_list = []
    lat_list = []
    if os.path.exists(xml_path):
        DomTree = xml.dom.minidom.parse(xml_path)
        annotation = DomTree.documentElement
        # print(annotation)

        # ProductMetaData = annotation.getElementsByTagName("ProductMetaData") #[<DOM Element: filename at 0x381f788>]
        # for ProductMetaData_iter in ProductMetaData:

        tmp_list = annotation.getElementsByTagName('TopLeftLatitude')
        lat1 = tmp_list[0].childNodes[0].data

        tmp_list = annotation.getElementsByTagName('TopLeftLongitude')
        lon1 = tmp_list[0].childNodes[0].data

        tmp_list = annotation.getElementsByTagName('TopRightLatitude')
        lat2 = tmp_list[0].childNodes[0].data

        tmp_list = annotation.getElementsByTagName('TopRightLongitude')
        lon2 = tmp_list[0].childNodes[0].data

        tmp_list = annotation.getElementsByTagName('BottomRightLatitude')
        lat4 = tmp_list[0].childNodes[0].data

        tmp_list = annotation.getElementsByTagName('BottomRightLongitude')
        lon4 = tmp_list[0].childNodes[0].data

        tmp_list = annotation.getElementsByTagName('BottomLeftLatitude')
        lat3 = tmp_list[0].childNodes[0].data

        tmp_list = annotation.getElementsByTagName('BottomLeftLongitude')
        lon3 = tmp_list[0].childNodes[0].data

        lon_list.append(float(lon1))
        lon_list.append(float(lon2))
        lon_list.append(float(lon3))
        lon_list.append(float(lon4))
        lat_list.append(float(lat1))
        lat_list.append(float(lat2))
        lat_list.append(float(lat3))
        lat_list.append(float(lat4))
    else :
        return False
    return lon_list, lat_list

def convert_img_loca_latlon(r,c,trans, zone_num,zone_let, w=7300, h=6908):
    # trans, zone_num,zone_let = calcu_trans(lon_list, lat_list, args.w, args.h)
    pixel_to_latlon(trans,r,c,zone_num,zone_let)
    return

def get_info_from_xml(xml_path):
    
    DomTree = xml.dom.minidom.parse(xml_path)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    json_list = []
    
    for objects in objectlist:


        # print objects
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data

      
        bndbox = objects.getElementsByTagName('bndbox')
  
        for box in bndbox:
            
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
           
            json_obj = {}
            json_obj['filename'] = filename
            json_obj['obj_name'] = objectname
            json_obj['xmin'] = x1
            json_obj['ymin'] = y1
            json_obj['xmax'] = x2
            json_obj['ymax'] = y2
            json_list.append(json_obj)
    return json_list

def get_dataset_path():
    img_dataset_path = '/home/room304/storage/datasets/GF2_yolo/images'
    latlon_xml_foler_path = '/home/room304/storage/datasets/unzip-GF2'
    imgpoints_folder_path = '/home/room304/storage/datasets/rgb_block_label/'
    saved_json_file_path = '/home/room304/storage/datasets/GF2_yolo/record_new_all.json'

    datatset_json = []
    
    img_list = os.listdir(img_dataset_path)

    with open(saved_json_file_path,'w') as f:
        for img_path_iter in img_list:
            json_tmp = {}
            img_name = img_path_iter.split('-')[0]
            # if not img_name == 'GF2_PMS1_E113.8_N22.6_20171227_L1A0002883457':
            #     continue


            img_num = int((img_path_iter.split('-')[1]).split('.')[0])
            col_num = int(img_num%7)
            row_num = int(img_num/7)
            # print(row_num*7 + col_num, '=? ', img_path_iter.split('-')[1])
            json_tmp['img_name'] = img_name
            json_tmp['col_num'] = col_num
            json_tmp['row_num'] = row_num
            # print('cols_num:', col_num, 'rows_num:', row_num)
            
            """
            get latlon from xml file
            """
            latlon_xml_path = os.path.join(latlon_xml_foler_path, img_name+'.', img_name + '-MSS1.xml')
            lon_list, lat_list = get_lat_lon_from_xml(latlon_xml_path)

            imgpoints_path = os.path.join(imgpoints_folder_path, img_name+'-label', img_path_iter[:-3] + 'xml')
            # print(imgpoints_path)
            if not os.path.exists(imgpoints_path):
                imgpoints_path = os.path.join(imgpoints_folder_path, img_name+'-label', img_path_iter[:-4])
                if not os.path.exists(imgpoints_path):
                    print('wht')
                    print(imgpoints_path)
            
            trans, zone_num, zone_let = calcu_trans(lon_list, lat_list, 7300, 6908)
            # print( col_num, row_num )
            print('1*5',pixel_to_latlon(trans,1024*(1),1024*(5),zone_num,zone_let))
            print('2*5',pixel_to_latlon(trans,1024*(2),1024*(5),zone_num,zone_let))
            print('5*2',pixel_to_latlon(trans,1024*(5),1024*(2),zone_num,zone_let))
            print('5*1',pixel_to_latlon(trans,1024*(5),1024*(1),zone_num,zone_let))
            # print('2',pixel_to_latlon(trans,1024*(col_num),1024*(row_num+1),zone_num,zone_let))
            # print('3',pixel_to_latlon(trans,1024*(col_num+1),1024*(row_num+1),zone_num,zone_let))
            # print('4',pixel_to_latlon(trans,1024*(col_num+1),1024*(row_num),zone_num,zone_let))

            # print('lonlist 0-- ', lon_list[0])
            # print('latlist 0-- ', lat_list[0])
            # print('lonlist 3-- ', lon_list[3])
            # print('latlist 3-- ', lat_list[3])

            # print(pixel_to_latlon(trans,0,0,zone_num,zone_let))
            # print('7,6',pixel_to_latlon(trans,7300,6908,zone_num,zone_let))
            # # print('6,7',pixel_to_latlon(trans,6908,7300,zone_num,zone_let))

            # print('lonlist 1-- ', lon_list[1])
            # print('latlist 1-- ', lat_list[1])
            # print('1',pixel_to_latlon(trans,0,6908,zone_num,zone_let))
            # print('2',pixel_to_latlon(trans,0,7300,zone_num,zone_let))
            # print('3',pixel_to_latlon(trans,6908,0,zone_num,zone_let))
            # print('4',pixel_to_latlon(trans,7300,0,zone_num,zone_let))

            # print('lonlist 2-- ', lon_list[2])
            # print('latlist 2-- ', lat_list[2])

            json_list = get_info_from_xml(imgpoints_path)
            for json_iter in json_list:
                json_iter['filename'] = img_path_iter[:-4]

                # json_iter['xmin'] += 1024*col_num
                # json_iter['ymin'] += 1024*row_num
                # json_iter['xmax'] += 1024*col_num
                # json_iter['ymax'] += 1024*row_num



                # lat1, lon1 = pixel_to_latlon(trans,json_iter['ymin']+ 1024*row_num,json_iter['xmin']+ 1024*col_num,zone_num,zone_let)
                # print(json_iter['ymin']+ 1024*row_num,json_iter['xmin']+ 1024*col_num, latlon_to_pixel(trans,lat1,lon1))
                # lat2, lon2 = pixel_to_latlon(trans,json_iter['ymin']+ 1024*row_num,json_iter['xmax']+ 1024*col_num,zone_num,zone_let)
                # lat3, lon3 = pixel_to_latlon(trans,json_iter['ymax']+ 1024*row_num,json_iter['xmax']+ 1024*col_num,zone_num,zone_let)
                # lat4, lon4 = pixel_to_latlon(trans,json_iter['ymax']+ 1024*row_num,json_iter['xmin']+ 1024*col_num,zone_num,zone_let)

                # lat1, lon1 = pixel_to_latlon(trans,json_iter['ymin']+ 1024*col_num,json_iter['xmin']+ 1024*row_num,zone_num,zone_let)
                # lat2, lon2 = pixel_to_latlon(trans,json_iter['ymin']+ 1024*col_num,json_iter['xmax']+ 1024*row_num,zone_num,zone_let)
                # lat3, lon3 = pixel_to_latlon(trans,json_iter['ymax']+ 1024*col_num,json_iter['xmax']+ 1024*row_num,zone_num,zone_let)
                # lat4, lon4 = pixel_to_latlon(trans,json_iter['ymax']+ 1024*col_num,json_iter['xmin']+ 1024*row_num,zone_num,zone_let)

                lat1, lon1 = pixel_to_latlon(trans,json_iter['xmin']+ 1024*col_num,json_iter['ymin']+ 1024*row_num,zone_num,zone_let)
                lat2, lon2 = pixel_to_latlon(trans,json_iter['xmax']+ 1024*col_num,json_iter['ymin']+ 1024*row_num,zone_num,zone_let)
                lat3, lon3 = pixel_to_latlon(trans,json_iter['xmax']+ 1024*col_num,json_iter['ymax']+ 1024*row_num,zone_num,zone_let)
                lat4, lon4 = pixel_to_latlon(trans,json_iter['xmin']+ 1024*col_num,json_iter['ymax']+ 1024*row_num,zone_num,zone_let)
                
                # print(json_iter['xmin']+ 1024*col_num, json_iter['ymin']+ 1024*row_num, latlon_to_pixel(trans,lat1,lon1))


                json_iter['lat1'] = lat1
                json_iter['lon1'] = lon1
                json_iter['lat2'] = lat2
                json_iter['lon2'] = lon2
                json_iter['lat3'] = lat3
                json_iter['lon3'] = lon3
                json_iter['lat4'] = lat4
                json_iter['lon4'] = lon4
                json_iter['col_num'] = col_num
                json_iter['row_num'] = row_num

                json_iter['original_lat1'] = lat_list[0]
                json_iter['original_lat2'] = lat_list[1]
                json_iter['original_lat3'] = lat_list[3]
                json_iter['original_lat4'] = lat_list[2]

                json_iter['original_lon1'] = lon_list[0]
                json_iter['original_lon2'] = lon_list[1]
                json_iter['original_lon3'] = lon_list[3]
                json_iter['original_lon4'] = lon_list[2]
                if 1024*(col_num+1) >=7300:
                    print(col_num)
                if 1024*(row_num+1) >=6908:
                    print(row_num)
                

                lat1, lon1 = pixel_to_latlon(trans,1024*(col_num),1024*(row_num),zone_num,zone_let)
                lat2, lon2 = pixel_to_latlon(trans,1024*(col_num),1024*(row_num+1),zone_num,zone_let)
                lat3, lon3 = pixel_to_latlon(trans,1024*(col_num+1),1024*(row_num+1),zone_num,zone_let)
                try:
                    lat4, lon4 = pixel_to_latlon(trans,1024*(col_num+1),1024*(row_num),zone_num,zone_let)
                except:
                    print(json_iter)
                    continue
                json_iter['block_lat1'] = lat1
                json_iter['block_lon1'] = lon1
                json_iter['block_lat2'] = lat2
                json_iter['block_lon2'] = lon2
                json_iter['block_lat3'] = lat3
                json_iter['block_lon3'] = lon3
                json_iter['block_lat4'] = lat4
                json_iter['block_lon4'] = lon4
                
                json.dump(json_iter,f)
                # f.write('\n')
                # print (json_iter)
   
        
    return

get_dataset_path()
# get_lat_lon_from_xml('/home/room304/storage/unzip-GF2/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623./GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.xml')
