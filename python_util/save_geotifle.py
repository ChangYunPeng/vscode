import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import os
import numpy as np
import tifffile

def isfolder( filepath ):

    save_file_path_list = filepath.split('/')

    if filepath[0] == '/' :
        save_file_path = '/'  + save_file_path_list[0]
    else:
        save_file_path = save_file_path_list[0]
    
    for idx in range(1,len(save_file_path_list)-1):
        save_file_path = os.path.join(save_file_path,save_file_path_list[idx])
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
    print(save_file_path)

    return

def save_file(save_path, data, bound, dst_crs='epsg:4326'):
    """
    每次运行脚本时，将生成的数据存储到一个临时位置，通过commit方式，将这个文件夹上传入库
    按照给出的影像信息，将影像数据保存为本地文件
    :param name: str,  保存的文件名称
    :param data:     numpy.arr 影像数据 shape：(band, height, width)
    :param bound:    [w, s, e, n]
    :param dst_crs:  输出影像的坐标系
    :return: bool 创建成功，返回 True; 创建失败，返回 False.
    """

    isfolder(save_path)
    try:
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        height_res = abs(bound[3] - bound[1]) / data.shape[1]
        width_res = abs(bound[2] - bound[0]) / data.shape[2]
        transform = from_origin(bound[0], bound[3], height_res, width_res)
        with rasterio.open(save_path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2],
                            count=data.shape[0],
                            dtype=data.dtype, crs=CRS({'init': dst_crs}), transform=transform) as dst:
            dst.write(data)

    except Exception as err:
        print(err)
        return False
    else:
        return True


if __name__ == '__main__':
    file_path = '/home/room304/storage/datasets/unzip-GF2/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623./GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.tiff'
    save_path = '/home/room304/storage/tmp/Gene_GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.tiff'
    bound = []
    bound.append(112.724)
    bound.append(23.2016)
    bound.append(113.012)
    bound.append(23.4585)
    data = tifffile.imread(file_path)
    data = np.transpose(data,axes=[2,0,1])
    print(data.shape)
    save_file(save_path=save_path,data=data,bound = bound)
