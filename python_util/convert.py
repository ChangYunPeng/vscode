# -*- encoding: utf-8 -*-

# from osgeo import gdal
# from osgeo import osr
import numpy as np
import utm
import argparse
from scipy.optimize import leastsq

##需要拟合的函数func :指定函数的形状
def func(p,x):
    
    return p.dot(x)

##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(x,A,y):
    # print 'x', x
    # print 'A', A
    # print 'y', y
    
    return A.dot(x)-y


def imagexy2geo(trans, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
#     trans = dataset.GetGeoTransform()
#     print trans
    px = trans[0] + row * trans[1] + col * trans[2]
    py = trans[3] + row * trans[4] + col * trans[5]
    return px, py

def pixel_to_latlon(trans, row, col, zone_num, zone_let):
    px,py = imagexy2geo(trans,row,col)
    lat,lon = utm.to_latlon(px, py, zone_num, zone_let)
    return lat, lon

def latlon_to_pixel(trans, lat, lon):
    utm_xy = utm.from_latlon(lat, lon)
    
    pxpy = geo2imagexy(trans, utm_xy[0], utm_xy[1])
    
    return pxpy


def geo2imagexy(trans, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    
    '''
    
    # print 'x', x
    # print 'y', y
#     trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    
    a= np.mat(a)


    
    b = np.mat(b)
    b = np.mat(b.T)
    # print 'b', b


    
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def array_a(w, h):
    tmp_tans = w
    w = h
    h = tmp_tans
    tmp = np.zeros(shape=(8, 6))
    tmp[0, 0] = 1
    tmp[1, 3] = 1

    tmp[2, 0] = 1
    tmp[2, 1] = h
    tmp[3, 3] = 1
    tmp[3, 4] = h
    
    tmp[4, 0] = 1
    tmp[4, 2] = w
    tmp[5, 3] = 1
    tmp[5, 5] = w

    tmp[6, 0] = 1
    tmp[6, 1] = h
    tmp[6, 2] = w
    tmp[7, 3] = 1
    tmp[7, 4] = h
    tmp[7, 5] = w
    return tmp

def lsm(a,y):
    at = np.mat(a.T)
    a = np.mat(a)
    y = np.mat(y)
    
    ata = at.dot(a)
    ata_inv = np.linalg.inv(ata)
#     ata_inv = np.mat(ata.I)
    aty = at.dot(y)
    
    return ata_inv.dot(aty)

def calcu_trans(lon_list=[],lat_list=[],w=7300,h=6908):
    lon = lon_list
    
    # lon.append(105.406)
    # lon.append(105.644)
    # lon.append(105.597)
    # lon.append(105.359)
    
#     print lon
    lat = lat_list
    
#     print lat
    # lat.append(25.9164)
    # lat.append(25.8733)
    # lat.append(25.6641)
    # lat.append(25.7072)

    W = w
    H = h
    # lon = [105.406, 105.644, 105.597, 105.359]
    # lat = [25.9164, 25.8733, 25.6641, 25.7072]

    arr_y = np.zeros(shape=(8,1))
    arr_A = array_a(W,H)
    list_A = [] 
    list_y = []
    ini_arr_x = np.ones(shape=(1,6))
    for idx in range(len(lon)):
#         print '%d' %idx 

        #EASTING, NORTHING
        utm_xy = utm.from_latlon(lat[idx], lon[idx])
        # print utm_xy
        arr_y[2*idx,0] = utm_xy[0]
        arr_y[2*idx+1,0] = utm_xy[1]
        # arr_y[2*idx,0] = utm_xy[1]
        # arr_y[2*idx+1,0] = utm_xy[0]
        list_y.append(utm_xy[0])
        list_y.append(utm_xy[1])
    
    for idx_shape in range(arr_A.shape[0]):
        list_A.append(arr_A[idx_shape,:])
    # print 'list_a', list_A
    # print 'list_y', list_y
    ini_arr_x[0,0] = arr_y[0,0]
    ini_arr_x[0,3] = arr_y[1,0]
    Para=leastsq(error,ini_arr_x,args=(np.array(list_A),np.array(list_y)))
    # print 'Para', Para[0]
    # return lsm(arr_A,arr_y), utm_xy[2], utm_xy[3]
    return Para[0], utm_xy[2], utm_xy[3]
    

#print longs
#print lats



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w', default=7300, type=float)
    parser.add_argument('--h', default=6908, type=float)

    parser.add_argument('--lon1', default=105.406, type=float)
    parser.add_argument('--lon2', default=105.644, type=float)
    parser.add_argument('--lon3', default=105.359, type=float)
    parser.add_argument('--lon4', default=105.597 , type=float)
    # 'python convert.py --w 7300 --h 6900 --lon1 105.406 --lon2 105.644 --lon3 105.359 --lon4 105.597 --lat1 25.9164 --lat2 25.8733 --lat3 25.7072 --lat4 25.6641 --lonlat2pxpy 1 --lon 105.406 --lat 25.9164'

    # 'python convert.py --w 7300 --h 6900 --lon1 105.406 --lon2 105.644 --lon3 105.359 --lon4 105.597 --lat1 25.9164 --lat2 25.8733 --lat3 25.7072 --lat4 25.6641 --pxpy2lonlat 1 --row 6908 --col 7300'
    parser.add_argument('--lat1', default=25.9164, type=float)
    parser.add_argument('--lat2', default=25.8733, type=float)
    parser.add_argument('--lat3', default=25.7072, type=float)
    parser.add_argument('--lat4', default=25.6641 , type=float)
    
    
    parser.add_argument('--lonlat2pxpy', default=0, type=int)
    parser.add_argument('--lon', default=105.406, type=float)
    parser.add_argument('--lat', default=25.9164, type=float)
    
    parser.add_argument('--pxpy2lonlat', default=0, type=int)
    parser.add_argument('--row', default=6908, type=float)
    parser.add_argument('--col', default=7300, type=float)


    args = parser.parse_args()
#     print args.data_dir

    lat_list = []
    lat_list.append(args.lat1)
    lat_list.append(args.lat2)
    lat_list.append(args.lat3)
    lat_list.append(args.lat4)

    lon_list = []
    lon_list.append(args.lon1)
    lon_list.append(args.lon2)
    lon_list.append(args.lon3)
    lon_list.append(args.lon4)
    
    trans, zone_num,zone_let = calcu_trans(lon_list, lat_list, args.w, args.h)
    # print trans
    
    if args.lonlat2pxpy:
        pxpy = latlon_to_pixel(trans, args.lat, args.lon)
        # print pxpy.shape
        # print 'px', np.asscalar(np.reshape(pxpy[0][0],newshape=[1]))
        # print 'py', np.asscalar(np.reshape(pxpy[1][0],newshape=[1]))
        tmp_json = {}
        tmp_json['px'] = np.asscalar(np.reshape(pxpy[0][0],newshape=[1]))
        tmp_json['py'] = np.asscalar(np.reshape(pxpy[1][0],newshape=[1]))
        print (tmp_json)
        # print 'pxpy', pxpy[0][0],pxpy[1][0]
        # lat,lon = pixel_to_latlon(trans, pxpy[0][0], pxpy[1][0], zone_num , zone_let )
        # print 'back', lat, lon
    else:
        print (zone_num, zone_let)
        lat,lon = pixel_to_latlon(trans, args.row, args.col, zone_num , zone_let )
        print (lat,lon)
#     print utm.to_latlon(pxpy[0], pxpy[1],48,'R')