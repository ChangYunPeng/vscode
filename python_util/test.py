# -*- coding:utf-8 -*-

import datetime
import tifffile
import numpy as np
import os
# import tensorflow as tf
import cv2
from ucsd_t1 import TestVideoFile as t1

video_path = '/home/room304/TB/TB/DATASET/Avenue_Dataset/testing_videos/01'
path_list = os.listdir(video_path)
path_list.sort()
print(path_list)
# img_size = False
# if img_size:
#     print(t1)
# else:
#     print(img_size)




# tmpp = [12,23]
# print (tmpp)
# def get_bound(detail_json):
#     bound_list = []
#     for i,(lat1,lat2,lat3,lat4,lon1,lon2,lon3,lon4) in enumerate(zip(detail_json['lat1'],detail_json['lat2'],detail_json['lat3'], detail_json['lat4'],detail_json['lon1'],detail_json['lon2'],detail_json['lon3'], detail_json['lon4'])):
#         w = min(lon1,lon2,lon3,lon4)
#         s = min(lat1,lat2,lat3,lat4)
#         e = max(lon1,lon2,lon3,lon4)
#         n = max(lat1,lat2,lat3,lat4)
#         bound = [w,s,e,n]
#         bound_list.append(bound)
#     return bound_list

# det = ({'lat4': [23.005910188532468], 'color_list': [{'b': '224', 'g': '42', 'r': '29'}, {'b': '119', 'g': '255', 'r': '0'}], 'images_url': ['/storage/geocloud/test/data/原始影像数据库/GF2/L1A/PMS/4m多光谱/GF2_PMS1_E113.2_N23.3_20171208_L1A0002831514-MSS1.tiff'], 'uid': 274, 'initialization': '/storage/WHU/model/train/132/', 'lat1': [23.32912158929669], 'lat3': [23.32912158929669], 'lon1': [113.68541791327642], 'app_images_uid': [448], 'lat2': [23.005910188532468], 'app_type_content': ['', 'lake', 'tree'], 'lon2': [113.68541791327642], 'result_list': ['/storage/WHU/result/274/GF2_PMS1_E113.2_N23.3_20171208_L1A0002831514-MSS1.tiff'], 'lon3': [113.22399213202642], 'lon4': [113.22399213202642], 'thumbnail_list': ['/storage/WHU/result/274/thumbnail/GF2_PMS1_E113.2_N23.3_20171208_L1A0002831514-MSS1.tiff']})
# print(get_bound(det))

# def _repeat(x, n_repeats):
#     # print(n_repeats)
#     with tf.variable_scope('repeat'):
#         rep = tf.transpose(
#             tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
#         rep = tf.cast(rep, 'int32')
#         # print('rep shape', rep.get_shape().as_list())
#         # print('x shape', x.get_shape().as_list())
#         # print('x tensor', x)
#         x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
#         return tf.reshape(x, [-1])

# x = tf.placeholder(shape=[None, None,None, 3],dtype = tf.float32)
# ker_init = tf.constant_initializer([[0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
#                                             [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
#                                             [0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
#                                             [0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
#                                             [0.000874, 0.006976, 0.01386, 0.006976, 0.000874]])
# smooth_x = tf.layers.conv2d(x,3, [5, 5], padding='same',  kernel_initializer=ker_init, use_bias=False, trainable=False)
# trainable_weights = tf.global_variables()
# with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
#     sess.run(tf.global_variables_initializer())
#     ker_inited = sess.run([trainable_weights],feed_dict={x:np.ones(shape=[4,100,100,3])})
#     ker_inited = ker_inited[0]
#     print(ker_inited)

# width = 15
# height = 15
# num_batch = 2
# dim1 = width*height
# x = tf.range(num_batch)*dim1
# base = _repeat(x , 15*15)

# with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
#     print(sess.run(base))

# print(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

# print(80300.0/7300.0)
# iomat = tifffile.imread('/Users/changyunpeng/CODE_BACKUP/GF2_PMS1_E113.4_N23.3_20170915_L1A0002600401/GF2_PMS1_E113.4_N23.3_20170915_L1A0002600401-MSS1.tif')
# print(iomat.shape)
# "block_lat1": 22.70506056480798, "block_lon1": 113.89879130296187, 
# "block_lat2": 22.69854275204969, "block_lon2": 113.93504680614278,
# "block_lat3": 22.668499103466488, "block_lon3": 113.92848666057257,
# "block_lat4": 22.675014653967132, "block_lon4": 113.89223841300341

# save_file_name = '/storage/geocloud/test/data/原始影像数据库/GF2/L1A/PMS/4m多光谱/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.tiff'

# save_file_path_list = save_file_name.split('/')

# if save_file_name[0] == '/' :
#     save_file_path = '/'  + save_file_path_list[0]
# else:
#     save_file_path = save_file_path_list[0]

# for idx in range(1,len(save_file_path_list)-1):
#     save_file_path = os.path.join(save_file_path,save_file_path_list[idx])
#     if not os.path.exists(save_file_path):
#         print (save_file_path)
#         # os.mkdir(save_file_path)
# print(save_file_path)

# result_list = []
# for idx in range(10):
#     result = np.ones(shape=(73000))
#     result = np.reshape(result,newshape=(-1,7300))
#     print(result.shape)
#     result_list.append(result)

# result_list = np.concatenate(result_list,axis=0)
# print(result_list.shape)