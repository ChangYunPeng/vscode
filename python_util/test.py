# -*- coding:utf-8 -*-

import datetime
import tifffile
import numpy as np
import os
import tensorflow as tf

def _repeat(x, n_repeats):
    # print(n_repeats)
    with tf.variable_scope('repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        # print('rep shape', rep.get_shape().as_list())
        # print('x shape', x.get_shape().as_list())
        # print('x tensor', x)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

width = 15
height = 15
num_batch = 2
dim1 = width*height
x = tf.range(num_batch)*dim1
base = _repeat(x , 15*15)

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
    print(sess.run(base))

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