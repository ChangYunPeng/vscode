import cv2
import numpy as np

im_data = cv2.imread('/home/room304/storage/datasets/guangzhou_png/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.png')
print ('shape 0',im_data.shape[0])
print ('shape 1',im_data.shape[1])

im_data_1 = cv2.resize(im_data,(1000,800))
print ('shape 0',im_data_1.shape[0])
print ('shape 1',im_data_1.shape[1])