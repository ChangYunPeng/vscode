import cv2
import os
import numpy as np

def get_rgb_np_of_optical_flow(optical_flow_np):
    hsv = np.zeros(shape=[optical_flow_np.shape[0],optical_flow_np.shape[1],3],dtype='uint8')
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(optical_flow_np[..., 0], optical_flow_np[..., 1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb

def save_batch_images(batches_input, save_path, save_name):
    if batches_input.shape[3] == 1:
        for img_idx in range(batches_input.shape[0]):
            img = batches_input[img_idx,:,:,0]* 255
            cv2.imwrite(os.path.join(save_path,'%d_'%img_idx + save_name), img)
    
    if batches_input.shape[3] == 2:
        for img_idx in range(batches_input.shape[0]):
            img = batches_input[img_idx,:,:,:]
            img = get_rgb_np_of_optical_flow(img)
            cv2.imwrite(os.path.join(save_path,'%d_'%img_idx + save_name), img)

    return