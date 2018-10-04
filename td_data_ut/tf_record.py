import os 
import tensorflow as tf 
from PIL import Image  
# import matplotlib.pyplot as plt 
import numpy as np

def get_img_path_list(dataset_path):
    img_list_path = os.listdir(dataset_path)
    img_path_list = []
    for img_iter in img_list_path:
        img_path_list.append(os.path.join(dataset_path, img_iter))

    # img_path_list = os.path.join(dataset_path,img_list_path)
    return img_path_list

filename = get_img_path_list('/home/room304/TB/DATASET/train2017')

writer= tf.python_io.TFRecordWriter("./train2017.tfrecords") 
for img_path_iter in filename:
    img=Image.open(img_path_iter)
    img= img.resize((256,256))
    img_raw=img.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        "no_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        'img_str': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    })) 
    writer.write(example.SerializeToString())
writer.close()
