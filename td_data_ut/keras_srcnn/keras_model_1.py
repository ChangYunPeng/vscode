import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
# import keras.layers as KKL
from tf_datasets import get_blockimgs_tf_datasets_from_pathlist as get_datasets
from tf_datasets import get_gray_rgb_block_path_list
import time
import os
from PIL import Image
import cv2
import numpy as np


def build_model(input_tensor, channel = 3, model_name = 'srcnn'):
    # Reset default graph. Keras leaves old ops in the graph,
    # which are ignored for execution but clutter graph
    # with tf.Graph().as_default():
    # inputs = input_tensor
    with tf.variable_scope(model_name) as vs:
        inputs = KL.Input(tensor=(input_tensor))
        x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",data_format= 'channels_last',
                        name="conv1")(inputs)
        x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                        name="conv2")(x)
        x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                        name="conv3")(x)
        x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                        name="conv4")(x)
        # x = KL.Conv2DTranspose(64,(4,4), strides = 2,padding="same",activation='relu',name="deconv")(x)
        outputs = KL.Conv2D(1, (3, 3), activation='tanh', padding="same",
                        name="final")(x)
        print(outputs)
        # outputs = KKL.Add()([outputs , 0.5*tf.ones_like(outputs,dtype=tf.float32)])
        outputs = KL.Lambda(lambda o: tf.add(o, 0.5) , name='add_tf')(outputs)
        # outputs = KL.add([outputs , 0.5*tf.ones_like(outputs,dtype=tf.float32)])
        print(outputs)
    
    variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=model_name)
    # outputs = KL.output(tensor=(outputs))
    return inputs, outputs, variable_list

def gene_img(input_numpy):
    

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    gray_datasets_path = '/home/room304/TB/DATASET/GF/GF2/png/gray_block'
    rgb_datasets_path = '/home/room304/TB/DATASET/GF/GF2/png/rgb_block'
    gray_filenames, rgb_filenames = get_gray_rgb_block_path_list(gray_datasets_path,rgb_datasets_path)

    # gray_filenames =[]
    # rgb_filenames = []
    # gray_filenames.append('/home/room304/storage/TB/DATASET/GF/GF2/png/gray/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-PAN1.png')
    # rgb_filenames.append('/home/room304/storage/TB/DATASET/GF/GF2/png/rgb/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.png')
    my_datasets = get_datasets( gray_filenames, rgb_filenames, batch_size=1, num_epochs = 1000)
    iterator = my_datasets.make_initializable_iterator()
    next_example, netx_label = iterator.get_next()
    next_example.set_shape([None, None, None, 3])
    netx_label.set_shape([None, None, None, 1])


    inputs,outputs,var_list = build_model(next_example)
    my_keras_model = tf.keras.Model(inputs,outputs)
    gpu_options = tf.GPUOptions(allow_growth=True)
    saver = tf.train.Saver(var_list)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess,'./tf_weights-100')
        my_keras_model.load_weights('./tf_k_block_rgb2gray_weights.h5')
        sess.run(iterator.initializer)
        for counters in range(100):
            # print (my_keras_model.load_weights('./tf_k_block_rgb2gray_weights.h5'))
            outnumpy = sess.run([outputs])
            outnumpy = outnumpy[0]
            print(outnumpy.shape)
            outnumpy = np.reshape(outnumpy,[outnumpy.shape[1],outnumpy.shape[2]])
            print('max : ', outnumpy.max())
            print('min : ', outnumpy.min())
            outnumpy = np.clip(outnumpy,0,1.0)*255.0

    print(outnumpy.shape)
    
    outnumpy = outnumpy.astype(np.uint8)
    img = Image.fromarray(outnumpy,mode='L')
    img.save('./tmp.png')
    return

# def gene_img_test1(input_tensor):
#     os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#     # input_tensor = tf.placeholder(dtype=tf.float64,shape=[None,None,None,3])
#     inputs,outputs,var_list = build_model(input_tensor)
#     my_keras_model = tf.keras.Model(inputs,outputs)
#     gpu_options = tf.GPUOptions(allow_growth=True)
#     saver = tf.train.Saver(var_list)
#     with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#         sess.run(tf.global_variables_initializer())
#         saver.restore(sess,'./tf_weights',global_step=100)
#         print (my_keras_model.load_weights('./tf_k_block_rgb2gray_weights.h5'))
#         outnumpy = sess.run([outputs])
#         outnumpy = outnumpy[0]
#     print(outnumpy.shape)
#     outnumpy = np.clip(outnumpy,0,1.0)*255.0
#     outnumpy = outnumpy.astye(np.uint8)
#     img = Image.fromarray(outnumpy,mode='L')
#     img.save('./tmp.png')
#     return

def start_train():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    gray_datasets_path = '/home/room304/TB/DATASET/GF/GF2/png/gray_block'
    rgb_datasets_path = '/home/room304/TB/DATASET/GF/GF2/png/rgb_block'
    gray_filenames, rgb_filenames = get_gray_rgb_block_path_list(gray_datasets_path,rgb_datasets_path)

    my_datasets = get_datasets( gray_filenames, rgb_filenames, batch_size=20, num_epochs = 1000)
    iterator = my_datasets.make_initializable_iterator()
    next_example, netx_label = iterator.get_next()
    next_example.set_shape([None, None, None, 3])
    netx_label.set_shape([None, None, None, 1])
    
    inputs,outputs,var_list = build_model(next_example)
    my_keras_model = tf.keras.Model(inputs,outputs)
    mse_loss = tf.reduce_mean(tf.square(outputs - netx_label))
    opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mse_loss)
    saver = tf.train.Saver(var_list)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # print(my_keras_model.load_weights('./tf_k_block_rgb2gray_weights.h5'))
        sess.run(iterator.initializer)
        for counters in range(100000):
            losses,_,inputs_arr, label_arr, output_arr = sess.run([mse_loss,opt, next_example, netx_label, outputs])
            print('num : ', counters)
            print(losses)
            print('max of inputs :', inputs_arr.max())
            print('max of labels :', label_arr.max())
            print('max of outputs :', output_arr.max())
            print('min of outputs :', output_arr.min())
        # saver.save(sess,'./tf_weights',global_step=100)
    # tf.keras.models.save_model(my_keras_model,'./tmp_weights.h5py')
        my_keras_model.save_weights('./tf_k_block_rgb2gray_weights.h5')
        my_keras_model.summary()

    return

if __name__ == "__main__":
    start_train()
    # img_path = '/home/room304/storage/TB/DATASET/GF/GF2/png/rgb/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-MSS1.png'
    img_path = '/home/room304/storage/TB/DATASET/GF/GF2/png/rgb_block/GF2_PMS1_E112.9_N23.3_20170831_L1A0002574623-1.png'
    img_numpy = np.array(Image.open(img_path))
    print(img_numpy.shape)
    img_numpy = img_numpy[np.newaxis,:,:,:]
    # image_string = tf.read_file(img_path)
    # image_decoded = tf.image.decode_jpeg(image_string)
    gene_img(img_numpy)
    
   