import sys
sys.path.append('/media/room304/TB/python-tensorflow-tu')
import tensorflow as tf
import time
import numpy as np
from PIL import Image
from tensorflow.contrib import keras

class C3D_Anormaly_Gray_OpticalFlow:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 2
        self.depth_stride = 2
        self.channel_num = [256,128,64,128,256,input_channel]
        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                with tf.variable_scope('layer1'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[0], kernel_size=[4, 4, 4],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4,training=training), name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer2'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[1], kernel_size=[4, 4, 4],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer3'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[2], kernel_size=[4, 4, 4],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

            with tf.variable_scope('decoder'):
                with tf.variable_scope('layer4'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[3], kernel_size=[4, 4, 4],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer5'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[4], kernel_size=[4, 4, 4],
                                                            strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                            padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer6'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[5], kernel_size=[4, 4, 4],
                                                            strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                            padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.sigmoid(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='sigmoid')
                    inputs = x_internal

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.encoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'encoder')
        self.decoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'decoder')
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        outputs = inputs
        print (outputs.shape)
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class C3D_Anormaly_Stander_Gray_OpticalFlow:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 2
        self.depth_stride = 2
        self.channel_num = [256,128,64,128,256,input_channel]

        """
         encoder para
        """
        self.encoder_channel_num = [256, 128, 64]
        self.encoder_stride = [[2, 2, 2], [2, 2, 2], [1, 2, 2]]
        self.encoder_kernel_size = [[4, 4, 4], [4, 4, 4], [3, 4, 4]]

        """
                 decoder para
                """
        self.decoder_channel_num = [128, 256, input_channel]
        self.decoder_stride = [[1, 2, 2], [2, 2, 2], [2, 2, 2]]
        self.decoder_kernel_size = [[3, 4, 4], [4, 4, 4], [4, 4, 4]]

        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                with tf.variable_scope('layer1'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.encoder_channel_num[0],
                                                  kernel_size=self.encoder_kernel_size[0],
                                                  strides=self.encoder_stride[0],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer2'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.encoder_channel_num[1],
                                                  kernel_size=self.encoder_kernel_size[1],
                                                  strides=self.encoder_stride[1],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer3'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.encoder_channel_num[2],
                                                  kernel_size=self.encoder_kernel_size[2],
                                                  strides=self.encoder_stride[2],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

            with tf.variable_scope('decoder'):
                with tf.variable_scope('layer4'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.decoder_channel_num[0],
                                                            kernel_size=self.decoder_kernel_size[0],
                                                            strides=self.decoder_stride[0],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer5'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.decoder_channel_num[1],
                                                            kernel_size=self.decoder_kernel_size[1],
                                                            strides=self.decoder_stride[1],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer6'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.decoder_channel_num[2],
                                                            kernel_size=self.decoder_kernel_size[2],
                                                            strides=self.decoder_stride[2],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.sigmoid(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                               name='sigmoid')
                    inputs = x_internal

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.encoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'encoder')
        self.decoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'decoder')
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.total_variable = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES,scope=self.model_scope_name)
        outputs = inputs
        print (outputs.shape)
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return


class Feature_MODEL_Sequential:
    def __init__(self, input_channel = 1,output_channel = 4, gray_tags = False):

        self.reuse = False

        self.h_w_stride = 1
        self.recurrent_num = 0

        self.model_scope_name = 'net_feature'

        self.channel_num = [8,output_channel]
        self.channel_size = [5,5]



        assert len(self.channel_num) == len(self.channel_size)
        self.initial = keras.initializers.he_normal()
        # self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('layer_feature'):
                x_internal = inputs
                for feature_layer_idx in range(len(self.channel_num)):
                    x_internal = tf.nn.tanh(
                        tf.layers.conv2d(x_internal, self.channel_num[feature_layer_idx],
                                         kernel_size=[self.channel_size[feature_layer_idx],
                                                      self.channel_size[feature_layer_idx]],
                                         strides=[self.h_w_stride, self.h_w_stride],
                                         padding='SAME', use_bias=True,
                                         kernel_initializer=self.initial,
                                         name='conv%d' % feature_layer_idx),
                                            name='tanh%d' % feature_layer_idx)

                inputs = x_internal


        self.total_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES,scope=self.model_scope_name)
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)

        outputs = inputs
        self.reuse = True
        return outputs

class C3D_Anormaly_autoencoder_head_return2:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 1
        self.depth_stride = 2

        self.encoder_channel_num = [4,
                                    8,
                                    16]
        self.encoder_stride_num = [[2, 2, 2],
                                   [2, 2, 2],
                                   [2, 2, 2]]

        self.decoder_channel_num = [8,
                                    4,
                                    input_channel]
        self.decoder_stride_num = [[2, 2, 2],
                                   [2, 2, 2],
                                   [2, 2, 2]]

        self.channel_num = [2,
                            4,
                            8,
                            4,
                            2,
                            input_channel]

        self.stride_num = [[2, 1, 1],
                           [2, 1, 1],
                           [2, 1, 1],
                           [2, 1, 1],
                           [2, 1, 1],
                           [2, 1, 1]]

        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('encoder'):
                for encoder_layer_idx in range(len(self.encoder_channel_num)):
                    with tf.variable_scope('encoder_layer%d'%encoder_layer_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv3d(x_internal, self.encoder_channel_num[encoder_layer_idx], kernel_size=[4, 4, 4],
                                                      strides=self.encoder_stride_num[encoder_layer_idx],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                        x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                                name='tanh')
                        inputs = x_internal
                outputs_encoder = inputs


            with tf.variable_scope('decoder'):
                for decoder_layer_idx in range(len(self.decoder_channel_num)):
                    with tf.variable_scope('decoder_layer%d'%decoder_layer_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv3d_transpose(x_internal, self.decoder_channel_num[decoder_layer_idx], kernel_size=[4, 4, 4],
                                                                strides=self.decoder_stride_num[decoder_layer_idx],
                                                                padding='SAME', use_bias=False,
                                                                kernel_initializer=self.initial,
                                                                name='conv')
                        x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                                name='tanh')
                        inputs = x_internal
                outputs_decoder = inputs

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.encoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'encoder')
        self.decoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'decoder')
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        print (outputs.shape)
        self.reuse = True
        return outputs_encoder,outputs_decoder

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return



class C3D_Anormaly_autoencoder_foot_return1:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 2
        self.depth_stride = 2

        self.encoder_channel_num = [64,16,4]
        self.decoder_channel_num = [16,64, input_channel]

        self.channel_num = [128,64,32,16,8,64,128,input_channel]
        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('encoder'):
                encoder_idx = 0
                for channel_num in (self.encoder_channel_num):
                    encoder_idx+=1
                    with tf.variable_scope('layer%d'%encoder_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv2d(x_internal, channel_num, kernel_size=[4, 4],
                                                      strides=[self.h_w_stride, self.h_w_stride],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                        x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=3, training=training),
                                                name='tanh')
                        inputs = x_internal

            with tf.variable_scope('decoder'):
                decoder_idx = 0
                for channel_num in (self.decoder_channel_num):
                    decoder_idx+=1
                    with tf.variable_scope('layer%d'%decoder_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv2d_transpose(x_internal, channel_num, kernel_size=[4, 4],
                                                      strides=[self.h_w_stride, self.h_w_stride],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                        x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=3, training=training),
                                                name='tanh')
                        inputs = x_internal

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.encoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'encoder')
        self.decoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'decoder')
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        print (outputs.shape)
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class C3D_Anormaly_App:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 1
        self.depth_stride = 2
        self.channel_num = [128,64,32,64,128,input_channel]
        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                with tf.variable_scope('layer1'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[0], kernel_size=[1, 4, 4],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer2'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[1], kernel_size=[1, 4, 4],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer3'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[2], kernel_size=[1, 4, 4],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

            with tf.variable_scope('decoder'):
                with tf.variable_scope('layer4'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[3], kernel_size=[1, 4, 4],
                                                            strides=[self.depth_stride, self.h_w_stride,
                                                                     self.h_w_stride],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer5'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[4], kernel_size=[1, 4, 4],
                                                            strides=[self.depth_stride, self.h_w_stride,
                                                                     self.h_w_stride],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer6'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[5], kernel_size=[1, 4, 4],
                                                            strides=[self.depth_stride, self.h_w_stride,
                                                                     self.h_w_stride],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.sigmoid(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                               name='sigmoid')
                    inputs = x_internal



        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.encoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'encoder')
        self.decoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'decoder')
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        print (outputs.shape)
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class C3D_Anormaly_Motion:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 2
        self.depth_stride = 1
        self.channel_num = [256,128,64,128,256,input_channel]
        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                with tf.variable_scope('layer1'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[0], kernel_size=[4, 2, 2],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer2'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[1], kernel_size=[4, 2, 2],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer3'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d(x_internal, self.channel_num[2], kernel_size=[4, 2, 2],
                                                  strides=[self.depth_stride, self.h_w_stride, self.h_w_stride],
                                                  padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                  name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

            with tf.variable_scope('decoder'):
                with tf.variable_scope('layer4'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[3], kernel_size=[4, 2, 2],
                                                            strides=[self.depth_stride, self.h_w_stride,
                                                                     self.h_w_stride],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer5'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[4], kernel_size=[4, 2, 2],
                                                            strides=[self.depth_stride, self.h_w_stride,
                                                                     self.h_w_stride],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                            name='tanh')
                    inputs = x_internal

                with tf.variable_scope('layer6'):
                    x_internal = inputs
                    x_internal = tf.layers.conv3d_transpose(x_internal, self.channel_num[5], kernel_size=[4, 2, 2],
                                                            strides=[self.depth_stride, self.h_w_stride,
                                                                     self.h_w_stride],
                                                            padding='SAME', use_bias=False,
                                                            kernel_initializer=self.initial,
                                                            name='conv')
                    x_internal = tf.nn.sigmoid(tf.layers.batch_normalization(x_internal, axis=4, training=training),
                                               name='sigmoid')
                    inputs = x_internal

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.encoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'encoder')
        self.decoder_trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name+'/'+'decoder')
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        print (outputs.shape)
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return