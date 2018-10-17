import sys
sys.path.append('/media/room304/TB/python-tensorflow-tu')
import tensorflow as tf
import time
import numpy as np
from PIL import Image
from tensorflow.contrib import keras

class C3D_WithoutBN:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False
        self.input_channel = input_channel

        self.h_w_stride = 1
        self.depth_stride = 2

        self.encoder_channel_num = [4,
                                    8,
                                    16]
        self.encoder_stride_num = [[2, 1, 1],
                                   [2, 1, 1],
                                   [2, 1, 1]]

        self.decoder_channel_num = [8,
                                    4,
                                    input_channel]
        self.decoder_stride_num = [[2, 1, 1],
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
                        x_internal = tf.nn.tanh(x_internal, name='tanh')
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
                        x_internal = tf.nn.tanh(x_internal,name='tanh')
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

class C3D:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False
        self.input_channel = input_channel

        self.h_w_stride = 1
        self.depth_stride = 2

        self.encoder_channel_num = [4,
                                    8,
                                    16]
        self.encoder_stride_num = [[2, 1, 1],
                                   [2, 1, 1],
                                   [2, 1, 1]]

        self.decoder_channel_num = [8,
                                    4,
                                    input_channel]
        self.decoder_stride_num = [[2, 1, 1],
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

class C2D_WithoutBN:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 2
        self.depth_stride = 2

        self.input_channel = input_channel

        self.encoder_channel_num = [64,16,4]
        self.encoder_stride = [[2, 2], [2, 2], [2, 2]]

        self.decoder_channel_num = [16,64, self.input_channel]
        self.decoder_stride = [[2, 2], [2, 2], [2, 2]]

        # self.channel_num = [128,64,32,16,8,64,128,input_channel]
        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        c,_,_,_ = inputs.shape

        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('encoder'):
                encoder_idx = 0
                for channel_num in (self.encoder_channel_num):
                    with tf.variable_scope('layer%d'%encoder_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv2d(x_internal, channel_num, kernel_size=[4, 4],
                                                      strides = self.encoder_stride[encoder_idx],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                        x_internal = tf.nn.tanh(x_internal, name='tanh')
                        inputs = x_internal
                    encoder_idx+=1

            with tf.variable_scope('decoder'):
                decoder_idx = 0
                for channel_num in (self.decoder_channel_num):
                    with tf.variable_scope('layer%d'%decoder_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv2d_transpose(x_internal, channel_num, kernel_size=[4, 4],
                                                      strides = self.decoder_stride[decoder_idx],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                        x_internal = tf.nn.tanh(x_internal, name='tanh')
                        inputs = x_internal
                    decoder_idx+=1

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

class C2D:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4):
        self.s_size = s_size
        self.reuse = False

        self.h_w_stride = 2
        self.depth_stride = 2

        self.input_channel = input_channel

        self.encoder_channel_num = [64,16,4]
        self.encoder_stride = [[2, 2], [2, 2], [2, 2]]

        self.decoder_channel_num = [16,64, self.input_channel]
        self.decoder_stride = [[2, 2], [2, 2], [2, 2]]

        # self.channel_num = [128,64,32,16,8,64,128,input_channel]
        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        c,_,_,_ = inputs.shape

        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('encoder'):
                encoder_idx = 0
                for channel_num in (self.encoder_channel_num):
                    with tf.variable_scope('layer%d'%encoder_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv2d(x_internal, channel_num, kernel_size=[4, 4],
                                                      strides = self.encoder_stride[encoder_idx],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                        x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=3, training=training),
                                                name='tanh')
                        inputs = x_internal                    
                    encoder_idx+=1

            with tf.variable_scope('decoder'):
                decoder_idx = 0
                for channel_num in (self.decoder_channel_num):
                    with tf.variable_scope('layer%d'%decoder_idx):
                        x_internal = inputs
                        x_internal = tf.layers.conv2d_transpose(x_internal, channel_num, kernel_size=[4, 4],
                                                      strides = self.decoder_stride[decoder_idx],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                        x_internal = tf.nn.tanh(tf.layers.batch_normalization(x_internal, axis=3, training=training),
                                                name='tanh')
                        inputs = x_internal
                    decoder_idx+=1

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