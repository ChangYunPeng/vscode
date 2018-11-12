import sys
sys.path.append('/media/room304/TB/python-tensorflow-tu')
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
from PIL import Image
from tensorflow.contrib import keras

class C3D_Test_ENCODER:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, bn_tag = True):
        self.s_size = s_size
        self.reuse = False
        self.input_channel = input_channel
        self.bn_tag = bn_tag

        self.h_w_stride = 1
        self.depth_stride = 2

        self.encoder_channel_num = [4,
                                    8,
                                    16]
        self.encoder_stride_num = [[2, 1, 1],
                                   [2, 1, 1],
                                   [2, 1, 1]]
        self.encoder_kernel_size= [[2, 3, 3],
                                   [2, 3, 3],
                                   [2, 3, 3]]

        self.model_scope_name = model_scope_name
        self.initial = keras.initializers.he_normal()

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('encoder'):

                for encoder_layer_idx,(channel_num, stride_num, kernel_size) in enumerate(zip(self.encoder_channel_num,self.encoder_stride_num,self.encoder_kernel_size)):

                    with tf.variable_scope('encoder_layer%d'%encoder_layer_idx):
                        x_internal = inputs
                        # if encoder_layer_idx == len(self.encoder_channel_num)-1:
                        #     padding = 'valid'
                        # else:
                        #     padding = 'same'
                        padding = 'same'
                        
                        x_internal = tf.layers.conv3d(x_internal, channel_num, kernel_size=kernel_size,
                                                    strides=stride_num,
                                                    padding=padding, use_bias=False, kernel_initializer=self.initial,
                                                    name='conv')
                        if self.bn_tag:
                            x_internal = tf.layers.batch_normalization(x_internal, axis=4, training=training)
                        
                        x_internal = tf.nn.tanh(x_internal, name='tanh') 
                        # if encoder_layer_idx == len(self.encoder_channel_num)-1 or encoder_layer_idx == 0:
                        #     x_internal = tf.nn.tanh(x_internal, name='tanh') 
                        # else:
                        #     x_internal = tf.nn.tanh(x_internal, name='tanh') + x_internal

                        inputs = x_internal
                outputs_encoder = inputs

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.model_scope_name)
        self.reuse = True
        return outputs_encoder

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)

class C2D_Test_ENCODER:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, bn_tag = True):
        self.s_size = s_size
        self.reuse = False
        self.input_channel = input_channel
        self.bn_tag = bn_tag

        self.h_w_stride = 1
        self.depth_stride = 2

        self.encoder_channel_num = [4,
                                    8,
                                    16]
        self.encoder_stride_num = [[1, 1],
                                   [1, 1],
                                   [1, 1]]
        self.encoder_kernel_size= [[3, 3],
                                   [3, 3],
                                   [3, 3]]

        self.model_scope_name = model_scope_name
        self.initial = keras.initializers.he_normal()

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('encoder'):

                for encoder_layer_idx,(channel_num, stride_num, kernel_size) in enumerate(zip(self.encoder_channel_num,self.encoder_stride_num,self.encoder_kernel_size)):

                    with tf.variable_scope('encoder_layer%d'%encoder_layer_idx):
                        x_internal = inputs
                        # if encoder_layer_idx == len(self.encoder_channel_num)-1:
                        #     padding = 'valid'
                        # else:
                        #     padding = 'same'
                        padding = 'same'
                        x_internal = tf.layers.conv2d(x_internal, channel_num, kernel_size=kernel_size,
                                                      strides=stride_num,
                                                      padding=padding, use_bias=False, name='conv')
                        if self.bn_tag:
                            x_internal = tf.layers.batch_normalization(x_internal, axis=4, training=training)
                        if encoder_layer_idx == len(self.encoder_channel_num)-1:
                            x_internal = tf.nn.sigmoid(x_internal, name='sigmoid')
                        else:
                            x_internal = tf.nn.relu(x_internal, name='tanh')

                        inputs = x_internal
                outputs_encoder = inputs

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.model_scope_name)
        self.reuse = True
        return outputs_encoder

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
    
class C2D_Test_DECODER:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, bn_tag = True):
        self.s_size = s_size
        self.reuse = False
        self.input_channel = input_channel
        self.bn_tag = bn_tag

        self.h_w_stride = 1
        self.depth_stride = 2

        self.decoder_channel_num = [4,
                                    8,
                                    16]
        self.decoder_stride_num = [[1, 1],
                                   [1, 1],
                                   [1, 1]]
        self.decoder_kernel_size= [[3, 3],
                                   [3, 3],
                                   [3, 3]]

        self.model_scope_name = model_scope_name
        self.initial = keras.initializers.he_normal()

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('decoder'):

                for decoder_layer_idx,(channel_num, stride_num, kernel_size) in enumerate(zip(self.decoder_channel_num,self.decoder_stride_num,self.decoder_kernel_size)):

                    with tf.variable_scope('decoder_layer%d'%decoder_layer_idx):
                        x_internal = inputs
                        padding = 'same'
                        x_internal = tf.layers.conv2d_transpose(x_internal, channel_num, kernel_size=kernel_size,
                                                      strides=stride_num,
                                                      padding=padding, use_bias=False, name='conv')
                        if self.bn_tag:
                            x_internal = tf.layers.batch_normalization(x_internal, axis=4, training=training)
                        if decoder_layer_idx == len(self.decoder_channel_num)-1:
                            x_internal = tf.nn.sigmoid(x_internal, name='sigmoid')
                        else:
                            x_internal = tf.nn.relu(x_internal, name='tanh')

                        inputs = x_internal
                outputs_decoder = inputs

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.model_scope_name)
        self.reuse = True
        return outputs_decoder

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)

class Attention_Test_Model:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, attention_uints = 256, attention_hops = 128):
        self.s_size = s_size
        self.reuse = False

        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.attention_hops))), trainable=False)
        # self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(batchsize,self.attention_hops))) , trainable=False)

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                batchsize, height, width, in_channels = inputs.get_shape().as_list()
                batchsize = tf.shape(inputs)[0]
                height = tf.shape(inputs)[1]
                width = tf.shape(inputs)[2]

                x_internal_h = tf.reshape(inputs,[-1,in_channels]) #bs*h*w,c
                x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1')) #bs*h*w,uints


                x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')/np.sqrt(self.attention_uints) #bs*h*w, hops

                x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, height*width, self.attention_hops]), perm=[0,2,1]) #bs, hops, h*w
                x_internal_a = tf.nn.softmax(tf.reshape(x_internal_a,[-1,height*width])) # bs*hops, h*w
                x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, height*width]) # bs, hops, h*w

                x_internal_h = tf.reshape(x_internal_h,[-1,height*width,in_channels]) # bs, h*w, c
                x_internal = tf.matmul( x_internal_a, x_internal_h) #bs, hops, c
                heat_map = tf.reshape(tf.reduce_sum(x_internal_a, axis=1),[-1,height,width,1])
                inputs = x_internal
                
                with tf.variable_scope('extra_loss'):
                    extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                    print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs, extra_loss, heat_map

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class C3D_ENCODER:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, bn_tag = True, relu_tag = False):
        self.s_size = s_size
        self.reuse = False
        self.input_channel = input_channel
        self.bn_tag = bn_tag
        self.relu_tag = relu_tag

        self.h_w_stride = 1
        self.depth_stride = 2

        self.encoder_channel_num = [4,
                                    8,
                                    16]
        self.encoder_stride_num = [[2, 1, 1],
                                   [2, 1, 1],
                                   [2, 1, 1]]

        self.model_scope_name = model_scope_name
        self.initial = keras.initializers.he_normal()

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

            with tf.variable_scope('encoder'):
                for encoder_layer_idx in range(len(self.encoder_channel_num)):
                    with tf.variable_scope('encoder_layer%d'%encoder_layer_idx):
                        x_internal = inputs

                        if self.bn_tag:
                            x_internal = tf.layers.batch_normalization(x_internal, axis=4, training=training)

                        x_internal = tf.layers.conv3d(x_internal, self.encoder_channel_num[encoder_layer_idx], kernel_size=[4, 4, 4],
                                                      strides=self.encoder_stride_num[encoder_layer_idx],
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv')
                       
                        
                        if self.relu_tag:                            
                            x_internal = tf.nn.relu(x_internal,name='relu')
                        else:
                            x_internal = tf.nn.tanh(x_internal, name='tanh')

                        inputs = x_internal
                outputs_encoder = inputs

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.model_scope_name)
        self.reuse = True
        return outputs_encoder

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)

class C3D_DECODER:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, bn_tag = True, relu_tag = False):
        self.s_size = s_size
        self.reuse = False
        self.relu_tag = relu_tag
        self.bn_tag = bn_tag
        self.input_channel = input_channel
        self.not_last_activation = False
        self.h_w_stride = 1
        self.depth_stride = 2
        self.decoder_channel_num = [8,
                                    4,
                                    input_channel]
        self.decoder_stride_num = [[2, 1, 1],
                                   [2, 1, 1],
                                   [2, 1, 1]]
        self.model_scope_name = model_scope_name
        self.initial = keras.initializers.he_normal()

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        activation = True
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('decoder'):
                for decoder_layer_idx in range(len(self.decoder_channel_num)):
                    if decoder_layer_idx+1 == len(self.decoder_channel_num) and self.not_last_activation:
                        activation = False
                    with tf.variable_scope('decoder_layer%d'%decoder_layer_idx):
                        x_internal = inputs
                        if self.bn_tag:
                            x_internal = tf.layers.batch_normalization(x_internal, axis=4, training=training)
                        x_internal = tf.layers.conv3d_transpose(x_internal, self.decoder_channel_num[decoder_layer_idx], kernel_size=[4, 4, 4],
                                                                strides=self.decoder_stride_num[decoder_layer_idx],
                                                                padding='SAME', use_bias=False,
                                                                kernel_initializer=self.initial,
                                                                name='conv')
                        x_internal = tf.layers.conv3d(x_internal, self.decoder_channel_num[decoder_layer_idx], kernel_size=[4, 4, 4],
                                                      strides=1,
                                                      padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                      name='conv_ap')
                        if activation == True:
                            if self.relu_tag:
                                x_internal = tf.nn.relu(x_internal,name='relu')
                            else:
                                x_internal = tf.nn.tanh(x_internal,name='tanh')
                        inputs = x_internal
                outputs_decoder = inputs
            

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.model_scope_name)
        self.reuse = True
        return outputs_decoder

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)


class C2D_ENCODER:
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

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class C2D_DECODER:
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

        self.model_scope_name = model_scope_name

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        c,_,_,_ = inputs.shape

        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):

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
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

def NonLocalBlock(input_x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock'):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g') as scope:
            g = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='g')
            if sub_sample:
                g = slim.max_pool2d(g, [2,2], stride=2, scope='g_max_pool')

        with tf.variable_scope('phi') as scope:
            phi = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='phi')
            if sub_sample:
                phi = slim.max_pool2d(phi, [2,2], stride=2, scope='phi_max_pool')

        with tf.variable_scope('theta') as scope:
            theta = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='theta')

        print(g.get_shape().as_list())
        # g = tf.transpose(g, [0,3,2,1])
        g_x = tf.reshape(g, [-1,height*width,out_channels])
        # g_x = tf.reshape(g, [batchsize,-1,out_channels])
        
        # g_x = tf.transpose(g_x, [0,2,1]) # after reshap, the axis h*w is fixed

        # theta = tf.transpose(theta, [0,3,2,1])
        theta_x = tf.reshape(theta, [-1,height*width,out_channels])
        # theta_x = tf.transpose(theta_x, [0,2,1])

        # phi = tf.transpose(phi, [0,3,2,1])
        phi_x = tf.reshape(phi, [-1,height*width,out_channels])
        phi_x = tf.transpose(phi_x, [0,2,1])

        f = tf.matmul(theta_x, phi_x)
        # ???
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        y = tf.reshape(y, [-1, height, width, out_channels])
        with tf.variable_scope('w') as scope:
            w_y = slim.conv2d(y, in_channels, [1,1], stride=1, scope='w')
            if is_bn:
                w_y = slim.batch_norm(w_y)
        z = w_y
        return z


class Attention_Model:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, attention_uints = 256, attention_hops = 128):
        self.s_size = s_size
        self.reuse = False

        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.attention_hops))), trainable=False)
        # self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(batchsize,self.attention_hops))) , trainable=False)

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                batchsize, height, width, in_channels = inputs.get_shape().as_list()
                batchsize = tf.shape(inputs)[0]

                x_internal_h = tf.reshape(inputs,[-1,in_channels]) #bs*h*w,c
                x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1')) #bs*h*w,uints


                x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')/np.sqrt(self.attention_uints) #bs*h*w, hops

                x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, height*width, self.attention_hops]), perm=[0,2,1]) #bs, hops, h*w
                x_internal_a = tf.nn.softmax(tf.reshape(x_internal_a,[-1,height*width])) # bs*hops, h*w
                x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, height*width]) # bs, hops, h*w

                heat_map = tf.reshape(tf.reduce_sum(x_internal_a, axis=1),[-1,height,width,1])

                x_internal_h = tf.reshape(x_internal_h,[-1,height*width,in_channels]) # bs, h*w, c
                x_internal = tf.matmul( x_internal_a, x_internal_h) #bs, hops, c
                x_out = tf.matmul( tf.transpose(x_internal_a,perm=[0,2,1]), x_internal) # bs, h*w, c
                print('x_out', x_out.get_shape().as_list())
                inputs = tf.reshape(x_out,[-1,height, width,in_channels])
                
                with tf.variable_scope('extra_loss'):
                    extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                    print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs, extra_loss, heat_map

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class Attention_sigmoid_Model:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, attention_uints = 256, attention_hops = 128):
        self.s_size = s_size
        self.reuse = False

        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.attention_hops))), trainable=False)
        # self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(batchsize,self.attention_hops))) , trainable=False)

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                batchsize, height, width, in_channels = inputs.get_shape().as_list()
                batchsize = tf.shape(inputs)[0]

                x_internal_h = tf.reshape(inputs,[-1,in_channels]) #bs*h*w,c
                x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1')) #bs*h*w,uints


                x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')/np.sqrt(self.attention_uints) #bs*h*w, hops

                x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, height*width, self.attention_hops]), perm=[0,2,1]) #bs, hops, h*w
                x_internal_a = tf.nn.sigmoid(tf.reshape(x_internal_a,[-1,height*width])) # bs*hops, h*w
                x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, height*width]) # bs, hops, h*w

                x_internal_h = tf.reshape(x_internal_h,[-1,height*width,in_channels]) # bs, h*w, c
                x_internal = tf.matmul( x_internal_a, x_internal_h) #bs, hops, c
                x_out = tf.matmul( tf.transpose(x_internal_a,perm=[0,2,1]), x_internal) # bs, h*w, c
                print('x_out', x_out.get_shape().as_list())
                inputs = tf.reshape(x_out,[-1,height, width,in_channels])
                
                with tf.variable_scope('extra_loss'):
                    extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                    print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs, extra_loss

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class Attention_Convert_Model:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, attention_uints = 256, attention_hops = 128):
        self.s_size = s_size
        self.reuse = False

        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.attention_hops))), trainable=False)

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                batchsize, height, width, in_channels = inputs.get_shape().as_list()
                batchsize = tf.shape(inputs)[0]
                x_internal = tf.reshape(inputs,[-1, height*width, in_channels]) 
                x_internal = tf.transpose(x_internal,perm=[0,2,1])
                x_internal_h = tf.reshape(x_internal, [-1,height*width]) #bs*c,h*w

                # x_internal_h = tf.reshape(inputs,[-1,in_channels]) #bs*h*w,c
                x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1')) #bs*c,uints

                x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')/np.sqrt(self.attention_uints) #bs*c, hops

                x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, in_channels, self.attention_hops]), perm=[0,2,1]) #bs, hops, c
                x_internal_a = tf.nn.softmax(tf.reshape(x_internal_a,[-1,in_channels])) # bs*hops, c
                x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, in_channels]) # bs, hops, c

                x_internal_h = tf.reshape(x_internal_h,[-1,in_channels,height*width]) # bs,c, h*w
                x_internal = tf.matmul( x_internal_a, x_internal_h) #bs, hops, h*w
                x_out = tf.transpose(x_internal,perm=[0,2,1]) #bs, h*w, hops
                # x_out = tf.matmul( tf.transpose(x_internal_a,perm=[0,2,1]), x_internal) # bs, h*w, c
                # print('x_out', x_out.get_shape().as_list())
                inputs = tf.reshape(x_out,[-1,height, width, self.attention_hops])
                
                with tf.variable_scope('extra_loss'):
                    extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                    print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs, extra_loss

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class Attention_Multiply_Model:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4, attention_uints = 256, attention_hops = 128):
        self.s_size = s_size
        self.reuse = False

        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.attention_hops))), trainable=False)
        # self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(batchsize,self.attention_hops))) , trainable=False)

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                batchsize, height, width, in_channels = inputs.get_shape().as_list()
                batchsize = tf.shape(inputs)[0]

                x_internal_h = tf.reshape(inputs,[-1,in_channels]) #bs*h*w,c
                x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1')) #bs*h*w,uints


                x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')/np.sqrt(self.attention_uints) #bs*h*w, hops

                x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, height*width, self.attention_hops]), perm=[0,2,1]) #bs, hops, h*w
                x_internal_a = tf.nn.softmax(tf.reshape(x_internal_a,[-1,height*width])) # bs*hops, h*w
                x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, height*width]) # bs, hops, h*w
                x_internal_a = tf.transpose(x_internal_a,perm=[0,2,1])# bs, h*w, hops
                x_internal_a = tf.reduce_sum(x_internal_a,axis=[2]) # bs, h*w, 1
                x_internal_a = tf.reshape(x_internal_a,[-1,1])
                x_internal_attention = tf.matmul( x_internal_a, x_internal_h)

                # x_internal_h = tf.reshape(x_internal_h,[-1,height*width,in_channels]) # bs, h*w, c
                # x_internal = tf.matmul( x_internal_a, x_internal_h) #bs, hops, c
                # x_out = tf.matmul( tf.transpose(x_internal_a,perm=[0,2,1]), x_internal) # bs, h*w, c
                # print('x_out', x_out.get_shape().as_list())
                inputs = tf.reshape(x_internal_attention,[-1,height, width,in_channels])
                
                with tf.variable_scope('extra_loss'):
                    extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                    print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs, extra_loss

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class Attention_Model_Xt:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4,xt_num = 64, attention_uints = 32, attention_hops = 8):
        self.s_size = s_size
        self.reuse = False

        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        self.xt_num = xt_num
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.xt_num * self.attention_hops))), trainable=False)

        self.initial = keras.initializers.he_normal()
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='c3d_anormaly')

    def __call__(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                batchsize, height, width, in_channels = inputs.get_shape().as_list()

                x_internal_h = tf.reshape(inputs,[-1,in_channels])
                x_internal_a_list = []

                for idx in range(self.xt_num):
                    with tf.variable_scope('attention_xt%d'%idx):
                        x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1'))
                        x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')
                        x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, height*width, self.attention_hops]), perm=[0,2,1])
                        x_internal_a = tf.nn.softmax(tf.reshape(x_internal_a,[-1,height*width]))
                        x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, height*width])
                        x_internal_a_list.append(x_internal_a)
                x_internal_a = tf.concat(x_internal_a_list, axis=1)
                x_internal_h = tf.reshape(x_internal_h,[-1,height*width,in_channels]) 
                x_internal = tf.matmul( x_internal_a, x_internal_h)
                x_out = tf.matmul( tf.transpose(x_internal_a,perm=[0,2,1]), x_internal)
                print('x_out', x_out.get_shape().as_list())
                inputs = tf.reshape(x_out,[-1,height, width,in_channels])

                with tf.variable_scope('extra_loss'):
                    extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                    print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        self.reuse = True
        return outputs, extra_loss

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class gan_model:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4,xt_num = 64, attention_uints = 512, attention_hops = 256, bn_tag = True, attention_tag = True, dense_tag = True, not_last_activation = True):
        self.s_size = s_size
        self.reuse = False
        self.not_last_activation = not_last_activation
        self.attention_tag = attention_tag
        self.dense_tag = dense_tag
        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        
        self.encoder_channel_num = [64,
                                    64,
                                    64]
        self.encoder_stride_num = [[2, 2],
                                   [2, 2],
                                   [2, 2]]
        
        self.dense_channel_num = [1024,
                                    256,
                                    1]

        self.xt_num = xt_num
        self.bn_tag = bn_tag
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.attention_hops))), trainable=False)
        self.initial = keras.initializers.he_normal()

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                
                # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                # batchsize, height, width, in_channels = tf.shape(inputs)
                batchsize, temporal, height, width, in_channels = inputs.get_shape().as_list()

                # batchsize, temporal, height, width, in_channels = tf.shape(inputs)
                print(' gray input tensor shape :' , batchsize, temporal, height, width, in_channels)
                inputs = tf.reshape(inputs, shape =[-1, height, width, in_channels] )
                print('reshaped',inputs.get_shape())

                with tf.variable_scope('encoder'):
                    encoder_idx = 0
                    for channel_num in self.encoder_channel_num:
                        with tf.variable_scope('layer%d'%encoder_idx):
                            print('layer%d'%encoder_idx, channel_num)
                            x_internal = inputs
                            x_internal = tf.layers.conv2d(x_internal, channel_num, kernel_size=[3, 3],
                                                        strides = self.encoder_stride_num[encoder_idx],
                                                        padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                        name='conv',data_format='channels_last')
                            if self.bn_tag:
                                x_internal = tf.layers.batch_normalization(x_internal, axis=4, training=training)
                            
                            inputs = tf.nn.tanh(x_internal,name='tanh')
          
                        encoder_idx+=1
                    
                if self.attention_tag:
                    batchsize, height, width, in_channels = inputs.get_shape().as_list()
                    with tf.variable_scope('attention'):
                        # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                        x_internal_h = tf.reshape(inputs,[-1,in_channels]) #bs*h*w,c
                        x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1')) #bs*h*w,uints
                        x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')/np.sqrt(self.attention_uints) #bs*h*w, hops
                        x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, height*width, self.attention_hops]), perm=[0,2,1]) #bs, hops, h*w
                        x_internal_a = tf.nn.softmax(tf.reshape(x_internal_a,[-1,height*width])) # bs*hops, h*w
                        x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, height*width]) # bs, hops, h*w
                        x_internal_h = tf.reshape(x_internal_h,[-1,height*width,in_channels]) # bs, h*w, c
                        inputs = tf.matmul( x_internal_a, x_internal_h) #bs, hops, c
                
                if self.dense_tag:
                    activation = True
                    inputs = tf.layers.flatten(inputs)
                    with tf.variable_scope('dense'):
                        for idx,(channel_num) in enumerate(self.dense_channel_num):
                            inputs = tf.layers.dense(inputs, channel_num, use_bias=True, kernel_initializer=self.initial, name='layer%d'%idx)
                            if idx+1 == len(self.dense_channel_num) and self.not_last_activation:
                                activation = False
                            if activation:
                                inputs = tf.nn.tanh(inputs,name='tanh')
                
                if self.attention_tag:
                    with tf.variable_scope('extra_loss'):
                        extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                        print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        print( outputs.get_shape().as_list())
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return

class m_gan_model:
    def __init__(self, input_channel = 3,model_scope_name = 'net',s_size=4,xt_num = 64, attention_uints = 512, attention_hops = 256, bn_tag = True, attention_tag = True, dense_tag = True, not_last_activation = True):
        self.s_size = s_size
        self.reuse = False
        self.not_last_activation = not_last_activation
        self.attention_tag = attention_tag
        self.dense_tag = dense_tag
        self.input_channel = input_channel
        self.model_scope_name = model_scope_name
        
        self.encoder_channel_num = [64,
                                    64,
                                    64]
        self.encoder_stride_num = [[2, 2],
                                   [2, 2],
                                   [2, 2]]
        
        self.dense_channel_num = [1024,
                                    256,
                                    1]

        self.xt_num = xt_num
        self.bn_tag = bn_tag
        self.attention_uints = attention_uints
        self.attention_hops = attention_hops
        self.I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(self.attention_hops))), trainable=False)
        self.initial = keras.initializers.he_normal()

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope(self.model_scope_name, reuse=self.reuse):
            with tf.variable_scope('encoder'):
                
                # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                # batchsize, height, width, in_channels = tf.shape(inputs)
                batchsize, temporal, height, width, in_channels = inputs.get_shape().as_list()

                # inputs = tf.transpose(inputs, perm=[0,2,3,4,1])
                # inputs = tf.reshape(inputs, shape =[-1, height, width, in_channels * temporal ] )

                # batchsize, temporal, height, width, in_channels = tf.shape(inputs)
                print(' gray input tensor shape :' , batchsize, temporal, height, width, in_channels)
                inputs = tf.reshape(inputs, shape =[-1, height, width, in_channels] )
                print('reshaped',inputs.get_shape())

                with tf.variable_scope('encoder'):
                    encoder_idx = 0
                    for channel_num in self.encoder_channel_num:
                        with tf.variable_scope('layer%d'%encoder_idx):
                            print('layer%d'%encoder_idx, channel_num)
                            x_internal = inputs
                            x_internal = tf.layers.conv2d(x_internal, channel_num, kernel_size=[3, 3],
                                                        strides = self.encoder_stride_num[encoder_idx],
                                                        padding='SAME', use_bias=False, kernel_initializer=self.initial,
                                                        name='conv',data_format='channels_last')
                            if self.bn_tag:
                                x_internal = tf.layers.batch_normalization(x_internal, axis=4, training=training)
                            
                            inputs = tf.nn.tanh(x_internal,name='tanh')
          
                        encoder_idx+=1
                    
                if self.attention_tag:
                    batchsize, height, width, in_channels = inputs.get_shape().as_list()
                    with tf.variable_scope('attention'):
                        # inputs = NonLocalBlock(input_x=inputs, out_channels=512, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                        x_internal_h = tf.reshape(inputs,[-1,in_channels]) #bs*h*w,c
                        x_internal = tf.nn.tanh(tf.layers.dense(x_internal_h, self.attention_uints, use_bias=False, kernel_initializer=self.initial, name='ws1')) #bs*h*w,uints
                        x_internal_a = tf.layers.dense(x_internal, self.attention_hops, use_bias=False,kernel_initializer=self.initial, name='ws2')/np.sqrt(self.attention_uints) #bs*h*w, hops
                        x_internal_a = tf.transpose(tf.reshape(x_internal_a,[-1, height*width, self.attention_hops]), perm=[0,2,1]) #bs, hops, h*w
                        x_internal_a = tf.nn.softmax(tf.reshape(x_internal_a,[-1,height*width])) # bs*hops, h*w
                        x_internal_a = tf.reshape(x_internal_a,[-1, self.attention_hops, height*width]) # bs, hops, h*w
                        x_internal_h = tf.reshape(x_internal_h,[-1,height*width,in_channels]) # bs, h*w, c
                        inputs = tf.matmul( x_internal_a, x_internal_h) #bs, hops, c
                
                if self.dense_tag:
                    activation = True
                    inputs = tf.layers.flatten(inputs)
                    with tf.variable_scope('dense'):
                        for idx,(channel_num) in enumerate(self.dense_channel_num):
                            inputs = tf.layers.dense(inputs, channel_num, use_bias=True, kernel_initializer=self.initial, name='layer%d'%idx)
                            if idx+1 == len(self.dense_channel_num) and self.not_last_activation:
                                activation = False
                            if activation:
                                inputs = tf.nn.tanh(inputs,name='tanh')
                
                if self.attention_tag:
                    with tf.variable_scope('extra_loss'):
                        extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), self.I_matrix)))
                        print('extra loss',tf.shape(extra_loss))

        self.trainable_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_name)
        self.update_variable = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_name)
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_name)
        outputs = inputs
        print( outputs.get_shape().as_list())
        self.reuse = True
        return outputs

    def summary(self):
        for var in self.update_variable:
            tf.summary.tensor_summary(var.name,var)
        for var in self.trainable_variable:
            tf.summary.tensor_summary(var.name,var)
        return