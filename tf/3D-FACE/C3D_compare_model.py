import sys
sys.path.append('/media/room304/TB/vscode/abnormly-code/anormly_utilize_code')

import tensorflow as tf
import time
import numpy as np
# import cv2
from PIL import Image
import os
from matplotlib import pyplot as plt
from tensorflow.contrib import keras
from TF_MODEL_Utils.C3D_MODEL import C3D_Anormaly_Stander_Gray_OpticalFlow as c3d_model
from TF_MODEL_Utils.C3D_Depart import C3D_WithoutBN as c3d_model
from TF_MODEL_Utils.C3D_Depart import C2D as c2d_model
from VideoSequenceUtils.DataSetImgSequence import Dataset_Frame_and_OPTICALFLOW_Batches as dataset_sequence
from ImageProcessUtils.save_img_util import save_c3d_text_opticalflow_result, save_c3d_text_frame_opticalflow_result, save_c3d_text_frame_result
from ImageProcessUtils.model_uilit import save_plot_img,mk_dirs, save_double_plot_img
from VideoSequenceUtils.DataSetImgSequence import Sequence_UCSD_Dataset_Frame_and_OPTICALFLOW_Batches as sequence_dataset_sequence
from VideoSequenceUtils.DataSetImgSequence import Sequence_Shanghai_Dataset_Frame_and_OPTICALFLOW_Batches as sequence_shanghai_dataset_sequence


class C3D_Running:
    def __init__(self):
        self.img_channel = 2

        self.optical_c3d_model = c3d_model(input_channel=2, model_scope_name='optical_flow_test')
        self.optical_c3d_model.encoder_channel_num = [64, 16]
        self.optical_c3d_model.encoder_stride_num = [[2, 2, 2], [2, 2, 2]]
        self.optical_c3d_model.decoder_channel_num = [32, self.optical_c3d_model.input_channel]
        self.optical_c3d_model.decoder_stride_num = [[2, 2, 2], [2, 2, 2]]

        self.gray_c3d_model = c3d_model(input_channel=1, model_scope_name='gray_test')
        self.gray_c3d_model.encoder_channel_num = [32, 8]
        self.gray_c3d_model.encoder_stride_num = [[2, 2, 2], [2, 2, 2]]
        self.gray_c3d_model.decoder_channel_num = [32, self.gray_c3d_model.input_channel]
        self.gray_c3d_model.decoder_stride_num = [[2, 2, 2], [2, 2, 2]]

        # self.mid_model = c2d_model(input_channel = 192,model_scope_name='concate_model')
        # self.mid_model.encoder_channel_num = [512,1024,2048]
        # self.mid_model.encoder_stride = [[2, 2], [2, 2], [2, 2]]
        # self.mid_model.decoder_channel_num = [1024,512, self.mid_model.input_channel]
        # self.mid_model.decoder_stride = [[2, 2], [2, 2], [2, 2]]

        self.batch_size = 8
        self.video_imgs_num = 4
        self.img_size_h = None
        self.img_size_w = None
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-4
        self.autoencoder_lr = 1e-3

        self.selected_gpu_num = 0

        self.build_model_adn_loss_opt()
        self.Data_Sequence = dataset_sequence(frame_tags = True,opticalflow_tags=True,img_num=self.video_imgs_num)

        self.root_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/COMPARED_VN4_L3_Separated/'

        self.optical_save_path = self.root_path + 'MODEL_OPTICAL/'
        self.gray_save_path = self.root_path + 'MODEL_GRAY/'
        self.mid_stage_save_path = self.root_path + 'MODEL_D/'

        self.summaries_dir = self.root_path + 'SUMMARY/'

        self.optical_img_save_path = self.root_path + 'RESULT/MODEL_OPTICAL/'
        self.gray_img_save_path = self.root_path + 'RESULT/MODEL_GRAY/'
        self.mid_stage_img_save_path = self.root_path + 'RESULT/MODEL_MID_STAGE/'
        self.full_datat_set_path = self.root_path + 'RESULT/Data_Set_UCSD_Path/'
        self.full_ucsd_datatset_path = self.root_path + 'RESULT/FULL_Data_Set/UCSD_Path/'
        self.full_shanghai_datatset_path = self.root_path + 'RESULT/FULL_Data_Set/Shanghai/'

        tmp_dir_list = []
        tmp_dir_list.append(self.optical_save_path)
        tmp_dir_list.append(self.gray_save_path)
        tmp_dir_list.append(self.mid_stage_save_path)
        tmp_dir_list.append(self.summaries_dir)
        tmp_dir_list.append(self.optical_img_save_path)
        tmp_dir_list.append(self.gray_img_save_path)
        tmp_dir_list.append(self.mid_stage_img_save_path)
        tmp_dir_list.append(self.full_datat_set_path)
        tmp_dir_list.append(self.full_ucsd_datatset_path)
        tmp_dir_list.append(self.full_shanghai_datatset_path)
        mk_dirs(tmp_dir_list)

        return

    def build_model_adn_loss_opt(self):

        self.mid_stage_loss_ratio = 10.0
        self.optical_flow_loss_ratio = 0.10
        self.gray_loss_ratio = 1.0
        with tf.device('/cpu:0'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optical_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                gray_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                mid_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)

                autoencoder_global_step = tf.Variable(0, trainable=False)
                autoencoder_lr_rate = tf.train.exponential_decay(self.autoencoder_lr, autoencoder_global_step, 10000, 0.99,staircase=True)
                autoencoder_opt = tf.train.AdamOptimizer(learning_rate=autoencoder_lr_rate)


                self.optical_train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 2], name='optical_in')
                self.gray_train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 1], name='gray_in')

                self.phase = tf.placeholder(tf.bool,name='is_training')
                                            

            with tf.device('/gpu:%d' % self.selected_gpu_num):
                with tf.name_scope('GPU%d' % self.selected_gpu_num) as scope:

                    optical_encoder,self.optical_train_out_ph = self.optical_c3d_model(self.optical_train_in_ph,self.phase)
                    gray_encoder,self.gray_train_out_ph = self.gray_c3d_model(self.gray_train_in_ph,self.phase)

                    # print 'mid_stage_squeeze'
                    # optical_encoder = tf.squeeze(optical_encoder,axis=1)
                    # gray_encoder = tf.squeeze(gray_encoder,axis=1)
                    #
                    # mid_stage_in = tf.concat([optical_encoder,gray_encoder],axis=3,name = 'mid_stage_in')
                    # mid_stage_out = self.mid_model(mid_stage_in,self.phase)

                    # print mid_stage_in
                    # print mid_stage_out

                    self.optical_loss = tf.reduce_mean(tf.square(self.optical_train_in_ph - self.optical_train_out_ph)) * self.optical_flow_loss_ratio
                    self.gray_loss = tf.reduce_mean(tf.square(self.gray_train_in_ph - self.gray_train_out_ph)) * self.gray_loss_ratio
                    # self.mid_stage_loss = tf.reduce_mean(tf.square(mid_stage_in - mid_stage_out))* self.mid_stage_loss_ratio

                    self.total_loss = self.optical_loss + self.gray_loss
                    
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        self.optical_apply = optical_ae_opt.minimize(self.optical_loss,var_list=self.optical_c3d_model.trainable_variable)
                        self.gray_apply = gray_ae_opt.minimize(self.gray_loss,var_list=self.gray_c3d_model.trainable_variable)
                        # self.mid_stage_apply = mid_ae_opt.minimize(self.mid_stage_loss,var_list=self.mid_model.trainable_variable)
                        self.c3d_apply = autoencoder_opt.minimize(self.optical_loss + self.gray_loss , var_list=self.optical_c3d_model.trainable_variable + self.gray_c3d_model.trainable_variable )
                       
        self.gray_c3d_model.summary()
        self.optical_c3d_model.summary()
        # self.mid_model.summary()
        tf.summary.scalar('optical_loss', self.optical_loss)
        tf.summary.scalar('gray_loss', self.gray_loss)
        # tf.summary.scalar('mid_stage_loss', self.mid_stage_loss)
        return

    def train_gray_optical_c3d(self,max_iteration = 20000):

        gray_saver = tf.train.Saver(var_list=self.gray_c3d_model.all_variables)
        optical_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)

        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        optical_save_path = self.optical_save_path + 'trainable_weights.cptk'

        summaries_dir = self.summaries_dir + 'SINGLE_GPU%d.CPTK' % time.time()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir)

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())

            for idx in range(max_iteration):  
                batch_data = []
                while(batch_data == []):
                    batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                
                batch_data_gray = batch_data[:,:,:,:,0:1]
                batch_data_optical = batch_data[:,:,:,:,1:3]

                sess.run([ self.optical_apply,self.gray_apply], feed_dict={self.optical_train_in_ph:batch_data_optical, self.gray_train_in_ph:batch_data_gray,self.phase : True})

                if (idx+1)%20 == 0:
                    batch_data = []
                    while(batch_data == []):
                        batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                    # print batch_data.shape
                    batch_data_gray = batch_data[:,:,:,:,0:1]
                    batch_data_optical = batch_data[:,:,:,:,1:3]
                    print batch_data.shape
                    print batch_data_gray.shape
                    print batch_data_optical.shape

                    optical_loss,gray_loss,sum_tmp = sess.run([self.optical_loss,self.gray_loss, merged], feed_dict={self.optical_train_in_ph:batch_data_optical, self.gray_train_in_ph:batch_data_gray,self.phase : True})
                    print 'Epoches Idx%d' %(idx + 1)
                    print 'gray_loss %.8f'%gray_loss
                    print 'optical_loss %.8f'%optical_loss

                    # train_writer.add_summary(sum_tmp, (idx+1))
                    # train_writer.flush()

                if (idx+1)%2000 == 0 or (idx+1) == max_iteration:
                    # self.test_model(sess)
                    print gray_saver.save(sess, gray_save_path)
                    print optical_saver.save(sess, optical_save_path)

        return

    def train_mid_stage_c3d(self,max_iteration = 10000):
        gray_saver = tf.train.Saver(var_list=self.gray_c3d_model.all_variables)
        optical_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)
        mid_stage_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)

        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        optical_save_path = self.optical_save_path + 'trainable_weights.cptk'
        mid_stage_save_path = self.mid_stage_save_path + 'trainable_weights.cptk'

        summaries_dir = self.summaries_dir + 'SINGLE_GPU%d.CPTK' % time.time()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir)

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            gray_saver.restore(sess,gray_save_path)
            optical_saver.restore(sess,optical_save_path)

            for idx in range(max_iteration):  
                batch_data = []
                while(batch_data == []):
                    batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                # print batch_data.shape
                batch_data_gray = batch_data[:,:,:,:,0:1]
                batch_data_optical = batch_data[:,:,:,:,1:3]

                sess.run([ self.mid_stage_apply ], feed_dict={self.optical_train_in_ph:batch_data_optical, self.gray_train_in_ph:batch_data_gray,self.phase : True})

                if (idx+1)%20 == 0:
                    batch_data = []
                    while(batch_data == []):
                        batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                    # print batch_data.shape
                    batch_data_gray = batch_data[:,:,:,:,0:1]
                    batch_data_optical = batch_data[:,:,:,:,1:3]

                    total_loss, optical_loss,gray_loss,mid_stage_loss,sum_tmp = sess.run([self.total_loss, self.optical_loss,self.gray_loss, self.mid_stage_loss,merged], feed_dict={self.optical_train_in_ph:batch_data_optical, self.gray_train_in_ph:batch_data_gray,self.phase : True})
                    print 'Epoches Idx%d' %(idx + 1)
                    print 'loss %.8f'%total_loss
                    print 'gray_loss %.8f'%gray_loss
                    print 'optical_loss %.8f'%optical_loss
                    print 'mid_stage_loss %.8f'%mid_stage_loss



                if (idx+1)%2000 == 0 or (idx+1) == max_iteration:
                    # self.test_model(sess)
                    # train_writer.add_summary(sum_tmp, (idx+1))
                    # train_writer.flush()
                    print mid_stage_saver.save(sess, mid_stage_save_path)
        return

    def train_c3d(self,max_iteration = 100000):
        gray_saver = tf.train.Saver(var_list=self.gray_c3d_model.all_variables)
        optical_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)
        mid_stage_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)

        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        optical_save_path = self.optical_save_path + 'trainable_weights.cptk'
        mid_stage_save_path = self.mid_stage_save_path + 'trainable_weights.cptk'

        summaries_dir = self.summaries_dir + 'SINGLE_GPU%d.CPTK' % time.time()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir)

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            # gray_saver.restore(sess,gray_save_path)
            # optical_saver.restore(sess,optical_save_path)
            # mid_stage_saver.restore(sess, mid_stage_save_path)

            for idx in range(max_iteration):  
                batch_data = []
                while(batch_data == []):
                    batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                # print batch_data.shape
                batch_data_gray = batch_data[:,:,:,:,0:1]
                batch_data_optical = batch_data[:,:,:,:,1:3]

                sess.run([ self.c3d_apply ], feed_dict={self.optical_train_in_ph:batch_data_optical, self.gray_train_in_ph:batch_data_gray,self.phase : True})

                if (idx+1)%20 == 0:
                    batch_data = []
                    while(batch_data == []):
                        batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                    # print batch_data.shape
                    batch_data_gray = batch_data[:,:,:,:,0:1]
                    batch_data_optical = batch_data[:,:,:,:,1:3]

                    total_loss, optical_loss,gray_loss,sum_tmp = sess.run([self.total_loss, self.optical_loss,self.gray_loss, merged], feed_dict={self.optical_train_in_ph:batch_data_optical, self.gray_train_in_ph:batch_data_gray,self.phase : True})
                    print 'Epoches Idx%d' %(idx + 1)
                    print 'loss %.8f'%total_loss
                    print 'gray_loss %.8f'%gray_loss
                    print 'optical_loss %.8f'%optical_loss



                if (idx+1)%100 == 0 and (idx+1) >=100:
                    # train_writer.add_summary(sum_tmp, (idx + 1))
                    # train_writer.flush()
                    self.test_model(sess)
                    print gray_saver.save(sess, gray_save_path)
                    print optical_saver.save(sess, optical_save_path)
                    print mid_stage_saver.save(sess, mid_stage_save_path)
        return

    def test_model(self, sess):
        test_video_dataset = self.Data_Sequence.get_test_frames_objects()
        test_video_dataset.batch_size = 1
        test_video_idx = 0
        total_losses = []
        gray_losses = []
        optical_losses = []
        # mid_stage_losses = []

        save_path = self.mid_stage_img_save_path + 'Time_%d/' % time.time()
        save_gray_path = save_path + 'GRAY'
        save_optical_path = save_path + 'OPTICAL_FLOW'
        if not os.path.exists(save_gray_path):
            os.makedirs(save_gray_path)
            os.makedirs(save_optical_path)

        while (test_video_dataset.continued_tags):
            batch_data = test_video_dataset.get_video_frame_batches()
            batch_data_gray = batch_data[:, :, :, :, 0:1]
            batch_data_optical = batch_data[:, :, :, :, 1:3]

            gray_output_np, optical_output_np, cur_total_loss, cur_gray_loss, cur_optical_loss = sess.run(
                [self.gray_train_out_ph, self.optical_train_out_ph, self.total_loss, self.gray_loss, self.optical_loss],
                feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
                           self.phase: False})
            total_losses.append(cur_total_loss)
            gray_losses.append(cur_gray_loss)
            optical_losses.append(cur_optical_loss)
            # mid_stage_losses.append(cur_mid_stage_loss)

            save_c3d_text_opticalflow_result(batch_data_optical, np.array(optical_output_np, dtype=float),
                                             cur_total_loss, 'global_%d' % (test_video_idx + 1),
                                             save_path=save_path)
            save_c3d_text_frame_result(batch_data_gray, np.array(gray_output_np, dtype=float), cur_total_loss,
                                       'global_%d' % (test_video_idx + 1), save_path=save_path)

            test_video_idx += 1

        save_plot_img(save_path + 'total_loss.jpg', total_losses)
        save_plot_img(save_path + 'gray_loss.jpg', gray_losses)
        save_plot_img(save_path + 'optical_loss.jpg', optical_losses)
        # save_plot_img(save_path + 'mid_stage_loss.jpg', mid_stage_losses)

        file = open(save_path + 'video_path.txt', 'w')
        file.write(test_video_dataset.video_path)

        return

    def test_ucsd_dataset_model(self):
        gray_saver = tf.train.Saver(var_list=self.gray_c3d_model.all_variables)
        optical_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)
        mid_stage_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)

        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        optical_save_path = self.optical_save_path + 'trainable_weights.cptk'
        mid_stage_save_path = self.mid_stage_save_path + 'trainable_weights.cptk'

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            gray_saver.restore(sess, gray_save_path)
            optical_saver.restore(sess, optical_save_path)
            # mid_stage_saver.restore(sess, mid_stage_save_path)


            full_test_dataset = sequence_dataset_sequence(img_num=4)
            while (full_test_dataset.continue_tags) :
            # for idx in range(1):
                save_path = self.full_ucsd_datatset_path + 'UCSD_Dataset_%d_Videoth_%d/' % ( full_test_dataset.selected_list_num, full_test_dataset.selected_video_num)
                test_video_dataset = full_test_dataset.get_test_frames_objects()
                self.test_cur_video_model(sess,test_video_dataset,save_path)
        return

    def test_cur_video_model(self, sess,test_video_dataset,save_path):
        # test_video_dataset = self.Data_Sequence.get_test_frames_objects()
        test_video_dataset.batch_size = 1
        test_video_idx = 0
        # total_losses = []
        gray_losses = []
        optical_losses = []
        # mid_stage_losses = []

        # save_path = self.mid_stage_img_save_path + 'Time_%d/' % time.time()
        save_gray_path = save_path + 'GRAY'
        save_optical_path = save_path + 'OPTICAL_FLOW'
        if not os.path.exists(save_gray_path):
            os.makedirs(save_gray_path)
            os.makedirs(save_optical_path)

        while (test_video_dataset.continued_tags):
            batch_data = test_video_dataset.get_video_frame_batches()
            batch_data_gray = batch_data[:, :, :, :, 0:1]
            batch_data_optical = batch_data[:, :, :, :, 1:3]

            gray_output_np, optical_output_np, cur_gray_loss, cur_optical_loss = sess.run(
                [self.gray_train_out_ph, self.optical_train_out_ph,  self.gray_loss, self.optical_loss ],
                feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
                           self.phase: False})
            # total_losses.append(cur_total_loss)
            gray_losses.append(cur_gray_loss)
            optical_losses.append(cur_optical_loss)
            # mid_stage_losses.append(cur_mid_stage_loss)

            # save_c3d_text_opticalflow_result(batch_data_optical, np.array(optical_output_np, dtype=float),
            #                                  cur_total_loss, 'global_%d' % (test_video_idx + 1),
            #                                  save_path=save_path)
            # save_c3d_text_frame_result(batch_data_gray, np.array(gray_output_np, dtype=float), cur_total_loss,
            #                            'global_%d' % (test_video_idx + 1), save_path=save_path)

            test_video_idx += 1

        # save_plot_img(save_path + 'total_loss.jpg', total_losses)
        save_plot_img(save_path + 'gray_loss.jpg', gray_losses)
        save_plot_img(save_path + 'optical_loss.jpg', optical_losses)
        # save_plot_img(save_path + 'mid_stage_loss.jpg', mid_stage_losses)
        # save_double_plot_img(save_path + 'total_loss.jpg', total_losses,label_loss)
        # np.save(save_path + 'total_loss.npy', total_losses)
        np.save(save_path + 'gray_loss.npy', gray_losses)
        np.save(save_path + 'optical_loss.npy', optical_losses)
        # np.save(save_path + 'label_loss.npy', label_loss)

        file = open(save_path + 'video_path.txt', 'w')
        file.write(test_video_dataset.video_path)

        return

    def test_shanghai_dataset_model(self):
        gray_saver = tf.train.Saver(var_list=self.gray_c3d_model.all_variables)
        optical_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)
        mid_stage_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)

        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        optical_save_path = self.optical_save_path + 'trainable_weights.cptk'
        mid_stage_save_path = self.mid_stage_save_path + 'trainable_weights.cptk'

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            gray_saver.restore(sess, gray_save_path)
            optical_saver.restore(sess, optical_save_path)
            # mid_stage_saver.restore(sess, mid_stage_save_path)


            full_test_dataset = sequence_shanghai_dataset_sequence(img_num = 4)
            while (full_test_dataset.continue_tags) :
            # for idx in range(1):
                save_path = self.full_shanghai_datatset_path + 'Shanghai_Dataset_%d_Videoth_%d/' % ( full_test_dataset.selected_list_num, full_test_dataset.selected_video_num)
                test_video_dataset,label_loss = full_test_dataset.get_test_frames_objects()
                self.test_cur_shanghai_video_model(sess,test_video_dataset,save_path,label_loss)
        return

    def test_cur_shanghai_video_model(self, sess,test_video_dataset,save_path,label_loss):
        # test_video_dataset = self.Data_Sequence.get_test_frames_objects()
        test_video_dataset.batch_size = 1
        test_video_idx = 0
        # total_losses = []
        gray_losses = []
        optical_losses = []

        # save_path = self.mid_stage_img_save_path + 'Time_%d/' % time.time()
        save_gray_path = save_path + 'GRAY'
        save_optical_path = save_path + 'OPTICAL_FLOW'
        if not os.path.exists(save_gray_path):
            os.makedirs(save_gray_path)
            os.makedirs(save_optical_path)

        while (test_video_dataset.continued_tags):
            batch_data = test_video_dataset.get_video_frame_batches()
            batch_data_gray = batch_data[:, :, :, :, 0:1]
            batch_data_optical = batch_data[:, :, :, :, 1:3]

            gray_output_np, optical_output_np,  cur_gray_loss, cur_optical_loss = sess.run(
                [self.gray_train_out_ph, self.optical_train_out_ph,  self.gray_loss, self.optical_loss],
                feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
                           self.phase: False})
            # total_losses.append(cur_total_loss)
            gray_losses.append(cur_gray_loss)
            optical_losses.append(cur_optical_loss)

            # save_c3d_text_opticalflow_result(batch_data_optical, np.array(optical_output_np, dtype=float),
            #                                  cur_total_loss, 'global_%d' % (test_video_idx + 1),
            #                                  save_path=save_path)
            # save_c3d_text_frame_result(batch_data_gray, np.array(gray_output_np, dtype=float), cur_total_loss,
            #                            'global_%d' % (test_video_idx + 1), save_path=save_path)

            test_video_idx += 1

        # save_plot_img(save_path + 'total_loss.jpg', total_losses)
        save_plot_img(save_path + 'gray_loss.jpg', gray_losses)
        save_plot_img(save_path + 'optical_loss.jpg', optical_losses)
        # save_plot_img(save_path + 'mid_stage_loss.jpg', mid_stage_losses)
        # save_double_plot_img(save_path + 'total_loss.jpg', total_losses,label_loss)
        # np.save(save_path + 'total_loss.npy', total_losses)
        np.save(save_path + 'gray_loss.npy', gray_losses)
        np.save(save_path + 'optical_loss.npy', optical_losses)
        np.save(save_path + 'label_loss.npy', label_loss)

        file = open(save_path + 'video_path.txt', 'w')
        file.write(test_video_dataset.video_path)

        return
        


run_model = C3D_Running()
# run_model.train_gray_optical_c3d(20000)
# run_model.train_mid_stage_c3d(1000)
# run_model.train_c3d(20000)
run_model.test_ucsd_dataset_model()
run_model.test_shanghai_dataset_model()