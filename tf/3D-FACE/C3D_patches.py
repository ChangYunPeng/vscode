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

        

        self.both_c3d_model = c3d_model(input_channel=3, model_scope_name='gray_test')
        self.both_c3d_model.encoder_channel_num = [96, 24]
        self.both_c3d_model.encoder_stride_num = [[2, 2, 2], [2, 2, 2]]
        self.both_c3d_model.decoder_channel_num = [96, self.both_c3d_model.input_channel]
        self.both_c3d_model.decoder_stride_num = [[2, 2, 2], [2, 2, 2]]

        # self.mid_model = c2d_model(input_channel = 192,model_scope_name='concate_model')
        # self.mid_model.encoder_channel_num = [512,1024,2048]
        # self.mid_model.encoder_stride = [[2, 2], [2, 2], [2, 2]]
        # self.mid_model.decoder_channel_num = [1024,512, self.mid_model.input_channel]
        # self.mid_model.decoder_stride = [[2, 2], [2, 2], [2, 2]]

        self.batch_size = 8
        self.video_imgs_num = 4
        self.img_size_h = 256
        self.img_size_w = 256
        self.patch_size = 8
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-4
        self.autoencoder_lr = 1e-3

        self.selected_gpu_num = 1

        self.build_model_adn_loss_opt()
        self.Data_Sequence = dataset_sequence(frame_tags = True,opticalflow_tags=True,img_num=self.video_imgs_num)

        self.root_path = '/media/room304/TB/TensorFlow_Saver/ANORMLY/COMPARED_VN4_Pathches_VS/'

       
        self.mid_stage_save_path = self.root_path + 'MODEL_D/'
        self.both_save_path = self.root_path + 'MODEL_BOTH/'

        self.summaries_dir = self.root_path + 'SUMMARY/'

        
        self.mid_stage_img_save_path = self.root_path + 'RESULT/MODEL_MID_STAGE/'
        self.full_datat_set_path = self.root_path + 'RESULT/Data_Set_UCSD_Path/'
        self.full_ucsd_datatset_path = self.root_path + 'RESULT/FULL_Data_Set/UCSD_Path/'
        self.full_shanghai_datatset_path = self.root_path + 'RESULT/FULL_Data_Set/Shanghai/'

        tmp_dir_list = []
        # tmp_dir_list.append(self.optical_save_path)
        # tmp_dir_list.append(self.gray_save_path)
        tmp_dir_list.append(self.mid_stage_save_path)
        tmp_dir_list.append(self.summaries_dir)
        # tmp_dir_list.append(self.optical_img_save_path)
        # tmp_dir_list.append(self.gray_img_save_path)
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

                autoencoder_global_step = tf.Variable(0, trainable=False)
                autoencoder_lr_rate = tf.train.exponential_decay(self.autoencoder_lr, autoencoder_global_step, 10000, 0.99,staircase=True)
                autoencoder_opt = tf.train.AdamOptimizer(learning_rate=autoencoder_lr_rate)


                # self.optical_train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 2], name='optical_in')
                # self.gray_train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 1], name='gray_in')
                self.both_train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 3], name='both_in')

                self.phase = tf.placeholder(tf.bool,name='is_training')
                                            

            with tf.device('/gpu:%d' % self.selected_gpu_num):
                with tf.name_scope('GPU%d' % self.selected_gpu_num) as scope:

                    # optical_encoder,self.optical_train_out_ph = self.optical_c3d_model(self.optical_train_in_ph,self.phase)
                    # gray_encoder, self.gray_train_out_ph = self.gray_c3d_model(self.gray_train_in_ph, self.phase)
                    h_bands = self.img_size_h/self.patch_size
                    w_bands = self.img_size_w/self.patch_size

                    

                    split_axis_h = tf.split(self.both_train_in_ph, num_or_size_splits=h_bands, axis=2)
                    # print split_axis_2
                    split_axis_hw_list = []
                    for split_axis_h_iter in split_axis_h:
                        split_axis_hw_list.append(tf.split(split_axis_h_iter, num_or_size_splits=w_bands, axis=3))
                    
                    merge_axis_hw_list = []
                    for split_axis_h_iter in split_axis_hw_list:
                        tmp_hw_list = []
                        for split_axis_hw_iter in split_axis_h_iter:
                            _,tmp_out = self.both_c3d_model(split_axis_hw_iter, self.phase)
                            # tmp_out = split_axis_hw_iter                            
                            tmp_hw_list.append(tmp_out)
                        merge_axis_hw_list.append(tmp_hw_list)

                    merge_axis_h_list = []
                    for split_axis_hw_iter in merge_axis_hw_list:
                        merge_axis_h_list.append(tf.concat(split_axis_hw_iter,axis=3))
                    
                    self.both_train_out_ph = tf.concat(merge_axis_h_list,axis=2)
                    
                    self.both_stage_loss = tf.reduce_mean(tf.square(self.both_train_in_ph - self.both_train_out_ph))

                    
                    
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        self.c3d_apply = autoencoder_opt.minimize(self.both_stage_loss , var_list=self.both_c3d_model.trainable_variable )
                       
        self.both_c3d_model.summary()
        tf.summary.scalar('optical_loss', self.both_stage_loss)
        return

   

    def train_both_stage_c3d(self,max_iteration = 20000):
        # gray_saver = tf.train.Saver(var_list=self.gray_c3d_model.all_variables)
        # optical_saver = tf.train.Saver(var_list=self.optical_c3d_model.all_variables)
        both_stage_saver = tf.train.Saver(var_list=self.both_c3d_model.all_variables)

        # gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        # optical_save_path = self.optical_save_path + 'trainable_weights.cptk'
        both_save_path = self.both_save_path + 'trainable_weights.cptk'

        summaries_dir = self.summaries_dir + 'SINGLE_GPU%d.CPTK' % time.time()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir)

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            # gray_saver.restore(sess,gray_save_path)
            # optical_saver.restore(sess,optical_save_path)

            for idx in range(max_iteration):  
                batch_data = []
                while(batch_data == []):
                    batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                # print batch_data.shape
                batch_data_gray = batch_data[:,:,:,:,0:1]
                batch_data_optical = batch_data[:,:,:,:,1:3]

                sess.run([ self.c3d_apply ], feed_dict={self.both_train_in_ph:batch_data, self.phase : True})

                if (idx+1)%20 == 0:
                    batch_data = []
                    while(batch_data == []):
                        batch_data = self.Data_Sequence.get_train_random_batches(batch_size=self.batch_size)
                    # print batch_data.shape
                    batch_data_gray = batch_data[:,:,:,:,0:1]
                    batch_data_optical = batch_data[:,:,:,:,1:3]

                    both_stage_loss,sum_tmp = sess.run([self.both_stage_loss,merged], feed_dict={self.both_train_in_ph:batch_data,self.phase : True})
                    print 'Epoches Idx%d' %(idx + 1)
                    print 'both_stage_loss %.8f'%both_stage_loss



                if (idx+1)%2000 == 0 or (idx+1) == max_iteration:
                    # self.test_model(sess)
                    # train_writer.add_summary(sum_tmp, (idx+1))
                    # train_writer.flush()
                    print both_stage_saver.save(sess, both_save_path)
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

        both_stage_saver = tf.train.Saver(var_list=self.both_c3d_model.all_variables)
        both_save_path = self.both_save_path + 'trainable_weights.cptk'

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            both_stage_saver.restore(sess, both_save_path)


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
        total_losses = []
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
            # batch_data_gray = batch_data[:, :, :, :, 0:1]
            # batch_data_optical = batch_data[:, :, :, :, 1:3]

            _, both_stage_loss = sess.run([self.both_train_out_ph,self.both_stage_loss],
                                                feed_dict={self.both_train_in_ph: batch_data, self.phase: True})
            # gray_output_np, optical_output_np, cur_total_loss, cur_gray_loss, cur_optical_loss = sess.run(
            #     [self.gray_train_out_ph, self.optical_train_out_ph, self.total_loss, self.gray_loss, self.optical_loss ],
            #     feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
            #                self.phase: False})
            total_losses.append(both_stage_loss)
            # gray_losses.append(cur_gray_loss)
            # optical_losses.append(cur_optical_loss)
            # mid_stage_losses.append(cur_mid_stage_loss)

            # save_c3d_text_opticalflow_result(batch_data_optical, np.array(optical_output_np, dtype=float),
            #                                  cur_total_loss, 'global_%d' % (test_video_idx + 1),
            #                                  save_path=save_path)
            # save_c3d_text_frame_result(batch_data_gray, np.array(gray_output_np, dtype=float), cur_total_loss,
            #                            'global_%d' % (test_video_idx + 1), save_path=save_path)

            test_video_idx += 1

        save_plot_img(save_path + 'both_loss.jpg', total_losses)
        # save_plot_img(save_path + 'gray_loss.jpg', gray_losses)
        # save_plot_img(save_path + 'optical_loss.jpg', optical_losses)
        # save_plot_img(save_path + 'mid_stage_loss.jpg', mid_stage_losses)
        # save_double_plot_img(save_path + 'total_loss.jpg', total_losses,label_loss)
        np.save(save_path + 'both_loss.npy', total_losses)
        # np.save(save_path + 'label_loss.npy', label_loss)

        file = open(save_path + 'video_path.txt', 'w')
        file.write(test_video_dataset.video_path)

        return

    def test_shanghai_dataset_model(self):
        both_stage_saver = tf.train.Saver(var_list=self.both_c3d_model.all_variables)
        both_save_path = self.both_save_path + 'trainable_weights.cptk'

        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            both_stage_saver.restore(sess, both_save_path)

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
        total_losses = []
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

            _, both_stage_loss = sess.run([self.both_train_out_ph, self.both_stage_loss],
                                          feed_dict={self.both_train_in_ph: batch_data, self.phase: True})

            # gray_output_np, optical_output_np, cur_total_loss, cur_gray_loss, cur_optical_loss = sess.run(
            #     [self.gray_train_out_ph, self.optical_train_out_ph, self.total_loss, self.gray_loss, self.optical_loss],
            #     feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
            #                self.phase: False})
            total_losses.append(both_stage_loss)
            # gray_losses.append(cur_gray_loss)
            # optical_losses.append(cur_optical_loss)

            # save_c3d_text_opticalflow_result(batch_data_optical, np.array(optical_output_np, dtype=float),
            #                                  cur_total_loss, 'global_%d' % (test_video_idx + 1),
            #                                  save_path=save_path)
            # save_c3d_text_frame_result(batch_data_gray, np.array(gray_output_np, dtype=float), cur_total_loss,
            #                            'global_%d' % (test_video_idx + 1), save_path=save_path)

            test_video_idx += 1

        save_plot_img(save_path + 'both_loss.jpg', total_losses)
        # save_plot_img(save_path + 'gray_loss.jpg', gray_losses)
        # save_plot_img(save_path + 'optical_loss.jpg', optical_losses)
        # save_plot_img(save_path + 'mid_stage_loss.jpg', mid_stage_losses)
        save_double_plot_img(save_path + 'both_loss.jpg', total_losses,label_loss)
        np.save(save_path + 'both_loss.npy', total_losses)
        np.save(save_path + 'label_loss.npy', label_loss)

        file = open(save_path + 'video_path.txt', 'w')
        file.write(test_video_dataset.video_path)

        return
        


run_model = C3D_Running()
run_model.train_both_stage_c3d(20000)
# run_model.train_mid_stage_c3d(1000)
# run_model.train_c3d(20000)
run_model.test_ucsd_dataset_model()
# run_model.test_shanghai_dataset_model()