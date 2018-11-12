import sys
sys.path.append('../anormly_utilize_code')
sys.path.append('./datasets')
sys.path.append('./model')

import tensorflow as tf
import time
import numpy as np
import cv2
from PIL import Image
import os
from matplotlib import pyplot as plt
from tensorflow.contrib import keras
# from TF_MODEL_Utils.C3D_MODEL import C3D_Anormaly_Stander_Gray_OpticalFlow as c3d_model
from TF_MODEL_Utils.C3D_Depart import C3D_WithoutBN as c3d_model
from TF_MODEL_Utils.C3D_Depart import C2D_WithoutBN as c2d_model
from VideoSequenceUtils.DataSetImgSequence import Dataset_Frame_and_OPTICALFLOW_Batches as dataset_sequence
from ImageProcessUtils.save_img_util import save_c3d_text_opticalflow_result, save_c3d_text_frame_opticalflow_result, save_c3d_text_frame_result
from ImageProcessUtils.model_uilit import save_plot_img,mk_dirs, save_double_plot_img, max_min_np, save_roc_auc_plot_img
from VideoSequenceUtils.DataSetImgSequence import Sequence_UCSD_Dataset_Frame_and_OPTICALFLOW_Batches as sequence_dataset_sequence
from VideoSequenceUtils.DataSetImgSequence import Sequence_Shanghai_Dataset_Frame_and_OPTICALFLOW_Batches as sequence_shanghai_dataset_sequence

from datasets_sequence import multi_train_datasets, multi_test_datasets
from C3D_MODEL import C3D_ENCODER, C3D_DECODER, C2D_ENCODER, C2D_DECODER, Attention_Model, Attention_Model_Xt
from model_utils import save_batch_images

def time_hms(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return h, m ,s

class C3D_Running:
    def __init__(self):
        self.img_channel = 2

        self.optical_encoder = C3D_ENCODER(input_channel=2,model_scope_name='optical_flow_encoder', bn_tag = False)
        self.optical_encoder.encoder_channel_num = [64,        16]
        self.optical_encoder.encoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.optical_decoder = C3D_DECODER(input_channel=2,model_scope_name='optical_flow_decoder', bn_tag = False)
        self.optical_decoder.not_last_activation = True
        self.optical_decoder.decoder_channel_num = [64,      self.optical_decoder.input_channel]
        self.optical_decoder.decoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.gray_encoder = C3D_ENCODER(input_channel=1,model_scope_name='gray_encoder', bn_tag = False)
        self.gray_encoder.encoder_channel_num = [32,          64]
        self.gray_encoder.encoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.gray_decoder = C3D_DECODER(input_channel=2,model_scope_name='gray_decoder', bn_tag = False)
        self.gray_decoder.decoder_channel_num = [32,          self.gray_decoder.input_channel]
        self.gray_decoder.decoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        # self.mid_model = Attention_Model_Xt(input_channel = 512,model_scope_name='vn4_concate_model',xt_num=64, attention_uints = 32, attention_hops = 16)
        self.mid_model = Attention_Model(input_channel = 512,model_scope_name='vn4_concate_model', attention_uints = 1024, attention_hops = 1024)
        # self.mid_model.xt_num=128

        self.batch_size = 8
        self.video_imgs_num = 4
        self.img_size_h = 112
        self.img_size_w = 112
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-4
        self.autoencoder_lr = 1e-3

        self.selected_gpu_num = 0

        self.build_model_adn_loss_opt()
        self.root_path = '/home/room304/TB/TB/TensorFlow_Saver/ANORMLY/Attention_xt_test2/'

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
    def restore_model_weghts(self, sess ):

        gray_cptk = tf.train.get_checkpoint_state(self.gray_save_path)
        optical_cptk = tf.train.get_checkpoint_state(self.optical_save_path)
        mid_stage_cptk = tf.train.get_checkpoint_state(self.mid_stage_save_path)
        print(gray_cptk.model_checkpoint_path)
        print(optical_cptk.model_checkpoint_path)
        print(mid_stage_cptk.model_checkpoint_path)
        self.gray_saver.restore(sess, gray_cptk.model_checkpoint_path)
        self.optical_saver.restore(sess, optical_cptk.model_checkpoint_path)
        self.mid_stage_saver.restore(sess, mid_stage_cptk.model_checkpoint_path)
        return
    
    def save_model_weghts(self, sess ):
        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        optical_save_path = self.optical_save_path + 'trainable_weights.cptk'
        mid_stage_save_path = self.mid_stage_save_path + 'trainable_weights.cptk'
        self.gray_saver.save(sess, gray_save_path)
        self.optical_saver.save(sess, optical_save_path)
        self.mid_stage_saver.save(sess, mid_stage_save_path)
        return

    def build_model_adn_loss_opt(self):

        self.mid_stage_loss_ratio = 1.0
        self.optical_flow_loss_ratio = 1.0
        self.gray_loss_ratio = 1.0
        self.extra_loss_ratio = 1.0
        with tf.device('/cpu:0'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optical_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                gray_ae_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
                mid_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)

                autoencoder_global_step = tf.Variable(0, trainable=False)
                autoencoder_lr_rate = tf.train.exponential_decay(self.autoencoder_lr, autoencoder_global_step, 10000, 0.99,staircase=True)
                autoencoder_opt = tf.train.RMSPropOptimizer(learning_rate=autoencoder_lr_rate)

                self.train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 2], name='batches_in')

                self.gray_train_in_ph  = self.train_in_ph
                self.optical_train_in_ph = self.train_in_ph
                self.phase = tf.placeholder(tf.bool,name='is_training')
                                            

            with tf.device('/gpu:%d' % self.selected_gpu_num):
                with tf.name_scope('GPU%d' % self.selected_gpu_num) as scope:

                    with tf.name_scope('encoder'):
                        optical_encoder = self.optical_encoder(self.optical_train_in_ph,self.phase)
                        gray_encoder = self.gray_encoder(self.gray_train_in_ph,self.phase)
                        print('optical_encoder',optical_encoder.shape)
                        print('gray_encoder',gray_encoder.shape)
                    
                    with tf.name_scope('mid_stage_squeeze'):
                        optical_encoder = tf.squeeze(optical_encoder,axis=1)
                        gray_encoder = tf.squeeze(gray_encoder,axis=1)
                        # mid_stage_in = tf.concat([optical_encoder,gray_encoder],axis=3,name = 'mid_stage_in')
                        mid_stage_in = gray_encoder
                        mid_stage_out, extraloss= self.mid_model(mid_stage_in)
                        mid_stage_out_p = tf.expand_dims(mid_stage_out,axis=1)
                        print('test')

                        # mid_stage_out_p = tf.expand_dims(mid_stage_in,axis=1)

                        mid_stage_out_opticalflow = tf.expand_dims(optical_encoder,axis=1)
                        mid_stage_out_gray =tf.expand_dims(gray_encoder,axis=1) #+  mid_stage_out_p

                        # mid_stage_out_opticalflow = mid_stage_out_p[:,:,:,:,0:optical_encoder.shape[3]]
                        # mid_stage_out_gray = mid_stage_out_p[:,:,:,:,optical_encoder.shape[3]:]

                        print('mid_stage_out_opticalflow',mid_stage_out_opticalflow.shape)
                        print('mid_stage_out_gray',mid_stage_out_gray.shape)
                    
                    with tf.name_scope('decoder'):
                        optical_decoder = self.optical_decoder(mid_stage_out_opticalflow,self.phase)
                        gray_decoder = self.gray_decoder(mid_stage_out_gray,self.phase)
                        self.optical_train_out_ph = optical_decoder
                        self.gray_train_out_ph = gray_decoder
                        print('optical flow output shape',tf.shape(self.optical_train_out_ph))
                        print('gray output shape',tf.shape(self.gray_train_out_ph))
                        print('optical_encoder',optical_encoder.shape)
                        print('gray_encoder',gray_encoder.shape)

                    with tf.name_scope('mean'):
                        with tf.name_scope('frame_level'):
                            with tf.name_scope('optical_flow'):
                                self.optical_loss_sequences_frame_mean = tf.reduce_mean(tf.square(self.optical_train_in_ph - self.optical_train_out_ph),axis=[2,3,4])
                            with tf.name_scope('gray'):
                                self.gray_loss_sequences_frame_mean = tf.reduce_mean(tf.square(self.gray_train_in_ph - self.gray_train_out_ph),axis=[2,3,4])
                            with tf.name_scope('mid_stage'):
                                self.midstage_loss_sequences_frame_mean = tf.reduce_mean(tf.square(mid_stage_in - mid_stage_out),axis=[1,2,3])
                        with tf.name_scope('pixel_level'):
                            with tf.name_scope('optical_flow'):
                                self.optical_loss_sequences_pixel_mean = tf.reduce_mean(tf.square(self.optical_train_in_ph - self.optical_train_out_ph),axis=4)
                            with tf.name_scope('gray'):
                                self.gray_loss_sequences_pixel_mean = tf.square(self.gray_train_in_ph - self.gray_train_out_ph)
                            with tf.name_scope('mid_stage'):
                                self.midstage_loss_sequences_pixel_mean = tf.reduce_mean(tf.square(mid_stage_in - mid_stage_out),axis=3)
                    
                    with tf.name_scope('sum'):
                        with tf.name_scope('frame_level'):
                            with tf.name_scope('optical_flow'):
                                self.optical_loss_sequences_frame_sum = tf.reduce_sum(tf.square(self.optical_train_in_ph - self.optical_train_out_ph),axis=[2,3,4])
                            with tf.name_scope('gray'):
                                self.gray_loss_sequences_frame_sum = tf.reduce_sum(tf.square(self.gray_train_in_ph - self.gray_train_out_ph),axis=[2,3,4])
                            with tf.name_scope('mid_stage'):
                                self.midstage_loss_sequences_frame_sum = tf.reduce_sum(tf.square(mid_stage_in - mid_stage_out),axis=[1,2,3])
                        with tf.name_scope('pixel_level'):
                            with tf.name_scope('optical_flow'):
                                self.optical_loss_sequences_pixel_sum = tf.reduce_sum(tf.square(self.optical_train_in_ph - self.optical_train_out_ph),axis=4)
                            with tf.name_scope('gray'):
                                self.gray_loss_sequences_pixel_sum = tf.square(self.gray_train_in_ph - self.gray_train_out_ph)
                            with tf.name_scope('mid_stage'):
                                self.midstage_loss_sequences_pixel_sum = tf.reduce_sum(tf.square(mid_stage_in - mid_stage_out),axis=3)
                    
                    

                    with tf.name_scope('net_variable'):
                        self.optical_trainable_variable = self.optical_encoder.trainable_variable + self.optical_decoder.trainable_variable
                        self.gray_trainable_variable = self.gray_encoder.trainable_variable + self.gray_decoder.trainable_variable
                        self.mid_trainable_variable = self.mid_model.trainable_variable
                        self.all_trainabel_variable = self.optical_trainable_variable + self.gray_trainable_variable + self.mid_trainable_variable
                        self.update_variable = self.optical_encoder.update_variable + \
                                                self.optical_decoder.update_variable + \
                                                self.gray_encoder.update_variable + \
                                                self.gray_decoder.update_variable + \
                                                self.mid_model.update_variable

                    with tf.name_scope('train_loss'):
                        self.optical_loss = tf.reduce_mean(self.optical_loss_sequences_frame_mean) 
                        self.gray_loss = tf.reduce_mean(self.gray_loss_sequences_frame_mean) 
                        self.mid_stage_loss = tf.reduce_mean(tf.square(mid_stage_in - mid_stage_out))
                        self.total_loss = self.optical_loss* self.optical_flow_loss_ratio + self.gray_loss* self.gray_loss_ratio 
                        
                        #+ extraloss * self.extra_loss_ratio 

                    with tf.control_dependencies(self.update_variable):
                        self.optical_apply = optical_ae_opt.minimize(self.optical_loss,var_list=self.optical_trainable_variable)
                        self.gray_apply = gray_ae_opt.minimize(self.gray_loss,var_list=self.gray_trainable_variable)
                        self.mid_stage_apply = mid_ae_opt.minimize(self.mid_stage_loss,var_list=self.mid_trainable_variable)
                        self.c3d_apply = autoencoder_opt.minimize(self.total_loss, var_list=self.all_trainabel_variable)
            
            with tf.name_scope('psnr'):
                self.gray_loss_sequences_frame_psnr1 = tf.image.psnr(self.gray_train_in_ph , self.gray_train_out_ph, max_val=1.0)
                self.gray_loss_sequences_frame_psnr2 = tf.image.psnr(self.gray_train_out_ph , self.gray_train_in_ph, max_val=1.0)
                        
        # self.gray_c3d_model.summary()
        # self.optical_c3d_model.summary()
        # self.mid_model.summary()
        self.gray_saver = tf.train.Saver(var_list=self.gray_trainable_variable)
        self.optical_saver = tf.train.Saver(var_list=self.optical_trainable_variable)
        self.mid_stage_saver = tf.train.Saver(var_list=self.mid_trainable_variable)
        tf.summary.scalar('optical_loss', self.optical_loss)
        tf.summary.scalar('gray_loss', self.gray_loss)
        tf.summary.scalar('mid_stage_loss', self.mid_stage_loss)
        tf.summary.scalar('extra loss', extraloss)
        self.summary_merged = tf.summary.merge_all()
        return
    
    def test_video(self):
        def reshape_batches(inputs_batches):
            return np.reshape(inputs_batches,newshape=[inputs_batches.shape[0]*inputs_batches.shape[1],inputs_batches.shape[2],inputs_batches.shape[3],inputs_batches.shape[4]])

        my_multi_test_datasets = multi_test_datasets(batch_size = 4, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = False,crop_size=4, img_size=self.img_size_h)
        
        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            self.restore_model_weghts(sess)
            video_label  =  my_multi_test_datasets.init_test_single_videos(seletced_dataset_idx=2, selected_video_idx=2)
            video_lenth = 0

            psnr1_list = []
            optical_loss_list = []
            gray_loss_list = []
            optical_frame_list = []
            gray_frame_list = []
            optical_frame_label_list = []
            gray_frame_label_list = []
            while 1:
                batches = my_multi_test_datasets.get_single_videos_batches()
                if not (batches == []):
                    print(batches.shape)
                    video_lenth += (batches.shape[0]*batches.shape[1])
                    batch_data_gray = batches[:, :, :, :, 0:1]

                    gray_loss, gray_frames, gray_psnr1 = sess.run([self.gray_loss_sequences_frame_mean, self.gray_train_out_ph ,self.gray_loss_sequences_frame_psnr1],
                        feed_dict={self.gray_train_in_ph: batch_data_gray,
                                self.phase: False})
                    # print('optical loss shape',optical_loss.shape)
                    print('gray loss shape',gray_loss.shape)

                    print('psnr shape',gray_psnr1.shape)
                    # print('gray loss shape',gray_loss.shape)

                    # optical_loss = optical_loss.flatten()
                    gray_loss = gray_loss.flatten()

                    gray_psnr1 = gray_psnr1.flatten()

                    # optical_frame_list.append(reshape_batches(optical_frames))
                    gray_frame_list.append(reshape_batches(gray_frames))

                    gray_frame_label_list.append(reshape_batches(batch_data_gray))
                    # optical_frame_label_list.append(reshape_batches(batch_data_optical))

                    # optical_loss_list.append(optical_loss)
                    gray_loss_list.append(gray_loss)
                    psnr1_list.append(gray_psnr1)
                    
                else:
                    break
            # print('optical-loss')
            # optical_loss_list = max_min_np(np.concatenate(optical_loss_list,axis=0))
            # save_roc_auc_plot_img('',optical_loss_list, video_label)
            print('gray-loss')
            gray_loss_list = max_min_np(np.concatenate(gray_loss_list,axis=0))
            print(gray_loss_list)
            print(video_label)
            save_roc_auc_plot_img('',gray_loss_list, video_label)

            print('psnr1-auc')
            gray_psnr1 = max_min_np(np.concatenate(psnr1_list,axis=0))
            # print(gray_psnr1)
            save_roc_auc_plot_img('',gray_psnr1, video_label)


            gray_frame_list = np.concatenate(gray_frame_list, axis=0)
            # optical_frame_list = np.concatenate(optical_frame_list, axis=0)
            gray_frame_label_list = np.concatenate(gray_frame_label_list, axis=0)
            # optical_frame_label_list = np.concatenate(optical_frame_label_list, axis=0)
            gray_frame_list = np.concatenate([gray_frame_list, gray_frame_label_list], axis=1)
            # optical_frame_list = np.concatenate([optical_frame_list, optical_frame_label_list], axis=2)
            save_batch_images(gray_frame_list,self.gray_img_save_path,'test_gray.jpg')
        return
    
    def fetch_net_loss(self, sess, batch_data):
        train_loss = {}
        total_loss, optical_loss,gray_loss,mid_stage_loss,tmp_sum = sess.run([self.total_loss, self.optical_loss,self.gray_loss, self.mid_stage_loss, self.summary_merged], feed_dict={self.train_in_ph : batch_data,self.phase : False})

        
        midstage_loss_sequences_pixel_mean =sess.run([self.midstage_loss_sequences_pixel_mean], feed_dict={self.train_in_ph : batch_data,self.phase : False})
        print( midstage_loss_sequences_pixel_mean[0].shape )
        # print(optical_out.shape)

        train_loss['optical_flow'] = optical_loss
        train_loss['gray'] = gray_loss
        train_loss['midstage'] = mid_stage_loss
        train_loss['total'] = total_loss
        train_loss['summary'] = tmp_sum
        return train_loss

    def train_c3d(self,max_iteration = 100000 , restore_tags = True, trainable_whole=True, trainable_mid = True, only_gray_tags = True ):
        if only_gray_tags:
            my_multi_train_datasets = multi_train_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = False, is_Optical = True,crop_size=4, img_size=self.img_size_h)
        else:
            my_multi_train_datasets = multi_train_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = True,crop_size=4, img_size=self.img_size_h)

        gpu_options = tf.GPUOptions(allow_growth=True)

        summaries_dir = self.summaries_dir + 'SINGLE_GPU%d.CPTK' % time.time()
        train_writer = tf.summary.FileWriter(summaries_dir)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            if restore_tags :
                self.restore_model_weghts(sess)
            
            start_time = time.time()

            for idx in range(max_iteration):  
                batch_data = my_multi_train_datasets.get_batches()
                if only_gray_tags:
                    sess.run([self.gray_apply ], feed_dict={self.train_in_ph : batch_data, self.phase : True})
                elif trainable_whole:
                    sess.run([ self.c3d_apply ], feed_dict={self.train_in_ph : batch_data,self.phase : True})
                elif trainable_mid:
                    sess.run([ self.mid_stage_apply ], feed_dict={self.train_in_ph : batch_data,self.phase : True})
                else:
                    sess.run([ self.optical_apply, self.gray_apply ], feed_dict={self.train_in_ph : batch_data, self.phase : True})

                if (idx+1)%100 == 0:
                    batch_data = my_multi_train_datasets.get_batches()
                    train_loss = self.fetch_net_loss(sess, batch_data)
                    print('Epoches Idx%d' %(idx + 1))
                    t_elp = time.time() - start_time
                    h,m,s =  time_hms(t_elp)
                    print('Time Elapsed : %d hours %d minutes %d seconds'%(h, m, s) )
                    t_eta = t_elp/(idx+1)*(max_iteration - idx -1)
                    h,m,s =  time_hms(t_eta)
                    print('Time ETA : %d hours %d minutes %d seconds'%(h, m, s) )
                    print ('loss %.8f, gray_loss %.8f, optical_loss %.8f, mid_stage_loss %.8f'%(train_loss['total'],train_loss['gray'] ,train_loss['optical_flow'] ,train_loss['midstage'] ))
                    train_writer.add_summary(train_loss['summary'], (idx+1))
                    train_writer.flush()

                if (idx+1)%500 == 0 and (idx+1) >=2000:
                # if (idx+1)%50 == 0:
                    self.save_model_weghts(sess)
        return

    def fetch_net_test_loss(self, sess, batch_data):
        batch_data_gray = batch_data[:, :, :, :, 0:1]
        batch_data_optical = batch_data[:, :, :, :, 1:3]

        optical_loss,gray_loss,mid_stage_loss, optical_frames, gray_frames, gray_psnr1 = sess.run([self.optical_loss_sequences_frame_mean, self.gray_loss_sequences_frame_mean, self.midstage_loss_sequences_frame_mean, self.optical_train_out_ph, self.gray_train_out_ph ,self.gray_loss_sequences_frame_psnr1],
            feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
                    self.phase: False})

        trib_loss = np.ones_like(optical_loss,dtype=np.float)
        for b_idx in range(optical_loss.shape[0]):
            trib_loss[b_idx,:] = optical_loss[b_idx,:] + gray_loss[b_idx,:] + mid_stage_loss[b_idx]
        
        net_batch_loss = {}
        net_batch_loss['optical_loss_sequence'] = optical_loss.flatten()
        net_batch_loss['gray_loss_sequence'] = gray_loss.flatten()
        net_batch_loss['mid_loss_sequence'] = trib_loss.flatten()
        net_batch_loss['psnr_sequence'] = gray_psnr1.flatten()

        return net_batch_loss

    def test_single_dataset_type2(self):
        my_multi_test_datasets = multi_test_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = True,crop_size=4, img_size=self.img_size_h)
        
        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            self.restore_model_weghts(sess)
            seletced_dataset_idx = 0
            
            datasets_op_list = []
            datasets_gr_list = []
            datasets_to_list = []
            datasets_tr_list = []
            datasets_la_list = []
            datasest_psnr_list = []
            
    
            for video_idx in range(my_multi_test_datasets.multi_datasets[seletced_dataset_idx].video_clips_num):
                video_label  =  my_multi_test_datasets.init_test_single_videos(seletced_dataset_idx,video_idx)
                video_lenth = 0
                optical_loss_list = []
                gray_loss_list = []
                trible_loss_list = []
                psnr_list = []
                while 1:
                    batches = my_multi_test_datasets.get_single_videos_batches()
                    if not (batches == []):
                        # print(batches.shape)
                        video_lenth += (batches.shape[0]*batches.shape[1])
                        test_loss = self.fetch_net_test_loss(sess,batches)                        
                        
                        optical_loss_list.append(test_loss['optical_loss_sequence'])
                        gray_loss_list.append(test_loss['gray_loss_sequence'])
                        trible_loss_list.append(test_loss['mid_loss_sequence'])
                        psnr_list.append(test_loss['psnr_sequence'])
                    else:
                        print('together-loss')
                        tog_loss = max_min_np(np.concatenate(optical_loss_list,axis=0)+np.concatenate(gray_loss_list,axis=0))
                        datasets_to_list.append(tog_loss)
                        print('optical-loss')
                        optical_loss_list = max_min_np(np.concatenate(optical_loss_list,axis=0))                        
                        datasets_op_list.append(optical_loss_list)
                        print('gray-loss')                        
                        gray_loss_list = max_min_np(np.concatenate(gray_loss_list,axis=0))
                        datasets_gr_list.append(gray_loss_list)
                        print('trible-loss')                        
                        trible_loss_list = max_min_np(np.concatenate(trible_loss_list,axis=0))
                        print('trible - loss - normalized',trible_loss_list)
                        datasets_tr_list.append(trible_loss_list)

                        print('psnr list')
                        psnr_list = max_min_np(np.concatenate(psnr_list,axis=0))
                        print(psnr_list.shape)
                        datasest_psnr_list.append(psnr_list)

                        datasets_la_list.append(video_label)
                        break
            
            datasets_op_list = np.concatenate(datasets_op_list,axis=0)
            datasets_gr_list = np.concatenate(datasets_gr_list,axis=0)
            datasets_to_list = np.concatenate(datasets_to_list,axis=0)
            datasets_tr_list = np.concatenate(datasets_tr_list,axis=0)
            datasets_la_list = np.concatenate(datasets_la_list,axis=0)
            datasest_psnr_list = np.concatenate(datasest_psnr_list,axis=0)
            print('optical-loss')
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_op_list, datasets_la_list)
           
            print('gray-loss')
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_gr_list, datasets_la_list)

            print('together-loss')
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_to_list, datasets_la_list)

            print('trible-loss')
            # print(datasets_tr_list)
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_tr_list, datasets_la_list)
            frame_auc, frame_eer = save_roc_auc_plot_img('',1-datasets_tr_list, datasets_la_list)

            print('psnr-loss')
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasest_psnr_list, datasets_la_list)

            print('test')
        return
    

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_model = C3D_Running()
# run_model.train_gray_optical_c3d(2000)
# run_model.train_mid_stage_c3d(1000)
run_model.train_c3d(800000, restore_tags=False)
# run_model.test_video()
# run_model.test_single_dataset_type2()
run_model.test_video()