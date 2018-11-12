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
import scipy.io as scio
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
from C3D_MODEL import C3D_ENCODER,C3D_DECODER,C2D_ENCODER,C2D_DECODER,Attention_Model, Attention_Model_Xt, Attention_Convert_Model, C3D_Test_ENCODER, C2D_Test_ENCODER, Attention_Test_Model
from model_utils import save_batch_images

def time_hms(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return h, m ,s

class C3D_Running:
    def __init__(self):
        self.img_channel = 2

        self.gray_encoder = C2D_Test_ENCODER(input_channel=1,model_scope_name='gray_encoder', bn_tag = False)
        self.gray_encoder.encoder_channel_num = [64,   1]
        self.gray_encoder.encoder_stride_num = [[ 1, 1],[ 1, 1] ]
        self.gray_encoder.encoder_kernel_size = [[ 3, 3],[ 1, 1]]
        # self.gray_encoder.encoder_channel_num = [64,          64,  1]
        # self.gray_encoder.encoder_stride_num = [[1, 1],[ 1, 1],[ 1, 1] ]
        # self.gray_encoder.encoder_kernel_size = [[3, 3],[ 3, 3],[ 3, 3]]

        self.gray_decoder = C3D_DECODER(input_channel=1,model_scope_name='gray_decoder', bn_tag = False)
        self.gray_decoder.decoder_channel_num = [8,          self.gray_decoder.input_channel]
        self.gray_decoder.decoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.mid_model = Attention_Test_Model( input_channel = 3,model_scope_name = 'net',s_size=4, attention_uints = 64, attention_hops = 16)

        self.batch_size = 1
        self.video_imgs_num = 4
        self.img_size_h = None
        self.img_size_w = None
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-4
        self.autoencoder_lr = 1e-3

        self.selected_gpu_num = 0

        self.build_model_adn_loss_opt()
        self.root_path = '/home/room304/TB/TB/TensorFlow_Saver/ANORMLY/Split_C2D_Without_Extraloss/'

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
        print(gray_cptk.model_checkpoint_path)
        self.gray_saver.restore(sess, gray_cptk.model_checkpoint_path)
        return
    
    def save_model_weghts(self, sess ):
        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        self.gray_saver.save(sess, gray_save_path)
        return

    def build_model_adn_loss_opt(self):

        self.mid_stage_loss_ratio = 1.0
        self.optical_flow_loss_ratio = 1.0
        self.gray_loss_ratio = 1.0
        self.extra_loss_ratio = 10.0
        with tf.device('/cpu:0'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optical_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                gray_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
                mid_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)

                autoencoder_global_step = tf.Variable(0, trainable=False)
                autoencoder_lr_rate = tf.train.exponential_decay(self.autoencoder_lr, autoencoder_global_step, 10000, 0.99,staircase=True)
                autoencoder_opt = tf.train.AdamOptimizer(learning_rate=autoencoder_lr_rate)

                self.train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 1], name='batches_in')

                self.gray_train_in1_ph = self.train_in_ph[:,0:4:2,:,:,:]
                self.gray_train_in2_ph = self.train_in_ph[:,1:4:2,:,:,:]

                self.phase = tf.placeholder(tf.bool,name='is_training')
                                            

            with tf.device('/gpu:%d' % self.selected_gpu_num):
                with tf.name_scope('GPU%d' % self.selected_gpu_num) as scope:

                    with tf.name_scope('encoder'):
                        gray_encoder1 = self.gray_train_in1_ph
                        gray_encoder2 = self.gray_train_in2_ph

                        gray_encoder1 = tf.transpose(gray_encoder1,perm=[0,4,2,3,1])
                        gray_encoder2 = tf.transpose(gray_encoder2,perm=[0,4,2,3,1])
                        gray_encoder1 = tf.squeeze(gray_encoder1,axis=1)
                        gray_encoder2 = tf.squeeze(gray_encoder2,axis=1)

                        gray_encoder1 = self.gray_encoder(gray_encoder1,self.phase)
                        gray_encoder2 = self.gray_encoder(gray_encoder2,self.phase)

                        att1,att1_extra_loss, att1_hp = self.mid_model(gray_encoder1)
                        att2,att2_extra_loss, att2_hp = self.mid_model(gray_encoder2)
                        
                        # gray_frame_0 = gray_encoder1[:,:,:,0:1]
                        # gray_frame_1 = gray_encoder2[:,:,:,1:2]
                        # gray_frame_2 = gray_encoder1[:,:,:,0:1]
                        # gray_frame_3 = gray_encoder2[:,:,:,1:2]

                    with tf.name_scope('net_variable'):
                        self.gray_trainable_variable = self.gray_encoder.trainable_variable + self.mid_model.trainable_variable
                        
                        self.update_variable = self.gray_encoder.update_variable

                    with tf.name_scope('train_loss'):
                        self.gray_loss = tf.reduce_mean(tf.square(gray_encoder1 - gray_encoder2))
                        self.att_loss = tf.reduce_mean(tf.square(att1 - att2))
                        self.total_loss =  -self.att_loss

                    with tf.control_dependencies(self.update_variable):
                        self.gray_apply = gray_ae_opt.minimize(self.total_loss,var_list=self.gray_trainable_variable)

                        
        # self.gray_c3d_model.summary()
        # self.optical_c3d_model.summary()
        # self.mid_model.summary()
        self.gray_saver = tf.train.Saver(var_list=self.gray_trainable_variable)
        tf.summary.scalar('gray_loss', self.gray_loss)
        tf.summary.scalar('att_loss', self.att_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('att1_extra_loss', att1_extra_loss)
        tf.summary.scalar('att2_extra_loss', att2_extra_loss)

        self.out_image_gray_encoder1 = gray_encoder1
        self.out_image_gray_encoder2 = gray_encoder2
        self.out_image_difference = tf.square(gray_encoder1 - gray_encoder2)
        self.out_image_input_frame1 = self.gray_train_in1_ph[:,0,:,:,:]
        self.out_image_input_frame2 = self.gray_train_in1_ph[:,1,:,:,:]
        self.out_image_input_frame3 = self.gray_train_in2_ph[:,0,:,:,:]
        self.out_image_input_frame4 = self.gray_train_in2_ph[:,1,:,:,:]
        self.out_image_heatmap_1_normalize = (att1_hp-tf.reduce_min(att1_hp))/(tf.reduce_max(att1_hp)-tf.reduce_min(att1_hp))
        self.out_image_heatmap_2_normalize = (att2_hp-tf.reduce_min(att2_hp))/(tf.reduce_max(att2_hp)-tf.reduce_min(att2_hp))
        self.out_image_heatmap_1_clip=tf.clip_by_value(att1_hp, 0, 1.0)
        self.out_image_heatmap_2_clip=tf.clip_by_value(att2_hp, 0, 1.0)

        self.out_image_con1 = tf.concat([self.out_image_input_frame1,self.out_image_input_frame2,self.out_image_heatmap_1_normalize,self.out_image_heatmap_1_clip], axis=2)
        self.out_image_con2 = tf.concat([self.out_image_input_frame3,self.out_image_input_frame4,self.out_image_heatmap_2_normalize,self.out_image_heatmap_2_clip], axis=2)

        tf.summary.image('out1',self.out_image_con1)
        tf.summary.image('out2',self.out_image_con2)


        # tf.summary.image('gray_encoder_f1', gray_encoder1)
        # tf.summary.image('gray_encoder_f2', gray_encoder2)
        # tf.summary.image('difference', tf.square(gray_encoder1 - gray_encoder2))

        # tf.summary.image('gray_inputer_f1', self.gray_train_in1_ph[:,0,:,:,:])
        # tf.summary.image('gray_inputer_f2', self.gray_train_in1_ph[:,1,:,:,:])
        # tf.summary.image('gray_inputer_f3', self.gray_train_in2_ph[:,0,:,:,:])
        # tf.summary.image('gray_inputer_f4', self.gray_train_in2_ph[:,1,:,:,:])
        

        # tf.summary.image('heatmap_1_normalize', (att1_hp-tf.reduce_min(att1_hp))/(tf.reduce_max(att1_hp)-tf.reduce_min(att1_hp)))
        # tf.summary.image('heatmap_2_normalize', (att2_hp-tf.reduce_min(att2_hp))/(tf.reduce_max(att2_hp)-tf.reduce_min(att2_hp)))

        # tf.summary.image('heatmap_1_clip', tf.clip_by_value(att1_hp, 0, 1.0))
        # tf.summary.image('heatmap_2_clip', tf.clip_by_value(att2_hp, 0, 1.0))

        # tf.summary.image('gray_inputer_difference1', tf.square((self.gray_train_in1_ph[:,0,:,:,:]-self.gray_train_in2_ph[:,0,:,:,:])))
        # tf.summary.tensor_summary('tensor_gray_inputer_difference1', (self.gray_train_in1_ph[:,0,:,:,:]-self.gray_train_in2_ph[:,0,:,:,:]))
        # tf.summary.tensor_summary('tensor_gray_inputer_difference2', (self.gray_train_in1_ph[:,1,:,:,:]-self.gray_train_in2_ph[:,1,:,:,:]))

        self.tensor_gray_inputer_difference1 = self.gray_train_in1_ph[:,0,:,:,:]-self.gray_train_in2_ph[:,0,:,:,:]
        self.tensor_gray_inputer_difference2 = self.gray_train_in1_ph[:,1,:,:,:]-self.gray_train_in2_ph[:,1,:,:,:]

        # tf.summary.image('gray_encoder_f3', gray_frame_2)
        # tf.summary.image('gray_encoder_f4', gray_frame_3)
        # tf.summary.scalar('total loss', self.total_loss)
        self.summary_merged = tf.summary.merge_all()
        for var_iter in self.gray_trainable_variable:
            print(var_iter)
        return

    def fetch_net_loss(self, sess, batch_data):
        train_loss = {}
        tensor_gray_inputer_difference1, tensor_gray_inputer_difference2, gray_loss,tmp_sum = sess.run([self.tensor_gray_inputer_difference1, self.tensor_gray_inputer_difference2, self.gray_loss,  self.summary_merged], feed_dict={self.train_in_ph : batch_data,self.phase : False})
        train_loss['gray'] = gray_loss
        train_loss['summary'] = tmp_sum
        # scio.savemat('./result.mat',{'dif1':tensor_gray_inputer_difference1})
        # scio.savemat('./result.mat',{'dif2':tensor_gray_inputer_difference2})
        return train_loss

    def train_c3d(self,max_iteration = 100000 , restore_tags = True, trainable_whole=True, trainable_mid = True, ):
        my_multi_train_datasets = multi_train_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = False,crop_size=4, img_size=0)
        gpu_options = tf.GPUOptions(allow_growth=True)

        summaries_dir = self.summaries_dir + 'slice_temporal_to_compare_model_without_extraloss%d.CPTK' % time.time()
        train_writer = tf.summary.FileWriter(summaries_dir)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            if restore_tags :
                self.restore_model_weghts(sess)
            
            start_time = time.time()
            # batch_data = my_multi_train_datasets.get_batches()
            # for idx in range(max_iteration):
            #     try:
            #         train_loss = self.fetch_net_loss(sess, batch_data)
            #         train_writer.add_summary(train_loss['summary'], (idx+1))
            #         train_writer.flush()
            #     except Exception:
            #         print(' fetch net loss failed')

            for idx in range(max_iteration):  
                batch_data = my_multi_train_datasets.get_batches()
                try:
                    sess.run([self.gray_apply ], feed_dict={self.train_in_ph : batch_data, self.phase : True})
                except Exception:                    print('train failed')
                if (idx+1)%200 == 0:
                    batch_data = my_multi_train_datasets.get_batches()
                    try:
                        train_loss = self.fetch_net_loss(sess, batch_data)
                        print('Epoches Idx%d' %(idx + 1))
                        t_elp = time.time() - start_time
                        h,m,s =  time_hms(t_elp)
                        print('Time Elapsed : %d hours %d minutes %d seconds'%(h, m, s) )
                        t_eta = t_elp/(idx+1)*(max_iteration - idx -1)
                        h,m,s =  time_hms(t_eta)
                        print('Time ETA : %d hours %d minutes %d seconds'%(h, m, s) )
                        print ('gray_loss %.8f'%(train_loss['gray']  ))
                        train_writer.add_summary(train_loss['summary'], (idx+1))
                        train_writer.flush()
                    except Exception:
                        print(' fetch net loss failed')
                    
                if (idx+1)%500 == 0 and (idx+1) >=2000:
                # if (idx+1)%50 == 0:
                    self.save_model_weghts(sess)
        return

    # def fetch_net_test_loss(self, sess, batch_data):
    #     batch_data_gray = batch_data[:, :, :, :, 0:1]
    #     batch_data_optical = batch_data[:, :, :, :, 1:3]

    #     optical_loss,gray_loss,mid_stage_loss = sess.run([self.optical_loss_sequences_frame_mean, self.gray_loss_sequences_frame_mean, self.midstage_loss_sequences_frame_mean],
    #         feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
    #                 self.phase: False})

    #     trib_loss = np.ones_like(optical_loss,dtype=np.float)
    #     for b_idx in range(optical_loss.shape[0]):
    #         trib_loss[b_idx,:] = optical_loss[b_idx,:] + gray_loss[b_idx,:] + mid_stage_loss[b_idx]
        
    #     net_batch_loss = {}
    #     net_batch_loss['optical_loss_sequence'] = optical_loss
    #     net_batch_loss['gray_loss_sequence'] = gray_loss
    #     net_batch_loss['trible_loss_sequence'] = mid_stage_loss
 
    #     return

    def save_single_video(self):
        my_multi_test_datasets = multi_test_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = False,crop_size=4, img_size=self.img_size_h)
        
        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            self.restore_model_weghts(sess)
            seletced_dataset_idx = 3
            selected_video_idx = 2
            
            video_label  =  my_multi_test_datasets.init_test_single_videos(seletced_dataset_idx,selected_video_idx)
            video_lenth = 0
            image_batches_list = []
            while 1:
                batches = my_multi_test_datasets.get_single_videos_batches()
                if not (batches == []):
                    # print(batches.shape)
                    video_lenth += (batches.shape[0]*batches.shape[1])

                    out_image1, out_image2 = sess.run([self.out_image_con1, self.out_image_con2],
                        feed_dict={ self.train_in_ph : batches, self.phase: False})
                    image_batches_list.append(out_image1)
                    image_batches_list.append(out_image2)
                else:
                    break
            
            image_batches_list = np.concatenate(image_batches_list,axis=0)
            save_batch_images(image_batches_list,self.gray_img_save_path,'datasets_%d_video_%d.jpg'%(seletced_dataset_idx,selected_video_idx))

            print('test')
        return

    # def test_single_dataset_type2(self):
    #     my_multi_test_datasets = multi_test_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = False,crop_size=4, img_size=self.img_size_h)
        
    #     gpu_options = tf.GPUOptions(allow_growth=True)

    #     with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         self.restore_model_weghts(sess)
    #         seletced_dataset_idx = 3
            
    #         datasets_op_list = []
    #         datasets_gr_list = []
    #         datasets_to_list = []
    #         datasets_tr_list = []
    #         datasets_la_list = []
            
    
    #         for video_idx in range(my_multi_test_datasets.multi_datasets[seletced_dataset_idx].video_clips_num):
    #             video_label  =  my_multi_test_datasets.init_test_single_videos(seletced_dataset_idx,video_idx)
    #             video_lenth = 0
    #             optical_loss_list = []
    #             gray_loss_list = []
    #             trible_loss_list = []
    #             while 1:
    #                 batches = my_multi_test_datasets.get_single_videos_batches()
    #                 if not (batches == []):
    #                     # print(batches.shape)
    #                     video_lenth += (batches.shape[0]*batches.shape[1])

    #                     gray_loss = sess.run([self.gray_loss_sequences_frame_mean],
    #                         feed_dict={ self.train_in_ph : batches, self.phase: False})
    #                     # print('optical loss shape', optical_loss.shape)
    #                     # print('gray loss shape', gray_loss.shape)
    #                     # print('mid stage loss', mid_stage_loss.shape)
    #                     # trib_loss = np.ones_like(optical_loss,dtype=np.float)
    #                     # for b_idx in range(optical_loss.shape[0]):
    #                     #     trib_loss[b_idx,:] = 0*optical_loss[b_idx,:] + 0*gray_loss[b_idx,:] + mid_stage_loss[b_idx]
    #                         # print(optical_loss[b_idx,:])
    #                         # print(gray_loss[b_idx,:])
    #                         # print(mid_stage_loss[b_idx])
    #                         # print(trib_loss[b_idx,:])
    #                     # optical_loss = optical_loss.flatten()
    #                     gray_loss = gray_loss[0].flatten()
    #                     # trib_loss = trib_loss.flatten()
    #                     # optical_loss_list.append(optical_loss)
    #                     gray_loss_list.append(gray_loss)
    #                     # trible_loss_list.append(trib_loss)
    #                 else:
    #                     print('optical-loss')
    #                     # tog_loss = max_min_np(np.concatenate(optical_loss_list,axis=0)+np.concatenate(gray_loss_list,axis=0))
    #                     # datasets_to_list.append(tog_loss)

    #                     # optical_loss_list = max_min_np(np.concatenate(optical_loss_list,axis=0))                        
    #                     # datasets_op_list.append(optical_loss_list)
                        
    #                     gray_loss_list = max_min_np(np.concatenate(gray_loss_list,axis=0))
    #                     datasets_gr_list.append(gray_loss_list)
                        
    #                     # trible_loss_list = max_min_np(np.concatenate(trible_loss_list,axis=0))
    #                     # print('trible - loss - normalized',trible_loss_list)
    #                     # datasets_tr_list.append(trible_loss_list)
    #                     datasets_la_list.append(video_label)
    #                     break
            
    #         # datasets_op_list = np.concatenate(datasets_op_list,axis=0)
    #         datasets_gr_list = np.concatenate(datasets_gr_list,axis=0)
    #         # datasets_to_list = np.concatenate(datasets_to_list,axis=0)
    #         # datasets_tr_list = np.concatenate(datasets_tr_list,axis=0)
    #         datasets_la_list = np.concatenate(datasets_la_list,axis=0)
    #         # print('optical-loss')
    #         # frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_op_list, datasets_la_list)
           
    #         print('gray-loss')
    #         # print(datasets_gr_list.shape)
    #         # print(datasets_gr_list.shape)
    #         frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_gr_list, datasets_la_list)

    #         # print('together-loss')
    #         # frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_to_list, datasets_la_list)

    #         print('trible-loss')
    #         # print(datasets_tr_list)
    #         print('label')
    #         # print(datasets_la_list)
    #         # frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_tr_list, datasets_la_list)
    #         # frame_auc, frame_eer = save_roc_auc_plot_img('',1-datasets_tr_list, datasets_la_list)

    #         print('test')
    #     return
    

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_model = C3D_Running()
# run_model.train_gray_optical_c3d(2000)
# run_model.train_mid_stage_c3d(1000)

# run_model.train_c3d(1000, restore_tags=True, trainable_whole=True, trainable_mid = True)

# run_model.train_c3d(10000, restore_tags=False, trainable_whole=False, trainable_mid = False)
# run_model.train_c3d(10000, restore_tags=True, trainable_whole=False, trainable_mid = True)
# run_model.test_single_dataset_type2()

run_model.save_single_video()