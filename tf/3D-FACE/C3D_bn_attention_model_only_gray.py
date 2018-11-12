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
from C3D_MODEL import C3D_ENCODER,C3D_DECODER,C2D_ENCODER,C2D_DECODER,Attention_Model

def time_hms(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return h, m ,s

class C3D_Running:
    def __init__(self):
        self.img_channel = 2

        self.optical_encoder = C3D_ENCODER(input_channel=2,model_scope_name='optical_flow_encoder', bn_tag = False , relu_tag=True)
        self.optical_encoder.encoder_channel_num = [64,        16]
        self.optical_encoder.encoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.optical_decoder = C3D_DECODER(input_channel=2,model_scope_name='optical_flow_decoder', bn_tag = False, relu_tag=True)
        self.optical_decoder.not_last_activation = True
        self.optical_decoder.decoder_channel_num = [64,      self.optical_decoder.input_channel]
        self.optical_decoder.decoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.gray_encoder = C3D_ENCODER(input_channel=1,model_scope_name='gray_encoder', bn_tag = False, relu_tag=True)
        self.gray_encoder.encoder_channel_num = [64,          32]
        self.gray_encoder.encoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.gray_decoder = C3D_DECODER(input_channel=1,model_scope_name='gray_decoder', bn_tag = False, relu_tag=True)
        self.gray_decoder.decoder_channel_num = [64,          self.gray_decoder.input_channel]
        self.gray_decoder.decoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.mid_model = Attention_Model(input_channel = 512,model_scope_name='vn4_concate_model', attention_uints = 512, attention_hops = 256)

        self.batch_size = 4
        self.video_imgs_num = 4
        self.img_size_h = 224
        self.img_size_w = 224
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-4
        self.autoencoder_lr = 1e-3

        self.selected_gpu_num = 0

        self.build_model_adn_loss_opt()
        self.root_path = '/home/room304/TB/TB/TensorFlow_Saver/ANORMLY/Attention_test_not_interval_only_gray/'

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
        with tf.device('/cpu:0'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optical_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                gray_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                mid_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)

                autoencoder_global_step = tf.Variable(0, trainable=False)
                autoencoder_lr_rate = tf.train.exponential_decay(self.autoencoder_lr, autoencoder_global_step, 10000, 0.99,staircase=True)
                autoencoder_opt = tf.train.RMSPropOptimizer(learning_rate=1e-3)

                self.train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 3], name='batches_in')

                self.optical_train_in_ph = self.train_in_ph[:,:,:,:,1:3]
                self.gray_train_in_ph = self.train_in_ph[:,:,:,:,0:1]
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
                        mid_stage_out, extraloss, heat_map = self.mid_model(mid_stage_in)
                        mid_stage_out_p = tf.expand_dims(mid_stage_out,axis=1)
                        print('test')

                        # mid_stage_out_p = tf.expand_dims(mid_stage_in,axis=1)

                        mid_stage_out_opticalflow = tf.expand_dims(optical_encoder,axis=1)
                        mid_stage_out_gray = tf.expand_dims(gray_encoder,axis=1)

                        # mid_stage_out_opticalflow = mid_stage_out_p[:,:,:,:,0:optical_encoder.shape[3]]
                        # mid_stage_out_gray = mid_stage_out_p[:,:,:,:,optical_encoder.shape[3]:]

                        print('mid_stage_out_opticalflow',mid_stage_out_opticalflow.shape)
                        print('mid_stage_out_gray',mid_stage_out_gray.shape)
                    
                    with tf.name_scope('decoder'):
                        optical_decoder = self.optical_decoder(mid_stage_out_opticalflow,self.phase)
                        gray_decoder = self.gray_decoder(mid_stage_out_gray,self.phase)
                        self.optical_train_out_ph = optical_decoder
                        self.gray_train_out_ph = gray_decoder
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

                        self.optical_all_variables = self.optical_encoder.all_variables + self.optical_decoder.all_variables
                        self.gray_all_variables = self.gray_encoder.all_variables + self.gray_decoder.all_variables
                        self.mid_all_variable = self.mid_model.all_variables

                        self.update_variable = self.optical_encoder.update_variable + \
                                                self.optical_decoder.update_variable + \
                                                self.gray_encoder.update_variable + \
                                                self.gray_decoder.update_variable + \
                                                self.mid_model.update_variable
                        

                    with tf.name_scope('train_loss'):
                        self.optical_loss = tf.reduce_mean(self.optical_loss_sequences_frame_mean) 
                        self.gray_loss = tf.reduce_mean(self.gray_loss_sequences_frame_mean) 
                        self.mid_stage_loss = tf.reduce_mean(tf.square(mid_stage_in - mid_stage_out))
                        self.total_loss = self.gray_loss* self.gray_loss_ratio + self.mid_stage_loss*self.mid_stage_loss_ratio
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    # print(update_ops)
                    # print(self.update_variable)
                    # print(self.gray_all_variables)

                    with tf.control_dependencies(update_ops):
                        self.optical_apply = optical_ae_opt.minimize(self.optical_loss,var_list=self.optical_trainable_variable)
                        self.gray_apply = gray_ae_opt.minimize(self.gray_loss,var_list=self.gray_trainable_variable)
                        self.mid_stage_apply = mid_ae_opt.minimize(self.mid_stage_loss,var_list=self.mid_trainable_variable)
                        self.c3d_apply = autoencoder_opt.minimize(self.total_loss, var_list=self.all_trainabel_variable)
                        
        # self.gray_c3d_model.summary()
        # self.optical_c3d_model.summary()
        # self.mid_model.summary()
        self.gray_saver = tf.train.Saver(var_list=self.gray_all_variables)
        self.optical_saver = tf.train.Saver(var_list=self.optical_all_variables)
        self.mid_stage_saver = tf.train.Saver(var_list=self.mid_all_variable)
        tf.summary.scalar('optical_loss', self.optical_loss)
        tf.summary.scalar('gray_loss', self.gray_loss)
        tf.summary.scalar('mid_stage_loss', self.mid_stage_loss)
        tf.summary.scalar('extra loss', extraloss)
        tf.summary.scalar('total loss', self.total_loss)
        self.extraloss = extraloss

        tf.summary.image('gray_output',tf.clip_by_value(self.gray_train_out_ph[0,0:2,:,:,:],0,1.0) )
        # heat_map = heat_map[0,:,:,:]
        # heat_map_max = tf.reduce_max(heat_map)
        # heat_map_min = tf.reduce_min(heat_map)
        # tf.summary.image('hear_map', tf.clip_by_value(heat_map, 0 , 1.0) )
        tf.summary.image('hear_map', (heat_map - tf.reduce_min(heat_map))/( tf.reduce_max(heat_map) - tf.reduce_min(heat_map) ))
        tf.summary.image('gray_pixel_loss',tf.clip_by_value(self.gray_loss_sequences_pixel_mean[0,0:2,:,:,:] , 0, 1.0))
        tf.summary.image('opticalflow_pixel_loss', tf.clip_by_value( tf.expand_dims(self.optical_loss_sequences_pixel_mean[0,0:2,:,:], axis=3), 0, 1.0))

        self.summary_merged = tf.summary.merge_all()

        return
    
    def fetch_net_loss(self, sess, batch_data):
        train_loss = {}
        total_loss,extraloss, optical_loss,gray_loss,mid_stage_loss,tmp_sum = sess.run([self.total_loss,self.extraloss, self.optical_loss,self.gray_loss, self.mid_stage_loss, self.summary_merged], feed_dict={self.train_in_ph : batch_data,self.phase : False})
        train_loss['optical_flow'] = optical_loss
        train_loss['gray'] = gray_loss
        train_loss['midstage'] = mid_stage_loss
        train_loss['total'] = total_loss
        train_loss['extraloss'] = extraloss
        train_loss['summary'] = tmp_sum
        return train_loss

    def train_c3d(self,max_iteration = 100000 , restore_tags = True, trainable_whole=True, trainable_mid = True, ):
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
                try:
                    batch_data = my_multi_train_datasets.get_batches()
                    if trainable_whole:
                        sess.run([ self.c3d_apply ], feed_dict={self.train_in_ph : batch_data,self.phase : True})
                    elif trainable_mid:
                        sess.run([ self.mid_stage_apply ], feed_dict={self.train_in_ph : batch_data,self.phase : True})
                    else:
                        sess.run([ self.optical_apply, self.gray_apply ], feed_dict={self.train_in_ph : batch_data, self.phase : True})

                except Exception:
                    print('training failed')
                

                if (idx+1)%200 == 0:
                    try:
                        batch_data = my_multi_train_datasets.get_batches()
                        train_loss = self.fetch_net_loss(sess, batch_data)
                        print('Epoches Idx%d' %(idx + 1))
                        t_elp = time.time() - start_time
                        h,m,s =  time_hms(t_elp)
                        print('Time Elapsed : %d hours %d minutes %d seconds'%(h, m, s) )
                        t_eta = t_elp/(idx+1)*(max_iteration - idx -1)
                        h,m,s =  time_hms(t_eta)
                        print('Time ETA : %d hours %d minutes %d seconds'%(h, m, s) )
                        print ('loss %.8f, extraloss %.8f,gray_loss %.8f, optical_loss %.8f, mid_stage_loss %.8f'%(train_loss['total'],train_loss['extraloss'],train_loss['gray'] ,train_loss['optical_flow'] ,train_loss['midstage'] ))
                        train_writer.add_summary(train_loss['summary'], (idx+1))
                        train_writer.flush()
                    except Exception:
                        print('fet net loss failed')
                    

                if (idx+1)%500 == 0 and (idx+1) >=2000:
                # if (idx+1)%50 == 0:
                    self.save_model_weghts(sess)
        return

    def fetch_net_test_loss(self, sess, batch_data):
        batch_data_gray = batch_data[:, :, :, :, 0:1]
        batch_data_optical = batch_data[:, :, :, :, 1:3]

        optical_loss,gray_loss,mid_stage_loss = sess.run([self.optical_loss_sequences_frame_mean, self.gray_loss_sequences_frame_mean, self.midstage_loss_sequences_frame_mean],
            feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
                    self.phase: False})

        trib_loss = np.ones_like(optical_loss,dtype=np.float)
        for b_idx in range(optical_loss.shape[0]):
            trib_loss[b_idx,:] = optical_loss[b_idx,:] + gray_loss[b_idx,:] + mid_stage_loss[b_idx]
        
        net_batch_loss = {}
        net_batch_loss['optical_loss_sequence'] = optical_loss
        net_batch_loss['gray_loss_sequence'] = gray_loss
        net_batch_loss['trible_loss_sequence'] = mid_stage_loss

        return

    def test_single_dataset_type2(self):
        my_multi_test_datasets = multi_test_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = True,crop_size=4, img_size=self.img_size_h)
        summaries_dir = self.summaries_dir + 'SINGLE_GPU%d.CPTK' % time.time()
        train_writer = tf.summary.FileWriter(summaries_dir)
        
        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            self.restore_model_weghts(sess)
            seletced_dataset_idx = 1
            
            datasets_op_list = []
            datasets_gr_list = []
            datasets_to_list = []
            datasets_tr_list = []
            datasets_la_list = []
            idx = 0
    
            for video_idx in range(my_multi_test_datasets.multi_datasets[seletced_dataset_idx].video_clips_num):
                video_label  =  my_multi_test_datasets.init_test_single_videos(seletced_dataset_idx,video_idx)
                video_lenth = 0
                optical_loss_list = []
                gray_loss_list = []
                trible_loss_list = []
                while 1:
                    idx = idx +1
                    batches = my_multi_test_datasets.get_single_videos_batches()
                    if not (batches == []):
                        # print(batches.shape)
                        video_lenth += (batches.shape[0]*batches.shape[1])

                        optical_loss,gray_loss,mid_stage_loss,tmp_sum = sess.run([self.optical_loss_sequences_frame_mean, self.gray_loss_sequences_frame_mean, self.midstage_loss_sequences_frame_mean, self.summary_merged],
                            feed_dict={ self.train_in_ph : batches, self.phase: False})
                        train_writer.add_summary(tmp_sum, (idx+1))
                        train_writer.flush()
                        # print('optical loss shape', optical_loss.shape)
                        # print('gray loss shape', gray_loss.shape)
                        # print('mid stage loss', mid_stage_loss.shape)
                        trib_loss = np.ones_like(optical_loss,dtype=np.float)
                        for b_idx in range(optical_loss.shape[0]):
                            trib_loss[b_idx,:] = optical_loss[b_idx,:] + gray_loss[b_idx,:] + mid_stage_loss[b_idx]
                            # print(optical_loss[b_idx,:])
                            # print(gray_loss[b_idx,:])
                            # print(mid_stage_loss[b_idx])
                            # print(trib_loss[b_idx,:])
                        optical_loss = optical_loss.flatten()
                        gray_loss = gray_loss.flatten()
                        trib_loss = trib_loss.flatten()
                        optical_loss_list.append(optical_loss)
                        gray_loss_list.append(gray_loss)
                        trible_loss_list.append(trib_loss)
                    else:
                        print('optical-loss')
                        tog_loss = max_min_np(np.concatenate(optical_loss_list,axis=0)+np.concatenate(gray_loss_list,axis=0))
                        datasets_to_list.append(tog_loss)

                        optical_loss_list = max_min_np(np.concatenate(optical_loss_list,axis=0))                        
                        datasets_op_list.append(optical_loss_list)
                        
                        gray_loss_list = max_min_np(np.concatenate(gray_loss_list,axis=0))
                        datasets_gr_list.append(gray_loss_list)
                        
                        trible_loss_list = max_min_np(np.concatenate(trible_loss_list,axis=0))
                        print('trible - loss - normalized',trible_loss_list)
                        datasets_tr_list.append(trible_loss_list)

                        datasets_la_list.append(video_label)
                        
                        break
            
            datasets_op_list = np.concatenate(datasets_op_list,axis=0)
            datasets_gr_list = np.concatenate(datasets_gr_list,axis=0)
            datasets_to_list = np.concatenate(datasets_to_list,axis=0)
            datasets_tr_list = np.concatenate(datasets_tr_list,axis=0)
            datasets_la_list = np.concatenate(datasets_la_list,axis=0)
            print('optical-loss')
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_op_list, datasets_la_list)
           
            print('gray-loss')
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_gr_list, datasets_la_list)

            print('together-loss')
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_to_list, datasets_la_list)

            print('trible-loss')
            # print(datasets_tr_list)
            print('label')
            # print(datasets_la_list)
            frame_auc, frame_eer = save_roc_auc_plot_img('',datasets_tr_list, datasets_la_list)
            frame_auc, frame_eer = save_roc_auc_plot_img('',1-datasets_tr_list, datasets_la_list)

            print('test')
        return
    
# current we add sqrt(uints) but not extra loss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_model = C3D_Running()
# run_model.train_gray_optical_c3d(2000)
# run_model.train_mid_stage_c3d(1000)
run_model.train_c3d(100000, restore_tags=False)
# run_model.test_single_dataset_type2()