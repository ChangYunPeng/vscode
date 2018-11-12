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
from C3D_MODEL import C3D_ENCODER,C3D_DECODER,C2D_ENCODER,C2D_DECODER,Attention_Model, Attention_Model_Xt, gan_model
from model_utils import save_batch_images

def time_hms(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return h, m ,s

class C3D_Running:
    def __init__(self):
        self.img_channel = 2
        self.gray_encoder = C3D_ENCODER(input_channel=1,model_scope_name='gray_encoder', bn_tag = False)
        self.gray_encoder.encoder_channel_num = [32,          64]
        self.gray_encoder.encoder_stride_num = [[2, 2, 2],[2, 2, 2]]
        self.gray_decoder = C3D_DECODER(input_channel=1,model_scope_name='gray_decoder', bn_tag = False)
        self.gray_decoder.decoder_channel_num = [32,          self.gray_decoder.input_channel]
        self.gray_decoder.decoder_stride_num = [[2, 2, 2],[2, 2, 2]]

        self.discriminator_model = gan_model(input_channel = 3,model_scope_name = 'discriminator',s_size=4,xt_num = 64, attention_uints = 512, attention_hops = 256, bn_tag = False, attention_tag = False, dense_tag=False,not_last_activation = True)
        self.discriminator_model.encoder_channel_num = [32,   32,   32]
        self.discriminator_model.encoder_stride_num = [[1, 1],[1, 1],[1, 1]]

        self.batch_size = 2
        self.video_imgs_num = 4
        self.img_size_h = 256
        self.img_size_w = 256
        self.encoder_lr = 1e-3
        self.decoder_lr = 1e-4
        self.autoencoder_lr = 1e-3

        self.selected_gpu_num = 0

        self.build_model_adn_loss_opt()
        self.root_path = '/home/room304/TB/TB/TensorFlow_Saver/ANORMLY/GAN_T1/'

        self.optical_save_path = self.root_path + 'MODEL_OPTICAL/'
        self.gray_save_path = self.root_path + 'MODEL_GRAY/'
        self.discriminator_save_path = self.root_path + 'MODEL_DISCRIMINATOR/'
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
        discriminator_cptk = tf.train.get_checkpoint_state(self.discriminator_save_path)
        print(gray_cptk.model_checkpoint_path)
        print(discriminator_cptk.model_checkpoint_path)
        self.gray_saver.restore(sess, gray_cptk.model_checkpoint_path)
        self.discriminator_saver.restore(sess,discriminator_cptk.model_checkpoint_path)
        return
    
    def save_model_weghts(self, sess ):
        gray_save_path = self.gray_save_path + 'trainable_weights.cptk'
        discriminator_save_path = self.discriminator_save_path + 'trainable_weights.cptk'
        self.gray_saver.save(sess, gray_save_path)
        self.discriminator_saver.save(sess, discriminator_save_path)
        return

    def build_model_adn_loss_opt(self):

        self.mid_stage_loss_ratio = 1.0
        self.optical_flow_loss_ratio = 1.0
        self.gray_loss_ratio = 1.0
        self.extra_loss_ratio = 1.0
        self.mse_ratio = 0.1
        LAMBDA = 10
        with tf.device('/cpu:0'):
            

            self.train_in_ph = tf.placeholder(dtype=tf.float32,shape=[None, self.video_imgs_num, self.img_size_h,self.img_size_w, 1], name='batches_in')

            self.gray_train_in_ph  = self.train_in_ph
            # self.optical_train_in_ph = self.train_in_ph[:,:,:,:,1:3]
            self.phase = tf.placeholder(tf.bool,name='is_training')
                                            

            with tf.device('/gpu:%d' % self.selected_gpu_num):
                with tf.name_scope('GPU%d' % self.selected_gpu_num) as scope:
                    with tf.name_scope('generator'):
                        gray_encoder = self.gray_encoder(self.gray_train_in_ph, self.phase)
                        gray_decoder = self.gray_decoder(gray_encoder,self.phase)
                        self.gray_train_out_ph = gray_decoder
                        # gray_decoder = tf.reshape(gray_decoder, tf.shape(self.gray_train_in_ph))


                    with tf.name_scope('discriminator'):
                        # batchsize, temporal, height, width, in_channels = tf.shape(self.gray_train_in_ph)
                        in_channels = tf.shape(self.gray_train_in_ph)[3]
                        print(' gray input tensor shape :' , in_channels)
                        # batchsize, temporal, height, width, in_channels = tf.shape(gray_decoder)
                        # print(' gray output tensor shape :' , batchsize, temporal, height, width, in_channels)

                        # disc_label_tensor = tf.reshape(self.gray_train_in_ph, shape =[batchsize*temporal, height, width, in_channels] )
                        # disc_gen_tensor = tf.reshape( gray_decoder, shape =[batchsize*temporal, height, width, in_channels] )
                        self.gray_train_in_ph.set_shape([None, self.video_imgs_num, self.img_size_h,self.img_size_w, 1])
                        gray_decoder.set_shape([None, self.video_imgs_num, self.img_size_h,self.img_size_w, 1])
                        disc_label_out = self.discriminator_model(self.gray_train_in_ph, self.phase)
                        disc_gen_out = self.discriminator_model(gray_decoder, self.phase)
                        
                        

                    with tf.name_scope('mean'):
                        with tf.name_scope('sequence'):
                            self.gray_loss_sequences_frame_mean = tf.reduce_mean(tf.square(self.gray_train_in_ph - self.gray_train_out_ph),axis=[2,3,4])
                            self.disc_loss_sequences_frame_mean = tf.reduce_mean(disc_gen_out - disc_label_out,axis=[1,2,3])

                        with tf.name_scope('generator'):
                            self.generator_loss = tf.reduce_mean(disc_label_out)-tf.reduce_mean(disc_gen_out) 
                            self.mse_loss = tf.reduce_mean(self.gray_loss_sequences_frame_mean)
                        with tf.name_scope('discrinimator'):
                            self.discriminator_loss = tf.reduce_mean(disc_gen_out) - tf.reduce_mean(disc_label_out)
                            # alpha = tf.random_uniform(
                            #     shape=tf.shape(self.gray_train_in_ph), 
                            #     minval=0.,
                            #     maxval=1.
                            # )
                            # differences = self.gray_train_out_ph - self.gray_train_in_ph
                            # interpolates = self.gray_train_in_ph + alpha*differences
                            # gradients = tf.gradients(self.discriminator_model(interpolates, self.phase), [interpolates])[0]
                            # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                            # gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                            # self.discriminator_loss += LAMBDA*gradient_penalty
                            
                    with tf.name_scope('net_variable'):
                        self.gray_trainable_variable = self.gray_encoder.trainable_variable + self.gray_decoder.trainable_variable
                        self.discriminator_trainabel_variable = self.discriminator_model.trainable_variable
                        self.update_variable = self.gray_encoder.update_variable + \
                                                self.gray_decoder.update_variable

                    with tf.control_dependencies(self.update_variable):
                        gray_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                        gray_mse_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                        discriminator_ae_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                        self.gray_apply = gray_ae_opt.minimize(self.generator_loss + self.mse_ratio*self.mse_loss,var_list=self.gray_trainable_variable)
                        self.gray_mse_apply = gray_mse_opt.minimize(self.mse_loss,var_list=self.gray_trainable_variable)
                        self.discriminator_apply = discriminator_ae_opt.minimize(self.discriminator_loss, var_list = self.discriminator_trainabel_variable)
            
            with tf.name_scope('psnr'):
                self.gray_loss_sequences_frame_psnr1 = tf.image.psnr(self.gray_train_in_ph , self.gray_train_out_ph, max_val=1.0)
                self.gray_loss_sequences_frame_psnr2 = tf.image.psnr(self.gray_train_out_ph , self.gray_train_in_ph, max_val=1.0)
                        
        # self.gray_c3d_model.summary()
        # self.optical_c3d_model.summary()
        # self.mid_model.summary()
        self.gray_saver = tf.train.Saver(var_list=self.gray_trainable_variable)
        self.discriminator_saver = tf.train.Saver(var_list=self.discriminator_trainabel_variable)
        tf.summary.scalar('generator_loss', self.generator_loss)
        tf.summary.scalar('discriminator_loss', self.discriminator_loss)
        self.summary_merged = tf.summary.merge_all()
        return
    
    def test_video(self):
        def reshape_batches(inputs_batches):
            return np.reshape(inputs_batches,newshape=[inputs_batches.shape[0]*inputs_batches.shape[1],inputs_batches.shape[2],inputs_batches.shape[3],inputs_batches.shape[4]])

        my_multi_test_datasets = multi_test_datasets(batch_size = 4, video_num = 4, frame_interval = 2, is_frame = True, is_Optical = True,crop_size=4, img_size=self.img_size_h)
        
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
                    batch_data_optical = batches[:, :, :, :, 1:3]

                    optical_loss,gray_loss, optical_frames, gray_frames, gray_psnr1 = sess.run([self.optical_loss_sequences_frame_mean, self.gray_loss_sequences_frame_mean, self.optical_train_out_ph, self.gray_train_out_ph ,self.gray_loss_sequences_frame_psnr1],
                        feed_dict={self.optical_train_in_ph: batch_data_optical, self.gray_train_in_ph: batch_data_gray,
                                self.phase: False})
                    print('optical loss shape',optical_loss.shape)
                    print('gray loss shape',gray_loss.shape)

                    print('psnr shape',gray_psnr1.shape)
                    # print('gray loss shape',gray_loss.shape)

                    optical_loss = optical_loss.flatten()
                    gray_loss = gray_loss.flatten()

                    gray_psnr1 = gray_psnr1.flatten()

                    optical_frame_list.append(reshape_batches(optical_frames))
                    gray_frame_list.append(reshape_batches(gray_frames))

                    gray_frame_label_list.append(reshape_batches(batch_data_gray))
                    optical_frame_label_list.append(reshape_batches(batch_data_optical))

                    optical_loss_list.append(optical_loss)
                    gray_loss_list.append(gray_loss)
                    psnr1_list.append(gray_psnr1)
                    
                else:
                    break
            print('optical-loss')
            optical_loss_list = max_min_np(np.concatenate(optical_loss_list,axis=0))
            save_roc_auc_plot_img('',optical_loss_list, video_label)
            print('gray-loss')
            gray_loss_list = max_min_np(np.concatenate(gray_loss_list,axis=0))
            print(gray_loss_list)
            print(video_label)
            save_roc_auc_plot_img('',gray_loss_list, video_label)

            print('psnr1-auc')
            gray_psnr1 = max_min_np(np.concatenate(psnr1_list,axis=0))
            print(gray_psnr1)
            save_roc_auc_plot_img('',gray_psnr1, video_label)


            gray_frame_list = np.concatenate(gray_frame_list, axis=0)
            optical_frame_list = np.concatenate(optical_frame_list, axis=0)
            gray_frame_label_list = np.concatenate(gray_frame_label_list, axis=0)
            optical_frame_label_list = np.concatenate(optical_frame_label_list, axis=0)
            gray_frame_list = np.concatenate([gray_frame_list, gray_frame_label_list], axis=2)
            optical_frame_list = np.concatenate([optical_frame_list, optical_frame_label_list], axis=2)
            save_batch_images(gray_frame_list,self.gray_img_save_path,'test_gray.jpg')
        return
    
    def fetch_net_loss(self, sess, batch_data):
        train_loss = {}
        gray_mse, disc_loss,generator_loss,discriminator_loss,tmp_sum = sess.run([self.gray_loss_sequences_frame_mean, self.disc_loss_sequences_frame_mean, self.generator_loss, self.discriminator_loss, self.summary_merged], feed_dict={self.train_in_ph : batch_data,self.phase : False})

        train_loss['generator'] = generator_loss
        train_loss['discriminator'] = discriminator_loss
        train_loss['gray_mse_loss'] = gray_mse
        train_loss['disc_sequence_loss'] = disc_loss
        train_loss['summary'] = tmp_sum
        return train_loss

    def train_c3d(self,max_iteration = 100000 , restore_tags = True, mse_tag = True ,gan_tag = True, g_iteration =5, d_iteration =1):
        self.g_iteration = g_iteration
        self.d_iteration = d_iteration
        my_multi_train_datasets = multi_train_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = False,crop_size=4, img_size=self.img_size_h)
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
                if mse_tag:
                    sess.run([ self.gray_mse_apply ], feed_dict={self.train_in_ph : batch_data,self.phase : True})
                if gan_tag:
                    for g_iter in range(self.g_iteration):
                        sess.run([ self.gray_apply ], feed_dict={self.train_in_ph : batch_data,self.phase : True})
                    for d_iter in range(self.d_iteration):
                        sess.run([ self.discriminator_apply ], feed_dict={self.train_in_ph : batch_data,self.phase : True})

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
                    print('generator %.8f, discriminator %.8f '%(train_loss['generator'],train_loss['discriminator']  ))
                    print('gray_mse_loss' ,train_loss['gray_mse_loss']  )
                    print('disc_sequence_loss',train_loss['disc_sequence_loss'] )
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
    

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_model = C3D_Running()
# run_model.train_gray_optical_c3d(2000)
# run_model.train_mid_stage_c3d(1000)
run_model.train_c3d(500, restore_tags=False, mse_tag = True ,gan_tag=True, g_iteration = 4, d_iteration =2)
run_model.train_c3d(80000, restore_tags=True, mse_tag = False ,gan_tag=True, g_iteration = 4, d_iteration =1)
# run_model.train_c3d(8000, restore_tags=True, mse_tag = False , g_iteration =2, d_iteration =1)
# run_model.test_video()
# # run_model.test_single_dataset_type2()

# run_model.test_video()