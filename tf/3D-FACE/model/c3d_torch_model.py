import sys
sys.path.append('.../anormly_utilize_code')
sys.path.append('../datasets')
sys.path.append('../model')
import os
from datasets_sequence import multi_train_datasets, multi_test_datasets
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from tensorboardX import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =  nn.Conv2d(2, 256, kernel_size=3,padding=True) 
        self.conv2 =  nn.Conv2d(256, 1, kernel_size=3,padding=True) 
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.sigmoid(self.conv2(x))
        
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        return

class c3d_t_m():
    def __init__(self):
        self.model = Net().cuda()
        lr = 1e-3
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.batch_size = 1
        self.video_imgs_num = 4
        self.img_size_h = 0
        self.img_size_w = 0
        self.root_path = '/home/room304/TB/TB/TensorFlow_Saver/ANORMLY/Torch_Split/'
        self.summaries_dir = self.root_path + 'SUMMARY/'
    
    def train(self, max_iteration, GPU_IN_USE):

        my_multi_train_datasets = multi_train_datasets(batch_size = self.batch_size, video_num = self.video_imgs_num, frame_interval = 2, is_frame = True, is_Optical = False,crop_size=4, img_size=self.img_size_h)
        loss_f = nn.MSELoss()
        summaries_dir = self.summaries_dir + 'SINGLE_GPU%d.CPTK' % time.time()
        writer = SummaryWriter(summaries_dir)

        for idx in range(max_iteration):  
            self.optimizer.zero_grad()
            batch_data = my_multi_train_datasets.get_batches()
            # batch_data = np.transpose(batch_data, (0,4,2,3,1))
            batch_data = np.squeeze(batch_data,4)
            # print(batch_data.shape)
            if GPU_IN_USE:
                # temp1 = Variable(ToTensor()(temp1)).view(1,-1,29,29)
                batch_data = Variable(torch.tensor(batch_data,dtype = torch.float)).cuda()
            else :
                batch_data = Variable(torch.tensor(batch_data), requires_grad=True).cpu()
            

            batch_data1 =  Variable(batch_data[:,0:4:2,:,:],requires_grad=True)
            # batch_data1 = np.transpose(batch_data1,(0,3,2,1))
            batch_data2 = Variable(batch_data[:,1:4:2,:,:],requires_grad=False)
            # batch_data2 = np.transpose(batch_data2,(0,3,2,1))


            output1 = self.model(batch_data1)
            output2 = self.model(batch_data2)
            output2 = Variable(output2,requires_grad=False)
            loss = -1.0 * torch.mean((output1 - output2)**2)
            # loss = loss * -1.0
            loss.backward()
            # print('%d\n'%(idx+1), loss.item())
            self.optimizer.step()
            if (idx+1)%200 ==0 :
                writer.add_scalar('loss', loss.item(), idx+1)
                writer.add_image('input1_image', batch_data1[0,0:1,:,:], idx+1)
                writer.add_image('input2_image', batch_data1[0,1:2,:,:], idx+1)
                writer.add_image('input3_image', batch_data2[0,0:1,:,:], idx+1)
                writer.add_image('input4_image', batch_data2[0,1:2,:,:], idx+1)
                writer.add_image('out1_image', output1[0,:,:,:], idx+1)
                writer.add_image('out2_image', output2[0,:,:,:], idx+1)
                writer.add_image('input1_diff_image',  (batch_data1[0,0:1,:,:] - batch_data2[0,0:1,:,:])**2, idx+1)
                writer.add_image('input2_diff_image', (batch_data1[0,1:2,:,:] - batch_data2[0,1:2,:,:])**2, idx+1)
                writer.add_image('out_difference', (output1[0,:,:,:] - output2[0,:,:,:])**2, idx+1)
                # writer.add_image('out2_image', output2[0,:,:,:], idx+1)
                print('%d\n'%(idx+1), loss.item())
                # writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_model = c3d_t_m()
run_model.train(100000, True)