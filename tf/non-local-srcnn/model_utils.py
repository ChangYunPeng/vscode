import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from tf_datasets import get_tf_datasets
import datetime

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
        g_x = tf.reshape(g, [batchsize,out_channels, -1])
        
        g_x = tf.transpose(g_x, [0,2,1])

        theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
        theta_x = tf.transpose(theta_x, [0,2,1])
        phi_x = tf.reshape(phi, [batchsize, out_channels, -1])

        f = tf.matmul(theta_x, phi_x)
        # ???
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        y = tf.reshape(y, [batchsize, height, width, out_channels])
        with tf.variable_scope('w') as scope:
            w_y = slim.conv2d(y, in_channels, [1,1], stride=1, scope='w')
            if is_bn:
                w_y = slim.batch_norm(w_y)
        z = input_x + w_y
        return z

def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    # print(I.get_shape().as_list())
    X = tf.reshape(I, (bsize, a, b, r,r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat(axis = 2, values = [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(X,b,1)  # b, [bsize, a*r, r]
    X = tf.concat(axis = 2, values = [tf.squeeze(x) for x in X])  #bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
  # Main OP that you can arbitrarily use in you tensorflow code

  if color:
      bsize, a, b, c = X.get_shape().as_list()
      Xc = tf.split(X, int(c/(r*r)), 3)
      X = tf.concat(axis = 3, values = [_phase_shift(x, r) for x in Xc])
  else:
      X = _phase_shift(X, r)
  return X

def EncoderLayerBlock(input_x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock',phrase = True):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        x = slim.conv2d(input_x, out_channels, [3,3], stride=1, scope='l1')
        x = slim.conv2d( x, out_channels, [5,5], stride=2, scope='l2')
        if is_bn :
            x = slim.batch_norm(x,is_training=phrase,scope='bc')
        return x

def DecoderLayerBlock(input_x, out_channels, up_sample=True, is_bn=True, scope='NonLocalBlock',phrase = True):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        x = PS(input_x,2,color=True)
        # x = slim.conv2d(input_x, out_channels, [3,3], stride=1, scope='l1')
        x = slim.conv2d( x, out_channels, [3,3], stride=1, scope='l2')

        if is_bn :
            x = slim.batch_norm(x,is_training=phrase,scope='bc')
        return x
    
class NonLocalNet:
    model_name = 'NonLocalNet.model'
    def __init__(self,
                config=None,
                sess=None,
                batchsize=32,
                input_height=28,
                input_width=28,
                input_channels=1,
                num_class=10):
        self.config = config
        self.batchsize =batchsize
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.lr = 1e-3
    
    def Net(self, input_x, is_training = True, scope='Nets'):
        batchsize, height, width, in_channels = input_x.get_shape().as_list()
        with tf.variable_scope(scope) as scope:
            with slim.arg_scope([slim.conv2d],
                                activation_fn = None,
                                normalizer_fn = None,
                                weights_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                weights_regularizer = None):
                with tf.name_scope('convolution') as sc_cnv:
                    with tf.name_scope('encoder1') as sc_cnv:
                        e1 = EncoderLayerBlock(input_x,1024,sub_sample=True,is_bn=True,scope='e1',phrase=is_training)
                    
                    with tf.name_scope('encoder2') as sc_cnv:
                        e2 = EncoderLayerBlock(e1,1024,sub_sample=True,is_bn=True,scope='e2',phrase=is_training)
                    
                    with tf.name_scope('encoder3') as sc_cnv:
                        e3 = EncoderLayerBlock(e2,1024,sub_sample=True,is_bn=True,scope='e3',phrase=is_training)
                    
                    with tf.name_scope('non-local') as sc_cnv:
                        nl = NonLocalBlock(e3, 1024, sub_sample=False, is_bn=True, scope='NonLocalBlock')
                    
                    with tf.name_scope('decoder1') as sc_cnv:
                        d1 = DecoderLayerBlock(e3,1024,up_sample=True,is_bn=True,scope='d1',phrase=is_training) + e2
                    
                    with tf.name_scope('decoder2') as sc_cnv:
                        d2 = DecoderLayerBlock(d1,1024,up_sample=True,is_bn=True,scope='d2',phrase=is_training) + e1
                    
                    with tf.name_scope('decoder3') as sc_cnv:
                        d3 = DecoderLayerBlock(d2,3,up_sample=True,is_bn=True,scope='d3',phrase=is_training)
                        out = 0.00001*tf.nn.tanh(d3,name = 'out') + input_x
                        # out = input_x
                    
        return out

    def build_model(self, input_images, input_labels):
        # mnist size
        # self.image_shape = [self.input_height*self.input_width*self.input_channels]
        # self.label_shape = [self.num_class]
        # input images & labels
        self.input_images = input_images
        self.input_labels = input_labels
       
        # prediction
        pred_outputs = self.Net(self.input_images)
        self.pred_outputs = pred_outputs
        # loss function
        self.loss = mse_loss = tf.reduce_mean(tf.square(pred_outputs - input_labels))
        # AdamOptimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        # accuracy rate
        # pred_outputs = tf.clip_by_value()
        self.psnr = tf.reduce_mean(tf.image.psnr(pred_outputs,input_labels,max_val=1.0))
        self.ssim = tf.reduce_mean(tf.image.ssim(pred_outputs,input_labels,max_val=1.0))



        # self.accuracy_counter = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred_softmax,1), tf.argmax(self.input_labels,1)), tf.float32))
        # self.accuracy = self.accuracy_counter/self.batchsize
        # add summary
        self.loss_summary = tf.summary.scalar('cross entropy loss', self.loss)
        # self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.summaries = tf.summary.merge_all()
        # self.summary_writer = tf.summary.FileWriter('./{}/{}'.format(self.config.log_dir, self.config.datasets), self.sess.graph)
        # save model
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=2)
    
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    nonlocalnet = NonLocalNet()

    train_num = 100000
    save_iter = 20
    batch_size = 8
    model_save_path = '/home/room304/storage/CODE_LOG/non_local/save_model'
    # input_x = tf.Variable(tf.random_normal([2,128,128,3]))
    # softmax = nonlocalnet.Net(input_x)
    # softmax_sum = tf.reduce_sum(softmax, -1)

    my_datasets = get_tf_datasets( batch_size=batch_size, num_epochs = 1000)
    iterator = my_datasets.make_initializable_iterator()
    next_example, netx_label = iterator.get_next()
    next_example.set_shape([batch_size, None, None, 3])
    netx_label.set_shape([batch_size, None, None, 3])
    nonlocalnet.build_model(next_example, netx_label)
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        for train_idx in range(1, train_num+1):
            if train_idx % (save_iter) == 0:
                print('%d-th'%train_idx,datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
                em_in, em_lab, em_out, psnr,ssim,loss = sess.run([next_example, netx_label,nonlocalnet.pred_outputs, nonlocalnet.psnr,nonlocalnet.ssim, nonlocalnet.loss])

                print('input max : ', em_in.max())
                print('input min : ', em_in.min())
                print('label max : ', em_lab.max())
                print('label min : ', em_lab.min())
                print('output max : ', em_out.max())
                print('output min : ', em_out.min())

                print('psnr:%.15f \n ssim:%.15f \n loss:%.15f \n'%(psnr,ssim,loss) )
                nonlocalnet.saver.save(sess,model_save_path)
            else :
                sess.run(nonlocalnet.optim)

        print(sess.run(nonlocalnet.psnr))
        print(sess.run(nonlocalnet.ssim))