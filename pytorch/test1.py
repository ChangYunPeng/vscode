import tensorflow as tf
import numpy as np

attention_hops = 512
x_internal_a = tf.placeholder(dtype=tf.float32,shape=[None, attention_hops,10], name='batches_in')
with tf.variable_scope('extra_loss'):
    I_matrix = tf.Variable(tf.matrix_diag(tf.ones(shape=(attention_hops))), trainable=False)
    extra_loss = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), I_matrix)))

    I_matrix = tf.matrix_diag(tf.ones(shape=(tf.shape(x_internal_a)[0],attention_hops)))
    extra_loss_2 = tf.norm((tf.subtract(tf.matmul(x_internal_a,tf.transpose(x_internal_a,perm=[0,2,1])), I_matrix)))


input_np = np.random.random_sample((1, attention_hops, 10))


input_np_2 = np.concatenate([input_np,input_np], axis=0)
input_np_3 = np.concatenate([input_np,input_np,input_np], axis=0)
print(input_np.shape)
# a = tf.constant(np.arange(1, 13, dtype=np.int32),
#                 shape=[2, 2, 3])
# tensor_list = []
# tensor_list.append(tf.constant(np.arange(1, 13, dtype=np.int32),shape=[8, 8, 3]))
# tensor_list.append(tf.constant(np.arange(1, 13, dtype=np.int32),shape=[4, 4, 3]))
# tensor_list.append(tf.constant(np.arange(1, 13, dtype=np.int32),shape=[2, 2, 3]))

# tensor_list = tensor_list + [tensor_list[0]]

# # 3-D tensor `b`
# # [[[13, 14],
# #   [15, 16],
# #   [17, 18]],
# #  [[19, 20],
# #   [21, 22],
# #   [23, 24]]]
# b = tf.constant(np.arange(13, 25, dtype=np.int32),
#                 shape=[2, 3, 2])

# # `a` * `b`
# # [[[ 94, 100],
# #   [229, 244]],
# #  [[508, 532],
# #   [697, 730]]]
# c = tf.matmul(a, b)

# json_points_input = []
# json_cls_input = []
# json_points_input.append('s')
# json_points_input.append('s')
# json_points_input.append('s')
# json_cls_input.append('s')
# json_cls_input.append('s')
# json_cls_input.append('s')
# for i,(json_points_list,json_cls_list) in enumerate(zip(json_points_input,json_cls_input)):
#             print (i,(json_points_list,json_cls_list))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cc1,cc12 = sess.run([extra_loss,extra_loss_2],feed_dict={x_internal_a:input_np})
    cc2,cc22 = sess.run([extra_loss,extra_loss_2],feed_dict={x_internal_a:input_np_2})
    cc3,cc23 = sess.run([extra_loss,extra_loss_2],feed_dict={x_internal_a:input_np_3})
    print(cc1,cc12)
    print(cc2,cc22)
    print(cc3,cc23)
