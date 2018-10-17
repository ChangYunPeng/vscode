import tensorflow as tf
import numpy as np
# 3-D tensor `a`
# [[[ 1,  2,  3],
#   [ 4,  5,  6]],
#  [[ 7,  8,  9],
#   [10, 11, 12]]]
a = tf.constant(np.arange(1, 24, dtype=np.int32),
                shape=[2, 3, 4])

hw_c_a = tf.reshape(a,[2*3,4])
hw_c_a_t = tf.transpose(hw_c_a,[1,0])


c_tra = tf.transpose(a,[2,1,0])
c_rea_re = tf.reshape(c_tra,[4,2*3])

c_h_w = tf.transpose(c_tra,[0,2,1])
c_hw_re = tf.reshape(c_h_w,[4,2*3])


# 3-D tensor `b`
# [[[13, 14],
#   [15, 16],
#   [17, 18]],
#  [[19, 20],
#   [21, 22],
#   [23, 24]]]
# b = tf.constant(np.arange(13, 25, dtype=np.int32),
#                 shape=[2, 3, 2])

# `a` * `b`
# [[[ 94, 100],
#   [229, 244]],
#  [[508, 532],
#   [697, 730]]]
# c = tf.matmul(a, b)

with tf.Session() as sess:
    c,c2, c3 = sess.run([a,hw_c_a, hw_c_a_t])
    print('sess strage 1')
    print(c[1,1,1])
    print(c2)
    print(c3)
    
    c,c2, c3 = sess.run([a,c_tra, c_rea_re])
    print('sess strage 2')
    print(c[1,1,1])
    print(c2)
    print(c3)

    c,c2, c3 = sess.run([a,c_h_w, c_hw_re])
    print('sess strage 3')
    print(c[1,1,1])
    print(c2)
    print(c3)
# print(1*13 + 2*15+ 3*17)