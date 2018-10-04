# import tensorflow as tf
# import numpy as np
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


# with tf.Session() as sess:
#     cc = sess.run(tensor_list)
#     for cc_iter in cc:
#         print(cc_iter.shape)
    # print(cc)

js = 'sadbkcf'
print js[-2:]