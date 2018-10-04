import tensorflow as tf
import os

def get_img_path_list(dataset_path):
    img_list_path = os.listdir(dataset_path)
    img_path_list = os.path.join(dataset_path,img_list_path)
    return img_path_list

# tf.train.string_input_producer()

filename = get_img_path_list('～/TB/DATASET/train2017')
with tf.Session() as sess:
    # filename = ['A.jpg', 'B.jpg', 'C.jpg']
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=1)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        image_data = sess.run(value)
        print(image_data.shape)
        # with open('read/test_%d.jpg' % i, 'wb') as f:
        #     f.write(image_data)

