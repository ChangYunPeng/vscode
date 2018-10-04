import tensorflow as tf
import cv2
import os


def get_img_path_list(dataset_path):
    img_list_path = os.listdir(dataset_path)
    img_path_list = []
    for img_iter in img_list_path:
        img_path_list.append(os.path.join(dataset_path, img_iter))

    # img_path_list = os.path.join(dataset_path,img_list_path)
    return img_path_list

filenames = get_img_path_list('/home/room304/TB/DATASET/train2017')
print(filenames[0])
num_epochs = 10

def _parse_function(filename):
  print(filename)
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_cropped = tf.image.resize_image_with_crop_or_pad(image_decoded,128,128)
  image_resized = tf.image.resize_images(image_cropped, [64, 64])
  return image_resized, image_cropped 

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
# labels = [0, 37, 29, 1, ...]
labels = filenames

dataset = tf.data.Dataset.from_tensor_slices(filenames)
# dataset = dataset.map(
#     lambda filename, label: tuple(tf.py_func(
#         _read_py_function, [filename, label], [tf.uint8, label.dtype])))
# dataset = dataset.map(_resize_function)
dataset = dataset.map(_parse_function)

dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(16)
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator()

# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()

# next_example, netx_label = iterator.get_next()
netx_inputs, next_labels = iterator.get_next()

with tf.Session() as sess:
    for counter in range(10000):
        exampel_iter1,exampel_iter2 = sess.run([netx_inputs, next_labels])
        print(counter)
        print(exampel_iter1.shape)
        print(exampel_iter2.shape)
        # print(label_iter)