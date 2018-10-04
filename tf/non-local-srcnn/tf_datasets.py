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

def get_gray_rgb_block_path_list(gray_dataset_path,rgb_dataset_path):
    img_list_path = os.listdir(gray_dataset_path)
    gray_img_path_list = []
    rgb_img_path_list = []
    for img_iter in img_list_path:
        gray_img_path_list.append(os.path.join(gray_dataset_path, img_iter))
        rgb_img_path_list.append(os.path.join(rgb_dataset_path, img_iter))

    # img_path_list = os.path.join(dataset_path,img_list_path)
    return gray_img_path_list, rgb_img_path_list

def _parse_function(filename):
  print(filename)
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_cropped = tf.image.resize_image_with_crop_or_pad(image_decoded,128,128)
  image_cropped = tf.image.convert_image_dtype(image_cropped,tf.float32)

  image_resized = tf.image.resize_images(image_cropped, [64, 64])
  image_resized = tf.image.resize_images(image_resized, [128, 128])

  image_cropped = tf.image.convert_image_dtype(image_cropped,tf.float32)
  image_resized = tf.image.convert_image_dtype(image_resized,tf.float32)
  return image_resized, image_cropped 

def _parse_gray_rgb_function(gray_filename, rgb_filename):
  
  image_string = tf.read_file(gray_filename)
  image_decoded = tf.image.decode_png(image_string)
  image_decoded = tf.image.convert_image_dtype(image_decoded,tf.float32)
  image_resized = tf.image.resize_images(image_decoded, [256, 256], method=tf.image.ResizeMethod.BICUBIC)
  img_gray = tf.image.convert_image_dtype(image_resized,tf.float32)

  image_string = tf.read_file(rgb_filename)
  image_decoded = tf.image.decode_png(image_string)
#   image_resized = tf.image.resize_images(image_cropped, [256, 256], method=ResizeMethod.BICUBIC)
  img_rgb = tf.image.convert_image_dtype(image_decoded,tf.float32)

  return img_rgb, img_gray  

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

def get_tf_datasets(datasets_path = '/home/room304/TB/TB/DATASET/train2017', batch_size = 128, num_epochs = 10):
    # datasets_path = '/home/room304/TB/DATASET/train2017'
    filenames = get_img_path_list(datasets_path)
    print(filenames[0])
    # num_epochs = 10
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
   
    dataset = dataset.map(_parse_function)

    dataset = dataset.shuffle(buffer_size=batch_size*20)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset

def get_blockimgs_tf_datasets(gray_datasets_path = '/home/room304/TB/DATASET/GF/GF2/png/gray_block',rgb_datasets_path = '/home/room304/TB/DATASET/GF/GF2/png/rgb_block', batch_size = 128, num_epochs = 10):

    gray_filenames, rgb_filenames = get_gray_rgb_block_path_list(gray_datasets_path, rgb_datasets_path)
   
    dataset = tf.data.Dataset.from_tensor_slices((gray_filenames, rgb_filenames))
   
    dataset = dataset.map(_parse_gray_rgb_function)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset

def get_blockimgs_tf_datasets_from_pathlist( gray_filenames, rgb_filenames, batch_size = 128, num_epochs = 10):
    # datasets_path = '/home/room304/TB/DATASET/train2017'
    # gray_filenames, rgb_filenames = get_gray_rgb_block_path_list(gray_datasets_path, rgb_datasets_path)
   
    dataset = tf.data.Dataset.from_tensor_slices((gray_filenames, rgb_filenames))
   
    dataset = dataset.map(_parse_gray_rgb_function)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset

if __name__ == "__main__":

    filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg"]
    # labels = [0, 37, 29, 1, ...]
    # labels = filenames

    # dataset = tf.data.Dataset.from_tensor_slices(filenames)
    # # dataset = dataset.map(
    # #     lambda filename, label: tuple(tf.py_func(
    # #         _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    # # dataset = dataset.map(_resize_function)
    # dataset = dataset.map(_parse_function)

    # dataset = dataset.shuffle(buffer_size=1000)
    # dataset = dataset.batch(16)
    # dataset = dataset.repeat(num_epochs)
    # iterator = dataset.make_one_shot_iterator()

    # # iterator = dataset.make_initializable_iterator()
    # sess.run(iterator.initializer)
    # # next_element = iterator.get_next()

    # # next_example, netx_label = iterator.get_next()
    # netx_inputs, next_labels = iterator.get_next()

    # with tf.Session() as sess:
    #     for counter in range(10000):
    #         exampel_iter1,exampel_iter2 = sess.run([netx_inputs, next_labels])
    #         print(counter)
    #         print(exampel_iter1.shape)
    #         print(exampel_iter2.shape)
            # print(label_iter)