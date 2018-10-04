import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tf_datasets import get_blockimgs_tf_datasets as get_datasets
import time

def build_model(input_tensor, channel = 3, model_name = 'srcnn'):
    # Reset default graph. Keras leaves old ops in the graph,
    # which are ignored for execution but clutter graph
    # visualization in TensorBoard.
    # with tf.Graph().as_default():
    # inputs = input_tensor
    inputs = KL.Input(tensor=(input_tensor))
    x = KL.Conv2D(32, (3, 3), activation='relu', padding="same",data_format= 'channels_last',
                    name="conv1")(inputs)
    x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                    name="conv2")(x)
    # x = KL.Conv2DTranspose(64,(4,4), strides = 2,padding="same",activation='relu',name="deconv")(x)
    outputs = KL.Conv2D(1, (3, 3), activation='sigmoid', padding="same",
                    name="conv4")(x)
    return inputs, outputs

"""
Mask R-CNN
Multi-GPU Support for Keras.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Ideas and a small code snippets from these sources:
https://github.com/fchollet/keras/issues/2436
https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
"""



class ParallelModel(tf.keras.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """

    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                           self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    zipped_inputs = zip(self.inner_model.input_names,
                                        self.inner_model.inputs)
                    inputs = [
                        KL.Lambda(lambda s: input_slices[name][i],
                                  output_shape=lambda s: (None,) + s[1:])(tensor)
                        for name, tensor in zipped_inputs]
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = KL.Lambda(lambda o: tf.add_n(o) / len(outputs), name=name)(outputs)
                else:
                    # Concatenate
                    m = KL.Concatenate(axis=0, name=name)(outputs)
                merged.append(m)
        return merged


if __name__ == "__main__":
    my_datasets = get_datasets(batch_size=32)
    iterator = my_datasets.make_initializable_iterator()
    next_example, netx_label = iterator.get_next()
    next_example.set_shape([None, None, None, 3])
    netx_label.set_shape([None, None, None, 3])
    
    inputs,outputs = build_model(next_example)
    my_keras_model = tf.keras.Model(inputs,outputs)
    mse_loss = tf.reduce_mean(tf.square(outputs - netx_label))
    opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mse_loss)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # my_keras_model.load_weights('./tf_k_weights.h5')
        sess.run(iterator.initializer)
        for counters in range(100):
            losses,_,inputs_arr, label_arr, output_arr = sess.run([mse_loss,opt, next_example, netx_label, outputs])
            print(losses)
            print('max of inputs :', inputs_arr.max())
            print('max of labels :', label_arr.max())
            print('max of outputs :', output_arr.max())
    # tf.keras.models.save_model(my_keras_model,'./tmp_weights.h5py')
        my_keras_model.save_weights('./tf_k_block_rgb2gray_weights.h5')
        my_keras_model.summary()
    # model = KM(inputs,outputs)
    # model = ParallelModel(model,gpu_count=2)
    # my_datasets = get_datasets()
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # model.compile(optimizer=optimizer, loss='mse')
    # # model.summary()
    
    # model.fit(my_datasets.make_one_shot_iterator(), steps_per_epoch=5 , epochs=10)
    # model.summary()