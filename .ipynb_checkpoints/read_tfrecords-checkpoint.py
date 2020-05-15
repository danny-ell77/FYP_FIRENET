import tensorflow as tf
import numpy as np
AUTO = tf.data.experimental.AUTOTUNE
GCS_OUTPUT = 'gs://qwiklabs-gcp-02-c7f1ded04a7e/fire_dataset/tfrecords-jpeg-192x192-2'  # prefix for output file names

TARGET_SIZE = [256, 256]


def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar

        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
        "label": tf.io.FixedLenFeature([], tf.string),  # one bytestring
        "size": tf.io.FixedLenFeature([2], tf.int64),  # two integers
        "one_hot_class": tf.io.VarLenFeature(tf.float32)  # a certain number of floats
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    # FixedLenFeature fields are now ready to use: exmple['size']
    # VarLenFeature fields require additional sparse_to_dense decoding

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.reshape(image, [*TARGET_SIZE, 3])

    class_num = example['class']

    label = example['label']
    height = example['size'][0]
    width = example['size'][1]
    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])
    return image, class_num, label, height, width, one_hot_class


# read from TFRecords. For optimal performance, read from multiple
# TFRecord files at once and set the option experimental_deterministic = False
# to allow order-altering optimizations.

option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False

filenames = tf.io.gfile.glob(GCS_OUTPUT + "*.tfrec")
dataset4 = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
dataset4 = dataset4.with_options(option_no_order)
dataset4 = dataset4.map(read_tfrecord, num_parallel_calls=AUTO)
for image, class_num, label, height, width, one_hot_class in dataset4.take(10):
    print("Image shape {}, {}x{} px, class={} ({:>10}, {})".format(image.numpy().shape, width, height, class_num,label.numpy().decode('utf8'),one_hot_class))