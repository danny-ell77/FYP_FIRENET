import tensorflow as tf
import numpy as np
import os
from Train import make_input_fn

AUTOTUNE = tf.data.experimantal.AUTOTUNE
GCS_PATTERN = 'gs://fire_dataset/*/*.jpg'
GCS_OUTPUT = 'gs://fire_dataset/tfrecords-jpeg-192x192-2/fire'  # prefix for output file names
SHARDS = 16
TARGET_SIZE = [192, 192]
CLASSES = [b'fire', b'normal']  # do not change, maps to the labels in the data (folder names)


def _input_fn(filenames, batch_size):
    # Create tf.data.Dataset from filename
    dataset = tf.data.Dataset.list_files(str(filenames / '*/*'))
    parts = tf.strings.splits(filenames, os.sep)
    label = parts[-2]

    dataset = dataset.batch(batch_size=batch_size) \
                     .shuffle(buffer_size=10 * batch_size) \
                     .prefetch(buffer_size=AUTOTUNE)
    return dataset, label


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(tfrec_filewriter, img_bytes, label, height, width):
    class_num = np.argmax(np.array(CLASSES) == label)  # 'roses' => 2 (order defined in CLASSES)
    one_hot_class = np.eye(len(CLASSES))[class_num]  # [0, 0, 1, 0, 0] for class #2, roses

    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "class": _int_feature([class_num]),  # one class in the list

        # additional (not very useful) fields to demonstrate TFRecord writing/reading of different types of data
        "label": _bytestring_feature([label]),  # fixed length (1) list of strings, the text label
        "size": _int_feature([height, width]),  # fixed length (2) list of ints
        "one_hot_class": _float_feature(one_hot_class.tolist())  # variable length  list of floats, n=len(CLASSES)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


print("Writing TFRecords")
for shard, (image, label, height, width) in enumerate(_input_fn()): #find out where the dataset is coming from in tutorial
    # batch size used as shard size here
    shard_size = image.numpy().shape[0]
# good practice to have the number of records in the filename
    filename = GCS_OUTPUT + "{:02d}-{}.tfrec".format(shard, shard_size)

    with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
            example = to_tfrecord(out_file,
                                  image.numpy()[i],  # re-compressed image: already a byte string
                                  label.numpy()[i],
                                  height.numpy()[i],
                                  width.numpy()[i])
            out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))