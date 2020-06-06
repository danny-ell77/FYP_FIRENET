import tensorflow as tf
import numpy as np
import argparse


HEIGHT = 180
WIDTH = 180
NUM_CHANNELS = 3
AUTO = tf.data.experimental.AUTOTUNE
GCS_PATTERN = 'gs://cloudfire_lyrical-edition-273206/fire_dataset/Eval/*/*.jpg'
GCS_OUTPUT = 'gs://cloudfire_lyrical-edition-273206/fire_dataset/tfrecords-dataset-eval/'
BATCH_SIZE = 20
# SHARDS = # To be determined
TARGET_SIZE = [100, 100]
CLASSES = [b'Fire', b'Normal']  # do not change, maps to the labels in the data (folder names)


nb_images = len(tf.io.gfile.glob(GCS_PATTERN))
#shard_size = math.ceil(nb_images / SHARDS)
#print("Pattern matches {} images which will be rewritten as {} .tfrec files containing {} images each.".format(nb_images, SHARDS, shard_size))



def read_jpeg_and_label(filename, augment=False):
    bits = tf.io.read_file(filename)   # parse  from containing directory
    image = tf.image.decode_jpeg(bits)
     #image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-2]
    return image, label
'''    
    #Augment the data
    image = tf.image.random_crop(value=image, size=[HEIGHT, WIDTH, NUM_CHANNELS])
    image = tf.image.random_flip_left_right(image=image)
    image = tf.image.random_brightness(image=image, max_delta=63.0 / 255.0)
    image = tf.image.random_contrast(image=image, lower=0.2, upper=1.8)
    image = tf.image.random_flip_up_down(image=image) 
'''    
    
    
        
def resize_and_crop_image(image, label):
    w = tf.shape(image)[0]    # Resize and crop using "fill" algorithm:
    h = tf.shape(image)[1]  # always make sure the resulting image
    tw = TARGET_SIZE[1]  # is cut out from the source image so that
    th = TARGET_SIZE[0]   # it fills the TARGET_SIZE entirely with no
    resize_crit = (w * th) / (h * tw)  # black bars and a preserved aspect ratio.
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w * tw / w, h * tw / w]),  # if true
                    lambda: tf.image.resize(image, [w * th / h, h * th / h])  # if false
                    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label


def recompress_image(image, label):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label, height, width

filenames = tf.data.Dataset.list_files(GCS_PATTERN, seed=35155)  # This also shuffles the images
dataset1 = filenames.map(read_jpeg_and_label, num_parallel_calls=AUTO)\
                    .map(resize_and_crop_image, num_parallel_calls=AUTO)\
                    .map(recompress_image, num_parallel_calls=AUTO)\
                    .batch(batch_size=BATCH_SIZE)
        
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(tfrec_filewriter, img_bytes, label, height, width):
    class_num = np.argmax(np.array(CLASSES) == label)  # fire => 0(order defined in CLASSES)
    one_hot_class = np.eye(len(CLASSES))[class_num]  # [1, 0] for class #1, fire

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
for shard, (image, label, height, width) in enumerate(dataset1): # loops through each line in the dataset 
    # batch size used as shard size here
    shard_size = image.numpy().shape[0] # batch size not specified
# good practice to have the number of records in the filename
    filename = GCS_OUTPUT + "shard_{:02d}-{}.tfrec".format(shard, shard_size)

    with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
            example = to_tfrecord(out_file,
                                  image.numpy()[i],  # re-compressed image: already a byte string
                                  label.numpy()[i],
                                  height.numpy()[i],
                                  width.numpy()[i])
            out_file.write(example.SerializeToString())
        print("Wrote files to {} containing {} records".format(filename, shard_size))


        
