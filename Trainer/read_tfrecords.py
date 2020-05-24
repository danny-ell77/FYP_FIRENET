import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE
# train_data_path = 'gs://cloudfire.../fire_dataset/tfrecords-dataset-train/'
# eval_data_path = 'gs://cloudfire.../fire_dataset/tfrecords-dataset-eval/'

TARGET_SIZE = [224, 224, 3]
#tf.compat.v1.enable_eager_execution()

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
    #image = tf.Reshape(image, [TARGET_SIZE])  # for cloud  define the variable with '*'!! e.g. [*TARGET_SIZE, 3]

    class_num = example['class']

    label = example['label']
    height = example['size'][0]
    width = example['size'][1]
    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])
    return  {"image":image}, label


# read from TFRecords. For optimal performance, read from multiple
# TFRecord files at once and set the option experimental_deterministic = False
# to allow order-altering optimizations.

#@tf.function
def input_fn(file_paths, batch_size, mode):

     def load_dataset():
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False

        filenames = tf.io.gfile.glob(file_paths + "*.tfrec")
        dataset4 = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        dataset4 = dataset4.with_options(option_no_order)
        dataset4 = dataset4.map(read_tfrecord, num_parallel_calls=AUTO)
        if mode == tf.estimator.ModeKeys.TRAIN:
             num_epochs = None  # indefinitely
        else:
             num_epochs = 1  # end-of-input after this
        dataset4 = dataset4.shuffle(buffer_size=10 * batch_size)\
                            .repeat(count=num_epochs)\
                            .batch(batch_size=batch_size)  #add prefectch later          
        dataset4 = tf.compat.v1.data.make_one_shot_iterator(dataset4) 
        return dataset4.get_next()
    
     return load_dataset    







