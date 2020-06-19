from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from .keras_squeezenet import FireEye 
from .read_tfrecords import input_fn
AUTO = tf.data.experimental.AUTOTUNE

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

CLASSES = [b'Fire', b'Normal']  # do not change, maps to the labels in the data (folder names)

def serve_preprocess(image_bytes):
    # Decode the image, end up with pixel values that are in the 0, 1 range
    image = tf.io.decode_jpeg(contents = image_bytes, channels = NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image = image, dtype = tf.float32) # 0-1
    image = tf.expand_dims(input = image, axis = 0) # resize_bilinear needs batches
    image = tf.compat.v1.image.resize_bilinear(images = image, size = [HEIGHT, WIDTH], align_corners = False)
    image = tf.squeeze(input = image, axis = 0) # remove batch dimension
    return {"image": image}


def serving_input_fn():
    # Note: only handles one image at a time 
    feature_placeholders = {"image_bytes": tf.compat.v1.placeholder(dtype = tf.string, shape = [])}
    image = serve_preprocess(tf.squeeze(input = feature_placeholders["image_bytes"]))
    features = {"input_1": tf.expand_dims(image["image"], axis = 0)}
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = feature_placeholders)

def train_and_evaluate(output_dir, hparams):
    
    EVAL_INTERVAL = 30
    
    model = FireEye.build()
    
    model.compile(loss = 'binary_crossentropy',
                    optimizer = 'Adam', 
                    metrics = [tf.keras.metrics.CategoricalAccuracy()]) 
    
    keras_estimator = tf.keras.estimator.model_to_estimator(
                                keras_model = model,
                                model_dir = output_dir,
                                config = tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL))

    train_spec = tf.estimator.TrainSpec(
            input_fn(
            hparams['train_data_path'],
            hparams['batch_size'],
            mode = tf.estimator.ModeKeys.TRAIN),
        max_steps = hparams["train_steps"])

    # Create exporter that uses serving_input_fn to create saved_model for serving
    exporter = tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=serving_input_fn)

    # Set estimator's eval_spec to use eval_input_fn and export saved_model
    eval_spec = tf.estimator.EvalSpec(
            input_fn(
            hparams['eval_data_path'],
            hparams['batch_size'],
            mode = tf.estimator.ModeKeys.EVAL),
        steps = None,
        exporters = exporter,
        start_delay_secs = EVAL_INTERVAL,
        throttle_secs = EVAL_INTERVAL)

    # Run train_and_evaluate loop
    tf.estimator.train_and_evaluate(
        estimator = keras_estimator,
        train_spec = train_spec,
        eval_spec = eval_spec)