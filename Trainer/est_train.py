from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .keras_squeezenet import FireEye 
from .read_tfrecords import input_fn
from . import config

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

LIST_OF_LABELS = "Fire, Normal".split(',')
HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
NCLASSES = 2

def serve_preprocess(image_bytes):
    # Decode the image, end up with pixel values that are in the 0, 1 range
    image = tf.io.decode_jpeg(contents = image_bytes, channels = config.NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image = image, dtype = tf.float32) # 0-1
    image = tf.expand_dims(input = image, axis = 0) # resize_bilinear needs batches
    image = tf.compat.v1.image.resize_bilinear(images = image, size = [config.HEIGHT, config.WIDTH], align_corners = False)
    image = tf.squeeze(input = image, axis = 0) # remove batch dimension
    return {"image": image}

def serving_input_fn():
    # Note: only handles one image at a time 
    feature_placeholders = {"image_bytes": tf.compat.v1.placeholder(dtype = tf.string, shape = [])}
    image = serve_preprocess(tf.squeeze(input = feature_placeholders["image_bytes"]))
    features = {"input_1": tf.expand_dims(image["image"], axis = 0)}
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = feature_placeholders)

def model_fn(features, labels, mode, params):
    """
    The ML architecture is built in Keras and its output tensor is
    used for building the prediction heads
    """
    model = FireEye.build(features)
    
    # Extract the output tensor of the Keras model
    logits = model.layers[1].output

    probabilities = tf.nn.softmax(logits = logits)
    class_int = tf.cast(x = tf.argmax(input = logits, axis = 1), dtype = tf.uint8)
    class_str = tf.gather(params = LIST_OF_LABELS, indices = tf.cast(x = class_int, dtype = tf.int32))
  
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        
        labels =  tf.lookup.KeyValueTensorInitializer(keys = labels, values = tf.constant(value = LIST_OF_LABELS, dtype = tf.string))
    
        loss = tf.reduce_mean(input_tensor = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits,
                                                        labels = tf.one_hot(indices =labels, depth = NCLASSES)))   
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            # This is needed for batch normalization, but has no effect otherwise
            update_ops = tf.compat.v1.get_collection(key = tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(control_inputs = update_ops):
                optimizer = tf.compat.v1.train.AdamOptimizer(params['learning_rate'])
                train_op = optimizer.minimize(loss, global_step = tf.compat.v1.train.get_global_step())
            eval_metric_ops = None
        else:
            train_op = None
            eval_metric_ops =  {'accuracy': tf.compat.v1.metrics.accuracy(labels = labels, predictions = class_int)}
    else:
        loss = None
        train_op = None
        eval_metric_ops = None
 
    return tf.estimator.EstimatorSpec(
        mode = mode,
        predictions = {"probabilities": probabilities, 
                       "classid": class_int, 
                       "class": class_str},
        loss = loss,
        train_op = train_op,
        eval_metric_ops = eval_metric_ops,
        export_outputs = {"classes": tf.estimator.export.PredictOutput(
            {"probabilities": probabilities, 
             "classid": class_int, 
             "class": class_str})}
    )

def train_and_evaluate(output_dir, arg):
    tf.compat.v1.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
        
    # Instantiate base estimator class for custom model function
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        params = arg,
        config = tf.estimator.RunConfig(
            save_checkpoints_secs = config.EVAL_INTERVAL),
            model_dir = output_dir)
    
    # Set estimator's train_spec to use train_input_fn and train for so many steps
    train_spec = tf.estimator.TrainSpec(
            input_fn(
            arg['train_data_path'],
            arg['batch_size'],
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=arg["train_steps"])

    # Create exporter that uses serving_input_fn to create saved_model for serving
    exporter = tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=serving_input_fn)

    # Set estimator's eval_spec to use eval_input_fn and export saved_model
    eval_spec = tf.estimator.EvalSpec(
            input_fn(
            arg['eval_data_path'],
            arg['batch_size'],
            mode=tf.estimator.ModeKeys.EVAL),
        steps=1,
        exporters=exporter,
        start_delay_secs=config.EVAL_INTERVAL,
        throttle_secs=config.EVAL_INTERVAL)

    # Run train_and_evaluate loop
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec)
