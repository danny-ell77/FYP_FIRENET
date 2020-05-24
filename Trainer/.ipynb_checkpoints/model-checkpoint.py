from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from .model_arch import cnn_model 
from .read_tfrecords import input_fn
AUTO = tf.data.experimental.AUTOTUNE

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
tf.compat.v1.logging.set_verbosity(v=tf.compat.v1.logging.INFO)

CLASSES = [b'Fire', b'Normal']  # do not change, maps to the labels in the data (folder names)


def serving_input_fn():
    # Note: only handles one image at a time
    feature_placeholders = {"image_bytes": tf.placeholder(dtype=tf.float32, shape=[HEIGHT, WIDTH, NUM_CHANNELS])}
    features = {"image": tf.expand_dims(input=feature_placeholders["image_bytes"], axis=0)}
    return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=feature_placeholders)


def image_classifier(features, labels, mode, params):
    model_function = cnn_model
    ylogits, nclasses = model_function(features["image"], mode, params)

    probabilities = tf.nn.softmax(logits=ylogits)
    class_int = tf.cast(x=tf.argmax(input=ylogits, axis=1), dtype=tf.uint8)
    class_str = tf.gather(params=CLASSES, indices=tf.cast(x=class_int, dtype=tf.int32))

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        class_num = np.argmax(np.array(CLASSES) == labels)  # fire => 0(order defined in CLASSES)
        one_hot_class = np.eye(len(CLASSES))[class_num]  # [1, 0] for class #1, fire

        loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(logits=ylogits,labels=one_hot_class)) 
        if mode == tf.estimator.ModeKeys.TRAIN:
            # This is needed for batch normalization, but has no effect otherwise
            update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(control_inputs=update_ops):
                train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=tf.train.get_global_step(),
                    learning_rate=params["learning_rate"],
                    optimizer=params["optimizer"])
            eval_metric_ops = None
        else:
            train_op = None
            eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=class_int)}
    else:
        loss = None
        train_op = None
        eval_metric_ops = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"probabilities": probabilities,
                     "classid": class_int,
                     "class": class_str},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={"classes": tf.estimator.export.PredictOutput(
            {"probabilities": probabilities,
             "classid": class_int,
             "class": class_str})}
    )


def train_and_evaluate(output_dir, hparams):
   # tf.summary.FileWriterCache.clear()  # ensure filewriter cache is clear for TensorBoard events file

    EVAL_INTERVAL = 300  # every 5 minutes

    # Instantiate base estimator class for custom model function
    estimator = tf.estimator.Estimator(
        model_fn=image_classifier,
        params=hparams,
        config=tf.estimator.RunConfig(
            save_checkpoints_secs=EVAL_INTERVAL),
        model_dir=output_dir)

    # Set estimator's train_spec to use train_input_fn and train for so many steps
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
        estimator = estimator,
        train_spec = train_spec,
        eval_spec = eval_spec)
