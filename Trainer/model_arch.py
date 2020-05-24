from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.compat.v1.layer


def cnn_model(img, mode, hparams):
    ksize1 = hparams.get("ksize1", 5)
    ksize2 = hparams.get("ksize2", 5)
    nfil1 = hparams.get("nfil1", 10)
    nfil2 = hparams.get("nfil2", 20)
    dprob = hparams.get("dprob", 0.25)

    c1 = tf.layers.conv2d(inputs = img, filters = nfil1,
                          kernel_size = ksize1, strides = 1,
                          padding = "same", activation = tf.nn.relu) # shape = (batch_size, HEIGHT, WIDTH, nfil1)
    
    p1 = tf.layers.max_pooling2d(inputs = c1, pool_size = 2, strides = 2) # shape = (batch_size, HEIGHT // 2, WIDTH // 2, nfil1)
    
    c2 = tf.layers.conv2d(inputs = p1, filters = nfil2,
                          kernel_size = ksize2, strides = 1, 
                          padding = "same", activation = tf.nn.relu) # shape = (batch_size, HEIGHT // 2, WIDTH // 2, nfil2)
    
    p2 = tf.layers.max_pooling2d(inputs = c2, pool_size = 2, strides = 2) # shape = (batch_size, HEIGHT // 4, WIDTH // 4, nfil2)

    outlen = p2.shape[1] * p2.shape[2] * p2.shape[3] # HEIGHT // 4 * WIDTH // 4 * nfil2
    p2flat = tf.reshape(tensor = p2, shape = [-1, outlen]) # shape = (batch_size, HEIGHT // 4 * WIDTH // 4 * nfil2)

    # Apply batch normalization
    if hparams["batch_norm"]:
        h3 = tf.layers.dense(inputs = p2flat, units = 300, activation = None)
        h3 = tf.layers.batch_normalization(inputs = h3, training = (mode == tf.estimator.ModeKeys.TRAIN)) # only batchnorm when training
        h3 = tf.nn.relu(features = h3)
    else:  
        h3 = tf.layers.dense(inputs = p2flat, units = 300, activation = tf.nn.relu)
  
    # Apply dropout
    h3d = tf.layers.dropout(inputs = h3, rate = dprob, training = (mode == tf.estimator.ModeKeys.TRAIN))

    ylogits = tf.layers.dense(inputs = h3d, units = NCLASSES, activation = None)
  
    # Apply batch normalization once more
    if hparams["batch_norm"]:
        ylogits = tf.layers.batch_normalization(inputs = ylogits, training = (mode == tf.estimator.ModeKeys.TRAIN))

    return ylogits, NCLASSES
