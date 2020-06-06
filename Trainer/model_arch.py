# import the necessary packages
import tensorflow as tf
NCLASSES = 2
IMAGE_SIZE = [192, 192]

class FireEye:
    def build():
        bnmomemtum=0.9
        def fire(x, squeeze, expand):
            y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
            y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
            y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)
            y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1)
            y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)
            y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3)
            return tf.keras.layers.concatenate([y1, y3])

        def fire_module(squeeze, expand):
            return lambda x: fire(x, squeeze, expand)

        x = tf.keras.layers.Input(shape=[*IMAGE_SIZE, 3]) # input is 192x192 pixels RGB

        y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(x)
        y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
        y = fire_module(24, 48)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = fire_module(48, 96)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = fire_module(64, 128)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = fire_module(48, 96)(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
        y = fire_module(24, 48)(y)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Dense(2, activation='softmax')(y)
        model = tf.keras.Model(x, y)
        return model
