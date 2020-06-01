# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer

NCLASSES = 2

class FireEye:
	def build():
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		inputShape = (224, 224, 3)
		chanDim = -1
		model = Sequential([
		# model.add(InputLayer(input_tensor = input_tensor, name = "image"))
		# CONV => RELU => POOL
		tf.keras.layers.SeparableConv2D(16, (7, 7), padding="same", activation='relu', input_shape=inputShape),
		tf.keras.layers.BatchNormalization(axis=chanDim),
		tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # CONV => RELU => POOL
		tf.keras.layers.SeparableConv2D(32, (3, 3), padding="same", activation='relu' ),
		tf.keras.layers.BatchNormalization(axis=chanDim),
		tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
		# (CONV => RELU) * 2 => POOL
		tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", activation='relu'),
		tf.keras.layers.BatchNormalization(axis=chanDim),
		tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", activation='relu'),
		tf.keras.layers.BatchNormalization(axis=chanDim),
		tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
		# first set of FC => RELU layers
		#tf.keras.layers.Flatten(),
		tf.keras.layers.GlobalAveragePooling2D(),
		# softmax classifier
		tf.keras.layers.Dense(NCLASSES, activation='softmax')
        ])
		# return the constructed network architecture
		return model