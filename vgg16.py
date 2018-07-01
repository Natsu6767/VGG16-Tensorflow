
"""This is a TensorFlow implementation of VGG16.

Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
(https://arxiv.org/abs/1409.1556)

Explanation on VGGNet can be found in my blog post:
https://mohitjain.me/2018/06/07/vggnet/

@author: Mohit Jain (contact: mohitjain1999(at)yahoo.com)
"""


import tensorflow as tf
import numpy as np


class VGG16(object):

	""" Implementation of VGG16 network """

	def __init__(self, x, keep_prob, num_classes):

		"""Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
        """

        # Parse input arguments into class variables
		self.X = x
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob

		# Call the create function to build the computational graph of AlexNet
		self.create()

	def create(self):

		"""Create the network graph."""

		conv1_1 = conv_layer(self.X, 64, 'conv1_1')
		conv1_2 = conv_layer(conv1_1, 64, 'conv1_2')
		pool1 = max_pool(conv1_2, 'pool1')

		conv2_1 = conv_layer(pool1, 128, 'conv2_1')
		conv2_2 = conv_layer(conv2_1, 128, 'conv2_2')
		pool2 = max_pool(conv2_2, 'pool2')

		conv3_1 = conv_layer(pool2, 256, 'conv3_1')
		conv3_2 = conv_layer(conv3_1, 256, 'conv3_2')
		conv3_3 = conv_layer(conv3_2, 256, 'conv3_3')
		pool3 = max_pool(conv3_3, 'pool3')

		conv4_1 = conv_layer(pool3, 512, 'conv4_1')
		conv4_2 = conv_layer(conv4_1, 512, 'conv4_2')
		conv4_3 = conv_layer(conv4_2, 512, 'conv4_3')
		pool4 = max_pool(conv4_3, 'pool4')

		conv5_1 = conv_layer(pool4, 512, 'conv5_1')
		conv5_2 = conv_layer(conv5_1, 512, 'conv5_2')
		conv5_3 = conv_layer(conv5_2, 512, 'conv5_3')
		pool5 = max_pool(conv5_3, 'pool5')

		flattened = tf.reshape(pool5, [-1, 1 * 1 * 512])
		fc6 = fc_layer(flattened, 1 * 1 * 512, 4096, name = 'fc6')
		dropout6 = dropout(fc6, self.KEEP_PROB)

		fc7 = fc_layer(dropout6, 4096, 4096, name = 'fc7')
		dropout7 = dropout(fc7, self.KEEP_PROB)

		self.fc8 = fc_layer(dropout7, 4096, self.NUM_CLASSES, relu = False, name = 'fc8')



def conv_layer(x, num_filters, name, filter_height = 3, filter_width = 3,
	stride = 1, padding = 'SAME'):

	"""Create a convolution layer."""
	
	# Get number of input channels
	input_channels = int(x.get_shape()[-1])

	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases of the conv layer
		W = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters],
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [num_filters], initializer = tf.constant_initializer(0.0))

		# Perform convolution.
		conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
		# Add the biases.
		z = tf.nn.bias_add(conv, b)
		# Apply ReLu non linearity.
		a = tf.nn.relu(z)

		return a

def fc_layer(x, input_size, output_size, name, relu = True):

	"""Create a fully connected layer."""
	
	with tf.variable_scope(name) as scope:

		# Create tf variables for the weights and biases.
		W = tf.get_variable('weights', shape = [input_size, output_size],
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))

		# Matrix multiply weights and inputs and add biases.
		z = tf.nn.bias_add(tf.matmul(x, W), b)

		if relu:
			# Apply ReLu non linearity.
			a = tf.nn.relu(z)
			return a

		else:
			return z

def max_pool(x, name, filter_height = 2, filter_width = 2,
	stride = 2, padding = 'VALID'):

	"""Create a max pooling layer."""

	return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
		strides = [1, stride, stride, 1], padding = padding, name = name)

def dropout(x, keep_prob):

	"""Create a dropout layer."""

	return tf.nn.dropout(x, keep_prob = keep_prob)
