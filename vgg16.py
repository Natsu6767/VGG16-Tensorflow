import tensorflow as tf
import numpy as np


class VGG16(object):

	""" Implementation of VGG16 network """


def conv_layer(x, filter_height, filter_width,
	num_filters, stride, name, padding = 'SAME'):

	input_channels = int(x.get_shape()[-1])

	with tf.variable_scope(name) as scope:

		W = tf.get_variable('weights', shape = [filter_height, filter_width],
			input_channels, initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

		b = tf.get_variable('biases', shape = [num_filters], initializer = tf.constant_initializer(0.0))

		conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
		z = tf.nn.bias_add(conv, b)
		a = tf.nn.relu(z)

		return a

def fc_layer(x, input_size, output_size, name, relu = True):

		with tf.variable_scope(name) as scope:

			W = tf.get_variable('weights', shape = [input_size, output_size],
				initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

			b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))

			z = tf.nn.bias_add(tf.matmul(x, W), b)

			if relu:
				a = tf.nn.relu(z)
				return a

			else:
				return z

def max_pool(x, name, filter_height = 2, filter_width = 2
	stride = 2, padding = 'VALID'):

	return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
		strides = [1, stride, stride, 1], padding = padding, name = name)

def dropout(x, keep_prob):

	return tf.nn.dropout(x, keep_prob = keep_prob)
