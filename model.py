import tensorflow as tf
import scipy

x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

def weight_variable(shape):
  initial_value = tf.truncated_normal(shape, stddev=0.1)
  k_weight = tf.Variable(initial_value)
  return k_weight

def bias_variable(shape):
  initial_value = tf.constant(0.1, shape=shape)
  k_bias = tf.Variable(initial_value)
  return k_bias

def conv2d(x, W, stride):
  conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
  return conv

#first convolutional layer
h_conv1 = tf.nn.relu(conv2d(x, weight_variable([5, 5, 3, 24]), 2) + bias_variable([24]))

#second convolutional layer
h_conv2 = tf.nn.relu(conv2d(h_conv1, weight_variable([5, 5, 24, 36]), 2) + bias_variable([36]) )

#third convolutional layer
h_conv3 = tf.nn.relu(conv2d(h_conv2, weight_variable([5, 5, 36, 48]), 2) + bias_variable([48]))

#fourth convolutional layer
h_conv4 = tf.nn.relu(conv2d(h_conv3, weight_variable([3, 3, 48, 64]), 1) + bias_variable([64]))

#fifth convolutional layer
h_conv5 = tf.nn.relu(conv2d(h_conv4, weight_variable([3, 3, 64, 64]), 1) + bias_variable([64]))

h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, weight_variable([1152, 1164])) + bias_variable([1164]))
keep_prob = tf.placeholder(tf.float32)

h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, weight_variable([1164, 100])) + bias_variable([100]))

h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob)
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_dropout, weight_variable([100, 50])) + bias_variable([50]))

h_fc3_dropout = tf.nn.dropout(h_fc3, keep_prob)

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_dropout, weight_variable([50, 10])) + bias_variable([10]))

h_fc4_dropout = tf.nn.dropout(h_fc4, keep_prob)

#Output
y = tf.multiply(tf.atan(tf.matmul(h_fc4_dropout, weight_variable([10, 1])) + bias_variable([1])), 2) 
