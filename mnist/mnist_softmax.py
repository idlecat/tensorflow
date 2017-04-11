# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  hidden_num = FLAGS.hidden_num
  x = tf.placeholder(tf.float32, [None, 784])
  W = weight_variable([784, hidden_num])
  b = bias_variable([hidden_num])
  z = tf.matmul(x, W) + b
  if FLAGS.use_sigmod:
    h = tf.sigmoid(z)
  else:
    h = tf.nn.relu(z)
  Wh2c = weight_variable([hidden_num, 10])
  Bh2c = bias_variable([10])
  y = tf.matmul(h, Wh2c) + Bh2c

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
  # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  train_step = tf.train.MomentumOptimizer(0.5, 0.01).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(FLAGS.iter):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  parser.add_argument('--iter', type=int, default=1000,
                      help='iteration number')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='batch size')
  parser.add_argument('--hidden_num', type=int, default=10,
                      help='hidden num')
  parser.add_argument('--use_sigmod', type=bool, default=False,
                      help='Use sigmod instead of relu')
  FLAGS = parser.parse_args()
  tf.app.run()
