
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util
from tensorflow.python.ops import rnn
@register_make_test_function("make_dynamic_rnn_tests")
@test_util.enable_control_flow_v2
def make_dynamic_rnn_tests(options):
  test_parameters = [
      {
          "dtype": [tf.float32],
          "num_batches": [4, 2],
          "time_step_size": [4, 3],
          "input_vec_size": [3, 2],
          "num_cells": [4, 2],
      },
  ]
  def build_graph(parameters):
    num_batches = parameters["num_batches"]
    time_step_size = parameters["time_step_size"]
    input_vec_size = parameters["input_vec_size"]
    num_cells = parameters["num_cells"]
    input_shape = (num_batches, time_step_size, input_vec_size)
    input_tensor = tf.placeholder(dtype=parameters["dtype"], shape=input_shape)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_cells, activation=tf.nn.relu)
    output, _ = rnn.dynamic_rnn(
        lstm_cell, input_tensor, dtype=parameters["dtype"])
    return [input_tensor], [output]
  def build_inputs(parameters, sess, inputs, outputs):
    sess.run(tf.global_variables_initializer())
    num_batches = parameters["num_batches"]
    time_step_size = parameters["time_step_size"]
    input_vec_size = parameters["input_vec_size"]
    input_shape = (num_batches, time_step_size, input_vec_size)
    input_value = create_tensor_data(parameters["dtype"], input_shape)
    output_values = sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))
    return [input_value], output_values
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      use_frozen_graph=True)
