
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util
from tensorflow.python.ops import rnn
@register_make_test_function("make_static_rnn_with_control_flow_v2_tests")
@test_util.enable_control_flow_v2
def make_static_rnn_with_control_flow_v2_tests(options):
  test_parameters = [
      {
          "dtype": [tf.float32],
          "num_batches": [4],
          "time_step_size": [4],
          "input_vec_size": [3],
          "num_cells": [4],
          "use_sequence_length": [True, False],
      },
  ]
  def build_graph(parameters):
    num_batches = parameters["num_batches"]
    time_step_size = parameters["time_step_size"]
    input_vec_size = parameters["input_vec_size"]
    num_cells = parameters["num_cells"]
    inputs_after_split = []
    for i in range(time_step_size):
      one_timestamp_input = tf.placeholder(
          dtype=parameters["dtype"],
          name="split_{}".format(i),
          shape=[num_batches, input_vec_size])
      inputs_after_split.append(one_timestamp_input)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_cells, activation=tf.nn.relu, state_is_tuple=True)
    sequence_length = None
    if parameters["use_sequence_length"]:
      sequence_length = [
          min(i + 1, time_step_size) for i in range(num_batches)
      ]
    cell_outputs, _ = rnn.static_rnn(
        lstm_cell,
        inputs_after_split,
        dtype=tf.float32,
        sequence_length=sequence_length)
    out = cell_outputs[-1]
    return inputs_after_split, [out]
  def build_inputs(parameters, sess, inputs, outputs):
    with tf.variable_scope("", reuse=True):
      kernel = tf.get_variable("rnn/basic_lstm_cell/kernel")
      bias = tf.get_variable("rnn/basic_lstm_cell/bias")
      kernel_values = create_tensor_data(parameters["dtype"],
                                         [kernel.shape[0], kernel.shape[1]], -1,
                                         1)
      bias_values = create_tensor_data(parameters["dtype"], [bias.shape[0]], 0,
                                       1)
      sess.run(tf.group(kernel.assign(kernel_values), bias.assign(bias_values)))
    num_batches = parameters["num_batches"]
    time_step_size = parameters["time_step_size"]
    input_vec_size = parameters["input_vec_size"]
    input_values = []
    for _ in range(time_step_size):
      tensor_data = create_tensor_data(parameters["dtype"],
                                       [num_batches, input_vec_size], 0, 1)
      input_values.append(tensor_data)
    out = sess.run(outputs, feed_dict=dict(zip(inputs, input_values)))
    return input_values, out
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      use_frozen_graph=True)
