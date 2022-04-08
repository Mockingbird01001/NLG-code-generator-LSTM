
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
def make_reduce_tests(reduce_op,
                      min_value=-10,
                      max_value=10,
                      boolean_tensor_only=False,
                      allow_fully_quantize=False):
  def f(options):
    test_parameters = [
        {
            "input_dtype": [tf.float32, tf.int32, tf.int64],
            "input_shape": [[3, 3, 2, 4]],
            "axis": [
                0,
                1,
                2,
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 1, 2],
                [1, 0],
                [2, 0],
                [2, 1],
                [2, 1, 0],
                [2, 0, 1],
                -1,
                -2,
                -3,
                [1, -1],
                [0, -1],
                [-1, 0],
                [-1, -2, -3],
            ],
            "const_axis": [True, False],
            "keepdims": [True, False],
            "fully_quantize": [False],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[1, 8, 8, 3]],
            "axis": [
                0,
                1,
                2,
                3,
                [1, 2],
                [0, 3],
                [1, 2, 3],
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [3, 1, 0, 2],
                [2, 0],
                [3, 0],
                [3, 1],
                [1, 0],
                -1,
                -2,
                -3,
                -4,
                [0, -2],
                [2, 3, 1, 0],
                [3, 1, 2],
                [3, -4],
            ],
            "const_axis": [True, False],
            "keepdims": [True, False],
            "fully_quantize": [False],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1, 8, 8, 3], [3, 2, 4]],
            "const_axis": [False],
            "keepdims": [True, False],
            "fully_quantize": [False],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1, 8, 8, 3], [3, 2, 4]],
            "const_axis": [True],
            "keepdims": [True, False],
            "fully_quantize": [False],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[3, 3, 2, 4]],
            "axis": [
                0,
                1,
                2,
                [0, 1],
                [0, 2],
                [1, 2],
                [0, 1, 2],
                [1, 0],
                [2, 0],
                [2, 1],
                [2, 1, 0],
                [2, 0, 1],
                -1,
                -2,
                -3,
                [1, -1],
                [0, -1],
                [-1, 0],
                [-1, -2, -3],
            ],
            "const_axis": [True],
            "keepdims": [True, False],
            "fully_quantize": [True],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[1, 8, 8, 4], [1, 8, 8, 3]],
            "axis": [
                0, 1, 2, 3, [0], [1], [2], [3], [-1], [-2], [-3], [1, 2],
                [0, 3], [1, 2, 3], [1, 3], [2, 3]
            ],
            "const_axis": [True],
            "keepdims": [True, False],
            "fully_quantize": [True],
        },
        {
            "input_dtype": [tf.float32, tf.int32],
            "input_shape": [[2, 0, 2], [0]],
            "axis": [0],
            "const_axis": [True],
            "keepdims": [True, False],
            "fully_quantize": [False],
        },
    ]
    if not allow_fully_quantize:
      test_parameters = [
          test_parameter for test_parameter in test_parameters
          if True not in test_parameter["fully_quantize"]
      ]
    def build_graph(parameters):
      dtype = parameters["input_dtype"]
      if boolean_tensor_only:
        dtype = tf.bool
      input_tensor = tf.compat.v1.placeholder(
          dtype=dtype, name="input", shape=parameters["input_shape"])
      if parameters["const_axis"]:
        axis = parameters["axis"]
        input_tensors = [input_tensor]
      else:
        if isinstance(parameters["axis"], list):
          shape = [len(parameters["axis"])]
        else:
        axis = tf.compat.v1.placeholder(
            dtype=tf.int32, name="axis", shape=shape)
        input_tensors = [input_tensor, axis]
      out = reduce_op(input_tensor, axis=axis, keepdims=parameters["keepdims"])
      return input_tensors, [out]
    def build_inputs(parameters, sess, inputs, outputs):
      dtype = parameters["input_dtype"]
      if boolean_tensor_only:
        dtype = tf.bool
      values = [
          create_tensor_data(
              dtype,
              parameters["input_shape"],
              min_value=min_value,
              max_value=max_value)
      ]
      if not parameters["const_axis"]:
        values.append(np.array(parameters["axis"]))
      return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))
    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
  return f
@register_make_test_function()
def make_mean_tests(options):
  return make_reduce_tests(
      tf.reduce_mean,
      min_value=-1,
      max_value=1,
      boolean_tensor_only=False,
      allow_fully_quantize=True)(
          options)
@register_make_test_function()
def make_sum_tests(options):
  return make_reduce_tests(
      tf.reduce_sum,
      min_value=-1,
      max_value=1,
      boolean_tensor_only=False,
      allow_fully_quantize=True)(
          options)
@register_make_test_function()
def make_reduce_prod_tests(options):
  return make_reduce_tests(tf.reduce_prod, -2, 2)(options)
@register_make_test_function()
def make_reduce_max_tests(options):
  return make_reduce_tests(
      tf.reduce_max, allow_fully_quantize=True, min_value=-1, max_value=1)(
          options)
@register_make_test_function()
def make_reduce_min_tests(options):
  return make_reduce_tests(
      tf.reduce_min, allow_fully_quantize=True, min_value=-1, max_value=1)(
          options)
@register_make_test_function()
def make_reduce_any_tests(options):
  return make_reduce_tests(tf.reduce_any, boolean_tensor_only=True)(options)
@register_make_test_function()
def make_reduce_all_tests(options):
  return make_reduce_tests(tf.reduce_all, boolean_tensor_only=True)(options)
