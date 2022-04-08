
from absl.testing import parameterized
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class DefunCollectionTest(test.TestCase, parameterized.TestCase):
  @parameterized.named_parameters(
      dict(testcase_name='Defun', function_decorator=function.defun),
      dict(
          testcase_name='DefFunction',
          function_decorator=def_function.function))
  def testCollectionValueAccess(self, function_decorator):
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = 2
        y = 5
        ops.add_to_collection('x', x)
        ops.add_to_collection('y', y)
        @function_decorator
        def fn():
          x_const = constant_op.constant(ops.get_collection('x')[0])
          y_const = constant_op.constant(ops.get_collection('y')[0])
          z = math_ops.add(x_const, y_const)
          ops.add_to_collection('z', 7)
          return z
        self.assertEqual(7, int(self.evaluate(fn())))
        self.assertEqual(ops.get_collection('x'), [2])
        self.assertEqual(ops.get_collection('y'), [5])
        self.assertEqual(ops.get_collection('z'), [])
  @parameterized.named_parameters(
      dict(testcase_name='Defun', function_decorator=function.defun),
      dict(
          testcase_name='DefFunction',
          function_decorator=def_function.function))
  def testCollectionVariableValueAccess(self, function_decorator):
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        v = resource_variable_ops.ResourceVariable(1.0)
        @function_decorator
        def f():
          return v.read_value()
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(1.0, float(self.evaluate(f())))
        self.assertLen(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), 1)
  def testCollectionVariableValueWrite(self):
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        @function.defun
        def f():
          v = resource_variable_ops.ResourceVariable(2.0)
          return v
        _ = f.get_concrete_function()
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(2.0, float(self.evaluate(f())))
        self.assertLen(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), 1)
if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={'CPU': 4}))
  test.main()
