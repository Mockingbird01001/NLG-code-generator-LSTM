
from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import session
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test
@test_util.with_control_flow_v2
class CondTest(xla_test.XLATestCase):
  def testCondAndTensorArrayInDefun(self):
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      @function.defun
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_ops.cond(
            constant_op.constant(True),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))
        return output.stack()
      output_t = f()
      self.assertAllEqual([5.], self.evaluate(output_t))
      xla_context.Exit()
  def testCondAndTensorArrayInDefun_constFolding(self):
    g = ops.Graph()
    with session.Session(graph=g), g.as_default(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      @function.defun
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_ops.cond(
            constant_op.constant(False),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))
        return output.stack()
      output_t = f()
      self.assertAllEqual([10.], self.evaluate(output_t))
      xla_context.Exit()
  def testCondAndTensorArray_xlaCompile(self):
    self.skipTest("b/127846988")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_ops.cond(
            constant_op.constant(True),
            lambda: ta.write(0, 5.), lambda: ta.write(0, 10.))
        return output.stack()
      output_t, = xla.compile(f)
      self.assertAllEqual([5.], self.evaluate(output_t))
      xla_context.Exit()
  def testCondConstPropagation(self):
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      x = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.int32)
      def if_true():
        return x[p]
      def if_false():
        return 5.
      output = control_flow_ops.cond(
          constant_op.constant(True), if_true, if_false)
      self.assertAllEqual(1.,
                          sess.run(output, feed_dict={
                              x: [0., 1., 2.],
                              p: 1
                          }))
      xla_context.Exit()
  def testCondConstPropagation_xlaCompile(self):
    self.skipTest("b/132430685")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      x = array_ops.placeholder_with_default([0., 1., 2.], shape=[3])
      p = constant_op.constant(1)
      def f():
        def if_true():
          return x[p]
        def if_false():
          return 5.
        return control_flow_ops.cond(
            constant_op.constant(True), if_true, if_false)
      output = xla.compile(f)
      self.assertAllEqual(1., self.evaluate(output))
      xla_context.Exit()
  def testCondConstPropagation_errorMsg(self):
    self.skipTest("b/132430685")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      x = array_ops.placeholder(dtypes.float32)
      p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)
      def if_true():
        return x[:p]
      def if_false():
        return array_ops.fill([p], 5.)
      output = control_flow_ops.cond(
          constant_op.constant(True), if_true, if_false)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must be a compile-time constant"):
        sess.run(
            output, feed_dict={
                x: [0., 1., 2.],
            })
      xla_context.Exit()
  def testCondConstPropagation_errorMsg_xlaCompile(self):
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      x = array_ops.placeholder(dtypes.float32)
      p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)
      condition = math_ops.cast(
          random_ops.random_uniform([], minval=0, maxval=2, dtype=dtypes.int32),
          dtypes.bool)
      def f():
        def if_true():
          return x[:p]
        def if_false():
          return array_ops.fill([p], 5.)
        return control_flow_ops.cond(condition, if_true, if_false)
      output = xla.compile(f)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "must be a compile-time constant"):
        sess.run(
            output, feed_dict={
                x: [0., 1., 2.],
            })
      xla_context.Exit()
  def testSwitchCaseAndTensorArrayInDefun(self):
    self.skipTest("b/127846988")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      @function.defun
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_ops.switch_case(
            constant_op.constant(1), {
                0: lambda: ta.write(0, 5.),
                1: lambda: ta.write(0, 10.),
                2: lambda: ta.write(0, 15.),
            })
        return output.stack()
      output_t = f()
      self.assertAllEqual([10.], self.evaluate(output_t))
      xla_context.Exit()
  def testSwitchCaseAndTensorArray_xlaCompile(self):
    self.skipTest("b/127846988")
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      def f():
        ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
        output = control_flow_ops.switch_case(
            constant_op.constant(1), {
                0: lambda: ta.write(0, 5.),
                1: lambda: ta.write(0, 10.),
                2: lambda: ta.write(0, 15.),
            })
        return output.stack()
      output_t, = xla.compile(f)
      self.assertAllEqual([10.], self.evaluate(output_t))
      xla_context.Exit()
  def testSwitchCaseConstPropagation(self):
    self.skipTest("b/127846988")
    with self.session() as sess, self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      x = array_ops.placeholder(dtypes.float32)
      p = array_ops.placeholder(dtypes.int32)
      def branch0():
        return 5.
      def branch1():
        return 15.
      def branch2():
        return x[p]
      output = control_flow_ops.switch_case(
          constant_op.constant(2), {
              0: branch0,
              1: branch1,
              2: branch2,
          })
      self.assertAllEqual(7.,
                          sess.run(output, feed_dict={
                              x: [0., 1., 7.],
                              p: 2,
                          }))
      xla_context.Exit()
  def testCondNoInputs(self):
    with self.session(), self.test_scope():
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      for pred in True, False:
        cond_out = control_flow_ops.cond(
            array_ops.placeholder_with_default(pred, []),
            lambda: constant_op.constant(2.),
            lambda: constant_op.constant(1.))
        self.assertEqual(int(pred) + 1., self.evaluate(cond_out))
      xla_context.Exit()
if __name__ == '__main__':
  test.main()