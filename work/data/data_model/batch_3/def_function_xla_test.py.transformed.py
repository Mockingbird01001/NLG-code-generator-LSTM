
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class DefFunctionTests(xla_test.XLATestCase):
  def testVarInitializedInFunction(self):
    with self.test_scope():
      v_holder = []
      @def_function.function
      def add_var(x):
        if not v_holder:
          v = variables.Variable([1., 2.])
          v_holder.append(v)
          already_initialized = variables.Variable(3.)
          with ops.init_scope():
            already_initialized.assign(10.)
          v_holder.append(already_initialized)
        return v_holder[0] + v_holder[1] + x
      self.assertAllClose([13., 14.], add_var(constant_op.constant(2.)))
if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
