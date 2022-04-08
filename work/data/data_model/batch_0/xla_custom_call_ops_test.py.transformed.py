
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
class XlaCustomCallOpTest(xla_test.XLATestCase):
  def testXlaCustomCallOp(self):
    with ops.device('device:{}:0'.format(self.device)):
      def f(x, y):
        return xla.custom_call(
            args=(x, y),
            target_name='my_call',
            dtype=dtypes.int32,
            shape=(3, 4, 5),
            backend_config='my_backend_config')
      compiled_f = def_function.function(f, jit_compile=True)
      x = random_ops.random_normal([1, 2, 3], dtype=dtypes.float32)
      y = random_ops.random_normal([], dtype=dtypes.float32)
      hlo = compiled_f.experimental_get_compiler_ir(x, y)(stage='hlo')
      self.assertIn('s32[3,4,5]{2,1,0} custom-call(f32[1,2,3]{2,1,0}', hlo)
      self.assertIn('custom_call_target="my_call"', hlo)
      self.assertIn('backend_config="my_backend_config"', hlo)
if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
