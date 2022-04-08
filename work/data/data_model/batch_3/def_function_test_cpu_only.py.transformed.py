
from absl.testing import parameterized
from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
class DefFunctionCpuOnlyTest(test.TestCase, parameterized.TestCase):
  def testJitCompileRaisesExceptionWhenXlaIsUnsupported(self):
    if test.is_built_with_rocm() or test_util.is_xla_enabled():
      return
    with self.assertRaisesRegex(errors.UnimplementedError,
                                'check target linkage'):
      @def_function.function(jit_compile=True)
      def fn(x):
        return x + x
      fn([1, 1, 2, 3])
if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
