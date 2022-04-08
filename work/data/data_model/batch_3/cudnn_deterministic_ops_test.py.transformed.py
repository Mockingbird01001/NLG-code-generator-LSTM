
from tensorflow.python.framework import config
from tensorflow.python.kernel_tests.nn_ops import cudnn_deterministic_base
from tensorflow.python.platform import test
ConvolutionTest = cudnn_deterministic_base.ConvolutionTest
if __name__ == '__main__':
  config.enable_op_determinism()
  test.main()
