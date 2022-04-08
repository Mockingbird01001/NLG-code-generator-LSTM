
from concurrent import futures
import numpy as np
from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training.tracking import tracking
class MultiThreadedTest(test.TestCase):
  def setUp(self):
    super(MultiThreadedTest, self).setUp()
    self.pool = futures.ThreadPoolExecutor(max_workers=4)
  def _convert_with_calibration(self):
    class ModelWithAdd(tracking.AutoTrackable):
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='x'),
          tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='y')
      ])
      def add(self, x, y):
        res = math_ops.add(x, y)
        return {'output': res}
    def data_gen():
      for _ in range(255):
        yield {
            'x':
                ops.convert_to_tensor(
                    np.random.uniform(size=(10)).astype('f4')),
            'y':
                ops.convert_to_tensor(
                    np.random.uniform(size=(10)).astype('f4'))
        }
    root = ModelWithAdd()
    temp_path = self.create_tempdir().full_path
    saved_model_save.save(
        root, temp_path, signatures=root.add.get_concrete_function())
    model = quantize_model.quantize(
        temp_path, ['serving_default'], [tag_constants.SERVING],
        optimization_method=quantize_model.OptimizationMethod
        .STATIC_RANGE_QUANT,
        representative_dataset=data_gen)
    return model
  def testMultipleConversionJobsWithCalibration(self):
    with self.pool:
      jobs = []
      for _ in range(10):
        jobs.append(self.pool.submit(self._convert_with_calibration))
      for job in jobs:
        self.assertIsNotNone(job.result())
if __name__ == '__main__':
  test.main()
