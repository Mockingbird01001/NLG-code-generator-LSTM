
import os
import numpy as np
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import flags
from tensorflow.python.tpu import tpu_strategy_util
FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")
def get_tpu_cluster_resolver():
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver
def get_tpu_strategy():
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  tpu_strategy_util.initialize_tpu_system(resolver)
  return tpu_lib.TPUStrategyV2(resolver)
class ConstOp(test.TestCase):
  def setUp(self):
    super(ConstOp, self).setUp()
    assert "TF_XLA_FLAGS" not in os.environ
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_disable_constant_folding=true"
  def testGiantConst(self):
    strategy = get_tpu_strategy()
    types = {
        dtypes.bool,
        dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
        dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64,
        dtypes.float16, dtypes.bfloat16,
        dtypes.float32, dtypes.float64,
    }
    for dtype in types:
      values = [True if dtype is dtypes.bool else 1]
      if dtype is dtypes.bool:
        values.append(False)
      elif dtype is not dtypes.float64:
        values.extend([dtype.min, dtype.max])
      for value in values:
        @def_function.function
        def train_step():
          def computation():
            const = constant_op.constant(value, dtype=dtype, shape=[1024]*4)
            return const[:1, :1, :1, :1]
          return strategy.run(computation, args=())
        output = strategy.experimental_local_results(train_step())[0]
        expected = np.full((1, 1, 1, 1), value)
        self.assertAllEqual(output, expected)
if __name__ == "__main__":
  test.main()
