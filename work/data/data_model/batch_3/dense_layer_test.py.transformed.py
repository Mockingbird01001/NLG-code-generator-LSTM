
import os
import numpy as np
from tensorflow.compiler.tests import test_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.compiler.xla import jit
from tensorflow.python.framework import ops
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
jit_scope = jit.experimental_jit_scope
def GetRunMetadataLabels(run_metadata):
  labels = []
  for dev_stats in run_metadata.step_stats.dev_stats:
    for node_stats in dev_stats.node_stats:
      labels.append(node_stats.timeline_label)
  return labels
def InLabels(labels, substr):
  return any(substr in x for x in labels)
class DenseLayerTest(test.TestCase):
  def countXlaOps(self, labels):
    xla_compile_count = sum("XlaCompile(" in x for x in labels)
    xla_run_count = sum("XlaRun(" in x for x in labels)
    self.assertEqual(xla_compile_count, xla_run_count)
    return xla_run_count
  def testDenseLayerAutoJit(self):
    os.environ["TF_XLA_FLAGS"] = (
        "--tf_xla_cpu_global_jit " + os.environ.get("TF_XLA_FLAGS", ""))
    config = config_pb2.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = (
        config_pb2.OptimizerOptions.ON_1)
    with self.session(config=config) as sess:
      x = array_ops.placeholder(shape=[None, None, 3], dtype=np.float32)
      y = layers.dense(x, 3)
      self.evaluate(variables.global_variables_initializer())
      run_metadata = config_pb2.RunMetadata()
      test_utils.RunWithWarmup(
          sess,
          y, {x: np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
    labels = GetRunMetadataLabels(run_metadata)
    self.assertEqual(1, self.countXlaOps(labels))
    self.assertFalse(InLabels(labels, "MatMult"))
  def testDenseLayerJitScopeDefinedShape(self):
    with self.session() as sess:
      x = array_ops.placeholder(shape=[2, 2, 3], dtype=np.float32)
      with jit_scope():
        y = layers.dense(x, 3)
      self.evaluate(variables.global_variables_initializer())
      run_metadata = config_pb2.RunMetadata()
      test_utils.RunWithWarmup(
          sess,
          y, {x: np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
    labels = GetRunMetadataLabels(run_metadata)
    self.assertEqual(1, self.countXlaOps(labels))
  def testDenseLayerJitScopeUndefinedShape(self):
    with self.session() as sess:
      x = array_ops.placeholder(shape=[None, None, 3], dtype=np.float32)
      with jit_scope():
        y = layers.dense(x, 3)
      self.evaluate(variables.global_variables_initializer())
      run_metadata = config_pb2.RunMetadata()
      test_utils.RunWithWarmup(
          sess,
          y, {x: np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])},
          run_metadata=run_metadata,
          options=config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE))
    labels = GetRunMetadataLabels(run_metadata)
    self.assertEqual(1, self.countXlaOps(labels))
    self.assertFalse(InLabels(labels, "MatMult"))
if __name__ == "__main__":
  os.environ["TF_XLA_FLAGS"] = ("--tf_xla_enable_lazy_compilation=true " +
                                os.environ.get("TF_XLA_FLAGS", ""))
  ops.disable_eager_execution()
  test.main()
