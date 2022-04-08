
import os
from absl import app
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import saved_model
def _gen_uninitialized_variable(base_dir):
  class SubModule(module.Module):
    def __init__(self):
      self.uninitialized_variable = resource_variable_ops.UninitializedVariable(
          name="uninitialized_variable", dtype=dtypes.int64)
  class Module(module.Module):
    def __init__(self):
      super(Module, self).__init__()
      self.sub_module = SubModule()
      self.initialized_variable = variables.Variable(
          1.0, name="initialized_variable")
      self.uninitialized_variable = resource_variable_ops.UninitializedVariable(
          name="uninitialized_variable", dtype=dtypes.float32)
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def compute(self, value):
      return self.initialized_variable + value
  to_save = Module()
  saved_model.save(
      to_save, export_dir=os.path.join(base_dir, "UninitializedVariable"))
def _gen_simple_while_loop(base_dir):
  class Module(module.Module):
    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def compute(self, value):
      acc, _ = control_flow_ops.while_loop(
          cond=lambda acc, i: i > 0,
          body=lambda acc, i: (acc + i, i - 1),
          loop_vars=(constant_op.constant(0.0), value))
      return acc
  to_save = Module()
  saved_model.save(
      to_save, export_dir=os.path.join(base_dir, "SimpleWhileLoop"))
def main(args):
  if len(args) != 2:
    raise app.UsageError("Expected one argument (base_dir).")
  _, base_dir = args
  _gen_uninitialized_variable(base_dir)
  _gen_simple_while_loop(base_dir)
if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)
