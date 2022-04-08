
from tensorflow.python.compiler.mlir import mlir
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.pywrap_mlir import import_graphdef
class MLIRGraphDefImportTest(test.TestCase):
  def testImport(self):
    mlir_module = mlir.convert_graph_def('')
    self.assertIn('func @main', mlir_module)
  def testInvalidPbtxt(self):
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Could not parse input proto'):
      mlir.convert_graph_def('some invalid proto')
  def testGraphDefToTf(self):
    tensor_shape = (10, 10)
    @def_function.function(
        input_signature=(
            tensor_spec.TensorSpec(shape=tensor_shape, dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=tensor_shape, dtype=dtypes.float32),
        ))
    def add_func(lhs, rhs):
      return math_ops.add(lhs, rhs)
    tf_graph_def = add_func.get_concrete_function().graph.as_graph_def()
    mlir_tf = import_graphdef(
        tf_graph_def,
        "tf-standard-pipeline",
        False,
        input_names=["lhs", "rhs"],
        input_data_types=["DT_FLOAT", "DT_FLOAT"],
        input_data_shapes=["10,10", "10,10"],
        output_names=["Add"])
    self.assertRegex(
        mlir_tf,
        r"func @main\(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>")
    self.assertRegex(mlir_tf, r'inputs = "lhs,rhs"')
    self.assertRegex(mlir_tf, r'outputs = "Add"')
    mlir_tf = import_graphdef(
        tf_graph_def,
        "tf-standard-pipeline",
        False,
        input_names=["lhs", "rhs"],
        input_data_types=["DT_FLOAT", "DT_FLOAT"],
        input_data_shapes=["", ""],
        output_names=["Add"])
    self.assertRegex(mlir_tf,
                     r"func @main\(%arg0: tensor<f32>, %arg1: tensor<f32>")
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Length of input node array and data type doesn't match"):
      import_graphdef(
          tf_graph_def,
          "tf-standard-pipeline",
          False,
          input_names=["lhs"],
          input_data_types=["DT_FLOAT", "DT_FLOAT"],
          input_data_shapes=["10,10", "10,10"],
          output_names=["Add"])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Dimensions must be equal"):
      import_graphdef(
          tf_graph_def,
          "tf-standard-pipeline",
          False,
          input_names=["lhs", "rhs"],
          input_data_types=["DT_FLOAT", "DT_FLOAT"],
          input_data_shapes=["10,11", "10,10"],
          output_names=["Add"])
class MLIRConcreteFunctionImportTest(test.TestCase):
  @test_util.run_v2_only
  def testImport(self):
    @def_function.function
    def sqr(i):
      return i * i
    concrete_function = sqr.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32))
    mlir_module = mlir.convert_function(concrete_function, show_debug_info=True)
    self.assertRegex(mlir_module, r'func @.*sqr.*\(')
    self.assertRegex(mlir_module, r'callsite\(".*mlir_test.py":')
  @test_util.run_v2_only
  def testImportWithCall(self):
    @def_function.function
    def callee(i):
      return i
    @def_function.function
    def caller(i):
      return callee(i)
    concrete_function = caller.get_concrete_function(
        tensor_spec.TensorSpec(None, dtypes.float32))
    mlir_module = mlir.convert_function(concrete_function)
    self.assertRegex(mlir_module, r'func @.*caller.*\(')
    self.assertRegex(mlir_module, r'func private @.*callee.*\(')
  @test_util.run_v2_only
  def testImportWithControlRet(self):
    @def_function.function
    def logging():
      logging_ops.print_v2('some message')
    concrete_function = logging.get_concrete_function()
    mlir_module = mlir.convert_function(concrete_function, pass_pipeline='')
    self.assertRegex(mlir_module, r'tf\.PrintV2')
    self.assertRegex(mlir_module, r'tf_executor.fetch.*: !tf_executor.control')
if __name__ == '__main__':
  test.main()
