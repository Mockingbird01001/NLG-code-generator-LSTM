
from tensorflow.python import Session
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.training.tracking import tracking
def GetGraph(input1, input2, var):
  add = input1 + var
  mul = input1 * add
  add = mul + add
  add = add + input2
  out = array_ops.identity(add, name="output")
  return out
def GenerateModelV2(tf_saved_model_dir, tftrt_saved_model_dir):
  class SimpleModel(tracking.AutoTrackable):
    def __init__(self):
      self.v = None
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32)
    ])
    def run(self, input1, input2):
      if self.v is None:
        self.v = variables.Variable([[[1.0]]], dtype=dtypes.float32)
      return GetGraph(input1, input2, self.v)
  root = SimpleModel()
  save(
      root,
      tf_saved_model_dir,
      {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: root.run})
  converter = trt_convert.TrtGraphConverterV2(
      input_saved_model_dir=tf_saved_model_dir)
  converter.convert()
  converter.save(tftrt_saved_model_dir)
def GenerateModelV1(tf_saved_model_dir, tftrt_saved_model_dir):
  def SimpleModel():
    def GraphFn():
      input1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      input2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      var = variables.Variable([[[1.0]]], dtype=dtypes.float32, name="v1")
      out = GetGraph(input1, input2, var)
      return g, var, input1, input2, out
    g = ops.Graph()
    with g.as_default():
      return GraphFn()
  g, var, input1, input2, out = SimpleModel()
  signature_def = signature_def_utils.build_signature_def(
      inputs={
          "input1": utils.build_tensor_info(input1),
          "input2": utils.build_tensor_info(input2)
      },
      outputs={"output": utils.build_tensor_info(out)},
      method_name=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
  saved_model_builder = builder.SavedModelBuilder(tf_saved_model_dir)
  with Session(graph=g) as sess:
    sess.run(var.initializer)
    saved_model_builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        })
  saved_model_builder.save()
  converter = trt_convert.TrtGraphConverter(
      input_saved_model_dir=tf_saved_model_dir, is_dynamic_op=True)
  converter.convert()
  converter.save(tftrt_saved_model_dir)
if __name__ == "__main__":
  GenerateModelV2(
      tf_saved_model_dir="tf_saved_model",
      tftrt_saved_model_dir="tftrt_saved_model")
