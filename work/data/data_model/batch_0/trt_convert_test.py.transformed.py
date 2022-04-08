
import gc
import os
import re
import tempfile
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.compiler.tensorrt import trt_convert
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util.lazy_loader import LazyLoader
_SAVED_MODEL_SIGNATURE_KEY = "mypredict"
gen_trt_ops = LazyLoader(
    "gen_trt_ops", globals(),
    "tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops")
class TrtConvertTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  _TRT_MAX_WORKSPACE_SIZE_BYTES = 2 << 20
  def mkdtemp(self):
    return tempfile.mkdtemp(dir=self.get_temp_dir())
  def testTRTEngineInstanceAvailable(self):
    assert hasattr(TRTEngineInstance(), "serialized_engine")
  def _GetConfigProto(self, rewriter_config=None):
    config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(allow_growth=True))
    if rewriter_config:
      config.graph_options.rewrite_options.CopyFrom(rewriter_config)
    return config
  @classmethod
  def _GetGraph(cls, inp1, inp2, var):
    add = inp1 + var
    mul = inp1 * add
    add = mul + add
    add = add + inp2
    out = array_ops.identity(add, name="output")
    return out
  def _GetModelForV2(self):
    class SimpleModel(tracking.AutoTrackable):
      def __init__(self):
        self.v = None
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[None, 1, 1], dtype=dtypes.float32)
      ])
      def run(self, inp1, inp2):
        if self.v is None:
          self.v = variables.Variable([[[1.0]]], dtype=dtypes.float32)
        return TrtConvertTest._GetGraph(inp1, inp2, self.v)
    return SimpleModel()
  def _GetGraphForV1(self, device):
    def _GraphFn():
      inp1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      inp2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      var = variables.Variable([[[1.0]]], dtype=dtypes.float32, name="v1")
      out = TrtConvertTest._GetGraph(inp1, inp2, var)
      return g, var, inp1, inp2, out
    g = ops.Graph()
    with g.as_default():
      if device:
        with g.device(device):
          return _GraphFn()
      return _GraphFn()
  def _GetGraphDefForV1(self, device):
    g, var, _, _, _ = self._GetGraphForV1(device)
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      graph_def = graph_util.convert_variables_to_constants(
          sess, g.as_graph_def(add_shapes=True), ["output"])
    node_name_to_op = {node.name: node.op for node in graph_def.node}
    self.assertEqual(
        {
            "v1": "Const",
            "add/ReadVariableOp": "Identity",
            "input1": "Placeholder",
            "input2": "Placeholder",
            "add": "AddV2",
            "mul": "Mul",
            "add_1": "AddV2",
            "add_2": "AddV2",
            "output": "Identity"
        }, node_name_to_op)
    return graph_def
  def _WriteInputSavedModelForV1(self, input_saved_model_dir, device):
    g, var, inp1, inp2, out = self._GetGraphForV1(device)
    signature_def = signature_def_utils.build_signature_def(
        inputs={
            "myinput1": utils.build_tensor_info(inp1),
            "myinput2": utils.build_tensor_info(inp2)
        },
        outputs={"myoutput": utils.build_tensor_info(out)},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    saved_model_builder = builder.SavedModelBuilder(input_saved_model_dir)
    with self.session(graph=g, config=self._GetConfigProto()) as sess:
      sess.run(var.initializer)
      saved_model_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING],
          signature_def_map={_SAVED_MODEL_SIGNATURE_KEY: signature_def})
    saved_model_builder.save()
  def _ConvertGraphV1(self,
                      output_saved_model_dir=None,
                      need_calibration=False,
                      max_batch_size=1,
                      minimum_segment_size=3,
                      is_dynamic_op=False,
                      maximum_cached_engines=1,
                      device=None):
    input_saved_model_dir = None
    if output_saved_model_dir:
      input_saved_model_dir = self.mkdtemp()
      self._WriteInputSavedModelForV1(input_saved_model_dir, device)
    if need_calibration:
      is_dynamic_op = True
    if is_dynamic_op:
      max_batch_size = None
    converter = trt_convert.TrtGraphConverter(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY,
        input_graph_def=None
        if input_saved_model_dir else self._GetGraphDefForV1(device),
        nodes_denylist=None if input_saved_model_dir else ["output"],
        max_batch_size=max_batch_size,
        max_workspace_size_bytes=TrtConvertTest._TRT_MAX_WORKSPACE_SIZE_BYTES,
        precision_mode=(trt_convert.TrtPrecisionMode.INT8 if need_calibration
                        else trt_convert.TrtPrecisionMode.FP32),
        minimum_segment_size=minimum_segment_size,
        is_dynamic_op=is_dynamic_op,
        maximum_cached_engines=maximum_cached_engines)
    output_graph_def = converter.convert()
    if need_calibration:
      class CalibrationData(object):
        def __init__(self):
          self._data = 0
        def next(self):
          self._data += 1
          return {"input1:0": [[[self._data]]], "input2:0": [[[self._data]]]}
      output_graph_def = converter.calibrate(
          fetch_names=["output:0"],
          num_runs=10,
          feed_dict_fn=CalibrationData().next)
    if output_saved_model_dir is not None:
      converter.save(output_saved_model_dir=output_saved_model_dir)
    return output_graph_def
  def _MayRemoveGraphSequenceNumber(self, name):
    prefix = re.search(r"TRTEngineOp_\d+_", name)
    if prefix and name.startswith(prefix.group(0)):
      parts = name.split("_", maxsplit=2)
      assert len(parts) == 3
      return parts[0] + "_" + parts[2]
    return name
  def _GetUniqueTRTEngineOp(self, graph_def):
    trt_engine_nodes = [
        node for node in graph_def.node if node.op == "TRTEngineOp"
    ]
    assert len(trt_engine_nodes) == 1
    return trt_engine_nodes[0]
  def _TestTrtGraphConverter(self,
                             device,
                             output_saved_model_dir=None,
                             need_calibration=False,
                             is_dynamic_op=False):
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir,
        need_calibration=need_calibration,
        is_dynamic_op=is_dynamic_op,
        device=device)
    graph_defs_to_verify = [output_graph_def]
    if output_saved_model_dir:
      saved_model_graph_def = saved_model_utils.get_meta_graph_def(
          output_saved_model_dir, tag_constants.SERVING).graph_def
      self.assertIsInstance(saved_model_graph_def, graph_pb2.GraphDef)
      graph_defs_to_verify.append(saved_model_graph_def)
    for graph_def in graph_defs_to_verify:
      node_name_to_op = {
          self._MayRemoveGraphSequenceNumber(node.name): node.op
          for node in graph_def.node
      }
      if device is not None and device.startswith("/CPU:"):
        self.assertEqual(
            {
                "add": "AddV2",
                "v1": "Const",
                "add_1": "AddV2",
                "add_2": "AddV2",
                "input1": "Placeholder",
                "input2": "Placeholder",
                "mul": "Mul",
                "output": "Identity"
            }, node_name_to_op)
      else:
        self.assertEqual(
            {
                "input1": "Placeholder",
                "input2": "Placeholder",
                "TRTEngineOp_000": "TRTEngineOp",
                "output": "Identity"
            }, node_name_to_op)
      if need_calibration:
        trt_engine_nodes = [
            node for node in graph_def.node if node.op == "TRTEngineOp"
        ]
        if device is not None and device.startswith("/CPU:"):
          self.assertEmpty(trt_engine_nodes)
          return
        self.assertNotEmpty(trt_engine_nodes)
        for node in trt_engine_nodes:
          self.assertTrue(len(node.attr["calibration_data"].s))
        with ops.Graph().as_default():
          importer.import_graph_def(graph_def, name="")
          with self.session(config=self._GetConfigProto()) as sess:
            for test_data in range(10):
              self.assertEqual((test_data + 1.0)**2 + test_data,
                               sess.run(
                                   "output:0",
                                   feed_dict={
                                       "input1:0": [[[test_data]]],
                                       "input2:0": [[[test_data]]]
                                   }))
  @parameterized.named_parameters([
      ("NoDeviceAssignment", None),
      ("GPU", "/GPU:0"),
      ("CPU", "/CPU:0"),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_OfflineConversion(self, device):
    for need_calibration in [False, True]:
      self._TestTrtGraphConverter(device)
      self._TestTrtGraphConverter(
          device,
          output_saved_model_dir=self.mkdtemp(),
          need_calibration=need_calibration)
  @parameterized.named_parameters([
      ("NoDeviceAssignment", None),
      ("GPU", "/device:GPU:0"),
      ("CPU", "/device:CPU:0"),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_OnlineConversion(self, device):
    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt_convert.TrtPrecisionMode.FP32)
    config = self._GetConfigProto(
        rewriter_config=trt_convert.get_tensorrt_rewriter_config(
            conversion_params,
            is_dynamic_op=False,
            max_batch_size=1,
            is_v2=False))
    with ops.Graph().as_default():
      inp1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input1")
      inp2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 1, 1], name="input2")
      if device:
        with ops.device(device):
          TrtConvertTest._GetGraph(inp1, inp2, inp1)
      else:
        TrtConvertTest._GetGraph(inp1, inp2, inp1)
      with self.session(config=config) as sess:
        self._TestRun(sess, batch_size=1)
  def _CreateConverterV2(
      self,
      input_saved_model_dir,
      input_saved_model_signature_key=_SAVED_MODEL_SIGNATURE_KEY,
      precision_mode=trt_convert.TrtPrecisionMode.FP32,
      maximum_cached_engines=2,
      allow_build_at_runtime=True):
    return trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        input_saved_model_signature_key=input_saved_model_signature_key,
        max_workspace_size_bytes=max_workspace_size_bytes,
        precision_mode=precision_mode,
        maximum_cached_engines=maximum_cached_engines,
        allow_build_at_runtime=allow_build_at_runtime)
  def _CheckTrtOps(self, concrete_func, check_fn=None, num_engines=1):
    graph_def = concrete_func.graph.as_graph_def()
    trt_op_names = []
    for node in graph_def.node:
      if node.op == "TRTEngineOp":
        trt_op_names.append(self._MayRemoveGraphSequenceNumber(node.name))
        if check_fn:
          check_fn(node)
    for func in graph_def.library.function:
      for node in func.node_def:
        if node.op == "TRTEngineOp":
          trt_op_names.append(self._MayRemoveGraphSequenceNumber(node.name))
          if check_fn:
            check_fn(node)
    self.assertLen(trt_op_names, num_engines)
  def _RandomInput(self, shape, dtype=np.float32):
    inp1 = np.random.random_sample(shape).astype(dtype)
    inp2 = np.random.random_sample(shape).astype(dtype)
    return inp1, inp2
  @test_util.run_v2_only
  def testTrtGraphConverter_DynamicConversion_v2(self):
    np_input1, np_input2 = self._RandomInput([4, 1, 1])
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    expected_output = root.run(np_input1, np_input2)
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})
    converter = self._CreateConverterV2(input_saved_model_dir)
    converter.convert()
    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    unexpected_asset_file = os.path.join(
        output_saved_model_dir,
        "assets/trt-serialized-engine." + trt_engine_name)
    self.assertFalse(os.path.exists(unexpected_asset_file))
    def _InputFn():
      yield np_input1, np_input2
    converter.build(input_fn=_InputFn)
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    expected_asset_file = os.path.join(
        output_saved_model_dir,
        "assets/trt-serialized-engine." + trt_engine_name)
    self.assertTrue(os.path.exists(expected_asset_file))
    self.assertTrue(os.path.getsize(expected_asset_file))
    del converter
    root_with_trt = load.load(output_saved_model_dir)
    converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
    self._CheckTrtOps(converted_signature)
    output_with_trt = converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))
    self.assertAllClose(
        expected_output,
        list(output_with_trt.values())[0],
        atol=1e-6,
        rtol=1e-6)
    del root_with_trt
  @test_util.run_v2_only
  def testTrtGraphConverter_ShapeOp_Int32InputOutput_v2(self):
    class ShapeOpModel(tracking.AutoTrackable):
      def __init__(self):
        self.v = None
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, None], dtype=dtypes.float32)
      ])
      def run(self, x):
        q = x + 1
        q_shape = array_ops.shape(q)
        q = math_ops.cumsum(q_shape)
        q = q * 2
        return array_ops.identity(q, name="output")
    np_input = np.random.random_sample([5, 3]).astype(np.float32)
    def _InputFunc():
      yield (np_input,)
    root = ShapeOpModel()
    expected_output = root.run(np_input)
    input_saved_model_dir = self.mkdtemp()
    save.save(root, input_saved_model_dir, signatures=root.run)
    conv_params = trt_convert.TrtConversionParams(minimum_segment_size=2)
    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        use_dynamic_shape=True,
        **conv_params._asdict())
    converter.convert()
    converter.build(_InputFunc)
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    root_with_trt = load.load(output_saved_model_dir)
    converted_signature = root_with_trt.signatures["serving_default"]
    self._CheckTrtOps(converted_signature, num_engines=2)
    output_with_trt = converted_signature(x=ops.convert_to_tensor(np_input))
    self.assertAllClose(expected_output, list(output_with_trt.values())[0])
  @test_util.run_v2_only
  def testTrtGraphConverter_Int8Conversion_v2(self):
    np_input1, np_input2 = self._RandomInput([4, 1, 1])
    input_saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    root = self._GetModelForV2()
    expected_output = root.run(np_input1, np_input2)
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})
    converter = self._CreateConverterV2(
        input_saved_model_dir,
        precision_mode=trt_convert.TrtPrecisionMode.INT8,
        maximum_cached_engines=3)
    def _CalibrationInputFn():
      yield np_input1, np_input2
    converter.convert(calibration_input_fn=_CalibrationInputFn)
    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name
    def _CheckFn(node):
      self.assertTrue(len(node.attr["calibration_data"].s), node.name)
    def _InputFn():
      yield self._RandomInput([5, 1, 1])
    converter.build(input_fn=_InputFn)
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    expected_asset_file = os.path.join(
        output_saved_model_dir,
        "assets/trt-serialized-engine." + trt_engine_name)
    self.assertTrue(os.path.exists(expected_asset_file))
    self.assertTrue(os.path.getsize(expected_asset_file))
    del converter
    root_with_trt = load.load(output_saved_model_dir)
    converted_signature = root_with_trt.signatures[_SAVED_MODEL_SIGNATURE_KEY]
    self._CheckTrtOps(converted_signature, _CheckFn)
    output_with_trt = converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))
    self.assertEqual(1, len(output_with_trt))
    self.assertAllClose(
        expected_output,
        list(output_with_trt.values())[0],
        atol=1e-6,
        rtol=1e-6)
    np_input1, np_input2 = self._RandomInput([6, 1, 1])
    converted_signature(
        inp1=ops.convert_to_tensor(np_input1),
        inp2=ops.convert_to_tensor(np_input2))
    del root_with_trt
  @test_util.run_v2_only
  def testTrtGraphConverter_DestroyEngineCache(self):
    np_input1, np_input2 = self._RandomInput([4, 1, 1])
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})
    converter = self._CreateConverterV2(input_saved_model_dir)
    converter.convert()
    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name
    def _InputFn():
      yield np_input1, np_input2
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    def _DestroyCache():
      with ops.device("GPU:0"):
        handle = gen_trt_ops.create_trt_resource_handle(
            resource_name=trt_engine_name)
        gen_resource_variable_ops.destroy_resource_op(
            handle, ignore_lookup_error=False)
    with self.assertRaisesRegex(errors.NotFoundError,
                                r"Resource .* does not exist."):
      _DestroyCache()
    root = load.load(output_saved_model_dir)
    _DestroyCache()
    with self.assertRaisesRegex(errors.NotFoundError,
                                r"Resource .* does not exist."):
      _DestroyCache()
    root = load.load(output_saved_model_dir)
    del root
    with self.assertRaisesRegex(errors.NotFoundError,
                                r"Resource .* does not exist."):
      _DestroyCache()
  def _CompareSavedModel(self, model_class):
    signature_key = "serving_default"
    def _GetModelPaths(model_class):
      input_saved_model_dir = self.mkdtemp()
      root = model_class()
      save.save(root, input_saved_model_dir)
      converter = self._CreateConverterV2(
          input_saved_model_dir, input_saved_model_signature_key=signature_key)
      converter.convert()
      output_saved_model_dir = self.mkdtemp()
      converter.save(output_saved_model_dir)
      return input_saved_model_dir, output_saved_model_dir
    def _GetSignatureDef(export_dir):
      saved_model_proto = loader_impl.parse_saved_model(export_dir)
      self.assertEqual(1, len(saved_model_proto.meta_graphs))
      meta_graph = saved_model_proto.meta_graphs[0]
      self.assertIn(signature_key, meta_graph.signature_def)
      return meta_graph.signature_def[signature_key]
    def _CompareSignatureDef(original_def, converted_def, is_input):
      endpoints = original_def.inputs if is_input else original_def.outputs
      converted_endpoints = (
          converted_def.inputs if is_input else converted_def.outputs)
      self.assertEqual(set(endpoints.keys()), set(converted_endpoints.keys()))
      for key in endpoints:
        original_input = endpoints[key]
        converted_input = converted_endpoints[key]
        self.assertEqual(original_input.name, converted_input.name)
        self.assertEqual(original_input.dtype, converted_input.dtype)
        self.assertEqual(
            tensor_shape.TensorShape(original_input.tensor_shape).as_list(),
            tensor_shape.TensorShape(converted_input.tensor_shape).as_list())
    def _GetStructuredOutputs(export_dir):
      root = load.load(export_dir)
      return root.signatures[signature_key].structured_outputs
    saved_model_path, converted_saved_model_path = _GetModelPaths(model_class)
    original_def = _GetSignatureDef(saved_model_path)
    converted_def = _GetSignatureDef(converted_saved_model_path)
    self.assertEqual(original_def.method_name, converted_def.method_name)
    _CompareSignatureDef(original_def, converted_def, True)
    _CompareSignatureDef(original_def, converted_def, False)
    self.assertEqual(
        _GetStructuredOutputs(saved_model_path),
        _GetStructuredOutputs(converted_saved_model_path))
  @test_util.run_v2_only
  def testRetainSignatureInfo_NoInputs(self):
    class _Model(tracking.AutoTrackable):
      @def_function.function(input_signature=[])
      def run(self):
        return array_ops.constant(1.0)
    self._CompareSavedModel(_Model)
  @test_util.run_v2_only
  def testRetainSignatureInfo_OneInput(self):
    class _Model(tracking.AutoTrackable):
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)
      ])
      def run(self, inp):
        return inp + inp * inp
    self._CompareSavedModel(_Model)
  @test_util.run_v2_only
  def testRetainSignatureInfo_TwoInputs(self):
    class _Model(tracking.AutoTrackable):
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32),
          tensor_spec.TensorSpec(shape=[None, 2], dtype=dtypes.float32)
      ])
      def run(self, inp1, inp2):
        return inp1 + inp2 * inp2
    self._CompareSavedModel(_Model)
  @test_util.run_v2_only
  def testRetainSignatureInfo_OneOutputSignatureKey(self):
    class _Model(tracking.AutoTrackable):
      @def_function.function(input_signature=[])
      def run(self):
        return {"my_output": array_ops.constant(1.0)}
    self._CompareSavedModel(_Model)
  @test_util.run_v2_only
  def testRetainSignatureInfo_TwoOutputSignatureKeys(self):
    class _Model(tracking.AutoTrackable):
      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[None, 1], dtype=dtypes.float32)
      ])
      def run(self, inp):
        return {
            "output_b": array_ops.constant(1.0),
            "output_a": inp + inp * inp
        }
    self._CompareSavedModel(_Model)
  def _TestRun(self, sess, batch_size):
    result = sess.run(
        "output:0",
        feed_dict={
            "input1:0": [[[1.0]]] * batch_size,
            "input2:0": [[[1.0]]] * batch_size
        })
    self.assertAllEqual([[[5.0]]] * batch_size, result)
  @parameterized.named_parameters([
      ("LargeSegmentSize", 7),
      ("NoMainGraphConversionSegmentSize", -1),
  ])
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_MinimumSegmentSize(self, minimum_segment_size):
    output_graph_def = self._ConvertGraphV1(
        minimum_segment_size=minimum_segment_size)
    node_name_to_op = {node.name: node.op for node in output_graph_def.node}
    self.assertEqual(
        {
            "v1": "Const",
            "input1": "Placeholder",
            "input2": "Placeholder",
            "add": "AddV2",
            "mul": "Mul",
            "add_1": "AddV2",
            "add_2": "AddV2",
            "output": "Identity"
        }, node_name_to_op)
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_DynamicOp(self):
    output_saved_model_dir = self.mkdtemp()
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir,
        is_dynamic_op=True,
        maximum_cached_engines=2)
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        self._TestRun(sess, 1)
        self._TestRun(sess, 2)
        self._TestRun(sess, 3)
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        self._TestRun(sess, 1)
        self._TestRun(sess, 2)
        self._TestRun(sess, 3)
  @test_util.deprecated_graph_mode_only
  def testTrtGraphConverter_StaticOp(self):
    output_saved_model_dir = self.mkdtemp()
    output_graph_def = self._ConvertGraphV1(
        output_saved_model_dir=output_saved_model_dir, maximum_cached_engines=1)
    with ops.Graph().as_default():
      importer.import_graph_def(output_graph_def, name="")
      with self.session(config=self._GetConfigProto()) as sess:
        self._TestRun(sess, 1)
        self._TestRun(sess, 2)
    with ops.Graph().as_default():
      with self.session(config=self._GetConfigProto()) as sess:
        loader.load(sess, [tag_constants.SERVING], output_saved_model_dir)
        self._TestRun(sess, 1)
        self._TestRun(sess, 2)
  @test_util.run_v2_only
  def testTrtGraphConverter_AllowEngineNativeSegmentExecution(self):
    np_input1, np_input2 = self._RandomInput([4, 1, 1])
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})
    def _InputFn():
      yield np_input1, np_input2
    converter = self._CreateConverterV2(
        input_saved_model_dir, max_workspace_size_bytes=1 << 20)
    converter.convert()
    os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"
    os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "True"
    with self.assertRaisesRegex(
        errors.AbortedError,
        r"User disallowed engine native segment execution"):
      try:
        converter.build(input_fn=_InputFn)
      finally:
        os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"
        os.environ["TF_TRT_ABORT_CUDA_ENGINE_BUILD"] = "False"
    converter.build(input_fn=_InputFn)
  @parameterized.parameters((True, True), (True, False), (False, True),
                            (False, False))
  @test_util.run_v2_only
  def testTrtGraphConverter_AllowBuildAtRuntime(self, build_offline,
                                                allow_build_at_runtime):
    if not is_tensorrt_enabled():
      return
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})
    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    def _InputFn():
      yield np_input1, np_input2
    converter = self._CreateConverterV2(
        input_saved_model_dir, allow_build_at_runtime=allow_build_at_runtime)
    converter.convert()
    if build_offline:
      converter.build(input_fn=_InputFn)
    output_saved_model_dir = self.mkdtemp()
    converter.save(output_saved_model_dir)
    saved_model_loaded = load.load(
        output_saved_model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[_SAVED_MODEL_SIGNATURE_KEY]
    def _CheckFn(node):
      self.assertEqual(node.attr["_allow_build_at_runtime"].b,
                       allow_build_at_runtime)
    self._CheckTrtOps(graph_func, _CheckFn)
    if not build_offline and not allow_build_at_runtime:
      with self.assertRaisesRegex(
          errors.AbortedError,
          r"User disallowed engine native segment execution"):
        try:
          os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "False"
          graph_func(inp1=np_input1, inp2=np_input2)
        finally:
          os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "True"
    else:
      output = graph_func(inp1=np_input1, inp2=np_input2)["output_0"]
      self.assertEqual(output.shape, (4, 1, 1))
      self.assertAllClose(
          np.asarray([5.0, 5.0, 5.0, 5.0]).reshape([4, 1, 1]), output)
  @test_util.run_v2_only
  def testBackwardCompatibility(self):
    model_dir = test.test_src_dir_path(
        "python/compiler/tensorrt/test/testdata/tftrt_2.0_saved_model")
    saved_model_loaded = load.load(model_dir, tags=[tag_constants.SERVING])
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    np_input1 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    np_input2 = ops.convert_to_tensor(np.ones([4, 1, 1]).astype(np.float32))
    output = graph_func(input1=np_input1, input2=np_input2)["output_0"]
    self.assertEqual(output.shape, (4, 1, 1))
    self.assertAllClose(
        np.asarray([5.0, 5.0, 5.0, 5.0]).reshape([4, 1, 1]), output)
  @parameterized.named_parameters([
      ("SaveGPUSpecificEngine", True),
      ("WithoutSaveGPUSpecificEngine", False),
  ])
  @test_util.run_v2_only
  def testTrtGraphConverter_SaveGPUSpecificEngine(self, save_engine_flag):
    np_input1, np_input2 = self._RandomInput([4, 1, 1])
    input_saved_model_dir = self.mkdtemp()
    root = self._GetModelForV2()
    save.save(root, input_saved_model_dir,
              {_SAVED_MODEL_SIGNATURE_KEY: root.run})
    converter = self._CreateConverterV2(
        input_saved_model_dir, precision_mode=trt_convert.TrtPrecisionMode.INT8)
    def CalibrationFn():
      yield np_input1, np_input2
    converter.convert(calibration_input_fn=CalibrationFn)
    self._CheckTrtOps(converter._converted_func)
    trt_engine_name = self._GetUniqueTRTEngineOp(
        converter._converted_graph_def).name
    output_saved_model_dir = self.mkdtemp()
    converter.save(
        output_saved_model_dir, save_gpu_specific_engines=save_engine_flag)
    expected_asset_file = os.path.join(
        output_saved_model_dir,
        "assets/trt-serialized-engine." + trt_engine_name)
    self.assertTrue(os.path.exists(expected_asset_file))
    if save_engine_flag:
      self.assertTrue(os.path.getsize(expected_asset_file))
    else:
      self.assertFalse(os.path.getsize(expected_asset_file))
    del converter
if __name__ == "__main__" and is_tensorrt_enabled():
  test.main()
