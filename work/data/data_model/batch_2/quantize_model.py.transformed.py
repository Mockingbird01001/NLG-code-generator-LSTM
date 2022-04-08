
import enum
import tempfile
import uuid
import warnings
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model as quantize_model_wrapper
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.load import load as saved_model_load
_INIT_OP_SIGNATURE_KEY = '__saved_model_init_op'
class OptimizationMethod(enum.Enum):
  STATIC_RANGE_QUANT = 'STATIC_RANGE_QUANTIZATION'
  DYNAMIC_RANGE_QUANT = 'DYNAMIC_RANGE_QUANTIZATION'
  AUTOMATIC_QUANT = 'AUTOMATIC_QUANTIZATION'
def _legalize_tensor_name(tensor_name: str) -> str:
  return tensor_name.replace(':', '__')
def _is_qat_saved_model(saved_model_path: str):
  saved_model_proto = saved_model_loader.parse_saved_model(saved_model_path)
  for meta_graph in saved_model_proto.meta_graphs:
    if any(
        node.op.startswith('FakeQuant') for node in meta_graph.graph_def.node):
      return True
    for function in meta_graph.graph_def.library.function:
      if any(node.op.startswith('FakeQuant') for node in function.node_def):
        return True
  return False
def _get_signatures_from_saved_model(saved_model_path: str,
                                     signature_keys=None,
                                     tags=None):
  if tags is None:
    tags = set([tag_constants.SERVING])
  loader = saved_model_loader.SavedModelLoader(saved_model_path)
  meta_graphdef = loader.get_meta_graph_def_from_tags(tags)
  signatures = {}
  for key, signature_def in meta_graphdef.signature_def.items():
    if key == _INIT_OP_SIGNATURE_KEY:
      continue
    if signature_keys is not None and key not in signature_keys:
      continue
    signatures[key] = signature_def
  return signatures
def _fix_tensor_names(signatures, exported_graph):
  if signatures is None:
    return None
  output_index_path_map = {}
  for op in exported_graph.get_operations():
    if (op.type == '_Retval' and
        op.get_attr('tf_saved_model.index_path') is not None):
      index_path_name = op.get_attr('tf_saved_model.index_path')[0]
      index_path_name = index_path_name.decode('utf-8')
      output_index_path_map[index_path_name] = op.inputs[0].name
  for signature_def in signatures.values():
    for tensor_info in signature_def.inputs.values():
      try:
        exported_graph.get_tensor_by_name(tensor_info.name)
      except KeyError:
        warnings.warn('Cannot find the tensor with name %s in the graph.' %
                      tensor_info.name)
        return None
    for tensor_info in signature_def.outputs.values():
      try:
        if tensor_info.name in output_index_path_map:
          tensor_info.name = output_index_path_map[tensor_info.name]
        else:
          return_node = exported_graph.get_operation_by_name(
              _legalize_tensor_name(tensor_info.name))
          tensor_info.name = return_node.inputs[0].name
      except KeyError:
        warnings.warn(
            'Cannot find the tensor or node with name %s in the graph.' %
            tensor_info.name)
        return None
  return signatures
def _static_range_quantize(saved_model_path: str,
                           signature_keys=None,
                           tags=None,
                           output_directory=None,
                           representative_dataset=None):
  """Quantizes the given SavedModel via static range quantization.
  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    signature_keys: List of keys identifying SignatureDef containing inputs and
      outputs.
    tags: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze.
    output_directory: The path to save the output SavedModel (must be an empty
      directory).
    representative_dataset: a generator that returns a dictionary in
      {input_name: input_tensor} format or a tuple with signature key and a
      dictionary in {input_name: input_tensor} format that feeds calibration
        data for quantizing model. This should be provided when the model is not
        a QAT model.
  Returns:
    A SavedModel object with TF quantization applied.
  Raises:
    ValueError: when representative_dataset is not provided for non-QAT model.
  """
  is_qat_saved_model = _is_qat_saved_model(saved_model_path)
  signatures = _get_signatures_from_saved_model(saved_model_path,
                                                signature_keys, tags)
  if representative_dataset is None and not is_qat_saved_model:
    raise ValueError(
        'When `representative_dataset` is not provided, the model should be '
        'trained with quantization-aware training (QAT).')
  if is_qat_saved_model:
    graph_def_serialized = (
        quantize_model_wrapper.quantize_qat_model(saved_model_path,
                                                  ','.join(signature_keys),
                                                  ','.join(tags)))
  else:
    graph_def_serialized = (
        quantize_model_wrapper.quantize_ptq_model_pre_calibration(
            saved_model_path, ','.join(signature_keys), ','.join(tags)))
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(graph_def_serialized)
    float_model_dir = tempfile.mkdtemp()
    v1_builder = builder.SavedModelBuilder(float_model_dir)
    with session.Session(graph=ops.Graph()) as sess:
      for function_def in graph_def.library.function:
        for node_def in function_def.node_def:
          if node_def.op == 'CustomAggregator':
            node_def.attr['id'].s = uuid.uuid4().hex.encode('ascii')
      importer.import_graph_def(graph_def, name='')
      working_graph = ops.get_default_graph()
      graph_def = working_graph.as_graph_def()
      signatures = _fix_tensor_names(signatures, working_graph)
      if signatures is None:
        raise ValueError(
            "The input SavedModel doesn't contain a valid signature")
      v1_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING], signature_def_map=signatures)
    v1_builder.save()
    float_model = saved_model_load(float_model_dir)
    for sample in representative_dataset():
      if isinstance(sample, tuple):
        if not isinstance(sample[1], dict):
          raise ValueError('You need to provide a dictionary with input '
                           'names and values in the second argument in the '
                           'tuple')
        signature_key = sample[0]
        input_data_map = sample[1]
      elif isinstance(sample, dict):
        if len(signature_keys) > 1:
          raise ValueError('When the model has multiple signatures, you need '
                           'to provide a tuple with signature key and a '
                           'dictionary with input names and values')
        signature_key = signature_keys[0]
        input_data_map = sample
      else:
        raise ValueError('You need to provide either a dictionary with input '
                         'names and values or a tuple with signature key and a '
                         'dictionary with input names and values')
      func = float_model.signatures[signature_key]
      func(**input_data_map)
    for function_def in graph_def.library.function:
      for node_def in function_def.node_def:
        if node_def.op == 'CustomAggregator':
          node_id = node_def.attr['id'].s
          try:
            min_val = quantize_model_wrapper.get_min_from_calibrator(node_id)
            max_val = quantize_model_wrapper.get_max_from_calibrator(node_id)
            quantize_model_wrapper.clear_data_from_calibrator(node_id)
            node_def.attr['min'].f = float(min_val)
            node_def.attr['max'].f = float(max_val)
          except ValueError:
            warnings.warn('%s does not have min/max values.' % node_id)
    calibrated_model_dir = tempfile.mkdtemp()
    v1_builder = builder.SavedModelBuilder(calibrated_model_dir)
    with session.Session(graph=ops.Graph()) as sess:
      importer.import_graph_def(graph_def, name='')
      working_graph = ops.get_default_graph()
      graph_def = working_graph.as_graph_def()
      v1_builder.add_meta_graph_and_variables(
          sess, [tag_constants.SERVING], signature_def_map=signatures)
    v1_builder.save()
    signatures = _get_signatures_from_saved_model(calibrated_model_dir,
                                                  signature_keys, tags)
    graph_def_serialized = (
        quantize_model_wrapper.quantize_ptq_model_post_calibration(
            calibrated_model_dir,
            ','.join(signature_keys),
            ','.join(tags),
        ))
  graph_def = graph_pb2.GraphDef()
  graph_def.ParseFromString(graph_def_serialized)
  if output_directory is None:
    output_directory = tempfile.mkdtemp()
  v1_builder = builder.SavedModelBuilder(output_directory)
  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(graph_def, name='')
    working_graph = ops.get_default_graph()
    signatures = _fix_tensor_names(signatures, working_graph)
    if signatures is None:
      raise ValueError("The input SavedModel doesn't contain a valid signature")
    v1_builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING], signature_def_map=signatures)
  v1_builder.save()
  return saved_model_load(output_directory)
def quantize(saved_model_path: str,
             signature_keys=None,
             tags=None,
             output_directory=None,
             optimization_method: OptimizationMethod = OptimizationMethod
             .AUTOMATIC_QUANT,
             representative_dataset=None):
  """Quantizes the given SavedModel.
  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    signature_keys: List of keys identifying SignatureDef containing inputs and
      outputs.
    tags: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze.
    output_directory: The path to save the output SavedModel (must be an empty
      directory).
    optimization_method: Optimization method to apply.
    representative_dataset: a generator that returns a dictionary in
      {input_name: input_tensor} format or a tuple with signature key and a
      dictionary in {input_name: input_tensor} format that feeds calibration
        data for quantizing model. This should be provided when the model is not
        a QAT model.
  Returns:
    A SavedModel object with TF quantization applied.
  Raises:
    ValueError: when representative_dataset is not provided for non QAT model
      for enabling static range quantization.
  """
  if tags is None:
    tags = set([tag_constants.SERVING])
  if signature_keys is None:
    signature_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  if optimization_method == OptimizationMethod.STATIC_RANGE_QUANT:
    return _static_range_quantize(
        saved_model_path=saved_model_path,
        signature_keys=signature_keys,
        tags=tags,
        output_directory=output_directory,
        representative_dataset=representative_dataset)
  else:
    raise NotImplementedError(
        'Optimization method "%s" is not implemented yet' %
        optimization_method.name)
