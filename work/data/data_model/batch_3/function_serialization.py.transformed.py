
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
def _serialize_function_spec(function_spec):
  if function_spec.is_method and not function_spec.fullargspec.args:
    raise NotImplementedError(
        "Cannot serialize a method function without a named "
        "'self' argument.")
  proto = saved_object_graph_pb2.FunctionSpec()
  proto.fullargspec.CopyFrom(
      nested_structure_coder.encode_structure(
          function_spec.fullargspec._replace(annotations={})))
  proto.is_method = function_spec.is_method
  proto.input_signature.CopyFrom(
      nested_structure_coder.encode_structure(function_spec.input_signature))
  proto.jit_compile = {
      None: saved_object_graph_pb2.FunctionSpec.JitCompile.DEFAULT,
      True: saved_object_graph_pb2.FunctionSpec.JitCompile.ON,
      False: saved_object_graph_pb2.FunctionSpec.JitCompile.OFF,
  }.get(function_spec.jit_compile)
  return proto
def serialize_concrete_function(concrete_function, node_ids):
  bound_inputs = []
  try:
    for capture in concrete_function.captured_inputs:
      bound_inputs.append(node_ids[capture])
  except KeyError:
    raise KeyError(
        f"Failed to add concrete function '{concrete_function.name}' to object-"
        f"based SavedModel as it captures tensor {capture!r} which is unsupported"
        " or not reachable from root. "
        "One reason could be that a stateful object or a variable that the "
        "function depends on is not assigned to an attribute of the serialized "
        "trackable object (see SaveTest.test_captures_unreachable_variable).")
  concrete_function_proto = saved_object_graph_pb2.SavedConcreteFunction()
  structured_outputs = func_graph_module.convert_structure_to_signature(
      concrete_function.structured_outputs)
  concrete_function_proto.canonicalized_input_signature.CopyFrom(
      nested_structure_coder.encode_structure(
          concrete_function.structured_input_signature))
  concrete_function_proto.output_signature.CopyFrom(
      nested_structure_coder.encode_structure(structured_outputs))
  concrete_function_proto.bound_inputs.extend(bound_inputs)
  return concrete_function_proto
def serialize_bare_concrete_function(concrete_function):
  proto = saved_object_graph_pb2.SavedBareConcreteFunction(
      concrete_function_name=concrete_function.name,
      allowed_positional_arguments=concrete_function._num_positional_args,
      argument_keywords=concrete_function._arg_keywords)
  if concrete_function._pre_initialized_function_spec is not None:
    proto.function_spec.CopyFrom(
        _serialize_function_spec(
            concrete_function._pre_initialized_function_spec))
  return proto
def serialize_function(function, concrete_functions):
  proto = saved_object_graph_pb2.SavedFunction()
  function_spec_proto = _serialize_function_spec(function.function_spec)
  proto.function_spec.CopyFrom(function_spec_proto)
  for concrete_function in concrete_functions:
    proto.concrete_functions.append(concrete_function.name)
  return proto
def wrap_cached_variables(concrete_function):
  outer_graph = func_graph_module.FuncGraph(
      "{}_no_cache".format(concrete_function.graph.name))
  mapped_captures = None
  remapped_captures = {}
  with outer_graph.as_default():
    for capture, placeholder in concrete_function.graph.captures:
      cached_variable = getattr(capture, "_cached_variable", None)
      if cached_variable is None:
        continue
      cached_variable = cached_variable()
      new_cached_value = cached_variable.read_value()
      remapped_captures[id(capture)] = captures[id(capture)]
      captures[id(capture)] = (new_cached_value, placeholder)
      mapped_captures = True
  if not mapped_captures:
    return concrete_function
  inner_concrete = defun.ConcreteFunction(concrete_function.graph)
  def wrap_function(*args):
  args = nest.flatten(concrete_function.structured_input_signature,
                      expand_composites=True)
  func_graph_module.func_graph_from_py_func(
      None, wrap_function, args=tuple(args), kwargs={},
      func_graph=outer_graph)
  fn = defun.ConcreteFunction(
      outer_graph, spec=concrete_function._function_spec)
  fn._arg_keywords = concrete_function._arg_keywords
  fn._num_positional_args = concrete_function._num_positional_args
  fn._pre_initialized_function_spec = (
      concrete_function._pre_initialized_function_spec)
  for key, capture in remapped_captures.items():
    captures[key] = capture
  return fn
