
import functools
import itertools
import weakref
import numpy as np
import six
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
class FunctionSpec(object):
  @classmethod
  def from_function_and_signature(cls, python_function,
                                  input_signature,
                                  is_pure=False,
                                  experimental_follow_type_hints=False,
                                  jit_compile=None):
    """Creates a FunctionSpec instance given a python function and signature.
    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
      will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`
      jit_compile: see `tf.function`
    Returns:
      instance of FunctionSpec
    """
    fullargspec = tf_inspect.getfullargspec(python_function)
    if (input_signature is not None and
        set(fullargspec.kwonlyargs) - set(fullargspec.kwonlydefaults or ())):
      nodefault_kwonlyargs = set(fullargspec.kwonlyargs)
      if fullargspec.kwonlydefaults is not None:
        nodefault_kwonlyargs -= set(fullargspec.kwonlydefaults)
      raise ValueError("Cannot build TF function from "
                       f"{python_function.__name__}: keyword-only arguments "
                       "must have default values when input_signature is "
                       "provided. Got keyword-only arguments without default "
                       f"values: {sorted(nodefault_kwonlyargs)}.")
    is_method = tf_inspect.isanytargetmethod(python_function)
    _, unwrapped = tf_decorator.unwrap(python_function)
    if isinstance(unwrapped, functools.partial):
      if fullargspec.defaults or fullargspec.kwonlydefaults:
        new_defaults = fullargspec.defaults
        new_args = fullargspec.args
        if fullargspec.defaults:
          old_args = fullargspec.args
          old_defaults = fullargspec.defaults
          no_default = object()
          num_args_without_defaults = len(old_args) - len(old_defaults)
          left_padding = tuple([no_default] * num_args_without_defaults)
          args_with_defaults = zip(old_args, left_padding + old_defaults)
          non_keyword_defaults_mask = [
              0 if key in unwrapped.keywords else 1 for key in old_args
          ]
          new_args_with_defaults = list(
              itertools.compress(args_with_defaults, non_keyword_defaults_mask))
          new_args = [arg for arg, _ in new_args_with_defaults]
          new_defaults = [
              default for _, default in new_args_with_defaults
              if default is not no_default
          ]
        fullargspec = tf_inspect.FullArgSpec(
            args=new_args,
            varargs=fullargspec.varargs,
            varkw=fullargspec.varkw,
            defaults=new_defaults,
            kwonlyargs=[],
            kwonlydefaults={},
            annotations=fullargspec.annotations)
    while isinstance(python_function, functools.partial):
      python_function = python_function.func
    name = getattr(python_function, "__name__", "f")
    return FunctionSpec(
        fullargspec,
        is_method,
        input_signature,
        is_pure=is_pure,
        jit_compile=jit_compile,
        experimental_follow_type_hints=experimental_follow_type_hints,
        name=name)
  def __init__(self,
               fullargspec,
               is_method,
               input_signature,
               is_pure=False,
               experimental_follow_type_hints=False,
               name=None,
               jit_compile=None):
    """Constructs a FunctionSpec describing a python function.
    Args:
      fullargspec: `tf_inspect.FullArgSpec` object describing the function.
      is_method: True if the function is a method.
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`.
      name: Name of the function
      jit_compile: see `tf.function`.
    """
    self._fullargspec = fullargspec
    self._is_method = is_method
    self._is_pure = is_pure
    self._jit_compile = jit_compile
    self._experimental_follow_type_hints = experimental_follow_type_hints
    self._name = name or "f"
    if self._is_method:
      args = fullargspec.args[1:]
    else:
      args = fullargspec.args
    self._args_to_indices = {arg: i for i, arg in enumerate(args)}
    self._arg_names = args
    default_values = fullargspec.defaults
    offset = len(args) - len(default_values or [])
    self._arg_indices_to_default_values = {
        offset + index: default
        for index, default in enumerate(default_values or [])
    }
    self._arg_indices_no_default_values = set(range(len(args))) - set(
        self._arg_indices_to_default_values)
    if input_signature is None:
      self._input_signature = None
    else:
      self._input_signature = tuple(input_signature)
      self._flat_input_signature = tuple(nest.flatten(input_signature,
                                                      expand_composites=True))
  @property
  def fullargspec(self):
    return self._fullargspec
  @property
  def is_method(self):
    return self._is_method
  @property
  def args_to_indices(self):
    return self._args_to_indices
  @property
  def kwargs_to_include(self):
    return self._kwargs_to_include
  @property
  def input_signature(self):
    return self._input_signature
  @property
  def flat_input_signature(self):
    return self._flat_input_signature
  @property
  def is_pure(self):
    return self._is_pure
  @property
  def jit_compile(self):
    return self._jit_compile
  @property
  def arg_names(self):
    return self._arg_names
  @property
  def vararg_name(self):
    return self._fullargspec.varargs
  @property
  def varkw_name(self):
    return self._fullargspec.varkw
  def signature_summary(self, default_values=False):
    args = list(self._arg_names)
    if default_values:
      for (i, default) in self._arg_indices_to_default_values.items():
        args[i] += "={}".format(default)
    if self._fullargspec.kwonlyargs:
      args.append("*")
      for arg_name in self._fullargspec.kwonlyargs:
        args.append(arg_name)
        if default_values and arg_name in self._fullargspec.kwonlydefaults:
          args[-1] += "={}".format(self._fullargspec.kwonlydefaults[arg_name])
    return f"{self._name}({', '.join(args)})"
  def _convert_annotated_args_to_tensors(self, args, kwargs):
    if self.input_signature is not None:
      return
    args = list(args)
    for i, arg in enumerate(args):
      if i < len(self._fullargspec.args):
        annotation_key = self._fullargspec.args[i]
      else:
        annotation_key = self._fullargspec.varargs
      arg_annotation = self._fullargspec.annotations.get(annotation_key, None)
      if arg_annotation == ops.Tensor:
        args[i] = _to_tensor_or_tensor_spec(arg)
    for kw, v in kwargs.items():
      if kw in self._fullargspec.kwonlyargs or kw in self._fullargspec.args:
        annotation_key = kw
      else:
        annotation_key = self._fullargspec.varkw
      kwarg_annotation = self._fullargspec.annotations.get(annotation_key, None)
      if kwarg_annotation == ops.Tensor:
        kwargs[kw] = _to_tensor_or_tensor_spec(v)
    return tuple(args), kwargs
  def _validate_inputs(self, flat_inputs):
    for inp in flat_inputs:
      if isinstance(inp, weakref.ref):
        raise ValueError(
            f"weakref input {inp} not supported for function {self._name}")
  def canonicalize_function_inputs(self, *args, **kwargs):
    """Canonicalizes `args` and `kwargs`.
    Canonicalize the inputs to the Python function using a `FunctionSpec`
    instance. In particular, we parse the varargs and kwargs that the
    original function was called with into a tuple corresponding to the
    Python function's positional (named) arguments and a dictionary
    corresponding to its kwargs.  Missing default arguments are added.
    If this `FunctionSpec` has an input signature, then it is used to convert
    arguments to tensors; otherwise, any inputs containing numpy arrays are
    converted to tensors.
    Additionally, any inputs containing numpy arrays are converted to Tensors.
    Args:
      *args: The varargs this object was called with.
      **kwargs: The keyword args this function was called with.
    Returns:
      A canonicalized ordering of the inputs, as well as full and filtered
      (Tensors and Variables only) versions of their concatenated flattened
      representations, represented by a tuple in the form (args, kwargs,
      flat_args, filtered_flat_args). Here: `args` is a full list of bound
      arguments, and `kwargs` contains only true keyword arguments, as opposed
      to named arguments called in a keyword-like fashion.
    Raises:
      ValueError: If a keyword in `kwargs` cannot be matched with a positional
        argument when an input signature is specified, or when the inputs
        do not conform to the input signature.
    """
    if self._is_pure:
      args, kwargs = _convert_variables_to_tensors(args, kwargs)
    if self._experimental_follow_type_hints:
      args, kwargs = self._convert_annotated_args_to_tensors(args, kwargs)
    arglen = len(args)
    if self._input_signature is not None:
      if arglen > len(self._input_signature):
        raise TypeError(f"{self.signature_summary()} specifies "
                        f"{len(self._input_signature)} positional arguments, "
                        f"but got {arglen}.")
      for arg in six.iterkeys(kwargs):
        index = self._args_to_indices.get(arg, None)
        if index is None:
          raise TypeError(f"{self.signature_summary()} got unexpected keyword "
                          f"argument `{arg}`.")
        if index >= len(self._input_signature):
          raise TypeError(
              f"{self.signature_summary()} got keyword argument `{arg}` that "
              "was not included in input_signature.")
    if not kwargs:
      inputs = args
      if self._arg_indices_to_default_values:
        try:
          inputs += tuple(self._arg_indices_to_default_values[i]
                          for i in range(arglen, len(self._arg_names)))
        except KeyError:
          missing_args = [
              self._arg_names[i]
              for i in range(arglen, len(self._arg_names))
              if i not in self._arg_indices_to_default_values
          ]
          raise TypeError(f"{self.signature_summary()} missing required "
                          f"arguments: {', '.join(missing_args)}.")
      if self._fullargspec.kwonlydefaults:
        kwargs.update(self._fullargspec.kwonlydefaults)
    else:
      arg_indices_to_values = {
          index: default for index, default in six.iteritems(
              self._arg_indices_to_default_values) if index >= arglen
      }
      consumed_args = []
      missing_arg_indices = self._arg_indices_no_default_values - set(
          range(arglen))
      for arg, value in six.iteritems(kwargs):
        index = self._args_to_indices.get(arg, None)
        if index is not None:
          if index < arglen:
            raise TypeError(f"{self.signature_summary()} got two values for "
                            f"{arg!r}.")
          arg_indices_to_values[index] = value
          missing_arg_indices.discard(index)
          consumed_args.append(arg)
      for arg in consumed_args:
        kwargs.pop(arg)
      inputs = args + _deterministic_dict_values(arg_indices_to_values)
      if missing_arg_indices:
        missing_args = [self._arg_names[i] for i in sorted(missing_arg_indices)]
        if len(missing_args) == 1:
          raise TypeError(f"{self.signature_summary()} missing 1 required "
                          f"argument: {missing_args[0]}.")
        else:
          raise TypeError(f"{self.signature_summary()} missing required "
                          f"arguments: {', '.join(missing_args)}.")
      if kwargs and self._input_signature is not None:
        raise TypeError("Keyword arguments are not supported when "
                        "input_signature is provided. Signature: "
                        f"{self.signature_summary()}. Keyword arguments: "
                        f"{kwargs}.")
      if self._fullargspec.kwonlydefaults:
        for (kwarg, default) in self._fullargspec.kwonlydefaults.items():
          kwargs.setdefault(kwarg, default)
    if self._input_signature is None:
      inputs, flat_inputs, filtered_flat_inputs = _convert_numpy_inputs(inputs)
      kwargs, flat_kwargs, filtered_flat_kwargs = _convert_numpy_inputs(kwargs)
      flat_inputs += flat_kwargs
      filtered_flat_inputs += filtered_flat_kwargs
    else:
      inputs, flat_inputs, filtered_flat_inputs = _convert_inputs_to_signature(
          inputs, self._input_signature, self._flat_input_signature)
    self._validate_inputs(flat_inputs)
    return inputs, kwargs, filtered_flat_inputs
def _to_tensor_or_tensor_spec(x):
  return (x if isinstance(x, (ops.Tensor, tensor_spec.TensorSpec)) else
          ops.convert_to_tensor(x))
def _deterministic_dict_values(dictionary):
  return tuple(dictionary[key] for key in sorted(dictionary))
def _convert_variables_to_tensors(args, kwargs):
  args = [_to_tensor_or_tensor_spec(x) for x in args]
  kwargs = {kw: _to_tensor_or_tensor_spec(x)
            for kw, x in kwargs.items()}
  return tuple(args), kwargs
def _convert_numpy_inputs(inputs):
  flat_inputs = nest.flatten(inputs, expand_composites=True)
  need_packing = False
  filtered_flat_inputs = []
  for index, value in enumerate(flat_inputs):
    if isinstance(value,
                  (ops.Tensor, resource_variable_ops.BaseResourceVariable)):
      filtered_flat_inputs.append(value)
    elif hasattr(value, "__array__") and not (
        hasattr(value, "_should_act_as_resource_variable") or
        isinstance(value, (np.str_, type, composite_tensor.CompositeTensor))):
      a = value.__array__()
      if not isinstance(a, np.ndarray):
        raise TypeError(f"The output of __array__ must be an np.ndarray, "
                        f"got {type(a)} from {value}.")
      flat_inputs[index] = constant_op.constant(a)
      filtered_flat_inputs.append(flat_inputs[index])
      need_packing = True
  if need_packing:
    return (nest.pack_sequence_as(
        structure=inputs, flat_sequence=flat_inputs,
        expand_composites=True), flat_inputs, filtered_flat_inputs)
  else:
    return inputs, flat_inputs, filtered_flat_inputs
def _convert_inputs_to_signature(inputs, input_signature, flat_input_signature):
  def format_error_message(inputs, input_signature):
    return ("  inputs: (\n" + "    " + ",\n    ".join(str(i) for i in inputs) +
            ")\n" + "  input_signature: (\n" + "    " +
            ",\n    ".join(str(i) for i in input_signature) + ")")
  try:
    flatten_inputs = nest.flatten_up_to(
        input_signature,
        inputs[:len(input_signature)],
        expand_composites=True,
  except ValueError:
    raise ValueError("Structure of Python function inputs does not match "
                     "input_signature:\n"
                     f"{format_error_message(inputs, input_signature)}.")
  need_packing = False
  for index, (value, spec) in enumerate(zip(flatten_inputs,
                                            flat_input_signature)):
    if (isinstance(spec, tensor_spec.TensorSpec) and
        not _pywrap_utils.IsTensor(value)):
      try:
        flatten_inputs[index] = ops.convert_to_tensor(
            value, dtype_hint=spec.dtype)
        need_packing = True
      except ValueError:
        raise ValueError("When input_signature is provided, all inputs to "
                         "the Python function must be convertible to "
                         "tensors:\n"
                         f"{format_error_message(inputs, input_signature)}.")
  if any(not spec.is_compatible_with(other) for spec, other in zip(
      flat_input_signature,
      flatten_inputs)):
    raise ValueError("Python inputs incompatible with input_signature:\n"
                     f"{format_error_message(inputs, input_signature)}.")
  if need_packing:
    inputs = nest.pack_sequence_as(
        structure=input_signature,
        flat_sequence=flatten_inputs,
        expand_composites=True)
  flat_inputs = nest.flatten(inputs, expand_composites=True)
  return (inputs, flat_inputs, [
      t for t in flat_inputs
      if isinstance(t, (ops.Tensor, resource_variable_ops.BaseResourceVariable))
  ])
