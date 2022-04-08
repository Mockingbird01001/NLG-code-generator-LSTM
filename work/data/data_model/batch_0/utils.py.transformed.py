
import itertools
import threading
import types
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.util import tf_decorator
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
def use_wrapped_call(layer, call_fn, default_training_value=None,
                     return_method=False):
  """Creates fn that adds the losses returned by call_fn & returns the outputs.
  Args:
    layer: A Keras layer object
    call_fn: tf.function that takes layer inputs (and possibly a training arg),
      and returns a tuple of (outputs, list of losses).
    default_training_value: Default value of the training kwarg. If `None`, the
      default is `K.learning_phase()`.
    return_method: Whether to return a method bound to the layer.
  Returns:
    function that calls call_fn and returns the outputs. Losses returned by
    call_fn are added to the layer losses.
  """
  expects_training_arg = layer_uses_training_bool(layer)
    original_call = call_fn.original_layer_call
    call_fn = call_fn.__call__
  else:
    original_call = call_fn
  fn, arg_spec = maybe_add_training_arg(
      original_call, call_fn, expects_training_arg, default_training_value)
  def return_outputs_and_add_losses(*args, **kwargs):
    if return_method:
      args = args[1:]
    outputs, losses = fn(*args, **kwargs)
    layer.add_loss(losses, inputs=True)
    if context.executing_eagerly():
      for i in layer._flatten_layers():
        if i is not layer:
          i._eager_losses = [base_layer_utils.REVIVED_LOSS_PLACEHOLDER]
    return outputs
  decorated = tf_decorator.make_decorator(
      target=call_fn,
      decorator_func=return_outputs_and_add_losses,
      decorator_argspec=arg_spec)
  if return_method:
    return types.MethodType(decorated, layer)
  else:
    return decorated
def layer_uses_training_bool(layer):
    return True
  visited = {layer}
  to_visit = list_all_layers(layer)
  while to_visit:
    layer = to_visit.pop()
    if layer in visited:
      continue
    if getattr(layer, '_expects_training_arg', True):
      return True
    visited.add(layer)
    to_visit.extend(list_all_layers(layer))
  return False
def list_all_layers(obj):
  if isinstance(obj, training_lib.Model):
    return obj.layers
  else:
def list_all_layers_and_sublayers(obj):
  s = set([obj])
  s.update(itertools.chain.from_iterable(
      list_all_layers_and_sublayers(layer) for layer in list_all_layers(obj)))
  return s
def maybe_add_training_arg(
    original_call, wrapped_call, expects_training_arg, default_training_value):
  """Decorate call and optionally adds training argument.
  If a layer expects a training argument, this function ensures that 'training'
  is present in the layer args or kwonly args, with the default training value.
  Args:
    original_call: Original call function.
    wrapped_call: Wrapped call function.
    expects_training_arg: Whether to include 'training' argument.
    default_training_value: Default value of the training kwarg to include in
      the arg spec. If `None`, the default is `K.learning_phase()`.
  Returns:
    Tuple of (
      function that calls `wrapped_call` and sets the training arg,
      Argspec of returned function or `None` if the argspec is unchanged)
  """
  if not expects_training_arg:
    return wrapped_call, None
  def wrap_with_training_arg(*args, **kwargs):
    training_arg_index = get_training_arg_index(original_call)
    training = get_training_arg(training_arg_index, args, kwargs)
    if training is None:
      training = default_training_value or K.learning_phase()
    args = list(args)
    kwargs = kwargs.copy()
    def replace_training_and_call(training):
      set_training_arg(training, training_arg_index, args, kwargs)
      return wrapped_call(*args, **kwargs)
    return control_flow_util.smart_cond(
        training, lambda: replace_training_and_call(True),
        lambda: replace_training_and_call(False))
  arg_spec = tf_inspect.getfullargspec(original_call)
  defaults = list(arg_spec.defaults) if arg_spec.defaults is not None else []
  kwonlyargs = arg_spec.kwonlyargs
  kwonlydefaults = arg_spec.kwonlydefaults or {}
  if 'training' not in arg_spec.args:
    kwonlyargs.append('training')
    kwonlydefaults['training'] = default_training_value
  else:
    index = arg_spec.args.index('training')
    training_default_index = len(arg_spec.args) - index
    if (arg_spec.defaults and
        len(arg_spec.defaults) >= training_default_index and
        defaults[-training_default_index] is None):
      defaults[-training_default_index] = default_training_value
  decorator_argspec = tf_inspect.FullArgSpec(
      args=arg_spec.args,
      varargs=arg_spec.varargs,
      varkw=arg_spec.varkw,
      defaults=defaults,
      kwonlyargs=kwonlyargs,
      kwonlydefaults=kwonlydefaults,
      annotations=arg_spec.annotations)
  return wrap_with_training_arg, decorator_argspec
def get_training_arg_index(call_fn):
  argspec = tf_inspect.getfullargspec(call_fn)
  if argspec.varargs:
    if 'training' in argspec.kwonlyargs or argspec.varkw:
      return -1
    return None
  else:
    arg_list = argspec.args
    if tf_inspect.ismethod(call_fn):
      arg_list = arg_list[1:]
    if 'training' in arg_list:
      return arg_list.index('training')
    elif 'training' in argspec.kwonlyargs or argspec.varkw:
      return -1
    return None
def set_training_arg(training, index, args, kwargs):
    kwargs['training'] = training
  else:
    args[index] = training
  return args, kwargs
def get_training_arg(index, args, kwargs):
    return kwargs.get('training', None)
  else:
    return args[index]
def remove_training_arg(index, args, kwargs):
    kwargs.pop('training', None)
  else:
    args.pop(index)
class SaveOptionsContext(threading.local):
  def __init__(self):
    super(SaveOptionsContext, self).__init__()
    self.save_traces = True
_save_options_context = SaveOptionsContext()
@tf_contextlib.contextmanager
def keras_option_scope(save_traces):
  previous_value = _save_options_context.save_traces
  try:
    _save_options_context.save_traces = save_traces
    yield
  finally:
    _save_options_context.save_traces = previous_value
def should_save_traces():
  return _save_options_context.save_traces
@tf_contextlib.contextmanager
def no_automatic_dependency_tracking_scope(obj):
  """A context that disables automatic dependency tracking when assigning attrs.
  Objects that inherit from Autotrackable automatically creates dependencies
  to trackable objects through attribute assignments, and wraps data structures
  (lists or dicts) with trackable classes. This scope may be used to temporarily
  disable this behavior. This works similar to the decorator
  `no_automatic_dependency_tracking`.
  Example usage:
  ```
  model = tf.keras.Model()
  with no_automatic_dependency_tracking_scope(model):
  ```
  Args:
    obj: A trackable object.
  Yields:
    a scope in which the object doesn't track dependencies.
  """
  previous_value = getattr(obj, '_setattr_tracking', True)
  try:
    yield
  finally:
