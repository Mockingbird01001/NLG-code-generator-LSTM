
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils.generic_utils import LazyLoader
training = LazyLoader(
    "training", globals(),
    "tensorflow.python.keras.engine.training")
training_v1 = LazyLoader(
    "training_v1", globals(),
    "tensorflow.python.keras.engine.training_v1")
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
base_layer_v1 = LazyLoader(
    "base_layer_v1", globals(),
    "tensorflow.python.keras.engine.base_layer_v1")
callbacks = LazyLoader(
    "callbacks", globals(),
    "tensorflow.python.keras.callbacks")
callbacks_v1 = LazyLoader(
    "callbacks_v1", globals(),
    "tensorflow.python.keras.callbacks_v1")
class ModelVersionSelector(object):
    use_v2 = should_use_v2()
    return super(ModelVersionSelector, cls).__new__(cls)
class LayerVersionSelector(object):
    use_v2 = should_use_v2()
    return super(LayerVersionSelector, cls).__new__(cls)
class TensorBoardVersionSelector(object):
    use_v2 = should_use_v2()
    start_cls = cls
    cls = swap_class(start_cls, callbacks.TensorBoard, callbacks_v1.TensorBoard,
                     use_v2)
    if start_cls == callbacks_v1.TensorBoard and cls == callbacks.TensorBoard:
      return cls(*args, **kwargs)
    return super(TensorBoardVersionSelector, cls).__new__(cls)
def should_use_v2():
  if context.executing_eagerly():
    return True
  elif ops.executing_eagerly_outside_functions():
    graph = ops.get_default_graph()
    if (getattr(graph, "name", False) and
        graph.name.startswith("wrapped_function")):
      return False
    return True
  else:
    return False
def swap_class(cls, v2_cls, v1_cls, use_v2):
  if cls == object:
    return cls
  if cls in (v2_cls, v1_cls):
    return v2_cls if use_v2 else v1_cls
  new_bases = []
  for base in cls.__bases__:
    if ((use_v2 and issubclass(base, v1_cls)
         or (not use_v2 and issubclass(base, v2_cls)))):
      new_base = swap_class(base, v2_cls, v1_cls, use_v2)
    else:
      new_base = base
    new_bases.append(new_base)
  cls.__bases__ = tuple(new_bases)
  return cls
def disallow_legacy_graph(cls_name, method_name):
  if not ops.executing_eagerly_outside_functions():
    error_msg = (
        "Calling `{cls_name}.{method_name}` in graph mode is not supported "
        "when the `{cls_name}` instance was constructed with eager mode "
        "enabled. Please construct your `{cls_name}` instance in graph mode or"
        " call `{cls_name}.{method_name}` with eager mode enabled.")
    error_msg = error_msg.format(cls_name=cls_name, method_name=method_name)
    raise ValueError(error_msg)
def is_v1_layer_or_model(obj):
  return isinstance(obj, (base_layer_v1.Layer, training_v1.Model))
