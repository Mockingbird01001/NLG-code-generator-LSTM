
import abc
import sys
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
if sys.version_info >= (3, 8):
  from typing import Protocol
  from typing import runtime_checkable
else:
  from typing_extensions import Protocol
  from typing_extensions import runtime_checkable
class CompositeTensorGradient(object, metaclass=abc.ABCMeta):
  """Class used to help compute gradients for CompositeTensors.
  This abstract base class defines two methods: `get_gradient_components`, which
  returns the components of a value that should be included in gradients; and
  `replace_gradient_components`, which replaces the gradient components in a
  value.  These methods can be used to compute the gradient of a `y` with
  respect to `x` (`grad(y, x)`) as follows:
  * If `y` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    `y.__composite_gradient__`, then `grad(y, x)` =
    `grad(cg.get_gradient_components(y), x)`.
  * If `x` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    'x.__composite_gradient__', then `grad(y, x)` =
    `cg.replace_gradient_components(x, grad(y, cg.get_gradient_components(x))`.
  """
  @abc.abstractmethod
  def get_gradient_components(self, value):
    raise NotImplementedError(
        f"{type(self).__name__}.get_gradient_components()")
  @abc.abstractmethod
  def replace_gradient_components(self, value, component_grads):
    """Replaces the gradient components in `value` with `component_grads`.
    This method may not call TensorFlow ops, since any new ops added to the
    graph would not be propertly tracked by the gradient mechanisms.
    Args:
      value: A value with its gradient components compatible with
        `component_grads`.
      component_grads: A nested structure of `Tensor` or `CompositeTensor` or
        `None` (for unconnected gradients).
    Returns:
      A copy of `value`, where the components that should be included in
      gradients have been replaced by `component_grads`; or `None` (if
      `component_grads` includes `None`).
    """
    raise NotImplementedError(
        f"{type(self).__name__}.replace_gradient_components()")
@runtime_checkable
class CompositeTensorGradientProtocol(Protocol):
  __composite_gradient__: CompositeTensorGradient
class WithValuesCompositeTensorGradient(CompositeTensorGradient):
  def get_gradient_components(self, value):
    return value.values
  def replace_gradient_components(self, value, component_grads):
    return value.with_values(component_grads)
def _get_tensors_for_gradient(x):
  if not isinstance(x, composite_tensor.CompositeTensor):
    return x
  if not isinstance(x, CompositeTensorGradientProtocol):
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source or "
        "gradient target.")
  composite_gradient = x.__composite_gradient__
  return nest.map_structure(_get_tensors_for_gradient,
                            composite_gradient.get_gradient_components(x))
def _replace_tensors_for_gradient(x, grad):
  """Replaces the tensors in `x` that should be differentiated with `grad`.
  Args:
    x: A `Tensor` or `CompositeTensor`.
    grad: A nested structure of `Tensor`, with the same structure as the value
      returned by `_get_tensors_for_gradient(x)`.
  Returns:
    A `Tensor` or `CompositeTensor`.
  """
  if not isinstance(x, composite_tensor.CompositeTensor):
    return grad
  if not isinstance(x, CompositeTensorGradientProtocol):
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source.")
  composite_gradient = x.__composite_gradient__
  x_components = composite_gradient.get_gradient_components(x)
  grad_components = nest.map_structure_up_to(x_components,
                                             _replace_tensors_for_gradient,
                                             x_components, grad)
  if grad_components is None:
    return None
  return composite_gradient.replace_gradient_components(x, grad_components)
def get_flat_tensors_for_gradients(xs):
  """Returns a flat list of Tensors that should be differentiated for `xs`.
  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.
  Returns:
    A flat list of `Tensor`s constructed from `xs`, where `Tensor` values are
    left as-is, and `CompositeTensor`s are replaced with
    `_get_tensors_for_gradient(x)`.
  """
  result = []
  for x in xs:
    if not isinstance(x, composite_tensor.CompositeTensor):
      result.append(x)
    else:
      x_tensors = nest.flatten(_get_tensors_for_gradient(x))
      for t in x_tensors:
        if not backprop_util.IsTrainable(t):
          logging.log_first_n(
              logging.WARN, "The dtype of differentiable component %s in %s "
              "must be floating (e.g., tf.float32), got %r.", 5, t, x, t.dtype)
      result.extend(x_tensors)
  return result
def replace_flat_tensors_for_gradients(xs, flat_grads):
  xs_structure = [_get_tensors_for_gradient(x) for x in xs]
  grads = nest.pack_sequence_as(xs_structure, flat_grads)
  return [_replace_tensors_for_gradient(x, grad) for x, grad in zip(xs, grads)]
