
import collections
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_should_use
_Resource = collections.namedtuple("_Resource",
                                   ["handle", "create", "is_initialized"])
def register_resource(handle, create_op, is_initialized_op, is_shared=True):
  resource = _Resource(handle, create_op, is_initialized_op)
  if is_shared:
    ops.add_to_collection(ops.GraphKeys.RESOURCES, resource)
  else:
    ops.add_to_collection(ops.GraphKeys.LOCAL_RESOURCES, resource)
def shared_resources():
  return ops.get_collection(ops.GraphKeys.RESOURCES)
def local_resources():
  return ops.get_collection(ops.GraphKeys.LOCAL_RESOURCES)
def report_uninitialized_resources(resource_list=None,
                                   name="report_uninitialized_resources"):
  """Returns the names of all uninitialized resources in resource_list.
  If the returned tensor is empty then all resources have been initialized.
  Args:
   resource_list: resources to check. If None, will use shared_resources() +
    local_resources().
   name: name for the resource-checking op.
  Returns:
   Tensor containing names of the handles of all resources which have not
   yet been initialized.
  """
  if resource_list is None:
    resource_list = shared_resources() + local_resources()
  with ops.name_scope(name):
    local_device = os.environ.get(
        "TF_DEVICE_FOR_UNINITIALIZED_VARIABLE_REPORTING", "/cpu:0")
    with ops.device(local_device):
      if not resource_list:
        return array_ops.constant([], dtype=dtypes.string)
      variables_mask = math_ops.logical_not(
          array_ops.stack([r.is_initialized for r in resource_list]))
      variable_names_tensor = array_ops.constant(
          [s.handle.name for s in resource_list])
      return array_ops.boolean_mask(variable_names_tensor, variables_mask)
@tf_should_use.should_use_result
def initialize_resources(resource_list, name="init"):
  if resource_list:
    return control_flow_ops.group(*[r.create for r in resource_list], name=name)
  return control_flow_ops.no_op(name=name)
