
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
get_resource_handle_data = ops.get_resource_handle_data
def copy_handle_data(source_t, target_t):
  if (target_t.dtype == dtypes.resource or
      target_t.dtype == dtypes.variant):
    if isinstance(source_t, ops.EagerTensor):
    else:
      handle_data = get_resource_handle_data(source_t)
    if (handle_data is not None
        and handle_data.is_set
        and handle_data.shape_and_type):
      set_handle_data(target_t, handle_data)
def set_handle_data(target_t, handle_data):
  if isinstance(target_t, ops.EagerTensor):
    target_t._handle_data = handle_data
    return
  pywrap_tf_session.SetHandleShapeAndType(target_t.graph._c_graph,
                                          target_t._as_tf_output(),
                                          handle_data.SerializeToString())
