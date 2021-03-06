
from tensorflow.python.ops import gen_manip_ops as _gen_manip_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('roll', v1=['roll', 'manip.roll'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('manip.roll')
  return _gen_manip_ops.roll(input, shift, axis, name)
roll.__doc__ = _gen_manip_ops.roll.__doc__
