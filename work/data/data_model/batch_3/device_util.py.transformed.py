
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
def canonicalize(d, default=None):
  if isinstance(d, context.LogicalDevice):
    d = tf_device.DeviceSpec.from_string(d.name)
  else:
    d = tf_device.DeviceSpec.from_string(d)
  assert d.device_type is None or d.device_type == d.device_type.upper(), (
      "Device type '%s' must be all-caps." % (d.device_type,))
  result = tf_device.DeviceSpec(
      replica=0, task=0, device_type="CPU", device_index=0)
  if ops.executing_eagerly_outside_functions():
    host_cpu = tf_device.DeviceSpec.from_string(
        config.list_logical_devices("CPU")[0].name)
    if host_cpu.job:
      result = result.make_merged_spec(host_cpu)
    else:
      result = result.replace(job="localhost")
  if default:
    result = result.make_merged_spec(
        tf_device.DeviceSpec.from_string(default))
  result = result.make_merged_spec(d)
  return result.to_string()
def canonicalize_without_job_and_task(d):
  canonicalized_device = canonicalize(d)
  spec = tf_device.DeviceSpec.from_string(canonicalized_device)
  spec = spec.replace(job=None, task=None, replica=0)
  return spec.to_string()
def resolve(d):
  return canonicalize(d, default=current())
class _FakeNodeDef(object):
  __slots__ = ["op", "name"]
  def __init__(self):
    self.op = ""
    self.name = ""
class _FakeOperation(object):
  def __init__(self):
    self.device = ""
    self.type = ""
    self.name = ""
    self.node_def = _FakeNodeDef()
  def _set_device(self, device):
  def _set_device_from_string(self, device_str):
    self.device = device_str
def current():
  if ops.executing_eagerly_outside_functions():
    d = context.context().device_name
  else:
    op = _FakeOperation()
    d = op.device
  return d
def get_host_for_device(device):
  spec = tf_device.DeviceSpec.from_string(device)
  return tf_device.DeviceSpec(
      job=spec.job, replica=spec.replica, task=spec.task,
      device_type="CPU", device_index=0).to_string()
def local_devices_from_num_gpus(num_gpus):
  return (tuple("/device:GPU:%d" % i for i in range(num_gpus)) or
          ("/device:CPU:0",))
