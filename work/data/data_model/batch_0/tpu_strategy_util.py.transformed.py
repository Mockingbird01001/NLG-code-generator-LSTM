
import gc
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute.cluster_resolver.tpu_cluster_resolver import TPUClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.tpu import tpu
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
_INITIALIZED_TPU_SYSTEMS = {}
_LOCAL_MASTERS = ("", "local")
@tf_export("tpu.experimental.initialize_tpu_system")
def initialize_tpu_system(cluster_resolver=None):
  logging.info("Deallocate tpu buffers before initializing tpu system.")
  context.context().clear_kernel_cache()
  gc.collect()
  job = None
  if cluster_resolver is None:
    if context.executing_eagerly():
      curr_device = device.DeviceSpec.from_string(context.context().device_name)
      if curr_device.job is not None:
        job = "{}/replica:0/task:0".format(curr_device.job)
    cluster_resolver = TPUClusterResolver("")
  assert isinstance(cluster_resolver, TPUClusterResolver)
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    logging.warning(
        "TPU system %s has already been initialized. "
        "Reinitializing the TPU can cause previously created "
        "variables on TPU to be lost.", tpu_name)
  logging.info("Initializing the TPU system: %s", tpu_name)
  if tpu_name not in _LOCAL_MASTERS:
    job = "{}/replica:0/task:0".format(cluster_resolver.get_job_name())
  if context.executing_eagerly():
    @function.defun
    def _tpu_init_fn():
      return tpu.initialize_system(
          job=job,
          compilation_failure_closes_chips=False,
          tpu_cancellation_closes_chips=False)
    try:
        output = _tpu_init_fn()
      context.async_wait()
    except errors.InvalidArgumentError as e:
      raise errors.NotFoundError(
          None, None,
          "TPUs not found in the cluster. Failed in initialization: "
          + str(e))
    serialized_topology = output.numpy()
  elif not ops.executing_eagerly_outside_functions():
    master = cluster_resolver.master()
    cluster_spec = cluster_resolver.cluster_spec()
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    if cluster_spec:
      session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    with ops.Graph().as_default():
      with session_lib.Session(config=session_config, target=master) as sess:
        serialized_topology = sess.run(tpu.initialize_system())
  else:
      serialized_topology = tpu.initialize_system(
          job=job, compilation_failure_closes_chips=False)
      return serialized_topology
  logging.info("Finished initializing TPU system.")
  tpu_topology = topology.Topology(serialized=serialized_topology)
  cluster_resolver.set_tpu_topology(serialized_topology)
  _INITIALIZED_TPU_SYSTEMS[tpu_name] = tpu_topology
  return tpu_topology
def get_initialized_tpu_systems():
  return _INITIALIZED_TPU_SYSTEMS.copy()
@tf_export("tpu.experimental.shutdown_tpu_system")
def shutdown_tpu_system(cluster_resolver=None):
  job = None
  if cluster_resolver is None:
    if context.executing_eagerly():
      curr_device = device.DeviceSpec.from_string(context.context().device_name)
      if curr_device.job is not None:
        job = "{}/replica:0/task:0".format(curr_device.job)
    cluster_resolver = TPUClusterResolver("")
  assert isinstance(cluster_resolver, TPUClusterResolver)
  if tpu_name not in _INITIALIZED_TPU_SYSTEMS:
    logging.warning("You are shutting down a TPU system %s that has not been "
                    "initialized." % tpu_name)
  logging.info("Shutting down the TPU system: %s", tpu_name)
  if context.executing_eagerly():
    if tpu_name not in _LOCAL_MASTERS:
      job = "{}/replica:0/task:0".format(cluster_resolver.get_job_name())
    @function.defun
    def _tpu_shutdown_fn():
      tpu.shutdown_system(job=job)
      _tpu_shutdown_fn()
    logging.info("Clearing out eager caches")
    context.context().clear_kernel_cache()
  elif not ops.executing_eagerly_outside_functions():
    master = cluster_resolver.master()
    cluster_spec = cluster_resolver.cluster_spec()
    session_config = config_pb2.ConfigProto(allow_soft_placement=True)
    if cluster_spec:
      session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    with ops.Graph().as_default():
      with session_lib.Session(config=session_config, target=master) as sess:
        sess.run(tpu.shutdown_system())
  else:
    raise RuntimeError(
        "initialize_tpu_system is not supported within "
        "tf.functions.  You should call initialize_tpu_system outside of your tf.function. "
    )
  logging.info("Finished shutting down TPU system.")
  if tpu_name in _INITIALIZED_TPU_SYSTEMS:
    del _INITIALIZED_TPU_SYSTEMS[tpu_name]
