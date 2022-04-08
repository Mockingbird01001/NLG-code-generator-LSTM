
from absl import logging
from tensorflow.compiler.xla.python.tpu_driver.client import tpu_client_extension as _tpu_client
class TpuBackend(object):
  _local_backend = None
  @staticmethod
  def create(worker=None, force=False):
    if worker is None:
      raise ValueError(
          'Failed to create TpuBackend. The `worker` parameter must not be '
          '`None`. Use `local` to connect to a local TPU or '
          '`grpc://host:port` to connect to a remote TPU.')
    if worker == 'local' or 'local://' in worker:
      if worker == 'local':
        worker = 'local://'
      if force:
        return _tpu_client.TpuClient.Get(worker)
      if TpuBackend._local_backend is None:
        logging.info('Starting the local TPU driver.')
        TpuBackend._local_backend = _tpu_client.TpuClient.Get(worker)
      return TpuBackend._local_backend
    else:
      return _tpu_client.TpuClient.Get(worker)
