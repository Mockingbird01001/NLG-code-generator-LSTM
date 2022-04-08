
import enum
import os
import requests
from six.moves.urllib import request
from tensorflow.python.eager import context
GCP_METADATA_HEADER = {'Metadata-Flavor': 'Google'}
_GCE_METADATA_URL_ENV_VARIABLE = 'GCE_METADATA_IP'
_RESTARTABLE_EXIT_CODE = 143
GRACE_PERIOD_GCE = 0
def request_compute_metadata(path):
  gce_metadata_endpoint = 'http://' + os.environ.get(
      _GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
  req = request.Request(
      '%s/computeMetadata/v1/%s' % (gce_metadata_endpoint, path),
      headers={'Metadata-Flavor': 'Google'})
  info = request.urlopen(req).read()
  if isinstance(info, bytes):
    return info.decode('utf-8')
  else:
    return info
def termination_watcher_function_gce():
  result = request_compute_metadata(
      'instance/maintenance-event') == 'TERMINATE_ON_HOST_MAINTENANCE'
  return result
def on_gcp():
  gce_metadata_endpoint = 'http://' + os.environ.get(
      _GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
  try:
    response = requests.get(
        '%s/computeMetadata/v1/%s' %
        (gce_metadata_endpoint, 'instance/hostname'),
        headers=GCP_METADATA_HEADER,
        timeout=5)
    return response.status_code == 200
  except requests.exceptions.RequestException:
    return False
@enum.unique
class PlatformDevice(enum.Enum):
  INTERNAL = 'internal'
  GCE_GPU = 'GCE_GPU'
  GCE_TPU = 'GCE_TPU'
  GCE_CPU = 'GCE_CPU'
  UNSUPPORTED = 'unsupported'
def detect_platform():
  if on_gcp():
    if context.context().list_physical_devices('GPU'):
      return PlatformDevice.GCE_GPU
    elif context.context().list_physical_devices('TPU'):
      return PlatformDevice.GCE_TPU
    else:
      return PlatformDevice.GCE_CPU
  else:
    return PlatformDevice.INTERNAL
