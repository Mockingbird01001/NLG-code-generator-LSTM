
import os
import shutil
import urllib.request
_OSS_URL_PREFIX = 'https://github.com/google/mediapipe/raw/master/'
def download_oss_model(model_path: str):
  mp_root_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-4])
  model_abspath = os.path.join(mp_root_path, model_path)
  if os.path.exists(model_abspath):
    return
  model_url = _OSS_URL_PREFIX + model_path
  print('Downloading model to ' + model_abspath)
  with urllib.request.urlopen(model_url) as response, open(model_abspath,
                                                           'wb') as out_file:
    if response.code != 200:
      raise ConnectionError('Cannot download ' + model_path +
                            ' from the MediaPipe Github repo.')
    shutil.copyfileobj(response, out_file)
