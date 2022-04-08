
import pathlib
import shutil
import tempfile
from typing import Iterable
from absl import app
from absl import flags
from tensorflow_docs.api_generator import gen_java
FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', '/tmp/lite_api/',
                    ("Use this branch as the root version and don't"
                     ' create in version directory'))
flags.DEFINE_string('site_path', 'lite/api_docs/java',
                    'Path prefix in the _toc.yaml')
flags.DEFINE_string('code_url_prefix', None,
                    '[UNUSED] The url prefix for links to code.')
flags.DEFINE_bool(
    'search_hints', True,
    '[UNUSED] Include metadata search hints in the generated files')
SOURCE_PATH_CORE = pathlib.Path('tensorflow/lite/java/src/main/java')
SOURCE_PATH_SUPPORT = pathlib.Path('tensorflow_lite_support/java/src/java')
SOURCE_PATH_METADATA = pathlib.Path(
    'tensorflow_lite_support/metadata/java/src/java')
SOURCE_PATH_ODML = pathlib.Path('tensorflow_lite_support/odml/java/image/src')
SOURCE_PATH_ANDROID_SDK = pathlib.Path('android/sdk/api/26.txt')
SECTION_LABELS = {
    'org.tensorflow.lite': 'Core',
    'org.tensorflow.lite.support': 'Support Library',
    'org.tensorflow.lite.task': 'Task Library',
    'com.google.android.odml.image': 'ODML',
}
EXTERNAL_APIS = {'https://developer.android.com': SOURCE_PATH_ANDROID_SDK}
def overlay(from_root: pathlib.Path, to_root: pathlib.Path):
  for from_path in from_root.rglob('*'):
    to_path = to_root / from_path.relative_to(from_root)
    if from_path.is_file():
      assert not to_path.exists(), f'{to_path} exists!'
      shutil.copyfile(from_path, to_path)
    else:
      to_path.mkdir(exist_ok=True)
def resolve_nested_dir(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
  nested = path.parts[0] / path
  root_path = root / path
  root_nested_path = root / nested
  if root_path.exists():
    return root_path
  elif root_nested_path.exists():
    return root_nested_path
  raise ValueError(f'Could not find {path} or {nested}')
def exists_maybe_nested(paths: Iterable[pathlib.Path],
                        root: pathlib.Path) -> bool:
  for path in paths:
    try:
      resolve_nested_dir(path, root)
    except ValueError:
      return False
  return True
def main(unused_argv):
  root = pathlib.Path(__file__).resolve()
  all_deps = [SOURCE_PATH_CORE, SOURCE_PATH_SUPPORT, SOURCE_PATH_ODML]
  while root.name and not exists_maybe_nested(all_deps, root):
    root = root.parent
  assert exists_maybe_nested(all_deps, root), 'Could not find dependencies.'
  with tempfile.TemporaryDirectory() as merge_tmp_dir:
    merged_temp_dir = pathlib.Path(merge_tmp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_CORE, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_SUPPORT, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_METADATA, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_ODML, root), merged_temp_dir)
    gen_java.gen_java_docs(
        package=['org.tensorflow.lite', 'com.google.android.odml'],
        source_path=merged_temp_dir,
        output_dir=pathlib.Path(FLAGS.output_dir),
        site_path=pathlib.Path(FLAGS.site_path),
        section_labels=SECTION_LABELS,
        federated_docs={k: root / v for k, v in EXTERNAL_APIS.items()})
if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
