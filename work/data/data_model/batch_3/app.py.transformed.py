
import sys as _sys
from absl.app import run as _run
from tensorflow.python.platform import flags
from tensorflow.python.util.tf_export import tf_export
def _parse_flags_tolerate_undef(argv):
  return flags.FLAGS(_sys.argv if argv is None else argv, known_only=True)
@tf_export(v1=['app.run'])
def run(main=None, argv=None):
  main = main or _sys.modules['__main__'].main
  _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
