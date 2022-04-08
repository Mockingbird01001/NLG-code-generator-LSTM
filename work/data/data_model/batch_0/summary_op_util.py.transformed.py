
import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
def collect(val, collections, default_collections):
  if collections is None:
    collections = default_collections
  for key in collections:
    ops.add_to_collection(key, val)
_INVALID_TAG_CHARACTERS = re.compile(r'[^-/\w\.]')
def clean_tag(name):
  if name is not None:
    new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
    if new_name != name:
      tf_logging.info('Summary name %s is illegal; using %s instead.' %
                      (name, new_name))
      name = new_name
  return name
@contextlib.contextmanager
def summary_scope(name, family=None, default_name=None, values=None):
  """Enters a scope used for the summary and yields both the name and tag.
  To ensure that the summary tag name is always unique, we create a name scope
  based on `name` and use the full scope name in the tag.
  If `family` is set, then the tag name will be '<family>/<scope_name>', where
  `scope_name` is `<outer_scope>/<family>/<name>`. This ensures that `family`
  is always the prefix of the tag (and unmodified), while ensuring the scope
  respects the outer scope from this summary was created.
  Args:
    name: A name for the generated summary node.
    family: Optional; if provided, used as the prefix of the summary tag name.
    default_name: Optional; if provided, used as default name of the summary.
    values: Optional; passed as `values` parameter to name_scope.
  Yields:
    A tuple `(tag, scope)`, both of which are unique and should be used for the
    tag and the scope for the summary to output.
  """
  name = clean_tag(name)
  family = clean_tag(family)
  scope_base_name = name if family is None else '{}/{}'.format(family, name)
  with ops.name_scope(
      scope_base_name, default_name, values, skip_on_eager=False) as scope:
    if family is None:
      tag = scope.rstrip('/')
    else:
      tag = '{}/{}'.format(family, scope.rstrip('/'))
    yield (tag, scope)
