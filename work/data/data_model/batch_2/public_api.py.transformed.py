
import re
import six
from tensorflow.python.util import tf_inspect
class PublicAPIVisitor(object):
  def __init__(self, visitor):
    self._visitor = visitor
    self._root_name = 'tf'
    self._private_map = {
        'tf': [
            'compiler',
            'core',
            'dtensor',
            'python',
        ],
        'tf.flags': ['cpp_flags'],
    }
    self._do_not_descend_map = {
        'tf': [
            'examples',
            'platform',
            'pywrap_tensorflow',
            'user_ops',
            'tools',
            'tensorboard',
        ],
        'tf.app': ['flags'],
        'tf.test': ['mock'],
    }
  @property
  def private_map(self):
    return self._private_map
  @property
  def do_not_descend_map(self):
    return self._do_not_descend_map
  def set_root_name(self, root_name):
    self._root_name = root_name
  def _is_private(self, path, name, obj=None):
    return ((path in self._private_map and name in self._private_map[path]) or
            (six.ensure_str(name).startswith('_') and
             not re.match('__.*__$', six.ensure_str(name)) or
             name in ['__base__', '__class__', '__next_in_mro__']))
  def _do_not_descend(self, path, name):
    return (path in self._do_not_descend_map and
            name in self._do_not_descend_map[path])
  def __call__(self, path, parent, children):
    if tf_inspect.ismodule(parent) and len(
        six.ensure_str(path).split('.')) > 10:
      raise RuntimeError('Modules nested too deep:\n%s.%s\n\nThis is likely a '
                         'problem with an accidental public import.' %
                         (self._root_name, path))
    full_path = '.'.join([self._root_name, path]) if path else self._root_name
    for name, child in list(children):
      if self._is_private(full_path, name, child):
        children.remove((name, child))
    self._visitor(path, parent, children)
    for name, child in list(children):
      if self._do_not_descend(full_path, name):
        children.remove((name, child))
