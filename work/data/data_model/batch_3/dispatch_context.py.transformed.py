
import collections
class DispatchContext(collections.namedtuple(
    'DispatchContext',
    ('options',))):
  def option(self, name):
    return self.options[name]
NO_CTX = DispatchContext(options={})
