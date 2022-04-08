
__author__ = 'matthewtoia@google.com (Matt Toia)'
import warnings
class Error(Exception):
  pass
class DescriptorDatabaseConflictingDefinitionError(Error):
class DescriptorDatabase(object):
  def __init__(self):
    self._file_desc_protos_by_file = {}
    self._file_desc_protos_by_symbol = {}
  def Add(self, file_desc_proto):
    proto_name = file_desc_proto.name
    if proto_name not in self._file_desc_protos_by_file:
      self._file_desc_protos_by_file[proto_name] = file_desc_proto
    elif self._file_desc_protos_by_file[proto_name] != file_desc_proto:
      raise DescriptorDatabaseConflictingDefinitionError(
          '%s already added, but with different descriptor.' % proto_name)
    else:
      return
    package = file_desc_proto.package
    for message in file_desc_proto.message_type:
      for name in _ExtractSymbols(message, package):
        self._AddSymbol(name, file_desc_proto)
    for enum in file_desc_proto.enum_type:
      self._AddSymbol(('.'.join((package, enum.name))), file_desc_proto)
      for enum_value in enum.value:
        self._file_desc_protos_by_symbol[
            '.'.join((package, enum_value.name))] = file_desc_proto
    for extension in file_desc_proto.extension:
      self._AddSymbol(('.'.join((package, extension.name))), file_desc_proto)
    for service in file_desc_proto.service:
      self._AddSymbol(('.'.join((package, service.name))), file_desc_proto)
  def FindFileByName(self, name):
    return self._file_desc_protos_by_file[name]
  def FindFileContainingSymbol(self, symbol):
    try:
      return self._file_desc_protos_by_symbol[symbol]
    except KeyError:
      top_level, _, _ = symbol.rpartition('.')
      try:
        return self._file_desc_protos_by_symbol[top_level]
      except KeyError:
        raise KeyError(symbol)
  def FindFileContainingExtension(self, extendee_name, extension_number):
    return None
  def FindAllExtensionNumbers(self, extendee_name):
    return []
  def _AddSymbol(self, name, file_desc_proto):
    if name in self._file_desc_protos_by_symbol:
      warn_msg = ('Conflict register for file "' + file_desc_proto.name +
                  '": ' + name +
                  ' is already defined in file "' +
                  self._file_desc_protos_by_symbol[name].name + '"')
      warnings.warn(warn_msg, RuntimeWarning)
    self._file_desc_protos_by_symbol[name] = file_desc_proto
def _ExtractSymbols(desc_proto, package):
  message_name = package + '.' + desc_proto.name if package else desc_proto.name
  yield message_name
  for nested_type in desc_proto.nested_type:
    for symbol in _ExtractSymbols(nested_type, message_name):
      yield symbol
  for enum_type in desc_proto.enum_type:
    yield '.'.join((message_name, enum_type.name))
