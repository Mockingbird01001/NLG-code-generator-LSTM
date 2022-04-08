
def ld(v):
  if isinstance(v, Undefined):
    return v.read()
  return v
def ldu(load_v, name):
  """Load variable operator that returns Undefined when failing to evaluate.
  Note: the name ("load or return undefined") is abbreviated to minimize
  the amount of clutter in generated code.
  This variant of `ld` is useful when loading symbols that may be undefined at
  runtime, such as composite symbols, and whether they are defined or not cannot
  be determined statically. For example `d['a']` is undefined when `d` is an
  empty dict.
  Args:
    load_v: Lambda that executes the actual read.
    name: Human-readable name of the symbol being read.
  Returns:
    Either the value of the symbol, or Undefined, if the symbol is not fully
    defined.
  """
  try:
    return load_v()
  except (KeyError, AttributeError, NameError):
    return Undefined(name)
class Undefined(object):
  """Represents an undefined symbol in Python.
  This is used to reify undefined symbols, which is required to use the
  functional form of loops.
  Example:
    while n > 0:
      n = n - 1
      s = n
  This is valid Python code and will not result in an error as long as n
  is positive. The use of this class is to stay as close to Python semantics
  as possible for staged code of this nature.
  Converted version of the above showing the possible usage of this class:
    s = Undefined('s')
    init_state = (s,)
    s = while_loop(cond, body, init_state)
  Attributes:
    symbol_name: Text, identifier for the undefined symbol
  """
  __slots__ = ('symbol_name',)
  def __init__(self, symbol_name):
    self.symbol_name = symbol_name
  def read(self):
    raise UnboundLocalError("'{}' is used before assignment".format(
        self.symbol_name))
  def __repr__(self):
    return self.symbol_name
  def __getattribute__(self, name):
    try:
      return object.__getattribute__(self, name)
    except AttributeError:
      return self
  def __getitem__(self, i):
    return self
class UndefinedReturnValue(object):
  pass
