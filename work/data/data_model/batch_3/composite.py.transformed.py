
class Composite(object):
  """A decorator to register a function as a composition for an TF operator.
  The argument to the decorator must be the name of a TF raw operator the
  function composites for. Decorated function must take positional arguments
  which corresponds to the input and attributes in OpDef of the TF operation.
  Example:
    @composite.Composite('AddN')
    def _compose_add_n(inputs, N):
      if N == 1:
        ....
  """
  def __init__(self,
               op_name,
               inputs=None,
               attrs=None,
               derived_attrs=None,
               outputs=None):
    self._op_name = op_name
    self._inputs = inputs
    self._attrs = attrs
    self._derived_attrs = derived_attrs
    self._outputs = outputs
  def __call__(self, compose_fn):
    setattr(compose_fn, '_tfr_op_name', self._op_name)
    return compose_fn
