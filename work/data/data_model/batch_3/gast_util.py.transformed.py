
import functools
import gast
GAST2 = hasattr(gast, 'Str')
GAST3 = not GAST2
def _is_constant_gast_2(node):
  return isinstance(node, (gast.Num, gast.Str, gast.Bytes, gast.Ellipsis,
                           gast.NameConstant))
def _is_constant_gast_3(node):
  return isinstance(node, gast.Constant)
def is_literal(node):
  if is_constant(node):
    return True
  if isinstance(node, gast.Name) and node.id in ['True', 'False', 'None']:
    return True
  return False
def _is_ellipsis_gast_2(node):
  return isinstance(node, gast.Ellipsis)
def _is_ellipsis_gast_3(node):
  return isinstance(node, gast.Constant) and node.value == Ellipsis
if GAST2:
  is_constant = _is_constant_gast_2
  is_ellipsis = _is_ellipsis_gast_2
  Module = gast.Module
  Name = gast.Name
  Str = gast.Str
elif GAST3:
  is_constant = _is_constant_gast_3
  is_ellipsis = _is_ellipsis_gast_3
else:
  assert False
