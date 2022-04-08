
"""Lowers list comprehensions into for and if statements.
Example:
  result = [x * x for x in xs]
becomes
  result = []
  for x in xs:
    elt = x * x
    result.append(elt)
"""
import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import templates
class ListCompTransformer(converter.Base):
  def visit_Assign(self, node):
    if not isinstance(node.value, gast.ListComp):
      return self.generic_visit(node)
    if len(node.targets) > 1:
      raise NotImplementedError('multiple assignments')
    target, = node.targets
    list_comp_node = node.value
    template =
    initialization = templates.replace(template, target=target)
    template = """
      target.append(elt)
    """
    body = templates.replace(template, target=target, elt=list_comp_node.elt)
    for gen in reversed(list_comp_node.generators):
      for gen_if in reversed(gen.ifs):
        template =
        body = templates.replace(template, test=gen_if, body=body)
      template =
      body = templates.replace(
          template, iter_=gen.iter, target=gen.target, body=body)
    return initialization + body
def transform(node, ctx):
  return ListCompTransformer(ctx).visit(node)
