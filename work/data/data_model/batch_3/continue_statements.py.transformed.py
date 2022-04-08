
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis.annos import NodeAnno
class _Continue(object):
  def __init__(self):
    self.used = False
    self.control_var_name = None
  def __repr__(self):
    return '<_Continue(used: {}, var: {})>'.format(self.used,
                                                   self.control_var_name)
class _Block(object):
  """Tracks information about lexical blocks as they are visited in the AST.
  Mainly, this object tracks the creation of block guards that replace
  `continue` statements (e.g. `if not continue_:`).
  Attributes:
    create_guard_current: bool, whether to create a guard for the current
      statement.
    create_guard_next: bool, whether to create a guard for the next
      statement.
    is_loop_type: bool, whether this block is the body of a loop.
  """
  def __init__(self):
    self.is_loop_type = False
    self.create_guard_current = False
    self.create_guard_next = False
class ContinueCanonicalizationTransformer(converter.Base):
  def visit_Continue(self, node):
    self.state[_Continue].used = True
    for block in reversed(self.state[_Block].stack):
      block.create_guard_next = True
      if block.is_loop_type:
        break
    template =
    return templates.replace(
        template, var_name=self.state[_Continue].control_var_name)
  def _postprocess_statement(self, node):
    if self.state[_Continue].used:
      block = self.state[_Block]
      should_wrap_current = block.create_guard_current
      block.create_guard_current = block.create_guard_next
      block.create_guard_next = False
      if should_wrap_current:
        template =
        cond, = templates.replace(
            template,
            var_name=self.state[_Continue].control_var_name,
            original_node=node)
        return cond, cond.body
    return node, None
  def _visit_loop_body(self, node, nodes):
    self.state[_Continue].enter()
    self.state[_Block].enter()
    self.state[_Block].is_loop_type = True
    scope = anno.getanno(node, NodeAnno.BODY_SCOPE)
    continue_var = self.ctx.namer.new_symbol('continue_', scope.referenced)
    self.state[_Continue].control_var_name = continue_var
    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    if self.state[_Continue].used:
      template =
      control_var_init = templates.replace(template, var_name=continue_var)
      nodes = control_var_init + nodes
    self.state[_Block].exit()
    self.state[_Continue].exit()
    return nodes
  def _visit_non_loop_body(self, nodes):
    self.state[_Block].enter()
    nodes = self.visit_block(nodes, after_visit=self._postprocess_statement)
    self.state[_Block].exit()
    return nodes
  def visit_While(self, node):
    node.test = self.visit(node.test)
    node.body = self._visit_loop_body(node, node.body)
    node.orelse = self._visit_non_loop_body(node.orelse)
    return node
  def visit_For(self, node):
    node.target = self.generic_visit(node.target)
    node.iter = self.generic_visit(node.iter)
    node.body = self._visit_loop_body(node, node.body)
    node.orelse = self._visit_non_loop_body(node.orelse)
    return node
  def visit_If(self, node):
    node.body = self._visit_non_loop_body(node.body)
    node.orelse = self._visit_non_loop_body(node.orelse)
    return node
  def visit_With(self, node):
    node.items = self.visit_block(node.items)
    node.body = self._visit_non_loop_body(node.body)
    return node
  def visit_Try(self, node):
    node.body = self._visit_non_loop_body(node.body)
    node.orelse = self._visit_non_loop_body(node.orelse)
    node.finalbody = self._visit_non_loop_body(node.finalbody)
    node.handlers = self.visit_block(node.handlers)
    return node
  def visit_ExceptHandler(self, node):
    node.body = self._visit_non_loop_body(node.body)
    return node
def transform(node, ctx):
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)
  node = ContinueCanonicalizationTransformer(ctx).visit(node)
  return node
