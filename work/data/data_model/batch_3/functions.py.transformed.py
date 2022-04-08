
import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
class _Function(object):
  def __init__(self):
    self.context_name = None
class FunctionTransformer(converter.Base):
  def _function_scope_options(self, fn_scope):
    if fn_scope.level == 2:
      return self.ctx.user.options
    return self.ctx.user.options.call_options()
  def visit_Lambda(self, node):
    with self.state[_Function] as fn_scope:
      node = self.generic_visit(node)
      if fn_scope.level > 2:
        return templates.replace_as_expression(
            'ag__.autograph_artifact(l)', l=node)
      scope = anno.getanno(node, anno.Static.SCOPE)
      function_context_name = self.ctx.namer.new_symbol('lscope',
                                                        scope.referenced)
      fn_scope.context_name = function_context_name
      anno.setanno(node, 'function_context_name', function_context_name)
      template = """
        ag__.with_function_scope(
            lambda function_context: body, function_context_name, options)
      """
      node.body = templates.replace_as_expression(
          template,
          options=self._function_scope_options(fn_scope).to_ast(),
          function_context=function_context_name,
          function_context_name=gast.Constant(function_context_name, kind=None),
          body=node.body)
      return node
  def visit_FunctionDef(self, node):
    with self.state[_Function] as fn_scope:
      scope = anno.getanno(node, annos.NodeAnno.BODY_SCOPE)
      function_context_name = self.ctx.namer.new_symbol('fscope',
                                                        scope.referenced)
      fn_scope.context_name = function_context_name
      anno.setanno(node, 'function_context_name', function_context_name)
      node = self.generic_visit(node)
      if fn_scope.level <= 2:
        node.decorator_list = []
      else:
        node.decorator_list.append(
            parser.parse_expression('ag__.autograph_artifact'))
      docstring_node = None
      if node.body:
        first_statement = node.body[0]
        if (isinstance(first_statement, gast.Expr) and
            isinstance(first_statement.value, gast.Constant)):
          docstring_node = first_statement
          node.body = node.body[1:]
      template = """
        with ag__.FunctionScope(
            function_name, context_name, options) as function_context:
          body
      """
      wrapped_body = templates.replace(
          template,
          function_name=gast.Constant(node.name, kind=None),
          context_name=gast.Constant(function_context_name, kind=None),
          options=self._function_scope_options(fn_scope).to_ast(),
          function_context=function_context_name,
          body=node.body)
      if docstring_node is not None:
        wrapped_body = [docstring_node] + wrapped_body
      node.body = wrapped_body
      return node
def transform(node, ctx):
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx, None)
  return FunctionTransformer(ctx).visit(node)
