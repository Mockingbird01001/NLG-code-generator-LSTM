
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import annos
class Analyzer(cfg.GraphVisitor):
  def __init__(self, graph, include_annotations):
    super(Analyzer, self).__init__(graph)
    self.include_annotations = include_annotations
  def init_state(self, _):
    return set()
  def visit_node(self, node):
    prev_live_in = self.in_[node]
    if anno.hasanno(node.ast_node, anno.Static.SCOPE):
      node_scope = anno.getanno(node.ast_node, anno.Static.SCOPE)
      gen = node_scope.read
      if not self.include_annotations:
        gen -= node_scope.annotations
      kill = node_scope.modified | node_scope.deleted
      live_out = set()
      for n in node.next:
        live_out |= self.in_[n]
      live_in = gen | (live_out - kill)
      reaching_functions = anno.getanno(
          node.ast_node, anno.Static.DEFINED_FNS_IN)
      for fn_ast_node in reaching_functions:
        if isinstance(fn_ast_node, gast.Lambda):
          continue
        fn_scope = anno.getanno(fn_ast_node, annos.NodeAnno.ARGS_AND_BODY_SCOPE)
        live_in |= (fn_scope.read - fn_scope.bound)
    else:
      assert self.can_ignore(node), (node.ast_node, node)
      live_out = set()
      for n in node.next:
        live_out |= self.in_[n]
      live_in = live_out
    self.in_[node] = live_in
    self.out[node] = live_out
    return prev_live_in != live_in
class TreeAnnotator(transformer.Base):
  """Runs liveness analysis on each of the functions defined in the AST.
  If a function defined other local functions, those will have separate CFGs.
  However, dataflow analysis needs to tie up these CFGs to properly emulate the
  effect of closures. In the case of liveness, the parent function's live
  variables must account for the variables that are live at the entry of each
  subfunction. For example:
    def foo():
      def bar():
        print(baz)
  This analyzer runs liveness analysis on each individual function, accounting
  for the effect above.
  """
  def __init__(self, source_info, graphs, include_annotations):
    super(TreeAnnotator, self).__init__(source_info)
    self.include_annotations = include_annotations
    self.allow_skips = False
    self.graphs = graphs
    self.current_analyzer = None
  def visit(self, node):
    node = super(TreeAnnotator, self).visit(node)
    if (self.current_analyzer is not None and
        isinstance(node, gast.stmt) and
        node in self.current_analyzer.graph.index):
      cfg_node = self.current_analyzer.graph.index[node]
      anno.setanno(node, anno.Static.LIVE_VARS_IN,
                   frozenset(self.current_analyzer.in_[cfg_node]))
    return node
  def _analyze_function(self, node, is_lambda):
    parent_analyzer = self.current_analyzer
    analyzer = Analyzer(self.graphs[node], self.include_annotations)
    analyzer.visit_reverse()
    self.current_analyzer = analyzer
    node = self.generic_visit(node)
    self.current_analyzer = parent_analyzer
    return node
  def visit_Lambda(self, node):
    return self._analyze_function(node, is_lambda=True)
  def visit_FunctionDef(self, node):
    return self._analyze_function(node, is_lambda=False)
  def _block_statement_live_out(self, node):
    successors = self.current_analyzer.graph.stmt_next[node]
    stmt_live_out = set()
    for s in successors:
      stmt_live_out.update(self.current_analyzer.in_[s])
    anno.setanno(node, anno.Static.LIVE_VARS_OUT, frozenset(stmt_live_out))
    return node
  def _block_statement_live_in(self, node, entry_node):
    if entry_node in self.current_analyzer.graph.index:
      cfg_node = self.current_analyzer.graph.index[entry_node]
      stmt_live_in = frozenset(self.current_analyzer.in_[cfg_node])
    else:
      assert anno.hasanno(entry_node, anno.Static.LIVE_VARS_IN), (
          'If not matching a CFG node, must be a block statement:'
          ' {}'.format(entry_node))
      stmt_live_in = anno.getanno(entry_node, anno.Static.LIVE_VARS_IN)
    anno.setanno(node, anno.Static.LIVE_VARS_IN, stmt_live_in)
    return node
  def visit_If(self, node):
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.test)
  def visit_For(self, node):
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.iter)
  def visit_While(self, node):
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.test)
  def visit_Try(self, node):
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.body[0])
  def visit_ExceptHandler(self, node):
    node = self.generic_visit(node)
    node = self._block_statement_live_out(node)
    return self._block_statement_live_in(node, node.body[0])
  def visit_With(self, node):
    node = self.generic_visit(node)
    return self._block_statement_live_in(node, node.items[0])
  def visit_Expr(self, node):
    node = self.generic_visit(node)
    cfg_node = self.current_analyzer.graph.index[node]
    anno.setanno(node, anno.Static.LIVE_VARS_OUT,
                 frozenset(self.current_analyzer.out[cfg_node]))
    return node
def resolve(node, source_info, graphs, include_annotations=True):
  node = TreeAnnotator(source_info, graphs, include_annotations).visit(node)
  return node
