
"""An analysis that determines the reach of a function definition.
A function definition is said to reach a statement if that function may exist
(and therefore may be called) when that statement executes.
"""
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
class Definition(object):
  def __init__(self, def_node):
    self.def_node = def_node
class _NodeState(object):
  def __init__(self, init_from=None):
    if init_from:
      self.value = set(init_from)
    else:
      self.value = set()
  def __eq__(self, other):
    return self.value == other.value
  def __ne__(self, other):
    return self.value != other.value
  def __or__(self, other):
    assert isinstance(other, _NodeState)
    result = _NodeState(self.value)
    result.value.update(other.value)
    return result
  def __add__(self, value):
    result = _NodeState(self.value)
    result.value.add(value)
    return result
  def __repr__(self):
    return 'NodeState[%s]=%s' % (id(self), repr(self.value))
class Analyzer(cfg.GraphVisitor):
  def __init__(self, graph, external_defs):
    super(Analyzer, self).__init__(graph)
    self.external_defs = external_defs
  def init_state(self, _):
    return _NodeState()
  def visit_node(self, node):
    prev_defs_out = self.out[node]
    if node is self.graph.entry:
      defs_in = _NodeState(self.external_defs)
    else:
      defs_in = prev_defs_out
    for n in node.prev:
      defs_in |= self.out[n]
    defs_out = defs_in
    if isinstance(node.ast_node, (gast.Lambda, gast.FunctionDef)):
      defs_out += node.ast_node
    self.in_[node] = defs_in
    self.out[node] = defs_out
    return prev_defs_out != defs_out
class TreeAnnotator(transformer.Base):
  """AST visitor that annotates each symbol name with its reaching definitions.
  Simultaneously, the visitor runs the dataflow analysis on each function node,
  accounting for the effect of closures. For example:
    def foo():
      def f():
        pass
      def g():
  """
  def __init__(self, source_info, graphs):
    super(TreeAnnotator, self).__init__(source_info)
    self.graphs = graphs
    self.allow_skips = False
    self.current_analyzer = None
  def _proces_function(self, node):
    parent_analyzer = self.current_analyzer
    subgraph = self.graphs[node]
    if (self.current_analyzer is not None
        and node in self.current_analyzer.graph.index):
      cfg_node = self.current_analyzer.graph.index[node]
      defined_in = self.current_analyzer.in_[cfg_node].value
    else:
      defined_in = ()
    analyzer = Analyzer(subgraph, defined_in)
    analyzer.visit_forward()
    self.current_analyzer = analyzer
    node = self.generic_visit(node)
    self.current_analyzer = parent_analyzer
    return node
  def visit_FunctionDef(self, node):
    return self._proces_function(node)
  def visit_Lambda(self, node):
    return self._proces_function(node)
  def visit(self, node):
    if (self.current_analyzer is not None
        and node in self.current_analyzer.graph.index):
      cfg_node = self.current_analyzer.graph.index[node]
      anno.setanno(node, anno.Static.DEFINED_FNS_IN,
                   self.current_analyzer.in_[cfg_node].value)
    extra_node = anno.getanno(node, anno.Basic.EXTRA_LOOP_TEST, default=None)
    if extra_node is not None:
      cfg_node = self.current_analyzer.graph.index[extra_node]
      anno.setanno(extra_node, anno.Static.DEFINED_FNS_IN,
                   self.current_analyzer.in_[cfg_node].value)
    return super(TreeAnnotator, self).visit(node)
def resolve(node, source_info, graphs):
  visitor = TreeAnnotator(source_info, graphs)
  node = visitor.visit(node)
  return node
