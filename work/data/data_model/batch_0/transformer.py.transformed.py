
import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
class AnalysisLevel(enum.IntEnum):
  NONE = 0
  ACTIVITY = 1
  DEFINEDNESS = 2
  LIVENESS = 3
class Context(object):
  def __init__(self, info, namer, user_context):
    self.info = info
    self.namer = namer
    self.current_origin = None
    self.user = user_context
class EntityInfo(
    collections.namedtuple(
        'EntityInfo',
        ('name', 'source_code', 'source_file', 'future_features', 'namespace'))
):
  """Contains information about a Python entity.
  Immutable.
  Examples of entities include functions and classes.
  Attributes:
    name: The name that identifies this entity.
    source_code: The entity's source code.
    source_file: The entity's source file.
    future_features: Tuple[Text], the future features that this entity was
      compiled with. See
    namespace: Dict[str, ], containing symbols visible to the entity (excluding
      parameters).
  """
  pass
class _StateStack(object):
  """Templated context manager.
  This class provides syntactic sugar for a stack of objects of known
  type. It allows accessing attributes of the object at the top of the stack
  directly against this object, which allows for very terse syntax.
  For example, this code:
    stack = _StateStack(Foo)
    stack.enter()
    stack.bar
  Is equivalent to:
    stack = []
    stack.append(Foo())
    foo = stack[-1]
    foo.bar
  See _State for more on how this is used.
  Attributes:
    type: Any, the type of objects that this stack holds
    level: int, the current stack depth
    stack: List[Any], the actual stack
    value: Any, the instance of the object at the top of the stack
  """
  def __init__(self, type_):
    object.__setattr__(self, 'type', type_)
    object.__setattr__(self, '_stack', [])
    if not hasattr(type_, 'no_root'):
      self.enter()
  def __enter__(self):
    self.enter()
    return self
  def __exit__(self, exc_type, exc_value, traceback):
    self.exit()
  def enter(self):
    self._stack.append(self.type())
  def exit(self):
    self._stack.pop()
  @property
  def stack(self):
    return self._stack
  @property
  def level(self):
    return len(self._stack)
  @property
  def value(self):
    return self._stack[-1]
  def __iter__(self):
    return iter(self._stack)
  def __getattr__(self, key):
    return getattr(self._stack[-1], key)
  def __setattr__(self, key, value):
    setattr(self._stack[-1], key, value)
class _State(object):
  """Syntactic sugar for accessing an instance of a StateStack context manager.
  This structure offers syntactic sugar over a dict of stacks of objects
  of known type. These structures are useful to keep state during AST walks.
  Multiple different scopes can be tracked in parallel. For example:
    s = _State()
    s[foo].enter()
  Element access has special semantics:
    * keys are a data type
    * element values are _StateStack(type=key) objects
    * missing elements are automatically added, similarly to defaultdict
  For example, the following block :
    _State s
    s[Foo]
  Is equivalent to:
    s = {}
    if Foo not in s:
      s[Foo] = Foo()
    s[Foo]
  See Base for how it's used.
  """
  def __init__(self):
    self._value = {}
  def __getitem__(self, key):
    if key not in self._value:
      self._value[key] = _StateStack(key)
    return self._value[key]
class NodeStateTracker(object):
  """Base class for general-purpose Python code transformation.
  This abstract class provides helpful functions, like state tracking within
  the scope of arbitrary node, helpers for processing code blocks, debugging,
  mapping of transformed code to original code, and others.
  Scope-local state tracking: to keep state across nodes, at the level of
  (possibly nested) scopes, use enter/exit_local_scope and set/get_local.
  You must call enter/exit_local_scope manually, but the transformer detects
  when they are not properly paired.
  The transformer allows keeping state across calls that is local
  to arbitrary nodes and their descendants, using the self.state attribute.
  Multiple independent scopes are allowed and automatically constructed.
  For example, to keep track of the `If` node that encloses any `Name` node,
  one can write:
  ```
    class FooType(object):
      def __init__(self):
        self.foo_property = None
    class DummyTransformer(NodeStateTracker, ast.NodeTransformer):
      def visit_If(self, node):
        self.state[FooType].enter()
        self.state[FooType].foo_property = node
        node = self.veneric_visit(node)
        self.state[FooType].exit()
        return node
      def visit_Name(self, node):
  ```
  Alternatively, the `enter()`/`exit()` calls can be managed by a `with`
  statement:
  ```
      def visit_If(self, node):
        with self.state[FooType] as foo:
          foo.foo_property = node
          return self.generic_visit(node)
  ```
  """
  def __init__(self, ctx):
    self._lineno = 0
    self._col_offset = 0
    self.ctx = ctx
    self.state = _State()
  def debug_print(self, node):
    if __debug__:
      print(pretty_printer.fmt(node))
    return node
  def debug_print_src(self, node):
    if __debug__:
      print(parser.unparse(node))
    return node
  def visit_block(self, nodes, before_visit=None, after_visit=None):
    """A more powerful version of generic_visit for statement blocks.
    An example of a block is the body of an if statement.
    This function allows specifying a postprocessing callback (the
    after_visit argument) argument which can be used to move nodes to a new
    destination. This is done by after_visit by returning a non-null
    second return value, e.g. return new_node, new_destination.
    For example, a transformer could perform the following move:
        foo()
        bar()
        baz()
        foo()
        if cond:
          bar()
          baz()
    The above could be done with a postprocessor of this kind:
        def after_visit(node):
          if node_is_function_call(bar):
            new_container_node = build_cond()
            new_container_node.body.append(node)
            return new_container_node, new_container_node.body
          else:
            return node, None
    Args:
      nodes: enumerable of AST node objects. If None, the function returns None.
      before_visit: optional callable that is called before visiting each item
        in nodes
      after_visit: optional callable that takes in an AST node and returns a
        tuple (new_node, new_destination). It is called after visiting each item
        in nodes. Is used in the same was as the
          visit_* methods: new_node will replace the node; if not None,
            new_destination must be a list, and subsequent nodes will be placed
            in this list instead of the list returned by visit_block.
    Returns:
      A list of AST node objects containing the transformed items fron nodes,
      except those nodes that have been relocated using after_visit.
    """
    if nodes is None:
      return None
    results = []
    node_destination = results
    for node in nodes:
      if before_visit:
        before_visit()
      replacement = self.visit(node)
      if after_visit and replacement:
        replacement, new_destination = after_visit(replacement)
      else:
        new_destination = None
      if replacement:
        if isinstance(replacement, (list, tuple)):
          node_destination.extend(replacement)
        else:
          node_destination.append(replacement)
      if new_destination is not None:
        node_destination = new_destination
    return results
class Base(NodeStateTracker, gast.NodeTransformer):
  def create_assignment(self, target, expression):
    template =
    return templates.replace(template, target=target, expression=expression)
  def apply_to_single_assignments(self, targets, values, apply_fn):
    """Applies a function to each individual assignment.
    This function can process a possibly-unpacked (e.g. a, b = c, d) assignment.
    It tries to break down the unpacking if possible. In effect, it has the same
    effect as passing the assigned values in SSA form to apply_fn.
    Examples:
    The following will result in apply_fn(a, c), apply_fn(b, d):
        a, b = c, d
    The following will result in apply_fn(a, c[0]), apply_fn(b, c[1]):
        a, b = c
    The following will result in apply_fn(a, (b, c)):
        a = b, c
    It uses the visitor pattern to allow subclasses to process single
    assignments individually.
    Args:
      targets: list, tuple of or individual AST node. Should be used with the
        targets field of an ast.Assign node.
      values: an AST node.
      apply_fn: a function of a single argument, which will be called with the
        respective nodes of each single assignment. The signature is
        apply_fn(target, value), no return value.
    """
    if not isinstance(targets, (list, tuple)):
      targets = (targets,)
    for target in targets:
      if isinstance(target, (gast.Tuple, gast.List)):
        for i in range(len(target.elts)):
          target_el = target.elts[i]
          if isinstance(values, (gast.Tuple, gast.List)):
            value_el = values.elts[i]
          else:
            value_el = gast.Subscript(values, i, ctx=gast.Store())
          self.apply_to_single_assignments(target_el, value_el, apply_fn)
      else:
        apply_fn(target, values)
  def visit(self, node):
    if not isinstance(node, gast.AST):
      msg = ('invalid value for "node": expected "ast.AST", got "{}"; to'
             ' visit lists of nodes, use "visit_block" instead').format(
                 type(node))
      raise ValueError(msg)
    if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
      return node
    parent_origin = self.ctx.current_origin
    if anno.hasanno(node, anno.Basic.ORIGIN):
      self.ctx.current_origin = anno.getanno(node, anno.Basic.ORIGIN)
    try:
      processing_expr_node = isinstance(node, gast.Expr)
      if processing_expr_node:
        entry_expr_value = node.value
      result = super(Base, self).visit(node)
      if (processing_expr_node and isinstance(result, gast.Expr) and
          (result.value is not entry_expr_value)):
        if isinstance(result.value,
                      (list, tuple, gast.Assign, gast.AugAssign)):
          result = result.value
      if result is not node and result is not None:
        inherited_origin = anno.getanno(
            node, anno.Basic.ORIGIN, default=parent_origin)
        if inherited_origin is not None:
          nodes_to_adjust = result
          if isinstance(result, (list, tuple)):
            nodes_to_adjust = result
          else:
            nodes_to_adjust = (result,)
          for n in nodes_to_adjust:
            if not anno.hasanno(n, anno.Basic.ORIGIN):
              anno.setanno(n, anno.Basic.ORIGIN, inherited_origin)
    finally:
      self.ctx.current_origin = parent_origin
    return result
class CodeGenerator(NodeStateTracker, gast.NodeVisitor):
  """Base class for general-purpose Python-to-string code transformation.
  Similar to Base, but outputs arbitrary strings instead of a Python AST.
  This uses the same visitor mechanism that the standard NodeVisitor uses,
  meaning that subclasses write handlers for the different kinds of nodes.
  New code is generated using the emit method, which appends to a code buffer
  that can be afterwards obtained from code_buffer.
  Example:
    class SimpleCodeGen(CodeGenerator):
      def visitIf(self, node):
        self.emit('if ')
        self.visit(node.test)
        self.emit(' { ')
        self.visit(node.body)
        self.emit(' } else { ')
        self.visit(node.orelse)
        self.emit(' } ')
    node = ast.parse(...)
    gen = SimpleCodeGen()
    gen.visit(node)
  """
  def __init__(self, ctx):
    super(CodeGenerator, self).__init__(ctx)
    self._output_code = ''
    self.source_map = {}
  def emit(self, code):
    self._output_code += code
  @property
  def code_buffer(self):
    return self._output_code
  def visit(self, node):
    if anno.hasanno(node, anno.Basic.SKIP_PROCESSING):
      return
    parent_origin = self.ctx.current_origin
    eof_before = len(self._output_code)
    if anno.hasanno(node, anno.Basic.ORIGIN):
      self.ctx.current_origin = anno.getanno(node, anno.Basic.ORIGIN)
    try:
      ret = super(CodeGenerator, self).visit(node)
      eof_after = len(self._output_code)
      if eof_before - eof_after:
        inherited_origin = anno.getanno(
            node, anno.Basic.ORIGIN, default=parent_origin)
        if inherited_origin is not None:
          self.source_map[(eof_before, eof_after)] = inherited_origin
      return ret
    finally:
      self.ctx.current_origin = parent_origin
