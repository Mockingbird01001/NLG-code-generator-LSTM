
"""Converter construction support.
This module contains a base class for all converters, as well as supporting
structures. These structures are referred to as contexts.
The class hierarchy is as follows:
    <your converter>
      [extends] converter.Base
        [extends] transformer.Base
            [extends] gast.nodeTransformer
          [uses] transformer.SourceInfo
        [uses] converter.EntityContext
          [uses] converter.ProgramContext
          [uses] transformer.SourceInfo
converter.Base is a specialization of transformer.Base for AutoGraph. It's a
very lightweight subclass that adds a `ctx` attribute holding the corresponding
EntityContext object (see below). Note that converters are not reusable, and
`visit` will raise an error if called more than once.
converter.EntityContext contains mutable state associated with an entity that
the converter processes.
converter.ProgramContext contains mutable state across related entities. For
example, when converting several functions that call one another, the
ProgramContext should be shared across these entities.
Below is the overall flow at conversion:
    program_ctx = ProgramContext(<entities to convert>, <global settings>, ...)
    while <program_ctx has more entities to convert>:
      entity, source_info = <get next entity from program_ctx>
      entity_ctx = EntityContext(program_ctx, source_info)
      for <each ConverterClass>:
        converter = ConverterClass(entity_ctx)
        entity = converter.visit(entity)
      <add entity's dependencies to program_ctx>
Note that pyct contains a small number of transformers used for static analysis.
These implement transformer.Base, rather than converter.Base, to avoid a
dependency on AutoGraph.
"""
import enum
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.util.tf_export import tf_export
@tf_export('autograph.experimental.Feature')
class Feature(enum.Enum):
  """This enumeration represents optional conversion options.
  These conversion options are experimental. They are subject to change without
  notice and offer no guarantees.
  _Example Usage_
  ```python
  optionals= tf.autograph.experimental.Feature.EQUALITY_OPERATORS
  @tf.function(experimental_autograph_options=optionals)
  def f(i):
      tf.print('i is zero')
  ```
  Attributes:
    ALL: Enable all features.
    AUTO_CONTROL_DEPS: Insert of control dependencies in the generated code.
    ASSERT_STATEMENTS: Convert Tensor-dependent assert statements to tf.Assert.
    BUILTIN_FUNCTIONS: Convert builtin functions applied to Tensors to
      their TF counterparts.
    EQUALITY_OPERATORS: Whether to convert the equality operator ('==') to
      tf.math.equal.
    LISTS: Convert list idioms, like initializers, slices, append, etc.
    NAME_SCOPES: Insert name scopes that name ops according to context, like the
      function they were defined in.
  """
  ALL = 'ALL'
  AUTO_CONTROL_DEPS = 'AUTO_CONTROL_DEPS'
  ASSERT_STATEMENTS = 'ASSERT_STATEMENTS'
  BUILTIN_FUNCTIONS = 'BUILTIN_FUNCTIONS'
  EQUALITY_OPERATORS = 'EQUALITY_OPERATORS'
  LISTS = 'LISTS'
  NAME_SCOPES = 'NAME_SCOPES'
  @classmethod
  def all(cls):
    return tuple(cls.__members__.values())
  @classmethod
  def all_but(cls, exclude):
    if not isinstance(exclude, (list, tuple, set)):
      exclude = (exclude,)
    return tuple(set(cls.all()) - set(exclude) - {cls.ALL})
class ConversionOptions(object):
  def __init__(self,
               recursive=False,
               user_requested=False,
               internal_convert_user_code=True,
               optional_features=Feature.ALL):
    self.recursive = recursive
    self.user_requested = user_requested
    self.internal_convert_user_code = internal_convert_user_code
    if optional_features is None:
      optional_features = ()
    elif isinstance(optional_features, Feature):
      optional_features = (optional_features,)
    optional_features = frozenset(optional_features)
    self.optional_features = optional_features
  def as_tuple(self):
    return (self.recursive, self.user_requested,
            self.internal_convert_user_code, self.optional_features)
  def __hash__(self):
    return hash(self.as_tuple())
  def __eq__(self, other):
    assert isinstance(other, ConversionOptions)
    return self.as_tuple() == other.as_tuple()
  def __str__(self):
    return 'ConversionOptions[{}]'
  def uses(self, feature):
    return (Feature.ALL in self.optional_features or
            feature in self.optional_features)
  def call_options(self):
    return ConversionOptions(
        recursive=self.recursive,
        user_requested=False,
        internal_convert_user_code=self.recursive,
        optional_features=self.optional_features)
  def to_ast(self):
    if self == STANDARD_OPTIONS:
      return parser.parse_expression('ag__.STD')
    template = """
      ag__.ConversionOptions(
          recursive=recursive_val,
          user_requested=user_requested_val,
          optional_features=optional_features_val,
          internal_convert_user_code=internal_convert_user_code_val)
    """
    def list_of_features(values):
      return parser.parse_expression('({})'.format(', '.join(
          'ag__.{}'.format(str(v)) for v in values)))
    expr_ast = templates.replace(
        template,
        recursive_val=parser.parse_expression(str(self.recursive)),
        user_requested_val=parser.parse_expression(str(self.user_requested)),
        internal_convert_user_code_val=parser.parse_expression(
            str(self.internal_convert_user_code)),
        optional_features_val=list_of_features(self.optional_features))
    return expr_ast[0].value
STANDARD_OPTIONS = ConversionOptions(
    recursive=True,
    user_requested=False,
    internal_convert_user_code=True,
    optional_features=None)
class ProgramContext(object):
  def __init__(self, options, autograph_module=None):
    self.options = options
    self.autograph_module = autograph_module
class Base(transformer.Base):
  def __init__(self, ctx):
    super(Base, self).__init__(ctx)
    self._used = False
    self._ast_depth = 0
  def get_definition_directive(self, node, directive, arg, default):
    """Returns the unique directive argument for a symbol.
    See lang/directives.py for details on directives.
    Example:
       ag.foo_directive(bar, baz=1)
       get_definition_directive(node, ag.foo_directive, 'baz')
    Args:
      node: ast.AST, the node representing the symbol for which the directive
        argument is needed.
      directive: Callable[..., Any], the directive to search.
      arg: str, the directive argument to return.
      default: Any
    Raises:
      ValueError: if conflicting annotations have been found
    """
    defs = anno.getanno(node, anno.Static.ORIG_DEFINITIONS, ())
    if not defs:
      return default
    arg_values_found = []
    for def_ in defs:
      if (directive in def_.directives and arg in def_.directives[directive]):
        arg_values_found.append(def_.directives[directive][arg])
    if not arg_values_found:
      return default
    if len(arg_values_found) == 1:
      return arg_values_found[0]
    first_value = arg_values_found[0]
    for other_value in arg_values_found[1:]:
      if not ast_util.matches(first_value, other_value):
        qn = anno.getanno(node, anno.Basic.QN)
        raise ValueError(
            '%s has ambiguous annotations for %s(%s): %s, %s' %
            (qn, directive.__name__, arg, parser.unparse(other_value).strip(),
             parser.unparse(first_value).strip()))
    return first_value
  def visit(self, node):
    if not self._ast_depth:
      if self._used:
        raise ValueError('converter objects cannot be reused')
      self._used = True
    self._ast_depth += 1
    try:
      return super(Base, self).visit(node)
    finally:
      self._ast_depth -= 1
