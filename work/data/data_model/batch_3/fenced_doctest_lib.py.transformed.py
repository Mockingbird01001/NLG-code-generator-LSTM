
import ast
import doctest
import os
import re
import textwrap
from typing import Any, Callable, Dict, Iterable, Optional
import astor
from tensorflow.tools.docs import tf_doctest_lib
def load_from_files(
    files,
    globs: Optional[Dict[str, Any]] = None,
    set_up: Optional[Callable[[Any], None]] = None,
    tear_down: Optional[Callable[[Any], None]] = None) -> doctest.DocFileSuite:
  if globs is None:
    globs = {}
  files = [os.fspath(f) for f in files]
  globs['_print_if_not_none'] = _print_if_not_none
  return doctest.DocFileSuite(
      *files,
      module_relative=False,
      parser=FencedCellParser(fence_label='python'),
      globs=globs,
      setUp=set_up,
      tearDown=tear_down,
      checker=FencedCellOutputChecker(),
      optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
                   | doctest.IGNORE_EXCEPTION_DETAIL
                   | doctest.DONT_ACCEPT_BLANKLINE),
  )
class FencedCellOutputChecker(tf_doctest_lib.TfDoctestOutputChecker):
  MESSAGE = textwrap.dedent("""\n
        """)
class FencedCellParser(doctest.DocTestParser):
  patched = False
  def __init__(self, fence_label='python'):
    super().__init__()
    if not self.patched:
      doctest.compile = _patch_compile
      print(
          textwrap.dedent("""
          *********************************************************************
          * Caution: `fenced_doctest` patches `doctest.compile` don't use this
          *   in the same binary as any other doctests.
          *********************************************************************
          """))
      type(self).patched = True
    no_fence = '(.(?<!```))*?'
    self.fence_cell_re = re.compile(
        r
,
        re.MULTILINE |
        re.DOTALL |
        re.VERBOSE)
  def get_examples(self,
                   string: str,
                   name: str = '<string>') -> Iterable[doctest.Example]:
    if re.search('<!--.*?doctest.*?skip.*?all.*?-->', string, re.IGNORECASE):
      return
    for match in self.fence_cell_re.finditer(string):
      if re.search('doctest.*skip', match.group(0), re.IGNORECASE):
        continue
      groups = match.groupdict()
      source = textwrap.dedent(groups['doctest'])
      want = groups['output']
      if want is not None:
        want = textwrap.dedent(want)
      yield doctest.Example(
          lineno=string[:match.start()].count('\n') + 1,
          source=source,
          want=want)
def _print_if_not_none(obj):
  if obj is not None:
    print(repr(obj))
def _patch_compile(source,
                   filename,
                   mode,
                   flags=0,
                   dont_inherit=False,
                   optimize=-1):
  """Patch `doctest.compile` to make doctest to behave like a notebook.
  Default settings for doctest are configured to run like a repl: one statement
  at a time. The doctest source uses `compile(..., mode="single")`
  So to let doctest act like a notebook:
  1. We need `mode="exec"` (easy)
  2. We need the last expression to be printed (harder).
  To print the last expression, just wrap the last expression in
  `_print_if_not_none(expr)`. To detect the last expression use `AST`.
  if the last node is an expression modify the ast to to call
  `_print_if_not_none` on it, convert the ast back to source and compile that.
  Args:
    source: Can either be a normal string, a byte string, or an AST object.
    filename: Argument should give the file from which the code was read; pass
      some recognizable value if it wasn’t read from a file ('<string>' is
      commonly used).
    mode: [Ignored] always use exec.
    flags: Compiler options.
    dont_inherit: Compiler options.
    optimize: Compiler options.
  Returns:
    The resulting code object.
  """
  del filename
  del mode
  source_ast = ast.parse(source)
  final = source_ast.body[-1]
  if isinstance(final, ast.Expr):
    print_it = ast.Expr(
        lineno=-1,
        col_offset=-1,
        value=ast.Call(
            func=ast.Name(
                id='_print_if_not_none',
                ctx=ast.Load(),
                lineno=-1,
                col_offset=-1),
            lineno=-1,
            col_offset=-1,
            keywords=[]))
    source_ast.body[-1] = print_it
    source = astor.to_source(source_ast)
  return compile(
      source,
      filename='dummy.py',
      mode='exec',
      flags=flags,
      dont_inherit=dont_inherit,
      optimize=optimize)
