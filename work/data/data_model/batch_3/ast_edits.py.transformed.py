
import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
import six
from six.moves import range
FIND_OPEN = re.compile(r"^\s*(\[).*$")
FIND_STRING_CHARS = re.compile(r"['\"]")
INFO = "INFO"
WARNING = "WARNING"
ERROR = "ERROR"
ImportRename = collections.namedtuple(
    "ImportRename", ["new_name", "excluded_prefixes"])
def full_name_node(name, ctx=ast.Load()):
  """Make an Attribute or Name node for name.
  Translate a qualified name into nested Attribute nodes (and a Name node).
  Args:
    name: The name to translate to a node.
    ctx: What context this name is used in. Defaults to Load()
  Returns:
    A Name or Attribute node.
  """
  names = six.ensure_str(name).split(".")
  names.reverse()
  node = ast.Name(id=names.pop(), ctx=ast.Load())
  while names:
    node = ast.Attribute(value=node, attr=names.pop(), ctx=ast.Load())
  node.ctx = ctx
  return node
def get_arg_value(node, arg_name, arg_pos=None):
  """Get the value of an argument from a ast.Call node.
  This function goes through the positional and keyword arguments to check
  whether a given argument was used, and if so, returns its value (the node
  representing its value).
  This cannot introspect *args or **args, but it safely handles *args in
  Python3.5+.
  Args:
    node: The ast.Call node to extract arg values from.
    arg_name: The name of the argument to extract.
    arg_pos: The position of the argument (in case it's passed as a positional
      argument).
  Returns:
    A tuple (arg_present, arg_value) containing a boolean indicating whether
    the argument is present, and its value in case it is.
  """
  if arg_name is not None:
    for kw in node.keywords:
      if kw.arg == arg_name:
        return (True, kw.value)
  if arg_pos is not None:
    idx = 0
    for arg in node.args:
      if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
      if idx == arg_pos:
        return (True, arg)
      idx += 1
  return (False, None)
def uses_star_args_in_call(node):
  if sys.version_info[:2] >= (3, 5):
    for arg in node.args:
      if isinstance(arg, ast.Starred):
        return True
  else:
    if node.starargs:
      return True
  return False
def uses_star_kwargs_in_call(node):
  if sys.version_info[:2] >= (3, 5):
    for keyword in node.keywords:
      if keyword.arg is None:
        return True
  else:
    if node.kwargs:
      return True
  return False
def uses_star_args_or_kwargs_in_call(node):
  return uses_star_args_in_call(node) or uses_star_kwargs_in_call(node)
def excluded_from_module_rename(module, import_rename_spec):
  """Check if this module import should not be renamed.
  Args:
    module: (string) module name.
    import_rename_spec: ImportRename instance.
  Returns:
    True if this import should not be renamed according to the
    import_rename_spec.
  """
  for excluded_prefix in import_rename_spec.excluded_prefixes:
    if module.startswith(excluded_prefix):
      return True
  return False
class APIChangeSpec(object):
  """This class defines the transformations that need to happen.
  This class must provide the following fields:
  * `function_keyword_renames`: maps function names to a map of old -> new
    argument names
  * `symbol_renames`: maps function names to new function names
  * `change_to_function`: a set of function names that have changed (for
    notifications)
  * `function_reorders`: maps functions whose argument order has changed to the
    list of arguments in the new order
  * `function_warnings`: maps full names of functions to warnings that will be
    printed out if the function is used. (e.g. tf.nn.convolution())
  * `function_transformers`: maps function names to custom handlers
  * `module_deprecations`: maps module names to warnings that will be printed
    if the module is still used after all other transformations have run
  * `import_renames`: maps import name (must be a short name without '.')
    to ImportRename instance.
  For an example, see `TFAPIChangeSpec`.
  """
    return root_node, [], []
  def clear_preprocessing(self):
    pass
class NoUpdateSpec(APIChangeSpec):
  def __init__(self):
    self.function_handle = {}
    self.function_reorders = {}
    self.function_keyword_renames = {}
    self.symbol_renames = {}
    self.function_warnings = {}
    self.change_to_function = {}
    self.module_deprecations = {}
    self.function_transformers = {}
    self.import_renames = {}
class _PastaEditVisitor(ast.NodeVisitor):
  def __init__(self, api_change_spec):
    self._api_change_spec = api_change_spec
  def visit(self, node):
    self._stack.append(node)
    super(_PastaEditVisitor, self).visit(node)
    self._stack.pop()
  @property
  def errors(self):
    return [log for log in self._log if log[0] == ERROR]
  @property
  def warnings(self):
    return [log for log in self._log if log[0] == WARNING]
  @property
  def warnings_and_errors(self):
    return [log for log in self._log if log[0] in (WARNING, ERROR)]
  @property
  def info(self):
    return [log for log in self._log if log[0] == INFO]
  @property
  def log(self):
    return self._log
  def add_log(self, severity, lineno, col, msg):
    self._log.append((severity, lineno, col, msg))
    print("%s line %d:%d: %s" % (severity, lineno, col, msg))
  def add_logs(self, logs):
    """Record a log and print it.
    The log should be a tuple `(severity, lineno, col_offset, msg)`, which will
    be printed and recorded. It is part of the log available in the `self.log`
    property.
    Args:
      logs: The logs to add. Must be a list of tuples
        `(severity, lineno, col_offset, msg)`.
    """
    self._log.extend(logs)
    for log in logs:
      print("%s line %d:%d: %s" % log)
  def _get_applicable_entries(self, transformer_field, full_name, name):
    function_transformers = getattr(self._api_change_spec,
                                    transformer_field, {})
    glob_name = "*." + six.ensure_str(name) if name else None
    transformers = []
    if full_name in function_transformers:
      transformers.append(function_transformers[full_name])
    if glob_name in function_transformers:
      transformers.append(function_transformers[glob_name])
    if "*" in function_transformers:
      transformers.append(function_transformers["*"])
    return transformers
  def _get_applicable_dict(self, transformer_field, full_name, name):
    function_transformers = getattr(self._api_change_spec,
                                    transformer_field, {})
    glob_name = "*." + six.ensure_str(name) if name else None
    transformers = function_transformers.get("*", {}).copy()
    transformers.update(function_transformers.get(glob_name, {}))
    transformers.update(function_transformers.get(full_name, {}))
    return transformers
  def _get_full_name(self, node):
    """Traverse an Attribute node to generate a full name, e.g., "tf.foo.bar".
    This is the inverse of `full_name_node`.
    Args:
      node: A Node of type Attribute.
    Returns:
      a '.'-delimited full-name or None if node was not Attribute or Name.
      i.e. `foo()+b).bar` returns None, while `a.b.c` would return "a.b.c".
    """
    curr = node
    items = []
    while not isinstance(curr, ast.Name):
      if not isinstance(curr, ast.Attribute):
        return None
      items.append(curr.attr)
      curr = curr.value
    items.append(curr.id)
    return ".".join(reversed(items))
  def _maybe_add_warning(self, node, full_name):
    function_warnings = self._api_change_spec.function_warnings
    if full_name in function_warnings:
      level, message = function_warnings[full_name]
      message = six.ensure_str(message).replace("<function name>", full_name)
      self.add_log(level, node.lineno, node.col_offset,
                   "%s requires manual check. %s" % (full_name, message))
      return True
    else:
      return False
  def _maybe_add_module_deprecation_warning(self, node, full_name, whole_name):
    warnings = self._api_change_spec.module_deprecations
    if full_name in warnings:
      level, message = warnings[full_name]
      message = six.ensure_str(message).replace("<function name>",
                                                six.ensure_str(whole_name))
      self.add_log(level, node.lineno, node.col_offset,
                   "Using member %s in deprecated module %s. %s" % (whole_name,
                                                                    full_name,
                                                                    message))
      return True
    else:
      return False
  def _maybe_add_call_warning(self, node, full_name, name):
    """Print a warning when specific functions are called with selected args.
    The function _print_warning_for_function matches the full name of the called
    function, e.g., tf.foo.bar(). This function matches the function name that
    is called, as long as the function is an attribute. For example,
    `tf.foo.bar()` and `foo.bar()` are matched, but not `bar()`.
    Args:
      node: ast.Call object
      full_name: The precomputed full name of the callable, if one exists, None
        otherwise.
      name: The precomputed name of the callable, if one exists, None otherwise.
    Returns:
      Whether an error was recorded.
    """
    warned = False
    if isinstance(node.func, ast.Attribute):
      warned = self._maybe_add_warning(node, "*." + six.ensure_str(name))
    arg_warnings = self._get_applicable_dict("function_arg_warnings",
                                             full_name, name)
    variadic_args = uses_star_args_or_kwargs_in_call(node)
    for (kwarg, arg), (level, warning) in sorted(arg_warnings.items()):
      present, _ = get_arg_value(node, kwarg, arg) or variadic_args
      if present:
        warned = True
        warning_message = six.ensure_str(warning).replace(
            "<function name>", six.ensure_str(full_name or name))
        template = "%s called with %s argument, requires manual check: %s"
        if variadic_args:
          template = ("%s called with *args or **kwargs that may include %s, "
                      "requires manual check: %s")
        self.add_log(level, node.lineno, node.col_offset,
                     template % (full_name or name, kwarg, warning_message))
    return warned
  def _maybe_rename(self, parent, node, full_name):
    new_name = self._api_change_spec.symbol_renames.get(full_name, None)
    if new_name:
      self.add_log(INFO, node.lineno, node.col_offset,
                   "Renamed %r to %r" % (full_name, new_name))
      new_node = full_name_node(new_name, node.ctx)
      ast.copy_location(new_node, node)
      pasta.ast_utils.replace_child(parent, node, new_node)
      return True
    else:
      return False
  def _maybe_change_to_function_call(self, parent, node, full_name):
    if full_name in self._api_change_spec.change_to_function:
      if not isinstance(parent, ast.Call):
        new_node = ast.Call(node, [], [])
        pasta.ast_utils.replace_child(parent, node, new_node)
        ast.copy_location(new_node, node)
        self.add_log(INFO, node.lineno, node.col_offset,
                     "Changed %r to a function call" % full_name)
        return True
    return False
  def _maybe_add_arg_names(self, node, full_name):
    function_reorders = self._api_change_spec.function_reorders
    if full_name in function_reorders:
      if uses_star_args_in_call(node):
        self.add_log(WARNING, node.lineno, node.col_offset,
                     "(Manual check required) upgrading %s may require "
                     "re-ordering the call arguments, but it was passed "
                     "variable-length positional *args. The upgrade "
                     "script cannot handle these automatically." % full_name)
      reordered = function_reorders[full_name]
      new_keywords = []
      idx = 0
      for arg in node.args:
        if sys.version_info[:2] >= (3, 5) and isinstance(arg, ast.Starred):
        keyword_arg = reordered[idx]
        keyword = ast.keyword(arg=keyword_arg, value=arg)
        new_keywords.append(keyword)
        idx += 1
      if new_keywords:
        self.add_log(INFO, node.lineno, node.col_offset,
                     "Added keywords to args of function %r" % full_name)
        node.args = []
        node.keywords = new_keywords + (node.keywords or [])
        return True
    return False
  def _maybe_modify_args(self, node, full_name, name):
    renamed_keywords = self._get_applicable_dict("function_keyword_renames",
                                                 full_name, name)
    if not renamed_keywords:
      return False
    if uses_star_kwargs_in_call(node):
      self.add_log(WARNING, node.lineno, node.col_offset,
                   "(Manual check required) upgrading %s may require "
                   "renaming or removing call arguments, but it was passed "
                   "variable-length *args or **kwargs. The upgrade "
                   "script cannot handle these automatically." %
                   (full_name or name))
    modified = False
    new_keywords = []
    for keyword in node.keywords:
      argkey = keyword.arg
      if argkey in renamed_keywords:
        modified = True
        if renamed_keywords[argkey] is None:
          lineno = getattr(keyword, "lineno", node.lineno)
          col_offset = getattr(keyword, "col_offset", node.col_offset)
          self.add_log(INFO, lineno, col_offset,
                       "Removed argument %s for function %s" % (
                           argkey, full_name or name))
        else:
          keyword.arg = renamed_keywords[argkey]
          lineno = getattr(keyword, "lineno", node.lineno)
          col_offset = getattr(keyword, "col_offset", node.col_offset)
          self.add_log(INFO, lineno, col_offset,
                       "Renamed keyword argument for %s from %s to %s" % (
                           full_name, argkey, renamed_keywords[argkey]))
          new_keywords.append(keyword)
      else:
        new_keywords.append(keyword)
    if modified:
      node.keywords = new_keywords
    return modified
    assert self._stack[-1] is node
    full_name = self._get_full_name(node.func)
    if full_name:
      name = full_name.split(".")[-1]
    elif isinstance(node.func, ast.Name):
      name = node.func.id
    elif isinstance(node.func, ast.Attribute):
      name = node.func.attr
    else:
      name = None
    self._maybe_add_call_warning(node, full_name, name)
    self._maybe_add_arg_names(node, full_name)
    self._maybe_modify_args(node, full_name, name)
    transformers = self._get_applicable_entries("function_transformers",
                                                full_name, name)
    parent = self._stack[-2]
    if transformers:
      if uses_star_args_or_kwargs_in_call(node):
        self.add_log(WARNING, node.lineno, node.col_offset,
                     "(Manual check required) upgrading %s may require "
                     "modifying call arguments, but it was passed "
                     "variable-length *args or **kwargs. The upgrade "
                     "script cannot handle these automatically." %
                     (full_name or name))
    for transformer in transformers:
      logs = []
      new_node = transformer(parent, node, full_name, name, logs)
      self.add_logs(logs)
      if new_node and new_node is not node:
        pasta.ast_utils.replace_child(parent, node, new_node)
        node = new_node
        self._stack[-1] = node
    self.generic_visit(node)
    assert self._stack[-1] is node
    full_name = self._get_full_name(node)
    if full_name:
      parent = self._stack[-2]
      self._maybe_add_warning(node, full_name)
      if self._maybe_rename(parent, node, full_name):
        return
      if self._maybe_change_to_function_call(parent, node, full_name):
        return
      i = 2
      while isinstance(self._stack[-i], ast.Attribute):
        i += 1
      whole_name = pasta.dump(self._stack[-(i-1)])
      self._maybe_add_module_deprecation_warning(node, full_name, whole_name)
    self.generic_visit(node)
    new_aliases = []
    import_updated = False
    import_renames = getattr(self._api_change_spec, "import_renames", {})
    max_submodule_depth = getattr(self._api_change_spec, "max_submodule_depth",
                                  1)
    inserts_after_imports = getattr(self._api_change_spec,
                                    "inserts_after_imports", {})
    for import_alias in node.names:
      all_import_components = six.ensure_str(import_alias.name).split(".")
      found_update = False
      for i in reversed(list(range(1, max_submodule_depth + 1))):
        import_component = all_import_components[0]
        for j in range(1, min(i, len(all_import_components))):
          import_component += "." + six.ensure_str(all_import_components[j])
        import_rename_spec = import_renames.get(import_component, None)
        if not import_rename_spec or excluded_from_module_rename(
            import_alias.name, import_rename_spec):
          continue
        new_name = (
            import_rename_spec.new_name +
            import_alias.name[len(import_component):])
        new_asname = import_alias.asname
        if not new_asname and "." not in import_alias.name:
          new_asname = import_alias.name
        new_alias = ast.alias(name=new_name, asname=new_asname)
        new_aliases.append(new_alias)
        import_updated = True
        found_update = True
        full_import = (import_alias.name, import_alias.asname)
        insert_offset = 1
        for line_to_insert in inserts_after_imports.get(full_import, []):
          assert self._stack[-1] is node
          parent = self._stack[-2]
          new_line_node = pasta.parse(line_to_insert)
          ast.copy_location(new_line_node, node)
          parent.body.insert(
              parent.body.index(node) + insert_offset, new_line_node)
          insert_offset += 1
          old_suffix = pasta.base.formatting.get(node, "suffix")
          if old_suffix is None:
            old_suffix = os.linesep
          if os.linesep not in old_suffix:
            pasta.base.formatting.set(node, "suffix",
                                      six.ensure_str(old_suffix) + os.linesep)
          pasta.base.formatting.set(new_line_node, "prefix",
                                    pasta.base.formatting.get(node, "prefix"))
          pasta.base.formatting.set(new_line_node, "suffix", os.linesep)
          self.add_log(
              INFO, node.lineno, node.col_offset,
              "Adding `%s` after import of %s" %
              (new_line_node, import_alias.name))
        if found_update:
          break
      if not found_update:
    if import_updated:
      assert self._stack[-1] is node
      parent = self._stack[-2]
      new_node = ast.Import(new_aliases)
      ast.copy_location(new_node, node)
      pasta.ast_utils.replace_child(parent, node, new_node)
      self.add_log(
          INFO, node.lineno, node.col_offset,
          "Changed import from %r to %r." %
          (pasta.dump(node), pasta.dump(new_node)))
    self.generic_visit(node)
    if not node.module:
      self.generic_visit(node)
      return
    from_import = node.module
    from_import_first_component = six.ensure_str(from_import).split(".")[0]
    import_renames = getattr(self._api_change_spec, "import_renames", {})
    import_rename_spec = import_renames.get(from_import_first_component, None)
    if not import_rename_spec:
      self.generic_visit(node)
      return
    updated_aliases = []
    same_aliases = []
    for import_alias in node.names:
      full_module_name = "%s.%s" % (from_import, import_alias.name)
      if excluded_from_module_rename(full_module_name, import_rename_spec):
        same_aliases.append(import_alias)
      else:
        updated_aliases.append(import_alias)
    if not updated_aliases:
      self.generic_visit(node)
      return
    assert self._stack[-1] is node
    parent = self._stack[-2]
    new_from_import = (
        import_rename_spec.new_name +
        from_import[len(from_import_first_component):])
    updated_node = ast.ImportFrom(new_from_import, updated_aliases, node.level)
    ast.copy_location(updated_node, node)
    pasta.ast_utils.replace_child(parent, node, updated_node)
    additional_import_log = ""
    if same_aliases:
      same_node = ast.ImportFrom(from_import, same_aliases, node.level,
                                 col_offset=node.col_offset, lineno=node.lineno)
      ast.copy_location(same_node, node)
      parent.body.insert(parent.body.index(updated_node), same_node)
      pasta.base.formatting.set(
          same_node, "prefix",
          pasta.base.formatting.get(updated_node, "prefix"))
      additional_import_log = " and %r" % pasta.dump(same_node)
    self.add_log(
        INFO, node.lineno, node.col_offset,
        "Changed import from %r to %r%s." %
        (pasta.dump(node),
         pasta.dump(updated_node),
         additional_import_log))
    self.generic_visit(node)
class AnalysisResult(object):
class APIAnalysisSpec(object):
  """This class defines how `AnalysisResult`s should be generated.
  It specifies how to map imports and symbols to `AnalysisResult`s.
  This class must provide the following fields:
  * `symbols_to_detect`: maps function names to `AnalysisResult`s
  * `imports_to_detect`: maps imports represented as (full module name, alias)
    tuples to `AnalysisResult`s
    notifications)
  For an example, see `TFAPIImportAnalysisSpec`.
  """
class PastaAnalyzeVisitor(_PastaEditVisitor):
  def __init__(self, api_analysis_spec):
    super(PastaAnalyzeVisitor, self).__init__(NoUpdateSpec())
    self._api_analysis_spec = api_analysis_spec
  @property
  def results(self):
    return self._results
  def add_result(self, analysis_result):
    self._results.append(analysis_result)
    full_name = self._get_full_name(node)
    if full_name:
      detection = self._api_analysis_spec.symbols_to_detect.get(full_name, None)
      if detection:
        self.add_result(detection)
        self.add_log(
            detection.log_level, node.lineno, node.col_offset,
            detection.log_message)
    self.generic_visit(node)
    for import_alias in node.names:
      full_import = (import_alias.name, import_alias.asname)
      detection = (self._api_analysis_spec
                   .imports_to_detect.get(full_import, None))
      if detection:
        self.add_result(detection)
        self.add_log(
            detection.log_level, node.lineno, node.col_offset,
            detection.log_message)
    self.generic_visit(node)
    if not node.module:
      self.generic_visit(node)
      return
    from_import = node.module
    for import_alias in node.names:
      full_module_name = "%s.%s" % (from_import, import_alias.name)
      full_import = (full_module_name, import_alias.asname)
      detection = (self._api_analysis_spec
                   .imports_to_detect.get(full_import, None))
      if detection:
        self.add_result(detection)
        self.add_log(
            detection.log_level, node.lineno, node.col_offset,
            detection.log_message)
    self.generic_visit(node)
class ASTCodeUpgrader(object):
  def __init__(self, api_change_spec):
    if not isinstance(api_change_spec, APIChangeSpec):
      raise TypeError("Must pass APIChangeSpec to ASTCodeUpgrader, got %s" %
                      type(api_change_spec))
    self._api_change_spec = api_change_spec
  def process_file(self,
                   in_filename,
                   out_filename,
                   no_change_to_outfile_on_error=False):
    with open(in_filename, "r") as in_file, \
        tempfile.NamedTemporaryFile("w", delete=False) as temp_file:
      ret = self.process_opened_file(in_filename, in_file, out_filename,
                                     temp_file)
    if no_change_to_outfile_on_error and ret[0] == 0:
      os.remove(temp_file.name)
    else:
      shutil.move(temp_file.name, out_filename)
    return ret
  def format_log(self, log, in_filename):
    log_string = "%d:%d: %s: %s" % (log[1], log[2], log[0], log[3])
    if in_filename:
      return six.ensure_str(in_filename) + ":" + log_string
    else:
      return log_string
  def update_string_pasta(self, text, in_filename):
    try:
      t = pasta.parse(text)
    except (SyntaxError, ValueError, TypeError):
      log = ["ERROR: Failed to parse.\n" + traceback.format_exc()]
      return 0, "", log, []
    t, preprocess_logs, preprocess_errors = self._api_change_spec.preprocess(t)
    visitor = _PastaEditVisitor(self._api_change_spec)
    visitor.visit(t)
    self._api_change_spec.clear_preprocessing()
    logs = [self.format_log(log, None) for log in (preprocess_logs +
                                                   visitor.log)]
    errors = [self.format_log(error, in_filename)
              for error in (preprocess_errors +
                            visitor.warnings_and_errors)]
    return 1, pasta.dump(t), logs, errors
  def _format_log(self, log, in_filename, out_filename):
    text = six.ensure_str("-" * 80) + "\n"
    text += "Processing file %r\n outputting to %r\n" % (in_filename,
                                                         out_filename)
    text += six.ensure_str("-" * 80) + "\n\n"
    text += "\n".join(log) + "\n"
    text += six.ensure_str("-" * 80) + "\n\n"
    return text
  def process_opened_file(self, in_filename, in_file, out_filename, out_file):
    """Process the given python file for incompatible changes.
    This function is split out to facilitate StringIO testing from
    tf_upgrade_test.py.
    Args:
      in_filename: filename to parse
      in_file: opened file (or StringIO)
      out_filename: output file to write to
      out_file: opened file (or StringIO)
    Returns:
      A tuple representing number of files processed, log of actions, errors
    """
    lines = in_file.readlines()
    processed_file, new_file_content, log, process_errors = (
        self.update_string_pasta("".join(lines), in_filename))
    if out_file and processed_file:
      out_file.write(new_file_content)
    return (processed_file,
            self._format_log(log, in_filename, out_filename),
            process_errors)
  def process_tree(self, root_directory, output_root_directory,
                   copy_other_files):
    if output_root_directory == root_directory:
      return self.process_tree_inplace(root_directory)
    if output_root_directory and os.path.exists(output_root_directory):
      print("Output directory %r must not already exist." %
            (output_root_directory))
      sys.exit(1)
    norm_root = os.path.split(os.path.normpath(root_directory))
    norm_output = os.path.split(os.path.normpath(output_root_directory))
    if norm_root == norm_output:
      print("Output directory %r same as input directory %r" %
            (root_directory, output_root_directory))
      sys.exit(1)
    files_to_process = []
    files_to_copy = []
    for dir_name, _, file_list in os.walk(root_directory):
      py_files = [f for f in file_list if six.ensure_str(f).endswith(".py")]
      copy_files = [
          f for f in file_list if not six.ensure_str(f).endswith(".py")
      ]
      for filename in py_files:
        fullpath = os.path.join(dir_name, filename)
        fullpath_output = os.path.join(output_root_directory,
                                       os.path.relpath(fullpath,
                                                       root_directory))
        files_to_process.append((fullpath, fullpath_output))
      if copy_other_files:
        for filename in copy_files:
          fullpath = os.path.join(dir_name, filename)
          fullpath_output = os.path.join(output_root_directory,
                                         os.path.relpath(
                                             fullpath, root_directory))
          files_to_copy.append((fullpath, fullpath_output))
    file_count = 0
    tree_errors = {}
    report = ""
    report += six.ensure_str(("=" * 80)) + "\n"
    report += "Input tree: %r\n" % root_directory
    report += six.ensure_str(("=" * 80)) + "\n"
    for input_path, output_path in files_to_process:
      output_directory = os.path.dirname(output_path)
      if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
      if os.path.islink(input_path):
        link_target = os.readlink(input_path)
        link_target_output = os.path.join(
            output_root_directory, os.path.relpath(link_target, root_directory))
        if (link_target, link_target_output) in files_to_process:
          os.symlink(link_target_output, output_path)
        else:
          report += "Copying symlink %s without modifying its target %s" % (
              input_path, link_target)
          os.symlink(link_target, output_path)
        continue
      file_count += 1
      _, l_report, l_errors = self.process_file(input_path, output_path)
      tree_errors[input_path] = l_errors
      report += l_report
    for input_path, output_path in files_to_copy:
      output_directory = os.path.dirname(output_path)
      if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
      shutil.copy(input_path, output_path)
    return file_count, report, tree_errors
  def process_tree_inplace(self, root_directory):
    files_to_process = []
    for dir_name, _, file_list in os.walk(root_directory):
      py_files = [
          os.path.join(dir_name, f)
          for f in file_list
          if six.ensure_str(f).endswith(".py")
      ]
      files_to_process += py_files
    file_count = 0
    tree_errors = {}
    report = ""
    report += six.ensure_str(("=" * 80)) + "\n"
    report += "Input tree: %r\n" % root_directory
    report += six.ensure_str(("=" * 80)) + "\n"
    for path in files_to_process:
      if os.path.islink(path):
        report += "Skipping symlink %s.\n" % path
        continue
      file_count += 1
      _, l_report, l_errors = self.process_file(path, path)
      tree_errors[path] = l_errors
      report += l_report
    return file_count, report, tree_errors
