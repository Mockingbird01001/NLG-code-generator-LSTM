
import copy
import os
import re
import sre_constants
import traceback
import numpy as np
import six
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
HELP_INDENT = "  "
EXPLICIT_USER_EXIT = "explicit_user_exit"
REGEX_MATCH_LINES_KEY = "regex_match_lines"
INIT_SCROLL_POS_KEY = "init_scroll_pos"
MAIN_MENU_KEY = "mm:"
class CommandLineExit(Exception):
  def __init__(self, exit_token=None):
    Exception.__init__(self)
    self._exit_token = exit_token
  @property
  def exit_token(self):
    return self._exit_token
class RichLine(object):
  """Rich single-line text.
  Attributes:
    text: A plain string, the raw text represented by this object.  Should not
      contain newlines.
    font_attr_segs: A list of (start, end, font attribute) triples, representing
      richness information applied to substrings of text.
  """
  def __init__(self, text="", font_attr=None):
    self.text = text
    if font_attr:
      self.font_attr_segs = [(0, len(text), font_attr)]
    else:
      self.font_attr_segs = []
  def __add__(self, other):
    ret = RichLine()
    if isinstance(other, six.string_types):
      ret.text = self.text + other
      ret.font_attr_segs = self.font_attr_segs[:]
      return ret
    elif isinstance(other, RichLine):
      ret.text = self.text + other.text
      ret.font_attr_segs = self.font_attr_segs[:]
      old_len = len(self.text)
      for start, end, font_attr in other.font_attr_segs:
        ret.font_attr_segs.append((old_len + start, old_len + end, font_attr))
      return ret
    else:
      raise TypeError("%r cannot be concatenated with a RichLine" % other)
  def __len__(self):
    return len(self.text)
def rich_text_lines_from_rich_line_list(rich_text_list, annotations=None):
  lines = []
  font_attr_segs = {}
  for i, rl in enumerate(rich_text_list):
    if isinstance(rl, RichLine):
      lines.append(rl.text)
      if rl.font_attr_segs:
        font_attr_segs[i] = rl.font_attr_segs
    else:
      lines.append(rl)
  return RichTextLines(lines, font_attr_segs, annotations=annotations)
def get_tensorflow_version_lines(include_dependency_versions=False):
  lines = ["TensorFlow version: %s" % pywrap_tf_session.__version__]
  lines.append("")
  if include_dependency_versions:
    lines.append("Dependency version(s):")
    lines.append("  numpy: %s" % np.__version__)
    lines.append("")
  return RichTextLines(lines)
class RichTextLines(object):
  """Rich multi-line text.
  Line-by-line text output, with font attributes (e.g., color) and annotations
  (e.g., indices in a multi-dimensional tensor). Used as the text output of CLI
  commands. Can be rendered on terminal environments such as curses.
  This is not to be confused with Rich Text Format (RTF). This class is for text
  lines only.
  """
  def __init__(self, lines, font_attr_segs=None, annotations=None):
    """Constructor of RichTextLines.
    Args:
      lines: A list of str or a single str, representing text output to
        screen. The latter case is for convenience when the text output is
        single-line.
      font_attr_segs: A map from 0-based row index to a list of 3-tuples.
        It lists segments in each row that have special font attributes, such
        as colors, that are not the default attribute. For example:
        {1: [(0, 3, "red"), (4, 7, "green")], 2: [(10, 20, "yellow")]}
        In each tuple, the 1st element is the start index of the segment. The
        2nd element is the end index, in an "open interval" fashion. The 3rd
        element is an object or a list of objects that represents the font
        attribute. Colors are represented as strings as in the examples above.
      annotations: A map from 0-based row index to any object for annotating
        the row. A typical use example is annotating rows of the output as
        indices in a multi-dimensional tensor. For example, consider the
        following text representation of a 3x2x2 tensor:
          [[[0, 0], [0, 0]],
           [[0, 0], [0, 0]],
           [[0, 0], [0, 0]]]
        The annotation can indicate the indices of the first element shown in
        each row, i.e.,
          {0: [0, 0, 0], 1: [1, 0, 0], 2: [2, 0, 0]}
        This information can make display of tensors on screen clearer and can
        help the user navigate (scroll) to the desired location in a large
        tensor.
    Raises:
      ValueError: If lines is of invalid type.
    """
    if isinstance(lines, list):
      self._lines = lines
    elif isinstance(lines, six.string_types):
      self._lines = [lines]
    else:
      raise ValueError("Unexpected type in lines: %s" % type(lines))
    self._font_attr_segs = font_attr_segs
    if not self._font_attr_segs:
      self._font_attr_segs = {}
    self._annotations = annotations
    if not self._annotations:
      self._annotations = {}
  @property
  def lines(self):
    return self._lines
  @property
  def font_attr_segs(self):
    return self._font_attr_segs
  @property
  def annotations(self):
    return self._annotations
  def num_lines(self):
    return len(self._lines)
  def slice(self, begin, end):
    """Slice a RichTextLines object.
    The object itself is not changed. A sliced instance is returned.
    Args:
      begin: (int) Beginning line index (inclusive). Must be >= 0.
      end: (int) Ending line index (exclusive). Must be >= 0.
    Returns:
      (RichTextLines) Sliced output instance of RichTextLines.
    Raises:
      ValueError: If begin or end is negative.
    """
    if begin < 0 or end < 0:
      raise ValueError("Encountered negative index.")
    lines = self.lines[begin:end]
    font_attr_segs = {}
    for key in self.font_attr_segs:
      if key >= begin and key < end:
        font_attr_segs[key - begin] = self.font_attr_segs[key]
    annotations = {}
    for key in self.annotations:
      if not isinstance(key, int):
        annotations[key] = self.annotations[key]
      elif key >= begin and key < end:
        annotations[key - begin] = self.annotations[key]
    return RichTextLines(
        lines, font_attr_segs=font_attr_segs, annotations=annotations)
  def extend(self, other):
    """Extend this instance of RichTextLines with another instance.
    The extension takes effect on the text lines, the font attribute segments,
    as well as the annotations. The line indices in the font attribute
    segments and the annotations are adjusted to account for the existing
    lines. If there are duplicate, non-line-index fields in the annotations,
    the value from the input argument "other" will override that in this
    instance.
    Args:
      other: (RichTextLines) The other RichTextLines instance to be appended at
        the end of this instance.
    """
    self._lines.extend(other.lines)
    for line_index in other.font_attr_segs:
      self._font_attr_segs[orig_num_lines + line_index] = (
          other.font_attr_segs[line_index])
    for key in other.annotations:
      if isinstance(key, int):
        self._annotations[orig_num_lines + key] = (other.annotations[key])
      else:
        self._annotations[key] = other.annotations[key]
  def _extend_before(self, other):
    """Add another RichTextLines object to the front.
    Args:
      other: (RichTextLines) The other object to add to the front to this
        object.
    """
    self._lines = other.lines + self._lines
    new_font_attr_segs = {}
    for line_index in self.font_attr_segs:
      new_font_attr_segs[other_num_lines + line_index] = (
          self.font_attr_segs[line_index])
    new_font_attr_segs.update(other.font_attr_segs)
    self._font_attr_segs = new_font_attr_segs
    new_annotations = {}
    for key in self._annotations:
      if isinstance(key, int):
        new_annotations[other_num_lines + key] = (self.annotations[key])
      else:
        new_annotations[key] = other.annotations[key]
    new_annotations.update(other.annotations)
    self._annotations = new_annotations
  def append(self, line, font_attr_segs=None):
    """Append a single line of text.
    Args:
      line: (str) The text to be added to the end.
      font_attr_segs: (list of tuples) Font attribute segments of the appended
        line.
    """
    self._lines.append(line)
    if font_attr_segs:
      self._font_attr_segs[len(self._lines) - 1] = font_attr_segs
  def append_rich_line(self, rich_line):
    self.append(rich_line.text, rich_line.font_attr_segs)
  def prepend(self, line, font_attr_segs=None):
    """Prepend (i.e., add to the front) a single line of text.
    Args:
      line: (str) The text to be added to the front.
      font_attr_segs: (list of tuples) Font attribute segments of the appended
        line.
    """
    other = RichTextLines(line)
    if font_attr_segs:
      other.font_attr_segs[0] = font_attr_segs
    self._extend_before(other)
  def write_to_file(self, file_path):
    """Write the object itself to file, in a plain format.
    The font_attr_segs and annotations are ignored.
    Args:
      file_path: (str) path of the file to write to.
    """
    with gfile.Open(file_path, "w") as f:
      for line in self._lines:
        f.write(line + "\n")
def regex_find(orig_screen_output, regex, font_attr):
  new_screen_output = RichTextLines(
      orig_screen_output.lines,
      font_attr_segs=copy.deepcopy(orig_screen_output.font_attr_segs),
      annotations=orig_screen_output.annotations)
  try:
    re_prog = re.compile(regex)
  except sre_constants.error:
    raise ValueError("Invalid regular expression: \"%s\"" % regex)
  regex_match_lines = []
  for i, line in enumerate(new_screen_output.lines):
    find_it = re_prog.finditer(line)
    match_segs = []
    for match in find_it:
      match_segs.append((match.start(), match.end(), font_attr))
    if match_segs:
      if i not in new_screen_output.font_attr_segs:
        new_screen_output.font_attr_segs[i] = match_segs
      else:
        new_screen_output.font_attr_segs[i].extend(match_segs)
        new_screen_output.font_attr_segs[i] = sorted(
            new_screen_output.font_attr_segs[i], key=lambda x: x[0])
      regex_match_lines.append(i)
  new_screen_output.annotations[REGEX_MATCH_LINES_KEY] = regex_match_lines
  return new_screen_output
def wrap_rich_text_lines(inp, cols):
  """Wrap RichTextLines according to maximum number of columns.
  Produces a new RichTextLines object with the text lines, font_attr_segs and
  annotations properly wrapped. This ought to be used sparingly, as in most
  cases, command handlers producing RichTextLines outputs should know the
  screen/panel width via the screen_info kwarg and should produce properly
  length-limited lines in the output accordingly.
  Args:
    inp: Input RichTextLines object.
    cols: Number of columns, as an int.
  Returns:
    1) A new instance of RichTextLines, with line lengths limited to cols.
    2) A list of new (wrapped) line index. For example, if the original input
      consists of three lines and only the second line is wrapped, and it's
      wrapped into two lines, this return value will be: [0, 1, 3].
  Raises:
    ValueError: If inputs have invalid types.
  """
  new_line_indices = []
  if not isinstance(inp, RichTextLines):
    raise ValueError("Invalid type of input screen_output")
  if not isinstance(cols, int):
    raise ValueError("Invalid type of input cols")
  out = RichTextLines([])
  for i, line in enumerate(inp.lines):
    new_line_indices.append(out.num_lines())
    if i in inp.annotations:
      out.annotations[row_counter] = inp.annotations[i]
    if len(line) <= cols:
      out.lines.append(line)
      if i in inp.font_attr_segs:
        out.font_attr_segs[row_counter] = inp.font_attr_segs[i]
      row_counter += 1
    else:
      osegs = []
      if i in inp.font_attr_segs:
        osegs = inp.font_attr_segs[i]
      idx = 0
      while idx < len(line):
        if idx + cols > len(line):
          rlim = len(line)
        else:
          rlim = idx + cols
        wlines.append(line[idx:rlim])
        for seg in osegs:
          if (seg[0] < rlim) and (seg[1] >= idx):
            if seg[0] >= idx:
              lb = seg[0] - idx
            else:
              lb = 0
            if seg[1] < rlim:
              rb = seg[1] - idx
            else:
              rb = rlim - idx
              wseg = (lb, rb, seg[2])
              if row_counter not in out.font_attr_segs:
                out.font_attr_segs[row_counter] = [wseg]
              else:
                out.font_attr_segs[row_counter].append(wseg)
        idx += cols
        row_counter += 1
      out.lines.extend(wlines)
  for key in inp.annotations:
    if not isinstance(key, int):
      out.annotations[key] = inp.annotations[key]
  return out, new_line_indices
class CommandHandlerRegistry(object):
  """Registry of command handlers for CLI.
  Handler methods (callables) for user commands can be registered with this
  class, which then is able to dispatch commands to the correct handlers and
  retrieve the RichTextLines output.
  For example, suppose you have the following handler defined:
    def echo(argv, screen_info=None):
      return RichTextLines(["arguments = %s" % " ".join(argv),
                            "screen_info = " + repr(screen_info)])
  you can register the handler with the command prefix "echo" and alias "e":
    registry = CommandHandlerRegistry()
    registry.register_command_handler("echo", echo,
        "Echo arguments, along with screen info", prefix_aliases=["e"])
  then to invoke this command handler with some arguments and screen_info, do:
    registry.dispatch_command("echo", ["foo", "bar"], screen_info={"cols": 80})
  or with the prefix alias:
    registry.dispatch_command("e", ["foo", "bar"], screen_info={"cols": 80})
  The call will return a RichTextLines object which can be rendered by a CLI.
  """
  HELP_COMMAND = "help"
  HELP_COMMAND_ALIASES = ["h"]
  VERSION_COMMAND = "version"
  VERSION_COMMAND_ALIASES = ["ver"]
  def __init__(self):
    self._handlers = {}
    self._alias_to_prefix = {}
    self._prefix_to_aliases = {}
    self._prefix_to_help = {}
    self._help_intro = None
    self.register_command_handler(
        self.HELP_COMMAND,
        self._help_handler,
        "Print this help message.",
        prefix_aliases=self.HELP_COMMAND_ALIASES)
    self.register_command_handler(
        self.VERSION_COMMAND,
        self._version_handler,
        "Print the versions of TensorFlow and its key dependencies.",
        prefix_aliases=self.VERSION_COMMAND_ALIASES)
  def register_command_handler(self,
                               prefix,
                               handler,
                               help_info,
                               prefix_aliases=None):
    """Register a callable as a command handler.
    Args:
      prefix: Command prefix, i.e., the first word in a command, e.g.,
        "print" as in "print tensor_1".
      handler: A callable of the following signature:
          foo_handler(argv, screen_info=None),
        where argv is the argument vector (excluding the command prefix) and
          screen_info is a dictionary containing information about the screen,
          such as number of columns, e.g., {"cols": 100}.
        The callable should return:
          1) a RichTextLines object representing the screen output.
        The callable can also raise an exception of the type CommandLineExit,
        which if caught by the command-line interface, will lead to its exit.
        The exception can optionally carry an exit token of arbitrary type.
      help_info: A help string.
      prefix_aliases: Aliases for the command prefix, as a list of str. E.g.,
        shorthands for the command prefix: ["p", "pr"]
    Raises:
      ValueError: If
        1) the prefix is empty, or
        2) handler is not callable, or
        3) a handler is already registered for the prefix, or
        4) elements in prefix_aliases clash with existing aliases.
        5) help_info is not a str.
    """
    if not prefix:
      raise ValueError("Empty command prefix")
    if prefix in self._handlers:
      raise ValueError(
          "A handler is already registered for command prefix \"%s\"" % prefix)
    if not callable(handler):
      raise ValueError("handler is not callable")
    if not isinstance(help_info, six.string_types):
      raise ValueError("help_info is not a str")
    if prefix_aliases:
      for alias in prefix_aliases:
        if self._resolve_prefix(alias):
          raise ValueError(
              "The prefix alias \"%s\" clashes with existing prefixes or "
              "aliases." % alias)
        self._alias_to_prefix[alias] = prefix
      self._prefix_to_aliases[prefix] = prefix_aliases
    self._handlers[prefix] = handler
    self._prefix_to_help[prefix] = help_info
  def dispatch_command(self, prefix, argv, screen_info=None):
    if not prefix:
      raise ValueError("Prefix is empty")
    resolved_prefix = self._resolve_prefix(prefix)
    if not resolved_prefix:
      raise ValueError("No handler is registered for command prefix \"%s\"" %
                       prefix)
    handler = self._handlers[resolved_prefix]
    try:
      output = handler(argv, screen_info=screen_info)
    except CommandLineExit as e:
      raise e
    except SystemExit as e:
      lines = ["Syntax error for command: %s" % prefix,
               "For help, do \"help %s\"" % prefix]
      output = RichTextLines(lines)
      lines = ["Error occurred during handling of command: %s %s:" %
               (resolved_prefix, " ".join(argv)), "%s: %s" % (type(e), str(e))]
      lines.append("")
      lines.extend(traceback.format_exc().split("\n"))
      output = RichTextLines(lines)
    if not isinstance(output, RichTextLines) and output is not None:
      raise ValueError(
          "Return value from command handler %s is not None or a RichTextLines "
          "instance" % str(handler))
    return output
  def is_registered(self, prefix):
    return self._resolve_prefix(prefix) is not None
  def get_help(self, cmd_prefix=None):
    if not cmd_prefix:
      help_info = RichTextLines([])
      if self._help_intro:
        help_info.extend(self._help_intro)
      sorted_prefixes = sorted(self._handlers)
      for cmd_prefix in sorted_prefixes:
        lines = self._get_help_for_command_prefix(cmd_prefix)
        lines.append("")
        lines.append("")
        help_info.extend(RichTextLines(lines))
      return help_info
    else:
      return RichTextLines(self._get_help_for_command_prefix(cmd_prefix))
  def set_help_intro(self, help_intro):
    """Set an introductory message to help output.
    Args:
      help_intro: (RichTextLines) Rich text lines appended to the
        beginning of the output of the command "help", as introductory
        information.
    """
    self._help_intro = help_intro
  def _help_handler(self, args, screen_info=None):
    """Command handler for "help".
    "help" is a common command that merits built-in support from this class.
    Args:
      args: Command line arguments to "help" (not including "help" itself).
      screen_info: (dict) Information regarding the screen, e.g., the screen
        width in characters: {"cols": 80}
    Returns:
      (RichTextLines) Screen text output.
    """
    if not args:
      return self.get_help()
    elif len(args) == 1:
      return self.get_help(args[0])
    else:
      return RichTextLines(["ERROR: help takes only 0 or 1 input argument."])
  def _version_handler(self, args, screen_info=None):
    return get_tensorflow_version_lines(include_dependency_versions=True)
  def _resolve_prefix(self, token):
    if token in self._handlers:
      return token
    elif token in self._alias_to_prefix:
      return self._alias_to_prefix[token]
    else:
      return None
  def _get_help_for_command_prefix(self, cmd_prefix):
    lines = []
    resolved_prefix = self._resolve_prefix(cmd_prefix)
    if not resolved_prefix:
      lines.append("Invalid command prefix: \"%s\"" % cmd_prefix)
      return lines
    lines.append(resolved_prefix)
    if resolved_prefix in self._prefix_to_aliases:
      lines.append(HELP_INDENT + "Aliases: " + ", ".join(
          self._prefix_to_aliases[resolved_prefix]))
    lines.append("")
    help_lines = self._prefix_to_help[resolved_prefix].split("\n")
    for line in help_lines:
      lines.append(HELP_INDENT + line)
    return lines
class TabCompletionRegistry(object):
  def __init__(self):
    self._comp_dict = {}
  def register_tab_comp_context(self, context_words, comp_items):
    if not isinstance(context_words, list):
      raise TypeError("Incorrect type in context_list: Expected list, got %s" %
                      type(context_words))
    if not isinstance(comp_items, list):
      raise TypeError("Incorrect type in comp_items: Expected list, got %s" %
                      type(comp_items))
    sorted_comp_items = sorted(comp_items)
    for context_word in context_words:
      self._comp_dict[context_word] = sorted_comp_items
  def deregister_context(self, context_words):
    """Deregister a list of context words.
    Args:
      context_words: A list of context words to deregister, as a list of str.
    Raises:
      KeyError: if there are word(s) in context_words that do not correspond
        to any registered contexts.
    """
    for context_word in context_words:
      if context_word not in self._comp_dict:
        raise KeyError("Cannot deregister unregistered context word \"%s\"" %
                       context_word)
    for context_word in context_words:
      del self._comp_dict[context_word]
  def extend_comp_items(self, context_word, new_comp_items):
    """Add a list of completion items to a completion context.
    Args:
      context_word: A single completion word as a string. The extension will
        also apply to all other context words of the same context.
      new_comp_items: (list of str) New completion items to add.
    Raises:
      KeyError: if the context word has not been registered.
    """
    if context_word not in self._comp_dict:
      raise KeyError("Context word \"%s\" has not been registered" %
                     context_word)
    self._comp_dict[context_word].extend(new_comp_items)
    self._comp_dict[context_word] = sorted(self._comp_dict[context_word])
  def remove_comp_items(self, context_word, comp_items):
    if context_word not in self._comp_dict:
      raise KeyError("Context word \"%s\" has not been registered" %
                     context_word)
    for item in comp_items:
      self._comp_dict[context_word].remove(item)
  def get_completions(self, context_word, prefix):
    """Get the tab completions given a context word and a prefix.
    Args:
      context_word: The context word.
      prefix: The prefix of the incomplete word.
    Returns:
      (1) None if no registered context matches the context_word.
          A list of str for the matching completion items. Can be an empty list
          of a matching context exists, but no completion item matches the
          prefix.
      (2) Common prefix of all the words in the first return value. If the
          first return value is None, this return value will be None, too. If
          the first return value is not None, i.e., a list, this return value
          will be a str, which can be an empty str if there is no common
          prefix among the items of the list.
    """
    if context_word not in self._comp_dict:
      return None, None
    comp_items = self._comp_dict[context_word]
    comp_items = sorted(
        [item for item in comp_items if item.startswith(prefix)])
    return comp_items, self._common_prefix(comp_items)
  def _common_prefix(self, m):
    """Given a list of str, returns the longest common prefix.
    Args:
      m: (list of str) A list of strings.
    Returns:
      (str) The longest common prefix.
    """
    if not m:
      return ""
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
      if c != s2[i]:
        return s1[:i]
    return s1
class CommandHistory(object):
  _HISTORY_FILE_NAME = ".tfdbg_history"
  def __init__(self, limit=100, history_file_path=None):
    """CommandHistory constructor.
    Args:
      limit: Maximum number of the most recent commands that this instance
        keeps track of, as an int.
      history_file_path: (str) Manually specified path to history file. Used in
        testing.
    """
    self._commands = []
    self._limit = limit
    self._history_file_path = (
        history_file_path or self._get_default_history_file_path())
    self._load_history_from_file()
  def _load_history_from_file(self):
    if os.path.isfile(self._history_file_path):
      try:
        with open(self._history_file_path, "rt") as history_file:
          commands = history_file.readlines()
        self._commands = [command.strip() for command in commands
                          if command.strip()]
        if len(self._commands) > self._limit:
          self._commands = self._commands[-self._limit:]
          with open(self._history_file_path, "wt") as history_file:
            for command in self._commands:
              history_file.write(command + "\n")
      except IOError:
        print("WARNING: writing history file failed.")
  def _add_command_to_history_file(self, command):
    try:
      with open(self._history_file_path, "at") as history_file:
        history_file.write(command + "\n")
    except IOError:
      pass
  @classmethod
  def _get_default_history_file_path(cls):
    return os.path.join(os.path.expanduser("~"), cls._HISTORY_FILE_NAME)
  def add_command(self, command):
    if self._commands and command == self._commands[-1]:
      return
    if not isinstance(command, six.string_types):
      raise TypeError("Attempt to enter non-str entry to command history")
    self._commands.append(command)
    if len(self._commands) > self._limit:
      self._commands = self._commands[-self._limit:]
    self._add_command_to_history_file(command)
  def most_recent_n(self, n):
    return self._commands[-n:]
  def lookup_prefix(self, prefix, n):
    commands = [cmd for cmd in self._commands if cmd.startswith(prefix)]
    return commands[-n:]
class MenuItem(object):
  def __init__(self, caption, content, enabled=True):
    """Menu constructor.
    TODO(cais): Nested menu is currently not supported. Support it.
    Args:
      caption: (str) caption of the menu item.
      content: Content of the menu item. For a menu item that triggers
        a command, for example, content is the command string.
      enabled: (bool) whether this menu item is enabled.
    """
    self._caption = caption
    self._content = content
    self._enabled = enabled
  @property
  def caption(self):
    return self._caption
  @property
  def type(self):
    return self._node_type
  @property
  def content(self):
    return self._content
  def is_enabled(self):
    return self._enabled
  def disable(self):
    self._enabled = False
  def enable(self):
    self._enabled = True
class Menu(object):
  def __init__(self, name=None):
    """Menu constructor.
    Args:
      name: (str or None) name of this menu.
    """
    self._name = name
    self._items = []
  def append(self, item):
    """Append an item to the Menu.
    Args:
      item: (MenuItem) the item to be appended.
    """
    self._items.append(item)
  def insert(self, index, item):
    self._items.insert(index, item)
  def num_items(self):
    return len(self._items)
  def captions(self):
    return [item.caption for item in self._items]
  def caption_to_item(self, caption):
    """Get a MenuItem from the caption.
    Args:
      caption: (str) The caption to look up.
    Returns:
      (MenuItem) The first-match menu item with the caption, if any.
    Raises:
      LookupError: If a menu item with the caption does not exist.
    """
    captions = self.captions()
    if caption not in captions:
      raise LookupError("There is no menu item with the caption \"%s\"" %
                        caption)
    return self._items[captions.index(caption)]
  def format_as_single_line(self,
                            prefix=None,
                            divider=" | ",
                            enabled_item_attrs=None,
                            disabled_item_attrs=None):
    """Format the menu as a single-line RichTextLines object.
    Args:
      prefix: (str) String added to the beginning of the line.
      divider: (str) The dividing string between the menu items.
      enabled_item_attrs: (list or str) Attributes applied to each enabled
        menu item, e.g., ["bold", "underline"].
      disabled_item_attrs: (list or str) Attributes applied to each
        disabled menu item, e.g., ["red"].
    Returns:
      (RichTextLines) A single-line output representing the menu, with
        font_attr_segs marking the individual menu items.
    """
    if (enabled_item_attrs is not None and
        not isinstance(enabled_item_attrs, list)):
      enabled_item_attrs = [enabled_item_attrs]
    if (disabled_item_attrs is not None and
        not isinstance(disabled_item_attrs, list)):
      disabled_item_attrs = [disabled_item_attrs]
    menu_line = prefix if prefix is not None else ""
    attr_segs = []
    for item in self._items:
      menu_line += item.caption
      item_name_begin = len(menu_line) - len(item.caption)
      if item.is_enabled():
        final_attrs = [item]
        if enabled_item_attrs:
          final_attrs.extend(enabled_item_attrs)
        attr_segs.append((item_name_begin, len(menu_line), final_attrs))
      else:
        if disabled_item_attrs:
          attr_segs.append(
              (item_name_begin, len(menu_line), disabled_item_attrs))
      menu_line += divider
    return RichTextLines(menu_line, font_attr_segs={0: attr_segs})
