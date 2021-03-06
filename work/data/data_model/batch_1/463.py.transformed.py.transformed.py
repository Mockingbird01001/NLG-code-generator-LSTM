from contextlib import contextmanager
from .termui import get_terminal_size
from .parser import split_opt
from ._compat import term_len
FORCED_WIDTH = None
def measure_table(rows):
    widths = {}
    for row in rows:
        for idx, col in enumerate(row):
            widths[idx] = max(widths.get(idx, 0), term_len(col))
    return tuple(y for x, y in sorted(widths.items()))
def iter_rows(rows, col_count):
    for row in rows:
        row = tuple(row)
        yield row + ('',) * (col_count - len(row))
def wrap_text(text, width=78, initial_indent='', subsequent_indent='',
              preserve_paragraphs=False):
    from ._textwrap import TextWrapper
    text = text.expandtabs()
    wrapper = TextWrapper(width, initial_indent=initial_indent,
                          subsequent_indent=subsequent_indent,
                          replace_whitespace=False)
    if not preserve_paragraphs:
        return wrapper.fill(text)
    p = []
    buf = []
    indent = None
    def _flush_par():
        if not buf:
            return
        if buf[0].strip() == '\b':
            p.append((indent or 0, True, '\n'.join(buf[1:])))
        else:
            p.append((indent or 0, False, ' '.join(buf)))
        del buf[:]
    for line in text.splitlines():
        if not line:
            _flush_par()
            indent = None
        else:
            if indent is None:
                orig_len = term_len(line)
                line = line.lstrip()
                indent = orig_len - term_len(line)
            buf.append(line)
    _flush_par()
    rv = []
    for indent, raw, text in p:
        with wrapper.extra_indent(' ' * indent):
            if raw:
                rv.append(wrapper.indent_only(text))
            else:
                rv.append(wrapper.fill(text))
    return '\n\n'.join(rv)
class HelpFormatter(object):
    def __init__(self, indent_increment=2, width=None, max_width=None):
        self.indent_increment = indent_increment
        if max_width is None:
            max_width = 80
        if width is None:
            width = FORCED_WIDTH
            if width is None:
                width = max(min(get_terminal_size()[0], max_width) - 2, 50)
        self.width = width
        self.current_indent = 0
        self.buffer = []
    def write(self, string):
        self.buffer.append(string)
    def indent(self):
        self.current_indent += self.indent_increment
    def dedent(self):
        self.current_indent -= self.indent_increment
    def write_usage(self, prog, args='', prefix='Usage: '):
        usage_prefix = '%*s%s ' % (self.current_indent, prefix, prog)
        text_width = self.width - self.current_indent
        if text_width >= (term_len(usage_prefix) + 20):
            indent = ' ' * term_len(usage_prefix)
            self.write(wrap_text(args, text_width,
                                 initial_indent=usage_prefix,
                                 subsequent_indent=indent))
        else:
            self.write(usage_prefix)
            self.write('\n')
            indent = ' ' * (max(self.current_indent, term_len(prefix)) + 4)
            self.write(wrap_text(args, text_width,
                                 initial_indent=indent,
                                 subsequent_indent=indent))
        self.write('\n')
    def write_heading(self, heading):
        self.write('%*s%s:\n' % (self.current_indent, '', heading))
    def write_paragraph(self):
        if self.buffer:
            self.write('\n')
    def write_text(self, text):
        text_width = max(self.width - self.current_indent, 11)
        indent = ' ' * self.current_indent
        self.write(wrap_text(text, text_width,
                             initial_indent=indent,
                             subsequent_indent=indent,
                             preserve_paragraphs=True))
        self.write('\n')
    def write_dl(self, rows, col_max=30, col_spacing=2):
        rows = list(rows)
        widths = measure_table(rows)
        if len(widths) != 2:
            raise TypeError('Expected two columns for definition list')
        first_col = min(widths[0], col_max) + col_spacing
        for first, second in iter_rows(rows, len(widths)):
            self.write('%*s%s' % (self.current_indent, '', first))
            if not second:
                self.write('\n')
                continue
            if term_len(first) <= first_col - col_spacing:
                self.write(' ' * (first_col - term_len(first)))
            else:
                self.write('\n')
                self.write(' ' * (first_col + self.current_indent))
            text_width = max(self.width - first_col - 2, 10)
            lines = iter(wrap_text(second, text_width).splitlines())
            if lines:
                self.write(next(lines) + '\n')
                for line in lines:
                    self.write('%*s%s\n' % (
                        first_col + self.current_indent, '', line))
            else:
                self.write('\n')
    @contextmanager
    def section(self, name):
        self.write_paragraph()
        self.write_heading(name)
        self.indent()
        try:
            yield
        finally:
            self.dedent()
    @contextmanager
    def indentation(self):
        self.indent()
        try:
            yield
        finally:
            self.dedent()
    def getvalue(self):
        return ''.join(self.buffer)
def join_options(options):
    rv = []
    any_prefix_is_slash = False
    for opt in options:
        prefix = split_opt(opt)[0]
        if prefix == '/':
            any_prefix_is_slash = True
        rv.append((len(prefix), opt))
    rv.sort(key=lambda x: x[0])
    rv = ', '.join(x[1] for x in rv)
    return rv, any_prefix_is_slash
