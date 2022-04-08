
import os
import warnings
class Extension:
    def __init__(self, name, sources,
                  include_dirs=None,
                  define_macros=None,
                  undef_macros=None,
                  library_dirs=None,
                  libraries=None,
                  runtime_library_dirs=None,
                  extra_objects=None,
                  extra_compile_args=None,
                  extra_link_args=None,
                  export_symbols=None,
                  swig_opts = None,
                  depends=None,
                  language=None,
                  optional=None,
                  **kw
                 ):
        if not isinstance(name, str):
            raise AssertionError("'name' must be a string")
        if not (isinstance(sources, list) and
                all(isinstance(v, str) for v in sources)):
            raise AssertionError("'sources' must be a list of strings")
        self.name = name
        self.sources = sources
        self.include_dirs = include_dirs or []
        self.define_macros = define_macros or []
        self.undef_macros = undef_macros or []
        self.library_dirs = library_dirs or []
        self.libraries = libraries or []
        self.runtime_library_dirs = runtime_library_dirs or []
        self.extra_objects = extra_objects or []
        self.extra_compile_args = extra_compile_args or []
        self.extra_link_args = extra_link_args or []
        self.export_symbols = export_symbols or []
        self.swig_opts = swig_opts or []
        self.depends = depends or []
        self.language = language
        self.optional = optional
        if len(kw) > 0:
            options = [repr(option) for option in kw]
            options = ', '.join(sorted(options))
            msg = "Unknown Extension options: %s" % options
            warnings.warn(msg)
    def __repr__(self):
            self.__class__.__module__,
            self.__class__.__qualname__,
            self.name,
            id(self))
def read_setup_file(filename):
    from distutils.sysconfig import (parse_makefile, expand_makefile_vars,
                                     _variable_rx)
    from distutils.text_file import TextFile
    from distutils.util import split_quoted
    vars = parse_makefile(filename)
    file = TextFile(filename,
                    strip_comments=1, skip_blanks=1, join_lines=1,
                    lstrip_ws=1, rstrip_ws=1)
    try:
        extensions = []
        while True:
            line = file.readline()
            if line is None:
                break
            if _variable_rx.match(line):
                continue
            if line[0] == line[-1] == "*":
                file.warn("'%s' lines not handled yet" % line)
                continue
            line = expand_makefile_vars(line, vars)
            words = split_quoted(line)
            module = words[0]
            ext = Extension(module, [])
            append_next_word = None
            for word in words[1:]:
                if append_next_word is not None:
                    append_next_word.append(word)
                    append_next_word = None
                    continue
                suffix = os.path.splitext(word)[1]
                switch = word[0:2] ; value = word[2:]
                if suffix in (".c", ".cc", ".cpp", ".cxx", ".c++", ".m", ".mm"):
                    ext.sources.append(word)
                elif switch == "-I":
                    ext.include_dirs.append(value)
                elif switch == "-D":
                    equals = value.find("=")
                    if equals == -1:
                        ext.define_macros.append((value, None))
                    else:
                        ext.define_macros.append((value[0:equals],
                                                  value[equals+2:]))
                elif switch == "-U":
                    ext.undef_macros.append(value)
                elif switch == "-C":
                    ext.extra_compile_args.append(word)
                elif switch == "-l":
                    ext.libraries.append(value)
                elif switch == "-L":
                    ext.library_dirs.append(value)
                elif switch == "-R":
                    ext.runtime_library_dirs.append(value)
                elif word == "-rpath":
                    append_next_word = ext.runtime_library_dirs
                elif word == "-Xlinker":
                    append_next_word = ext.extra_link_args
                elif word == "-Xcompiler":
                    append_next_word = ext.extra_compile_args
                elif switch == "-u":
                    ext.extra_link_args.append(word)
                    if not value:
                        append_next_word = ext.extra_link_args
                elif suffix in (".a", ".so", ".sl", ".o", ".dylib"):
                    ext.extra_objects.append(word)
                else:
                    file.warn("unrecognized argument '%s'" % word)
            extensions.append(ext)
    finally:
        file.close()
    return extensions
