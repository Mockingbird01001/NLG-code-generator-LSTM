
import optparse
import os
import re
import shlex
import urllib.parse
from optparse import Values
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple
from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.urls import get_url_scheme, url_to_path
if TYPE_CHECKING:
    from typing import NoReturn
    from pip._internal.index.package_finder import PackageFinder
__all__ = ['parse_requirements']
ReqFileLines = Iterator[Tuple[int, str]]
LineParser = Callable[[str], Tuple[str, Values]]
SCHEME_RE = re.compile(r'^(http|https|file):', re.I)
ENV_VAR_RE = re.compile(r'(?P<var>\$\{(?P<name>[A-Z0-9_]+)\})')
SUPPORTED_OPTIONS = [
    cmdoptions.index_url,
    cmdoptions.extra_index_url,
    cmdoptions.no_index,
    cmdoptions.constraints,
    cmdoptions.requirements,
    cmdoptions.editable,
    cmdoptions.find_links,
    cmdoptions.no_binary,
    cmdoptions.only_binary,
    cmdoptions.prefer_binary,
    cmdoptions.require_hashes,
    cmdoptions.pre,
    cmdoptions.trusted_host,
    cmdoptions.use_new_feature,
]
SUPPORTED_OPTIONS_REQ = [
    cmdoptions.install_options,
    cmdoptions.global_options,
    cmdoptions.hash,
]
SUPPORTED_OPTIONS_REQ_DEST = [str(o().dest) for o in SUPPORTED_OPTIONS_REQ]
class ParsedRequirement:
    def __init__(
        self,
        requirement,
        is_editable,
        comes_from,
        constraint,
        options=None,
        line_source=None,
    ):
        self.requirement = requirement
        self.is_editable = is_editable
        self.comes_from = comes_from
        self.options = options
        self.constraint = constraint
        self.line_source = line_source
class ParsedLine:
    def __init__(
        self,
        filename,
        lineno,
        args,
        opts,
        constraint,
    ):
        self.filename = filename
        self.lineno = lineno
        self.opts = opts
        self.constraint = constraint
        if args:
            self.is_requirement = True
            self.is_editable = False
            self.requirement = args
        elif opts.editables:
            self.is_requirement = True
            self.is_editable = True
            self.requirement = opts.editables[0]
        else:
            self.is_requirement = False
def parse_requirements(
    filename,
    session,
    finder=None,
    options=None,
    constraint=False,
):
    line_parser = get_line_parser(finder)
    parser = RequirementsFileParser(session, line_parser)
    for parsed_line in parser.parse(filename, constraint):
        parsed_req = handle_line(
            parsed_line,
            options=options,
            finder=finder,
            session=session
        )
        if parsed_req is not None:
            yield parsed_req
def preprocess(content):
    lines_enum = enumerate(content.splitlines(), start=1)
    lines_enum = join_lines(lines_enum)
    lines_enum = ignore_comments(lines_enum)
    lines_enum = expand_env_variables(lines_enum)
    return lines_enum
def handle_requirement_line(
    line,
    options=None,
):
    line_comes_from = '{} {} (line {})'.format(
        '-c' if line.constraint else '-r', line.filename, line.lineno,
    )
    assert line.is_requirement
    if line.is_editable:
        return ParsedRequirement(
            requirement=line.requirement,
            is_editable=line.is_editable,
            comes_from=line_comes_from,
            constraint=line.constraint,
        )
    else:
        if options:
            cmdoptions.check_install_build_global(options, line.opts)
        req_options = {}
        for dest in SUPPORTED_OPTIONS_REQ_DEST:
            if dest in line.opts.__dict__ and line.opts.__dict__[dest]:
                req_options[dest] = line.opts.__dict__[dest]
        line_source = f'line {line.lineno} of {line.filename}'
        return ParsedRequirement(
            requirement=line.requirement,
            is_editable=line.is_editable,
            comes_from=line_comes_from,
            constraint=line.constraint,
            options=req_options,
            line_source=line_source,
        )
def handle_option_line(
    opts,
    filename,
    lineno,
    finder=None,
    options=None,
    session=None,
):
    if options:
        if opts.require_hashes:
            options.require_hashes = opts.require_hashes
        if opts.features_enabled:
            options.features_enabled.extend(
                f for f in opts.features_enabled
                if f not in options.features_enabled
            )
    if finder:
        find_links = finder.find_links
        index_urls = finder.index_urls
        if opts.index_url:
            index_urls = [opts.index_url]
        if opts.no_index is True:
            index_urls = []
        if opts.extra_index_urls:
            index_urls.extend(opts.extra_index_urls)
        if opts.find_links:
            value = opts.find_links[0]
            req_dir = os.path.dirname(os.path.abspath(filename))
            relative_to_reqs_file = os.path.join(req_dir, value)
            if os.path.exists(relative_to_reqs_file):
                value = relative_to_reqs_file
            find_links.append(value)
        if session:
            session.update_index_urls(index_urls)
        search_scope = SearchScope(
            find_links=find_links,
            index_urls=index_urls,
        )
        finder.search_scope = search_scope
        if opts.pre:
            finder.set_allow_all_prereleases()
        if opts.prefer_binary:
            finder.set_prefer_binary()
        if session:
            for host in opts.trusted_hosts or []:
                source = f'line {lineno} of {filename}'
                session.add_trusted_host(host, source=source)
def handle_line(
    line,
    options=None,
    finder=None,
    session=None,
):
    if line.is_requirement:
        parsed_req = handle_requirement_line(line, options)
        return parsed_req
    else:
        handle_option_line(
            line.opts,
            line.filename,
            line.lineno,
            finder,
            options,
            session,
        )
        return None
class RequirementsFileParser:
    def __init__(
        self,
        session,
        line_parser,
    ):
        self._session = session
        self._line_parser = line_parser
    def parse(self, filename, constraint):
        yield from self._parse_and_recurse(filename, constraint)
    def _parse_and_recurse(self, filename, constraint):
        for line in self._parse_file(filename, constraint):
            if (
                not line.is_requirement and
                (line.opts.requirements or line.opts.constraints)
            ):
                if line.opts.requirements:
                    req_path = line.opts.requirements[0]
                    nested_constraint = False
                else:
                    req_path = line.opts.constraints[0]
                    nested_constraint = True
                if SCHEME_RE.search(filename):
                    req_path = urllib.parse.urljoin(filename, req_path)
                elif not SCHEME_RE.search(req_path):
                    req_path = os.path.join(
                        os.path.dirname(filename), req_path,
                    )
                yield from self._parse_and_recurse(req_path, nested_constraint)
            else:
                yield line
    def _parse_file(self, filename, constraint):
        _, content = get_file_content(filename, self._session)
        lines_enum = preprocess(content)
        for line_number, line in lines_enum:
            try:
                args_str, opts = self._line_parser(line)
            except OptionParsingError as e:
                msg = f'Invalid requirement: {line}\n{e.msg}'
                raise RequirementsFileParseError(msg)
            yield ParsedLine(
                filename,
                line_number,
                args_str,
                opts,
                constraint,
            )
def get_line_parser(finder):
    def parse_line(line):
        parser = build_parser()
        defaults = parser.get_default_values()
        defaults.index_url = None
        if finder:
            defaults.format_control = finder.format_control
        args_str, options_str = break_args_options(line)
        opts, _ = parser.parse_args(shlex.split(options_str), defaults)
        return args_str, opts
    return parse_line
def break_args_options(line):
    tokens = line.split(' ')
    args = []
    options = tokens[:]
    for token in tokens:
        if token.startswith('-') or token.startswith('--'):
            break
        else:
            args.append(token)
            options.pop(0)
    return ' '.join(args), ' '.join(options)
class OptionParsingError(Exception):
    def __init__(self, msg):
        self.msg = msg
def build_parser():
    parser = optparse.OptionParser(add_help_option=False)
    option_factories = SUPPORTED_OPTIONS + SUPPORTED_OPTIONS_REQ
    for option_factory in option_factories:
        option = option_factory()
        parser.add_option(option)
    def parser_exit(self, msg):
        raise OptionParsingError(msg)
    parser.exit = parser_exit
    return parser
def join_lines(lines_enum):
    primary_line_number = None
    new_line = []
    for line_number, line in lines_enum:
        if not line.endswith('\\') or COMMENT_RE.match(line):
            if COMMENT_RE.match(line):
                line = ' ' + line
            if new_line:
                new_line.append(line)
                assert primary_line_number is not None
                yield primary_line_number, ''.join(new_line)
                new_line = []
            else:
                yield line_number, line
        else:
            if not new_line:
                primary_line_number = line_number
            new_line.append(line.strip('\\'))
    if new_line:
        assert primary_line_number is not None
        yield primary_line_number, ''.join(new_line)
def ignore_comments(lines_enum):
    for line_number, line in lines_enum:
        line = COMMENT_RE.sub('', line)
        line = line.strip()
        if line:
            yield line_number, line
def expand_env_variables(lines_enum):
    for line_number, line in lines_enum:
        for env_var, var_name in ENV_VAR_RE.findall(line):
            value = os.getenv(var_name)
            if not value:
                continue
            line = line.replace(env_var, value)
        yield line_number, line
def get_file_content(url, session):
    scheme = get_url_scheme(url)
    if scheme in ['http', 'https']:
        resp = session.get(url)
        raise_for_status(resp)
        return resp.url, resp.text
    elif scheme == 'file':
        url = url_to_path(url)
    try:
        with open(url, 'rb') as f:
            content = auto_decode(f.read())
    except OSError as exc:
        raise InstallationError(
            f'Could not open requirements file: {exc}'
        )
    return url, content
