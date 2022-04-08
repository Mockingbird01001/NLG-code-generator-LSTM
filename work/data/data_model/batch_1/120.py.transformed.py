
import logging
import os
import re
from typing import Any, Dict, Optional, Set, Tuple, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.specifiers import Specifier
from pip._vendor.pkg_resources import RequirementParseError, parse_requirements
from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.pyproject import make_pyproject_path
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import is_url, vcs
__all__ = [
    "install_req_from_editable", "install_req_from_line",
    "parse_editable"
]
logger = logging.getLogger(__name__)
operators = Specifier._operators.keys()
def _strip_extras(path):
    m = re.match(r'^(.+)(\[[^\]]+\])$', path)
    extras = None
    if m:
        path_no_extras = m.group(1)
        extras = m.group(2)
    else:
        path_no_extras = path
    return path_no_extras, extras
def convert_extras(extras):
    if not extras:
        return set()
    return Requirement("placeholder" + extras.lower()).extras
def parse_editable(editable_req):
    url = editable_req
    url_no_extras, extras = _strip_extras(url)
    if os.path.isdir(url_no_extras):
        setup_py = os.path.join(url_no_extras, 'setup.py')
        setup_cfg = os.path.join(url_no_extras, 'setup.cfg')
        if not os.path.exists(setup_py) and not os.path.exists(setup_cfg):
            msg = (
                'File "setup.py" or "setup.cfg" not found. Directory cannot be '
                'installed in editable mode: {}'
                .format(os.path.abspath(url_no_extras))
            )
            pyproject_path = make_pyproject_path(url_no_extras)
            if os.path.isfile(pyproject_path):
                msg += (
                    '\n(A "pyproject.toml" file was found, but editable '
                    'mode currently requires a setuptools-based build.)'
                )
            raise InstallationError(msg)
        url_no_extras = path_to_url(url_no_extras)
    if url_no_extras.lower().startswith('file:'):
        package_name = Link(url_no_extras).egg_fragment
        if extras:
            return (
                package_name,
                url_no_extras,
                Requirement("placeholder" + extras.lower()).extras,
            )
        else:
            return package_name, url_no_extras, set()
    for version_control in vcs:
        if url.lower().startswith(f'{version_control}:'):
            url = f'{version_control}+{url}'
            break
    link = Link(url)
    if not link.is_vcs:
        backends = ", ".join(vcs.all_schemes)
        raise InstallationError(
            f'{editable_req} is not a valid editable requirement. '
            f'It should either be a path to a local project or a VCS URL '
            f'(beginning with {backends}).'
        )
    package_name = link.egg_fragment
    if not package_name:
        raise InstallationError(
            "Could not detect requirement name for '{}', please specify one "
        )
    return package_name, url, set()
def deduce_helpful_msg(req):
    msg = ""
    if os.path.exists(req):
        msg = " The path does exist. "
        try:
            with open(req) as fp:
                next(parse_requirements(fp.read()))
                msg += (
                    "The argument you provided "
                    "({}) appears to be a"
                    " requirements file. If that is the"
                    " case, use the '-r' flag to install"
                    " the packages specified within it."
                ).format(req)
        except RequirementParseError:
            logger.debug(
                "Cannot parse '%s' as requirements file", req, exc_info=True
            )
    else:
        msg += f" File '{req}' does not exist."
    return msg
class RequirementParts:
    def __init__(
            self,
            requirement,
            link,
            markers,
            extras,
    ):
        self.requirement = requirement
        self.link = link
        self.markers = markers
        self.extras = extras
def parse_req_from_editable(editable_req):
    name, url, extras_override = parse_editable(editable_req)
    if name is not None:
        try:
            req = Requirement(name)
        except InvalidRequirement:
            raise InstallationError(f"Invalid requirement: '{name}'")
    else:
        req = None
    link = Link(url)
    return RequirementParts(req, link, None, extras_override)
def install_req_from_editable(
    editable_req,
    comes_from=None,
    use_pep517=None,
    isolated=False,
    options=None,
    constraint=False,
    user_supplied=False,
):
    parts = parse_req_from_editable(editable_req)
    return InstallRequirement(
        parts.requirement,
        comes_from=comes_from,
        user_supplied=user_supplied,
        editable=True,
        link=parts.link,
        constraint=constraint,
        use_pep517=use_pep517,
        isolated=isolated,
        install_options=options.get("install_options", []) if options else [],
        global_options=options.get("global_options", []) if options else [],
        hash_options=options.get("hashes", {}) if options else {},
        extras=parts.extras,
    )
def _looks_like_path(name):
    if os.path.sep in name:
        return True
    if os.path.altsep is not None and os.path.altsep in name:
        return True
    if name.startswith("."):
        return True
    return False
def _get_url_from_path(path, name):
    if _looks_like_path(name) and os.path.isdir(path):
        if is_installable_dir(path):
            return path_to_url(path)
        raise InstallationError(
            f"Directory {name!r} is not installable. Neither 'setup.py' "
            "nor 'pyproject.toml' found."
        )
    if not is_archive_file(path):
        return None
    if os.path.isfile(path):
        return path_to_url(path)
    urlreq_parts = name.split('@', 1)
    if len(urlreq_parts) >= 2 and not _looks_like_path(urlreq_parts[0]):
        return None
    logger.warning(
        'Requirement %r looks like a filename, but the '
        'file does not exist',
        name
    )
    return path_to_url(path)
def parse_req_from_line(name, line_source):
    if is_url(name):
        marker_sep = '; '
    else:
        marker_sep = ';'
    if marker_sep in name:
        name, markers_as_string = name.split(marker_sep, 1)
        markers_as_string = markers_as_string.strip()
        if not markers_as_string:
            markers = None
        else:
            markers = Marker(markers_as_string)
    else:
        markers = None
    name = name.strip()
    req_as_string = None
    path = os.path.normpath(os.path.abspath(name))
    link = None
    extras_as_string = None
    if is_url(name):
        link = Link(name)
    else:
        p, extras_as_string = _strip_extras(path)
        url = _get_url_from_path(p, name)
        if url is not None:
            link = Link(url)
    if link:
        if link.scheme == 'file' and re.search(r'\.\./', link.url):
            link = Link(
                path_to_url(os.path.normpath(os.path.abspath(link.path))))
        if link.is_wheel:
            wheel = Wheel(link.filename)
            req_as_string = f"{wheel.name}=={wheel.version}"
        else:
            req_as_string = link.egg_fragment
    else:
        req_as_string = name
    extras = convert_extras(extras_as_string)
    def with_source(text):
        if not line_source:
            return text
        return f'{text} (from {line_source})'
    def _parse_req_string(req_as_string: str) -> Requirement:
        try:
            req = Requirement(req_as_string)
        except InvalidRequirement:
            if os.path.sep in req_as_string:
                add_msg = "It looks like a path."
                add_msg += deduce_helpful_msg(req_as_string)
            elif ('=' in req_as_string and
                  not any(op in req_as_string for op in operators)):
                add_msg = "= is not a valid operator. Did you mean == ?"
            else:
                add_msg = ''
            msg = with_source(
                f'Invalid requirement: {req_as_string!r}'
            )
            if add_msg:
                msg += f'\nHint: {add_msg}'
            raise InstallationError(msg)
        else:
            for spec in req.specifier:
                spec_str = str(spec)
                if spec_str.endswith(']'):
                    msg = f"Extras after version '{spec_str}'."
                    raise InstallationError(msg)
        return req
    if req_as_string is not None:
        req = _parse_req_string(req_as_string)
    else:
        req = None
    return RequirementParts(req, link, markers, extras)
def install_req_from_line(
    name,
    comes_from=None,
    use_pep517=None,
    isolated=False,
    options=None,
    constraint=False,
    line_source=None,
    user_supplied=False,
):
    parts = parse_req_from_line(name, line_source)
    return InstallRequirement(
        parts.requirement, comes_from, link=parts.link, markers=parts.markers,
        use_pep517=use_pep517, isolated=isolated,
        install_options=options.get("install_options", []) if options else [],
        global_options=options.get("global_options", []) if options else [],
        hash_options=options.get("hashes", {}) if options else {},
        constraint=constraint,
        extras=parts.extras,
        user_supplied=user_supplied,
    )
def install_req_from_req_string(
    req_string,
    comes_from=None,
    isolated=False,
    use_pep517=None,
    user_supplied=False,
):
    try:
        req = Requirement(req_string)
    except InvalidRequirement:
        raise InstallationError(f"Invalid requirement: '{req_string}'")
    domains_not_allowed = [
        PyPI.file_storage_domain,
        TestPyPI.file_storage_domain,
    ]
    if (req.url and comes_from and comes_from.link and
            comes_from.link.netloc in domains_not_allowed):
        raise InstallationError(
            "Packages installed from PyPI cannot depend on packages "
            "which are not also hosted on PyPI.\n"
            "{} depends on {} ".format(comes_from.name, req)
        )
    return InstallRequirement(
        req,
        comes_from,
        isolated=isolated,
        use_pep517=use_pep517,
        user_supplied=user_supplied,
    )
def install_req_from_parsed_requirement(
    parsed_req,
    isolated=False,
    use_pep517=None,
    user_supplied=False,
):
    if parsed_req.is_editable:
        req = install_req_from_editable(
            parsed_req.requirement,
            comes_from=parsed_req.comes_from,
            use_pep517=use_pep517,
            constraint=parsed_req.constraint,
            isolated=isolated,
            user_supplied=user_supplied,
        )
    else:
        req = install_req_from_line(
            parsed_req.requirement,
            comes_from=parsed_req.comes_from,
            use_pep517=use_pep517,
            isolated=isolated,
            options=parsed_req.options,
            constraint=parsed_req.constraint,
            line_source=parsed_req.line_source,
            user_supplied=user_supplied,
        )
    return req
def install_req_from_link_and_ireq(link, ireq):
    return InstallRequirement(
        req=ireq.req,
        comes_from=ireq.comes_from,
        editable=ireq.editable,
        link=link,
        markers=ireq.markers,
        use_pep517=ireq.use_pep517,
        isolated=ireq.isolated,
        install_options=ireq.install_options,
        global_options=ireq.global_options,
        hash_options=ireq.hash_options,
    )
