import distutils.util
import logging
import os
import sys
import sysconfig
import typing
from pip._internal.exceptions import InvalidSchemeCombination, UserInstallationInvalid
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import get_major_minor_version
logger = logging.getLogger(__name__)
_AVAILABLE_SCHEMES = set(sysconfig.get_scheme_names())
def _infer_prefix():
    implementation_suffixed = f"{sys.implementation.name}_{os.name}"
    if implementation_suffixed in _AVAILABLE_SCHEMES:
        return implementation_suffixed
    if sys.implementation.name in _AVAILABLE_SCHEMES:
        return sys.implementation.name
    suffixed = f"{os.name}_prefix"
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    if os.name in _AVAILABLE_SCHEMES:
        return os.name
    return "posix_prefix"
def _infer_user():
    suffixed = f"{os.name}_user"
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    if "posix_user" not in _AVAILABLE_SCHEMES:
        raise UserInstallationInvalid()
    return "posix_user"
def _infer_home():
    suffixed = f"{os.name}_home"
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    return "posix_home"
_HOME_KEYS = [
    "installed_base",
    "base",
    "installed_platbase",
    "platbase",
    "prefix",
    "exec_prefix",
]
if sysconfig.get_config_var("userbase") is not None:
    _HOME_KEYS.append("userbase")
def get_scheme(
    dist_name,
    user=False,
    home=None,
    root=None,
    isolated=False,
    prefix=None,
):
    if user and prefix:
        raise InvalidSchemeCombination("--user", "--prefix")
    if home and prefix:
        raise InvalidSchemeCombination("--home", "--prefix")
    if home is not None:
        scheme_name = _infer_home()
    elif user:
        scheme_name = _infer_user()
    else:
        scheme_name = _infer_prefix()
    if home is not None:
        variables = {k: home for k in _HOME_KEYS}
    elif prefix is not None:
        variables = {k: prefix for k in _HOME_KEYS}
    else:
        variables = {}
    paths = sysconfig.get_paths(scheme=scheme_name, vars=variables)
    if running_under_virtualenv():
        if user:
            base = variables.get("userbase", sys.prefix)
        else:
            base = variables.get("base", sys.prefix)
        python_xy = f"python{get_major_minor_version()}"
        paths["include"] = os.path.join(base, "include", "site", python_xy)
    elif not dist_name:
        dist_name = "UNKNOWN"
    scheme = Scheme(
        platlib=paths["platlib"],
        purelib=paths["purelib"],
        headers=os.path.join(paths["include"], dist_name),
        scripts=paths["scripts"],
        data=paths["data"],
    )
    if root is not None:
        for key in SCHEME_KEYS:
            value = distutils.util.change_root(root, getattr(scheme, key))
            setattr(scheme, key, value)
    return scheme
def get_bin_prefix():
    if sys.platform[:6] == "darwin" and sys.prefix[:16] == "/System/Library/":
        return "/usr/local/bin"
    return sysconfig.get_paths()["scripts"]
def get_purelib():
    return sysconfig.get_paths()["purelib"]
def get_platlib():
    return sysconfig.get_paths()["platlib"]
def get_prefixed_libs(prefix):
    paths = sysconfig.get_paths(vars={"base": prefix, "platbase": prefix})
    return (paths["purelib"], paths["platlib"])
