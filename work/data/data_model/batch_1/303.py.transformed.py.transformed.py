
import os
import sys
from distutils.cmd import Command as DistutilsCommand
from distutils.command.install import SCHEME_KEYS
from distutils.command.install import install as distutils_install_command
from distutils.sysconfig import get_python_lib
from typing import Dict, List, Optional, Tuple, Union, cast
from pip._internal.models.scheme import Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import get_major_minor_version
def _distutils_scheme(
    dist_name, user=False, home=None, root=None, isolated=False, prefix=None
):
    from distutils.dist import Distribution
    dist_args = {"name": dist_name}
    if isolated:
        dist_args["script_args"] = ["--no-user-cfg"]
    d = Distribution(dist_args)
    d.parse_config_files()
    obj = None
    obj = d.get_command_obj("install", create=True)
    assert obj is not None
    i = cast(distutils_install_command, obj)
    assert not (user and prefix), f"user={user} prefix={prefix}"
    assert not (home and prefix), f"home={home} prefix={prefix}"
    i.user = user or i.user
    if user or home:
        i.prefix = ""
    i.prefix = prefix or i.prefix
    i.home = home or i.home
    i.root = root or i.root
    i.finalize_options()
    scheme = {}
    for key in SCHEME_KEYS:
        scheme[key] = getattr(i, "install_" + key)
    if "install_lib" in d.get_option_dict("install"):
        scheme.update(dict(purelib=i.install_lib, platlib=i.install_lib))
    if running_under_virtualenv():
        scheme["headers"] = os.path.join(
            i.prefix,
            "include",
            "site",
            f"python{get_major_minor_version()}",
            dist_name,
        )
        if root is not None:
            path_no_drive = os.path.splitdrive(os.path.abspath(scheme["headers"]))[1]
            scheme["headers"] = os.path.join(
                root,
                path_no_drive[1:],
            )
    return scheme
def get_scheme(
    dist_name,
    user=False,
    home=None,
    root=None,
    isolated=False,
    prefix=None,
):
    scheme = _distutils_scheme(dist_name, user, home, root, isolated, prefix)
    return Scheme(
        platlib=scheme["platlib"],
        purelib=scheme["purelib"],
        headers=scheme["headers"],
        scripts=scheme["scripts"],
        data=scheme["data"],
    )
def get_bin_prefix():
    if WINDOWS:
        bin_py = os.path.join(sys.prefix, "Scripts")
        if not os.path.exists(bin_py):
            bin_py = os.path.join(sys.prefix, "bin")
        return bin_py
    if sys.platform[:6] == "darwin" and sys.prefix[:16] == "/System/Library/":
        return "/usr/local/bin"
    return os.path.join(sys.prefix, "bin")
def get_purelib():
    return get_python_lib(plat_specific=False)
def get_platlib():
    return get_python_lib(plat_specific=True)
def get_prefixed_libs(prefix):
    return (
        get_python_lib(plat_specific=False, prefix=prefix),
        get_python_lib(plat_specific=True, prefix=prefix),
    )
