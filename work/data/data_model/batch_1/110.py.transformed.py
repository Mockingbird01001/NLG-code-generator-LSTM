
import re
from typing import TYPE_CHECKING, List, Optional, Tuple
from pip._vendor.packaging.tags import (
    Tag,
    compatible_tags,
    cpython_tags,
    generic_tags,
    interpreter_name,
    interpreter_version,
    mac_platforms,
)
if TYPE_CHECKING:
    from pip._vendor.packaging.tags import PythonVersion
_osx_arch_pat = re.compile(r"(.+)_(\d+)_(\d+)_(.+)")
def version_info_to_nodot(version_info):
    return "".join(map(str, version_info[:2]))
def _mac_platforms(arch):
    match = _osx_arch_pat.match(arch)
    if match:
        name, major, minor, actual_arch = match.groups()
        mac_version = (int(major), int(minor))
        arches = [
            "{}_{}".format(name, arch[len("macosx_") :])
            for arch in mac_platforms(mac_version, actual_arch)
        ]
    else:
        arches = [arch]
    return arches
def _custom_manylinux_platforms(arch):
    arches = [arch]
    arch_prefix, arch_sep, arch_suffix = arch.partition("_")
    if arch_prefix == "manylinux2014":
        if arch_suffix in {"i686", "x86_64"}:
            arches.append("manylinux2010" + arch_sep + arch_suffix)
            arches.append("manylinux1" + arch_sep + arch_suffix)
    elif arch_prefix == "manylinux2010":
        arches.append("manylinux1" + arch_sep + arch_suffix)
    return arches
def _get_custom_platforms(arch):
    arch_prefix, arch_sep, arch_suffix = arch.partition("_")
    if arch.startswith("macosx"):
        arches = _mac_platforms(arch)
    elif arch_prefix in ["manylinux2014", "manylinux2010"]:
        arches = _custom_manylinux_platforms(arch)
    else:
        arches = [arch]
    return arches
def _expand_allowed_platforms(platforms):
    if not platforms:
        return None
    seen = set()
    result = []
    for p in platforms:
        if p in seen:
            continue
        additions = [c for c in _get_custom_platforms(p) if c not in seen]
        seen.update(additions)
        result.extend(additions)
    return result
def _get_python_version(version):
    if len(version) > 1:
        return int(version[0]), int(version[1:])
    else:
        return (int(version[0]),)
def _get_custom_interpreter(implementation=None, version=None):
    if implementation is None:
        implementation = interpreter_name()
    if version is None:
        version = interpreter_version()
    return f"{implementation}{version}"
def get_supported(
    version=None,
    platforms=None,
    impl=None,
    abis=None,
):
    supported = []
    python_version = None
    if version is not None:
        python_version = _get_python_version(version)
    interpreter = _get_custom_interpreter(impl, version)
    platforms = _expand_allowed_platforms(platforms)
    is_cpython = (impl or interpreter_name()) == "cp"
    if is_cpython:
        supported.extend(
            cpython_tags(
                python_version=python_version,
                abis=abis,
                platforms=platforms,
            )
        )
    else:
        supported.extend(
            generic_tags(
                interpreter=interpreter,
                abis=abis,
                platforms=platforms,
            )
        )
    supported.extend(
        compatible_tags(
            python_version=python_version,
            interpreter=interpreter,
            platforms=platforms,
        )
    )
    return supported
