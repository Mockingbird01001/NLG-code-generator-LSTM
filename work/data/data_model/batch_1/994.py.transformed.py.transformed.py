import sys
from typing import List, Optional, Sequence
_SETUPTOOLS_SHIM = (
    "import io, os, sys, setuptools, tokenize; sys.argv[0] = {0!r}; __file__={0!r};"
    "f = getattr(tokenize, 'open', open)(__file__) "
    "if os.path.exists(__file__) "
    "else io.StringIO('from setuptools import setup; setup()');"
    "code = f.read().replace('\\r\\n', '\\n');"
    "f.close();"
    "exec(compile(code, __file__, 'exec'))"
)
def make_setuptools_shim_args(
    setup_py_path,
    global_options=None,
    no_user_config=False,
    unbuffered_output=False,
):
    args = [sys.executable]
    if unbuffered_output:
        args += ["-u"]
    args += ["-c", _SETUPTOOLS_SHIM.format(setup_py_path)]
    if global_options:
        args += global_options
    if no_user_config:
        args += ["--no-user-cfg"]
    return args
def make_setuptools_bdist_wheel_args(
    setup_py_path,
    global_options,
    build_options,
    destination_dir,
):
    args = make_setuptools_shim_args(
        setup_py_path, global_options=global_options, unbuffered_output=True
    )
    args += ["bdist_wheel", "-d", destination_dir]
    args += build_options
    return args
def make_setuptools_clean_args(
    setup_py_path,
    global_options,
):
    args = make_setuptools_shim_args(
        setup_py_path, global_options=global_options, unbuffered_output=True
    )
    args += ["clean", "--all"]
    return args
def make_setuptools_develop_args(
    setup_py_path,
    global_options,
    install_options,
    no_user_config,
    prefix,
    home,
    use_user_site,
):
    assert not (use_user_site and prefix)
    args = make_setuptools_shim_args(
        setup_py_path,
        global_options=global_options,
        no_user_config=no_user_config,
    )
    args += ["develop", "--no-deps"]
    args += install_options
    if prefix:
        args += ["--prefix", prefix]
    if home is not None:
        args += ["--install-dir", home]
    if use_user_site:
        args += ["--user", "--prefix="]
    return args
def make_setuptools_egg_info_args(
    setup_py_path,
    egg_info_dir,
    no_user_config,
):
    args = make_setuptools_shim_args(setup_py_path, no_user_config=no_user_config)
    args += ["egg_info"]
    if egg_info_dir:
        args += ["--egg-base", egg_info_dir]
    return args
def make_setuptools_install_args(
    setup_py_path,
    global_options,
    install_options,
    record_filename,
    root,
    prefix,
    header_dir,
    home,
    use_user_site,
    no_user_config,
    pycompile,
):
    assert not (use_user_site and prefix)
    assert not (use_user_site and root)
    args = make_setuptools_shim_args(
        setup_py_path,
        global_options=global_options,
        no_user_config=no_user_config,
        unbuffered_output=True,
    )
    args += ["install", "--record", record_filename]
    args += ["--single-version-externally-managed"]
    if root is not None:
        args += ["--root", root]
    if prefix is not None:
        args += ["--prefix", prefix]
    if home is not None:
        args += ["--home", home]
    if use_user_site:
        args += ["--user", "--prefix="]
    if pycompile:
        args += ["--compile"]
    else:
        args += ["--no-compile"]
    if header_dir:
        args += ["--install-headers", header_dir]
    args += install_options
    return args
