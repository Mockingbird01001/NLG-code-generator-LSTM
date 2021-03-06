
import logging
from typing import List, Optional, Sequence
from pip._internal.build_env import BuildEnvironment
from pip._internal.utils.logging import indent_log
from pip._internal.utils.setuptools_build import make_setuptools_develop_args
from pip._internal.utils.subprocess import call_subprocess
logger = logging.getLogger(__name__)
def install_editable(
    install_options,
    global_options,
    prefix,
    home,
    use_user_site,
    name,
    setup_py_path,
    isolated,
    build_env,
    unpacked_source_directory,
):
    logger.info('Running setup.py develop for %s', name)
    args = make_setuptools_develop_args(
        setup_py_path,
        global_options=global_options,
        install_options=install_options,
        no_user_config=isolated,
        prefix=prefix,
        home=home,
        use_user_site=use_user_site,
    )
    with indent_log():
        with build_env:
            call_subprocess(
                args,
                cwd=unpacked_source_directory,
            )
