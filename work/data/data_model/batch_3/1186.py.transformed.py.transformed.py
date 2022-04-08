import logging
import os
import re
import site
import sys
from typing import List, Optional
logger = logging.getLogger(__name__)
_INCLUDE_SYSTEM_SITE_PACKAGES_REGEX = re.compile(
    r"include-system-site-packages\s*=\s*(?P<value>true|false)"
)
def _running_under_venv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)
def _running_under_regular_virtualenv():
    return hasattr(sys, "real_prefix")
def running_under_virtualenv():
    return _running_under_venv() or _running_under_regular_virtualenv()
def _get_pyvenv_cfg_lines():
    pyvenv_cfg_file = os.path.join(sys.prefix, "pyvenv.cfg")
    try:
        with open(pyvenv_cfg_file, encoding="utf-8") as f:
            return f.read().splitlines()
    except OSError:
        return None
def _no_global_under_venv():
    cfg_lines = _get_pyvenv_cfg_lines()
    if cfg_lines is None:
        logger.warning(
            "Could not access 'pyvenv.cfg' despite a virtual environment "
            "being active. Assuming global site-packages is not accessible "
            "in this environment."
        )
        return True
    for line in cfg_lines:
        match = _INCLUDE_SYSTEM_SITE_PACKAGES_REGEX.match(line)
        if match is not None and match.group("value") == "false":
            return True
    return False
def _no_global_under_regular_virtualenv():
    site_mod_dir = os.path.dirname(os.path.abspath(site.__file__))
    no_global_site_packages_file = os.path.join(
        site_mod_dir,
        "no-global-site-packages.txt",
    )
    return os.path.exists(no_global_site_packages_file)
def virtualenv_no_global():
    if _running_under_venv():
        return _no_global_under_venv()
    if _running_under_regular_virtualenv():
        return _no_global_under_regular_virtualenv()
    return False
