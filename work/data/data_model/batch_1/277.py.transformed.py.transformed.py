
import logging
import warnings
from typing import Any, Optional, TextIO, Type, Union
from pip._vendor.packaging.version import parse
from pip import __version__ as current_version
DEPRECATION_MSG_PREFIX = "DEPRECATION: "
class PipDeprecationWarning(Warning):
    pass
_original_showwarning = None
def _showwarning(
    message,
    category,
    filename,
    lineno,
    file=None,
    line=None,
):
    if file is not None:
        if _original_showwarning is not None:
            _original_showwarning(message, category, filename, lineno, file, line)
    elif issubclass(category, PipDeprecationWarning):
        logger = logging.getLogger("pip._internal.deprecations")
        logger.warning(message)
    else:
        _original_showwarning(message, category, filename, lineno, file, line)
def install_warning_logger():
    warnings.simplefilter("default", PipDeprecationWarning, append=True)
    global _original_showwarning
    if _original_showwarning is None:
        _original_showwarning = warnings.showwarning
        warnings.showwarning = _showwarning
def deprecated(reason, replacement, gone_in, issue=None):
    sentences = [
        (reason, DEPRECATION_MSG_PREFIX + "{}"),
        (gone_in, "pip {} will remove support for this functionality."),
        (replacement, "A possible replacement is {}."),
        (
            issue,
            (
                "You can find discussion regarding this at "
                "https://github.com/pypa/pip/issues/{}."
            ),
        ),
    ]
    message = " ".join(
        template.format(val) for val, template in sentences if val is not None
    )
    if gone_in is not None and parse(current_version) >= parse(gone_in):
        raise PipDeprecationWarning(message)
    warnings.warn(message, category=PipDeprecationWarning, stacklevel=2)
