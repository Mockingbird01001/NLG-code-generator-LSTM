
import locale
import logging
import os
import sys
from typing import List, Optional
from pip._internal.cli.autocompletion import autocomplete
from pip._internal.cli.main_parser import parse_command
from pip._internal.commands import create_command
from pip._internal.exceptions import PipError
from pip._internal.utils import deprecation
logger = logging.getLogger(__name__)
def main(args=None):
    if args is None:
        args = sys.argv[1:]
    deprecation.install_warning_logger()
    autocomplete()
    try:
        cmd_name, cmd_args = parse_command(args)
    except PipError as exc:
        sys.stderr.write(f"ERROR: {exc}")
        sys.stderr.write(os.linesep)
        sys.exit(1)
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error as e:
        logger.debug("Ignoring error %s when setting locale", e)
    command = create_command(cmd_name, isolated=("--isolated" in cmd_args))
    return command.main(cmd_args)
