import sys
from typing import List, Optional
from pip._internal.cli.main import main
def _wrapper(args=None):
    sys.stderr.write(
        "WARNING: pip is being invoked by an old script wrapper. This will "
        "fail in a future version of pip.\n"
        "Please see https://github.com/pypa/pip/issues/5599 for advice on "
        "fixing the underlying issue.\n"
        "To avoid this problem you can invoke Python with '-m pip' instead of "
        "running pip directly.\n"
    )
    return main(args)
