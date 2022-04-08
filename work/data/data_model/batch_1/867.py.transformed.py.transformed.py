import sys
import subprocess
def __optim_args_from_interpreter_flags():
    args = []
    value = sys.flags.optimize
    if value > 0:
        args.append("-" + "O" * value)
    return args
_optim_args_from_interpreter_flags = getattr(
    subprocess,
    "_optim_args_from_interpreter_flags",
    __optim_args_from_interpreter_flags,
)
