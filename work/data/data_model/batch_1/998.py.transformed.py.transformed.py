
import os
import shlex
import subprocess
try:
    from shlex import quote
except ImportError:
    from pipes import quote
__all__ = ['WindowsParser', 'PosixParser', 'NativeParser']
class CommandLineParser:
    @staticmethod
    def join(argv):
        raise NotImplementedError
    @staticmethod
    def split(cmd):
        raise NotImplementedError
class WindowsParser:
    @staticmethod
    def join(argv):
        return subprocess.list2cmdline(argv)
    @staticmethod
    def split(cmd):
        import ctypes
        try:
            ctypes.windll
        except AttributeError:
            raise NotImplementedError
        if not cmd:
            return []
        cmd = 'dummy ' + cmd
        CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
        CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))
        nargs = ctypes.c_int()
        lpargs = CommandLineToArgvW(cmd, ctypes.byref(nargs))
        args = [lpargs[i] for i in range(nargs.value)]
        assert not ctypes.windll.kernel32.LocalFree(lpargs)
        assert args[0] == "dummy"
        return args[1:]
class PosixParser:
    @staticmethod
    def join(argv):
        return ' '.join(quote(arg) for arg in argv)
    @staticmethod
    def split(cmd):
        return shlex.split(cmd, posix=True)
if os.name == 'nt':
    NativeParser = WindowsParser
elif os.name == 'posix':
    NativeParser = PosixParser
