
from __future__ import division, print_function
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import defaultdict
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
try:
    from socket import AF_INET6
except ImportError:
    AF_INET6 = None
try:
    from socket import AF_UNIX
except ImportError:
    AF_UNIX = None
if sys.version_info >= (3, 4):
    import enum
else:
    enum = None
PY3 = sys.version_info[0] == 3
__all__ = [
    'FREEBSD', 'BSD', 'LINUX', 'NETBSD', 'OPENBSD', 'MACOS', 'OSX', 'POSIX',
    'SUNOS', 'WINDOWS',
    'CONN_CLOSE', 'CONN_CLOSE_WAIT', 'CONN_CLOSING', 'CONN_ESTABLISHED',
    'CONN_FIN_WAIT1', 'CONN_FIN_WAIT2', 'CONN_LAST_ACK', 'CONN_LISTEN',
    'CONN_NONE', 'CONN_SYN_RECV', 'CONN_SYN_SENT', 'CONN_TIME_WAIT',
    'NIC_DUPLEX_FULL', 'NIC_DUPLEX_HALF', 'NIC_DUPLEX_UNKNOWN',
    'STATUS_DEAD', 'STATUS_DISK_SLEEP', 'STATUS_IDLE', 'STATUS_LOCKED',
    'STATUS_RUNNING', 'STATUS_SLEEPING', 'STATUS_STOPPED', 'STATUS_SUSPENDED',
    'STATUS_TRACING_STOP', 'STATUS_WAITING', 'STATUS_WAKE_KILL',
    'STATUS_WAKING', 'STATUS_ZOMBIE', 'STATUS_PARKED',
    'ENCODING', 'ENCODING_ERRS', 'AF_INET6',
    'pconn', 'pcputimes', 'pctxsw', 'pgids', 'pio', 'pionice', 'popenfile',
    'pthread', 'puids', 'sconn', 'scpustats', 'sdiskio', 'sdiskpart',
    'sdiskusage', 'snetio', 'snicaddr', 'snicstats', 'sswap', 'suser',
    'conn_tmap', 'deprecated_method', 'isfile_strict', 'memoize',
    'parse_environ_block', 'path_exists_strict', 'usage_percent',
    'supports_ipv6', 'sockfam_to_enum', 'socktype_to_enum', "wrap_numbers",
    'bytes2human', 'conn_to_ntuple', 'debug',
    'hilite', 'term_supports_colors', 'print_color',
]
POSIX = os.name == "posix"
WINDOWS = os.name == "nt"
LINUX = sys.platform.startswith("linux")
MACOS = sys.platform.startswith("darwin")
OSX = MACOS
FREEBSD = sys.platform.startswith("freebsd")
OPENBSD = sys.platform.startswith("openbsd")
NETBSD = sys.platform.startswith("netbsd")
BSD = FREEBSD or OPENBSD or NETBSD
SUNOS = sys.platform.startswith(("sunos", "solaris"))
AIX = sys.platform.startswith("aix")
STATUS_RUNNING = "running"
STATUS_SLEEPING = "sleeping"
STATUS_DISK_SLEEP = "disk-sleep"
STATUS_STOPPED = "stopped"
STATUS_TRACING_STOP = "tracing-stop"
STATUS_ZOMBIE = "zombie"
STATUS_DEAD = "dead"
STATUS_WAKE_KILL = "wake-kill"
STATUS_WAKING = "waking"
STATUS_IDLE = "idle"
STATUS_LOCKED = "locked"
STATUS_WAITING = "waiting"
STATUS_SUSPENDED = "suspended"
STATUS_PARKED = "parked"
CONN_ESTABLISHED = "ESTABLISHED"
CONN_SYN_SENT = "SYN_SENT"
CONN_SYN_RECV = "SYN_RECV"
CONN_FIN_WAIT1 = "FIN_WAIT1"
CONN_FIN_WAIT2 = "FIN_WAIT2"
CONN_TIME_WAIT = "TIME_WAIT"
CONN_CLOSE = "CLOSE"
CONN_CLOSE_WAIT = "CLOSE_WAIT"
CONN_LAST_ACK = "LAST_ACK"
CONN_LISTEN = "LISTEN"
CONN_CLOSING = "CLOSING"
CONN_NONE = "NONE"
if enum is None:
    NIC_DUPLEX_FULL = 2
    NIC_DUPLEX_HALF = 1
    NIC_DUPLEX_UNKNOWN = 0
else:
    class NicDuplex(enum.IntEnum):
        NIC_DUPLEX_FULL = 2
        NIC_DUPLEX_HALF = 1
        NIC_DUPLEX_UNKNOWN = 0
    globals().update(NicDuplex.__members__)
if enum is None:
    POWER_TIME_UNKNOWN = -1
    POWER_TIME_UNLIMITED = -2
else:
    class BatteryTime(enum.IntEnum):
        POWER_TIME_UNKNOWN = -1
        POWER_TIME_UNLIMITED = -2
    globals().update(BatteryTime.__members__)
ENCODING = sys.getfilesystemencoding()
if not PY3:
    ENCODING_ERRS = "replace"
else:
    try:
        ENCODING_ERRS = sys.getfilesystemencodeerrors()
    except AttributeError:
        ENCODING_ERRS = "surrogateescape" if POSIX else "replace"
sswap = namedtuple('sswap', ['total', 'used', 'free', 'percent', 'sin',
                             'sout'])
sdiskusage = namedtuple('sdiskusage', ['total', 'used', 'free', 'percent'])
sdiskio = namedtuple('sdiskio', ['read_count', 'write_count',
                                 'read_bytes', 'write_bytes',
                                 'read_time', 'write_time'])
sdiskpart = namedtuple('sdiskpart', ['device', 'mountpoint', 'fstype', 'opts',
                                     'maxfile', 'maxpath'])
snetio = namedtuple('snetio', ['bytes_sent', 'bytes_recv',
                               'packets_sent', 'packets_recv',
                               'errin', 'errout',
                               'dropin', 'dropout'])
suser = namedtuple('suser', ['name', 'terminal', 'host', 'started', 'pid'])
sconn = namedtuple('sconn', ['fd', 'family', 'type', 'laddr', 'raddr',
                             'status', 'pid'])
snicaddr = namedtuple('snicaddr',
                      ['family', 'address', 'netmask', 'broadcast', 'ptp'])
snicstats = namedtuple('snicstats', ['isup', 'duplex', 'speed', 'mtu'])
scpustats = namedtuple(
    'scpustats', ['ctx_switches', 'interrupts', 'soft_interrupts', 'syscalls'])
scpufreq = namedtuple('scpufreq', ['current', 'min', 'max'])
shwtemp = namedtuple(
    'shwtemp', ['label', 'current', 'high', 'critical'])
sbattery = namedtuple('sbattery', ['percent', 'secsleft', 'power_plugged'])
sfan = namedtuple('sfan', ['label', 'current'])
pcputimes = namedtuple('pcputimes',
                       ['user', 'system', 'children_user', 'children_system'])
popenfile = namedtuple('popenfile', ['path', 'fd'])
pthread = namedtuple('pthread', ['id', 'user_time', 'system_time'])
puids = namedtuple('puids', ['real', 'effective', 'saved'])
pgids = namedtuple('pgids', ['real', 'effective', 'saved'])
pio = namedtuple('pio', ['read_count', 'write_count',
                         'read_bytes', 'write_bytes'])
pionice = namedtuple('pionice', ['ioclass', 'value'])
pctxsw = namedtuple('pctxsw', ['voluntary', 'involuntary'])
pconn = namedtuple('pconn', ['fd', 'family', 'type', 'laddr', 'raddr',
                             'status'])
addr = namedtuple('addr', ['ip', 'port'])
conn_tmap = {
    "all": ([AF_INET, AF_INET6, AF_UNIX], [SOCK_STREAM, SOCK_DGRAM]),
    "tcp": ([AF_INET, AF_INET6], [SOCK_STREAM]),
    "tcp4": ([AF_INET], [SOCK_STREAM]),
    "udp": ([AF_INET, AF_INET6], [SOCK_DGRAM]),
    "udp4": ([AF_INET], [SOCK_DGRAM]),
    "inet": ([AF_INET, AF_INET6], [SOCK_STREAM, SOCK_DGRAM]),
    "inet4": ([AF_INET], [SOCK_STREAM, SOCK_DGRAM]),
    "inet6": ([AF_INET6], [SOCK_STREAM, SOCK_DGRAM]),
}
if AF_INET6 is not None:
    conn_tmap.update({
        "tcp6": ([AF_INET6], [SOCK_STREAM]),
        "udp6": ([AF_INET6], [SOCK_DGRAM]),
    })
if AF_UNIX is not None:
    conn_tmap.update({
        "unix": ([AF_UNIX], [SOCK_STREAM, SOCK_DGRAM]),
    })
class Error(Exception):
    __module__ = 'psutil'
    def __init__(self, msg=""):
        Exception.__init__(self, msg)
        self.msg = msg
    def __repr__(self):
        ret = "psutil.%s %s" % (self.__class__.__name__, self.msg)
        return ret.strip()
    __str__ = __repr__
class NoSuchProcess(Error):
    __module__ = 'psutil'
    def __init__(self, pid, name=None, msg=None):
        Error.__init__(self, msg)
        self.pid = pid
        self.name = name
        self.msg = msg
        if msg is None:
            if name:
                details = "(pid=%s, name=%s)" % (self.pid, repr(self.name))
            else:
                details = "(pid=%s)" % self.pid
            self.msg = "process no longer exists " + details
class ZombieProcess(NoSuchProcess):
    __module__ = 'psutil'
    def __init__(self, pid, name=None, ppid=None, msg=None):
        NoSuchProcess.__init__(self, msg)
        self.pid = pid
        self.ppid = ppid
        self.name = name
        self.msg = msg
        if msg is None:
            args = ["pid=%s" % pid]
            if name:
                args.append("name=%s" % repr(self.name))
            if ppid:
                args.append("ppid=%s" % self.ppid)
            details = "(%s)" % ", ".join(args)
            self.msg = "process still exists but it's a zombie " + details
class AccessDenied(Error):
    __module__ = 'psutil'
    def __init__(self, pid=None, name=None, msg=None):
        Error.__init__(self, msg)
        self.pid = pid
        self.name = name
        self.msg = msg
        if msg is None:
            if (pid is not None) and (name is not None):
                self.msg = "(pid=%s, name=%s)" % (pid, repr(name))
            elif (pid is not None):
                self.msg = "(pid=%s)" % self.pid
            else:
                self.msg = ""
class TimeoutExpired(Error):
    __module__ = 'psutil'
    def __init__(self, seconds, pid=None, name=None):
        Error.__init__(self, "timeout after %s seconds" % seconds)
        self.seconds = seconds
        self.pid = pid
        self.name = name
        if (pid is not None) and (name is not None):
            self.msg += " (pid=%s, name=%s)" % (pid, repr(name))
        elif (pid is not None):
            self.msg += " (pid=%s)" % self.pid
def usage_percent(used, total, round_=None):
    try:
        ret = (float(used) / total) * 100
    except ZeroDivisionError:
        return 0.0
    else:
        if round_ is not None:
            ret = round(ret, round_)
        return ret
def memoize(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(sorted(kwargs.items())))
        try:
            return cache[key]
        except KeyError:
            ret = cache[key] = fun(*args, **kwargs)
            return ret
    def cache_clear():
        cache.clear()
    cache = {}
    wrapper.cache_clear = cache_clear
    return wrapper
def memoize_when_activated(fun):
    @functools.wraps(fun)
    def wrapper(self):
        try:
            ret = self._cache[fun]
        except AttributeError:
            return fun(self)
        except KeyError:
            ret = self._cache[fun] = fun(self)
        return ret
    def cache_activate(proc):
        proc._cache = {}
    def cache_deactivate(proc):
        try:
            del proc._cache
        except AttributeError:
            pass
    wrapper.cache_activate = cache_activate
    wrapper.cache_deactivate = cache_deactivate
    return wrapper
def isfile_strict(path):
    try:
        st = os.stat(path)
    except OSError as err:
        if err.errno in (errno.EPERM, errno.EACCES):
            raise
        return False
    else:
        return stat.S_ISREG(st.st_mode)
def path_exists_strict(path):
    try:
        os.stat(path)
    except OSError as err:
        if err.errno in (errno.EPERM, errno.EACCES):
            raise
        return False
    else:
        return True
@memoize
def supports_ipv6():
    if not socket.has_ipv6 or AF_INET6 is None:
        return False
    try:
        sock = socket.socket(AF_INET6, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(("::1", 0))
        return True
    except socket.error:
        return False
def parse_environ_block(data):
    ret = {}
    pos = 0
    WINDOWS_ = WINDOWS
    while True:
        next_pos = data.find("\0", pos)
        if next_pos <= pos:
            break
        equal_pos = data.find("=", pos, next_pos)
        if equal_pos > pos:
            key = data[pos:equal_pos]
            value = data[equal_pos + 1:next_pos]
            if WINDOWS_:
                key = key.upper()
            ret[key] = value
        pos = next_pos + 1
    return ret
def sockfam_to_enum(num):
    if enum is None:
        return num
    else:
        try:
            return socket.AddressFamily(num)
        except ValueError:
            return num
def socktype_to_enum(num):
    if enum is None:
        return num
    else:
        try:
            return socket.SocketKind(num)
        except ValueError:
            return num
def conn_to_ntuple(fd, fam, type_, laddr, raddr, status, status_map, pid=None):
    if fam in (socket.AF_INET, AF_INET6):
        if laddr:
            laddr = addr(*laddr)
        if raddr:
            raddr = addr(*raddr)
    if type_ == socket.SOCK_STREAM and fam in (AF_INET, AF_INET6):
        status = status_map.get(status, CONN_NONE)
    else:
        status = CONN_NONE
    fam = sockfam_to_enum(fam)
    type_ = socktype_to_enum(type_)
    if pid is None:
        return pconn(fd, fam, type_, laddr, raddr, status)
    else:
        return sconn(fd, fam, type_, laddr, raddr, status, pid)
def deprecated_method(replacement):
    def outer(fun):
        msg = "%s() is deprecated and will be removed; use %s() instead" % (
            fun.__name__, replacement)
        if fun.__doc__ is None:
            fun.__doc__ = msg
        @functools.wraps(fun)
        def inner(self, *args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            return getattr(self, replacement)(*args, **kwargs)
        return inner
    return outer
class _WrapNumbers:
    def __init__(self):
        self.lock = threading.Lock()
        self.cache = {}
        self.reminders = {}
        self.reminder_keys = {}
    def _add_dict(self, input_dict, name):
        assert name not in self.cache
        assert name not in self.reminders
        assert name not in self.reminder_keys
        self.cache[name] = input_dict
        self.reminders[name] = defaultdict(int)
        self.reminder_keys[name] = defaultdict(set)
    def _remove_dead_reminders(self, input_dict, name):
        old_dict = self.cache[name]
        gone_keys = set(old_dict.keys()) - set(input_dict.keys())
        for gone_key in gone_keys:
            for remkey in self.reminder_keys[name][gone_key]:
                del self.reminders[name][remkey]
            del self.reminder_keys[name][gone_key]
    def run(self, input_dict, name):
        if name not in self.cache:
            self._add_dict(input_dict, name)
            return input_dict
        self._remove_dead_reminders(input_dict, name)
        old_dict = self.cache[name]
        new_dict = {}
        for key in input_dict.keys():
            input_tuple = input_dict[key]
            try:
                old_tuple = old_dict[key]
            except KeyError:
                new_dict[key] = input_tuple
                continue
            bits = []
            for i in range(len(input_tuple)):
                input_value = input_tuple[i]
                old_value = old_tuple[i]
                remkey = (key, i)
                if input_value < old_value:
                    self.reminders[name][remkey] += old_value
                    self.reminder_keys[name][key].add(remkey)
                bits.append(input_value + self.reminders[name][remkey])
            new_dict[key] = tuple(bits)
        self.cache[name] = input_dict
        return new_dict
    def cache_clear(self, name=None):
        with self.lock:
            if name is None:
                self.cache.clear()
                self.reminders.clear()
                self.reminder_keys.clear()
            else:
                self.cache.pop(name, None)
                self.reminders.pop(name, None)
                self.reminder_keys.pop(name, None)
    def cache_info(self):
        with self.lock:
            return (self.cache, self.reminders, self.reminder_keys)
def wrap_numbers(input_dict, name):
    with _wn.lock:
        return _wn.run(input_dict, name)
_wn = _WrapNumbers()
wrap_numbers.cache_clear = _wn.cache_clear
wrap_numbers.cache_info = _wn.cache_info
def open_binary(fname, **kwargs):
    return open(fname, "rb", **kwargs)
def open_text(fname, **kwargs):
    if PY3:
        kwargs.setdefault('encoding', ENCODING)
        kwargs.setdefault('errors', ENCODING_ERRS)
    return open(fname, "rt", **kwargs)
def bytes2human(n, format="%(value).1f%(symbol)s"):
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)
def get_procfs_path():
    return sys.modules['psutil'].PROCFS_PATH
if PY3:
    def decode(s):
        return s.decode(encoding=ENCODING, errors=ENCODING_ERRS)
else:
    def decode(s):
        return s
@memoize
def term_supports_colors(file=sys.stdout):
    if os.name == 'nt':
        return True
    try:
        import curses
        assert file.isatty()
        curses.setupterm()
        assert curses.tigetnum("colors") > 0
    except Exception:
        return False
    else:
        return True
def hilite(s, color=None, bold=False):
    if not term_supports_colors():
        return s
    attr = []
    colors = dict(green='32', red='91', brown='33', yellow='93', blue='34',
                  violet='35', lightblue='36', grey='37', darkgrey='30')
    colors[None] = '29'
    try:
        color = colors[color]
    except KeyError:
        raise ValueError("invalid color %r; choose between %s" % (
            list(colors.keys())))
    attr.append(color)
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), s)
def print_color(
        s, color=None, bold=False, file=sys.stdout):
    if not term_supports_colors():
        print(s, file=file)
    elif POSIX:
        print(hilite(s, color, bold), file=file)
    else:
        import ctypes
        DEFAULT_COLOR = 7
        GetStdHandle = ctypes.windll.Kernel32.GetStdHandle
        SetConsoleTextAttribute =            ctypes.windll.Kernel32.SetConsoleTextAttribute
        colors = dict(green=2, red=4, brown=6, yellow=6)
        colors[None] = DEFAULT_COLOR
        try:
            color = colors[color]
        except KeyError:
            raise ValueError("invalid color %r; choose between %r" % (
                color, list(colors.keys())))
        if bold and color <= 7:
            color += 8
        handle_id = -12 if file is sys.stderr else -11
        GetStdHandle.restype = ctypes.c_ulong
        handle = GetStdHandle(handle_id)
        SetConsoleTextAttribute(handle, color)
        try:
            print(s, file=file)
        finally:
            SetConsoleTextAttribute(handle, DEFAULT_COLOR)
if bool(os.getenv('PSUTIL_DEBUG', 0)):
    import inspect
    def debug(msg):
        fname, lineno, func_name, lines, index = inspect.getframeinfo(
            inspect.currentframe().f_back)
        print("psutil-debug [%s:%s]> %s" % (fname, lineno, msg),
              file=sys.stderr)
else:
    def debug(msg):
        pass
