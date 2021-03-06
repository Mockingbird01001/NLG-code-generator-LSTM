
import functools
import glob
import os
import re
import subprocess
import sys
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_aix as cext
from . import _psutil_posix as cext_posix
from ._common import AccessDenied
from ._common import conn_to_ntuple
from ._common import get_procfs_path
from ._common import memoize_when_activated
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import NoSuchProcess
from ._common import usage_percent
from ._common import ZombieProcess
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import PY3
__extra__all__ = ["PROCFS_PATH"]
HAS_THREADS = hasattr(cext, "proc_threads")
HAS_NET_IO_COUNTERS = hasattr(cext, "net_io_counters")
HAS_PROC_IO_COUNTERS = hasattr(cext, "proc_io_counters")
PAGE_SIZE = cext_posix.getpagesize()
AF_LINK = cext_posix.AF_LINK
PROC_STATUSES = {
    cext.SIDL: _common.STATUS_IDLE,
    cext.SZOMB: _common.STATUS_ZOMBIE,
    cext.SACTIVE: _common.STATUS_RUNNING,
    cext.SSWAP: _common.STATUS_RUNNING,
    cext.SSTOP: _common.STATUS_STOPPED,
}
TCP_STATUSES = {
    cext.TCPS_ESTABLISHED: _common.CONN_ESTABLISHED,
    cext.TCPS_SYN_SENT: _common.CONN_SYN_SENT,
    cext.TCPS_SYN_RCVD: _common.CONN_SYN_RECV,
    cext.TCPS_FIN_WAIT_1: _common.CONN_FIN_WAIT1,
    cext.TCPS_FIN_WAIT_2: _common.CONN_FIN_WAIT2,
    cext.TCPS_TIME_WAIT: _common.CONN_TIME_WAIT,
    cext.TCPS_CLOSED: _common.CONN_CLOSE,
    cext.TCPS_CLOSE_WAIT: _common.CONN_CLOSE_WAIT,
    cext.TCPS_LAST_ACK: _common.CONN_LAST_ACK,
    cext.TCPS_LISTEN: _common.CONN_LISTEN,
    cext.TCPS_CLOSING: _common.CONN_CLOSING,
    cext.PSUTIL_CONN_NONE: _common.CONN_NONE,
}
proc_info_map = dict(
    ppid=0,
    rss=1,
    vms=2,
    create_time=3,
    nice=4,
    num_threads=5,
    status=6,
    ttynr=7)
pmem = namedtuple('pmem', ['rss', 'vms'])
pfullmem = pmem
scputimes = namedtuple('scputimes', ['user', 'system', 'idle', 'iowait'])
svmem = namedtuple('svmem', ['total', 'available', 'percent', 'used', 'free'])
def virtual_memory():
    total, avail, free, pinned, inuse = cext.virtual_mem()
    percent = usage_percent((total - avail), total, round_=1)
    return svmem(total, avail, percent, inuse, free)
def swap_memory():
    total, free, sin, sout = cext.swap_mem()
    used = total - free
    percent = usage_percent(used, total, round_=1)
    return _common.sswap(total, used, free, percent, sin, sout)
def cpu_times():
    ret = cext.per_cpu_times()
    return scputimes(*[sum(x) for x in zip(*ret)])
def per_cpu_times():
    ret = cext.per_cpu_times()
    return [scputimes(*x) for x in ret]
def cpu_count_logical():
    try:
        return os.sysconf("SC_NPROCESSORS_ONLN")
    except ValueError:
        return None
def cpu_count_physical():
    cmd = "lsdev -Cc processor"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if PY3:
        stdout, stderr = [x.decode(sys.stdout.encoding)
                          for x in (stdout, stderr)]
    if p.returncode != 0:
        raise RuntimeError("%r command error\n%s" % (cmd, stderr))
    processors = stdout.strip().splitlines()
    return len(processors) or None
def cpu_stats():
    ctx_switches, interrupts, soft_interrupts, syscalls = cext.cpu_stats()
    return _common.scpustats(
        ctx_switches, interrupts, soft_interrupts, syscalls)
disk_io_counters = cext.disk_io_counters
disk_usage = _psposix.disk_usage
def disk_partitions(all=False):
    retlist = []
    partitions = cext.disk_partitions()
    for partition in partitions:
        device, mountpoint, fstype, opts = partition
        if device == 'none':
            device = ''
        if not all:
            if not disk_usage(mountpoint).total:
                continue
        maxfile = maxpath = None
        ntuple = _common.sdiskpart(device, mountpoint, fstype, opts,
                                   maxfile, maxpath)
        retlist.append(ntuple)
    return retlist
net_if_addrs = cext_posix.net_if_addrs
if HAS_NET_IO_COUNTERS:
    net_io_counters = cext.net_io_counters
def net_connections(kind, _pid=-1):
    cmap = _common.conn_tmap
    if kind not in cmap:
        raise ValueError("invalid %r kind argument; choose between %s"
                         % (kind, ', '.join([repr(x) for x in cmap])))
    families, types = _common.conn_tmap[kind]
    rawlist = cext.net_connections(_pid)
    ret = []
    for item in rawlist:
        fd, fam, type_, laddr, raddr, status, pid = item
        if fam not in families:
            continue
        if type_ not in types:
            continue
        nt = conn_to_ntuple(fd, fam, type_, laddr, raddr, status,
                            TCP_STATUSES, pid=pid if _pid == -1 else None)
        ret.append(nt)
    return ret
def net_if_stats():
    duplex_map = {"Full": NIC_DUPLEX_FULL,
                  "Half": NIC_DUPLEX_HALF}
    names = set([x[0] for x in net_if_addrs()])
    ret = {}
    for name in names:
        isup, mtu = cext.net_if_stats(name)
        duplex = ""
        speed = 0
        p = subprocess.Popen(["/usr/bin/entstat", "-d", name],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if PY3:
            stdout, stderr = [x.decode(sys.stdout.encoding)
                              for x in (stdout, stderr)]
        if p.returncode == 0:
            re_result = re.search(
                r"Running: (\d+) Mbps.*?(\w+) Duplex", stdout)
            if re_result is not None:
                speed = int(re_result.group(1))
                duplex = re_result.group(2)
        duplex = duplex_map.get(duplex, NIC_DUPLEX_UNKNOWN)
        ret[name] = _common.snicstats(isup, duplex, speed, mtu)
    return ret
def boot_time():
    return cext.boot_time()
def users():
    retlist = []
    rawlist = cext.users()
    localhost = (':0.0', ':0')
    for item in rawlist:
        user, tty, hostname, tstamp, user_process, pid = item
        if not user_process:
            continue
        if hostname in localhost:
            hostname = 'localhost'
        nt = _common.suser(user, tty, hostname, tstamp, pid)
        retlist.append(nt)
    return retlist
def pids():
    return [int(x) for x in os.listdir(get_procfs_path()) if x.isdigit()]
def pid_exists(pid):
    return os.path.exists(os.path.join(get_procfs_path(), str(pid), "psinfo"))
def wrap_exceptions(fun):
    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        try:
            return fun(self, *args, **kwargs)
        except (FileNotFoundError, ProcessLookupError):
            if not pid_exists(self.pid):
                raise NoSuchProcess(self.pid, self._name)
            else:
                raise ZombieProcess(self.pid, self._name, self._ppid)
        except PermissionError:
            raise AccessDenied(self.pid, self._name)
    return wrapper
class Process(object):
    __slots__ = ["pid", "_name", "_ppid", "_procfs_path", "_cache"]
    def __init__(self, pid):
        self.pid = pid
        self._name = None
        self._ppid = None
        self._procfs_path = get_procfs_path()
    def oneshot_enter(self):
        self._proc_basic_info.cache_activate(self)
        self._proc_cred.cache_activate(self)
    def oneshot_exit(self):
        self._proc_basic_info.cache_deactivate(self)
        self._proc_cred.cache_deactivate(self)
    @wrap_exceptions
    @memoize_when_activated
    def _proc_basic_info(self):
        return cext.proc_basic_info(self.pid, self._procfs_path)
    @wrap_exceptions
    @memoize_when_activated
    def _proc_cred(self):
        return cext.proc_cred(self.pid, self._procfs_path)
    @wrap_exceptions
    def name(self):
        if self.pid == 0:
            return "swapper"
        return cext.proc_name(self.pid, self._procfs_path).rstrip("\x00")
    @wrap_exceptions
    def exe(self):
        cmdline = self.cmdline()
        if not cmdline:
            return ''
        exe = cmdline[0]
        if os.path.sep in exe:
            if not os.path.isabs(exe):
                exe = os.path.abspath(os.path.join(self.cwd(), exe))
            if (os.path.isabs(exe) and
                    os.path.isfile(exe) and
                    os.access(exe, os.X_OK)):
                return exe
            exe = os.path.basename(exe)
        for path in os.environ["PATH"].split(":"):
            possible_exe = os.path.abspath(os.path.join(path, exe))
            if (os.path.isfile(possible_exe) and
                    os.access(possible_exe, os.X_OK)):
                return possible_exe
        return ''
    @wrap_exceptions
    def cmdline(self):
        return cext.proc_args(self.pid)
    @wrap_exceptions
    def environ(self):
        return cext.proc_environ(self.pid)
    @wrap_exceptions
    def create_time(self):
        return self._proc_basic_info()[proc_info_map['create_time']]
    @wrap_exceptions
    def num_threads(self):
        return self._proc_basic_info()[proc_info_map['num_threads']]
    if HAS_THREADS:
        @wrap_exceptions
        def threads(self):
            rawlist = cext.proc_threads(self.pid)
            retlist = []
            for thread_id, utime, stime in rawlist:
                ntuple = _common.pthread(thread_id, utime, stime)
                retlist.append(ntuple)
            if not retlist:
                os.stat('%s/%s' % (self._procfs_path, self.pid))
            return retlist
    @wrap_exceptions
    def connections(self, kind='inet'):
        ret = net_connections(kind, _pid=self.pid)
        if not ret:
            os.stat('%s/%s' % (self._procfs_path, self.pid))
        return ret
    @wrap_exceptions
    def nice_get(self):
        return cext_posix.getpriority(self.pid)
    @wrap_exceptions
    def nice_set(self, value):
        return cext_posix.setpriority(self.pid, value)
    @wrap_exceptions
    def ppid(self):
        self._ppid = self._proc_basic_info()[proc_info_map['ppid']]
        return self._ppid
    @wrap_exceptions
    def uids(self):
        real, effective, saved, _, _, _ = self._proc_cred()
        return _common.puids(real, effective, saved)
    @wrap_exceptions
    def gids(self):
        _, _, _, real, effective, saved = self._proc_cred()
        return _common.puids(real, effective, saved)
    @wrap_exceptions
    def cpu_times(self):
        cpu_times = cext.proc_cpu_times(self.pid, self._procfs_path)
        return _common.pcputimes(*cpu_times)
    @wrap_exceptions
    def terminal(self):
        ttydev = self._proc_basic_info()[proc_info_map['ttynr']]
        ttydev = (((ttydev & 0x0000FFFF00000000) >> 16) | (ttydev & 0xFFFF))
        for dev in glob.glob("/dev/**/*"):
            if os.stat(dev).st_rdev == ttydev:
                return dev
        return None
    @wrap_exceptions
    def cwd(self):
        procfs_path = self._procfs_path
        try:
            result = os.readlink("%s/%s/cwd" % (procfs_path, self.pid))
            return result.rstrip('/')
        except FileNotFoundError:
            os.stat("%s/%s" % (procfs_path, self.pid))
            return None
    @wrap_exceptions
    def memory_info(self):
        ret = self._proc_basic_info()
        rss = ret[proc_info_map['rss']] * 1024
        vms = ret[proc_info_map['vms']] * 1024
        return pmem(rss, vms)
    memory_full_info = memory_info
    @wrap_exceptions
    def status(self):
        code = self._proc_basic_info()[proc_info_map['status']]
        return PROC_STATUSES.get(code, '?')
    def open_files(self):
        p = subprocess.Popen(["/usr/bin/procfiles", "-n", str(self.pid)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if PY3:
            stdout, stderr = [x.decode(sys.stdout.encoding)
                              for x in (stdout, stderr)]
        if "no such process" in stderr.lower():
            raise NoSuchProcess(self.pid, self._name)
        procfiles = re.findall(r"(\d+): S_IFREG.*\s*.*name:(.*)\n", stdout)
        retlist = []
        for fd, path in procfiles:
            path = path.strip()
            if path.startswith("//"):
                path = path[1:]
            if path.lower() == "cannot be retrieved":
                continue
            retlist.append(_common.popenfile(path, int(fd)))
        return retlist
    @wrap_exceptions
    def num_fds(self):
        if self.pid == 0:
            return 0
        return len(os.listdir("%s/%s/fd" % (self._procfs_path, self.pid)))
    @wrap_exceptions
    def num_ctx_switches(self):
        return _common.pctxsw(
            *cext.proc_num_ctx_switches(self.pid))
    @wrap_exceptions
    def wait(self, timeout=None):
        return _psposix.wait_pid(self.pid, timeout, self._name)
    if HAS_PROC_IO_COUNTERS:
        @wrap_exceptions
        def io_counters(self):
            try:
                rc, wc, rb, wb = cext.proc_io_counters(self.pid)
            except OSError:
                if not pid_exists(self.pid):
                    raise NoSuchProcess(self.pid, self._name)
                raise
            return _common.pio(rc, wc, rb, wb)
