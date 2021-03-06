
import contextlib
import errno
import functools
import os
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_osx as cext
from . import _psutil_posix as cext_posix
from ._common import AccessDenied
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import isfile_strict
from ._common import memoize_when_activated
from ._common import NoSuchProcess
from ._common import parse_environ_block
from ._common import usage_percent
from ._common import ZombieProcess
from ._compat import PermissionError
from ._compat import ProcessLookupError
__extra__all__ = []
PAGESIZE = cext_posix.getpagesize()
AF_LINK = cext_posix.AF_LINK
TCP_STATUSES = {
    cext.TCPS_ESTABLISHED: _common.CONN_ESTABLISHED,
    cext.TCPS_SYN_SENT: _common.CONN_SYN_SENT,
    cext.TCPS_SYN_RECEIVED: _common.CONN_SYN_RECV,
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
PROC_STATUSES = {
    cext.SIDL: _common.STATUS_IDLE,
    cext.SRUN: _common.STATUS_RUNNING,
    cext.SSLEEP: _common.STATUS_SLEEPING,
    cext.SSTOP: _common.STATUS_STOPPED,
    cext.SZOMB: _common.STATUS_ZOMBIE,
}
kinfo_proc_map = dict(
    ppid=0,
    ruid=1,
    euid=2,
    suid=3,
    rgid=4,
    egid=5,
    sgid=6,
    ttynr=7,
    ctime=8,
    status=9,
    name=10,
)
pidtaskinfo_map = dict(
    cpuutime=0,
    cpustime=1,
    rss=2,
    vms=3,
    pfaults=4,
    pageins=5,
    numthreads=6,
    volctxsw=7,
)
scputimes = namedtuple('scputimes', ['user', 'nice', 'system', 'idle'])
svmem = namedtuple(
    'svmem', ['total', 'available', 'percent', 'used', 'free',
              'active', 'inactive', 'wired'])
pmem = namedtuple('pmem', ['rss', 'vms', 'pfaults', 'pageins'])
pfullmem = namedtuple('pfullmem', pmem._fields + ('uss', ))
def virtual_memory():
    total, active, inactive, wired, free, speculative = cext.virtual_mem()
    avail = inactive + free
    used = active + wired
    free -= speculative
    percent = usage_percent((total - avail), total, round_=1)
    return svmem(total, avail, percent, used, free,
                 active, inactive, wired)
def swap_memory():
    total, used, free, sin, sout = cext.swap_mem()
    percent = usage_percent(used, total, round_=1)
    return _common.sswap(total, used, free, percent, sin, sout)
def cpu_times():
    user, nice, system, idle = cext.cpu_times()
    return scputimes(user, nice, system, idle)
def per_cpu_times():
    ret = []
    for cpu_t in cext.per_cpu_times():
        user, nice, system, idle = cpu_t
        item = scputimes(user, nice, system, idle)
        ret.append(item)
    return ret
def cpu_count_logical():
    return cext.cpu_count_logical()
def cpu_count_physical():
    return cext.cpu_count_phys()
def cpu_stats():
    ctx_switches, interrupts, soft_interrupts, syscalls, traps =        cext.cpu_stats()
    return _common.scpustats(
        ctx_switches, interrupts, soft_interrupts, syscalls)
def cpu_freq():
    curr, min_, max_ = cext.cpu_freq()
    return [_common.scpufreq(curr, min_, max_)]
disk_usage = _psposix.disk_usage
disk_io_counters = cext.disk_io_counters
def disk_partitions(all=False):
    retlist = []
    partitions = cext.disk_partitions()
    for partition in partitions:
        device, mountpoint, fstype, opts = partition
        if device == 'none':
            device = ''
        if not all:
            if not os.path.isabs(device) or not os.path.exists(device):
                continue
        maxfile = maxpath = None
        ntuple = _common.sdiskpart(device, mountpoint, fstype, opts,
                                   maxfile, maxpath)
        retlist.append(ntuple)
    return retlist
def sensors_battery():
    try:
        percent, minsleft, power_plugged = cext.sensors_battery()
    except NotImplementedError:
        return None
    power_plugged = power_plugged == 1
    if power_plugged:
        secsleft = _common.POWER_TIME_UNLIMITED
    elif minsleft == -1:
        secsleft = _common.POWER_TIME_UNKNOWN
    else:
        secsleft = minsleft * 60
    return _common.sbattery(percent, secsleft, power_plugged)
net_io_counters = cext.net_io_counters
net_if_addrs = cext_posix.net_if_addrs
def net_connections(kind='inet'):
    ret = []
    for pid in pids():
        try:
            cons = Process(pid).connections(kind)
        except NoSuchProcess:
            continue
        else:
            if cons:
                for c in cons:
                    c = list(c) + [pid]
                    ret.append(_common.sconn(*c))
    return ret
def net_if_stats():
    names = net_io_counters().keys()
    ret = {}
    for name in names:
        try:
            mtu = cext_posix.net_if_mtu(name)
            isup = cext_posix.net_if_is_running(name)
            duplex, speed = cext_posix.net_if_duplex_speed(name)
        except OSError as err:
            if err.errno != errno.ENODEV:
                raise
        else:
            if hasattr(_common, 'NicDuplex'):
                duplex = _common.NicDuplex(duplex)
            ret[name] = _common.snicstats(isup, duplex, speed, mtu)
    return ret
def boot_time():
    return cext.boot_time()
def users():
    retlist = []
    rawlist = cext.users()
    for item in rawlist:
        user, tty, hostname, tstamp, pid = item
        if tty == '~':
            continue
        if not tstamp:
            continue
        nt = _common.suser(user, tty or None, hostname or None, tstamp, pid)
        retlist.append(nt)
    return retlist
def pids():
    ls = cext.pids()
    if 0 not in ls:
        try:
            Process(0).create_time()
            ls.insert(0, 0)
        except NoSuchProcess:
            pass
        except AccessDenied:
            ls.insert(0, 0)
    return ls
pid_exists = _psposix.pid_exists
def is_zombie(pid):
    try:
        st = cext.proc_kinfo_oneshot(pid)[kinfo_proc_map['status']]
        return st == cext.SZOMB
    except Exception:
        return False
def wrap_exceptions(fun):
    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        try:
            return fun(self, *args, **kwargs)
        except ProcessLookupError:
            if is_zombie(self.pid):
                raise ZombieProcess(self.pid, self._name, self._ppid)
            else:
                raise NoSuchProcess(self.pid, self._name)
        except PermissionError:
            raise AccessDenied(self.pid, self._name)
        except cext.ZombieProcessError:
            raise ZombieProcess(self.pid, self._name, self._ppid)
    return wrapper
@contextlib.contextmanager
def catch_zombie(proc):
    try:
        yield
    except (OSError, RuntimeError) as err:
        if isinstance(err, RuntimeError) or err.errno == errno.ESRCH:
            try:
                status = proc.status()
            except NoSuchProcess:
                raise err
            else:
                if status == _common.STATUS_ZOMBIE:
                    raise ZombieProcess(proc.pid, proc._name, proc._ppid)
                else:
                    raise AccessDenied(proc.pid, proc._name)
        else:
            raise
class Process(object):
    __slots__ = ["pid", "_name", "_ppid", "_cache"]
    def __init__(self, pid):
        self.pid = pid
        self._name = None
        self._ppid = None
    @wrap_exceptions
    @memoize_when_activated
    def _get_kinfo_proc(self):
        ret = cext.proc_kinfo_oneshot(self.pid)
        assert len(ret) == len(kinfo_proc_map)
        return ret
    @wrap_exceptions
    @memoize_when_activated
    def _get_pidtaskinfo(self):
        with catch_zombie(self):
            ret = cext.proc_pidtaskinfo_oneshot(self.pid)
        assert len(ret) == len(pidtaskinfo_map)
        return ret
    def oneshot_enter(self):
        self._get_kinfo_proc.cache_activate(self)
        self._get_pidtaskinfo.cache_activate(self)
    def oneshot_exit(self):
        self._get_kinfo_proc.cache_deactivate(self)
        self._get_pidtaskinfo.cache_deactivate(self)
    @wrap_exceptions
    def name(self):
        name = self._get_kinfo_proc()[kinfo_proc_map['name']]
        return name if name is not None else cext.proc_name(self.pid)
    @wrap_exceptions
    def exe(self):
        with catch_zombie(self):
            return cext.proc_exe(self.pid)
    @wrap_exceptions
    def cmdline(self):
        with catch_zombie(self):
            return cext.proc_cmdline(self.pid)
    @wrap_exceptions
    def environ(self):
        with catch_zombie(self):
            return parse_environ_block(cext.proc_environ(self.pid))
    @wrap_exceptions
    def ppid(self):
        self._ppid = self._get_kinfo_proc()[kinfo_proc_map['ppid']]
        return self._ppid
    @wrap_exceptions
    def cwd(self):
        with catch_zombie(self):
            return cext.proc_cwd(self.pid)
    @wrap_exceptions
    def uids(self):
        rawtuple = self._get_kinfo_proc()
        return _common.puids(
            rawtuple[kinfo_proc_map['ruid']],
            rawtuple[kinfo_proc_map['euid']],
            rawtuple[kinfo_proc_map['suid']])
    @wrap_exceptions
    def gids(self):
        rawtuple = self._get_kinfo_proc()
        return _common.puids(
            rawtuple[kinfo_proc_map['rgid']],
            rawtuple[kinfo_proc_map['egid']],
            rawtuple[kinfo_proc_map['sgid']])
    @wrap_exceptions
    def terminal(self):
        tty_nr = self._get_kinfo_proc()[kinfo_proc_map['ttynr']]
        tmap = _psposix.get_terminal_map()
        try:
            return tmap[tty_nr]
        except KeyError:
            return None
    @wrap_exceptions
    def memory_info(self):
        rawtuple = self._get_pidtaskinfo()
        return pmem(
            rawtuple[pidtaskinfo_map['rss']],
            rawtuple[pidtaskinfo_map['vms']],
            rawtuple[pidtaskinfo_map['pfaults']],
            rawtuple[pidtaskinfo_map['pageins']],
        )
    @wrap_exceptions
    def memory_full_info(self):
        basic_mem = self.memory_info()
        uss = cext.proc_memory_uss(self.pid)
        return pfullmem(*basic_mem + (uss, ))
    @wrap_exceptions
    def cpu_times(self):
        rawtuple = self._get_pidtaskinfo()
        return _common.pcputimes(
            rawtuple[pidtaskinfo_map['cpuutime']],
            rawtuple[pidtaskinfo_map['cpustime']],
            0.0, 0.0)
    @wrap_exceptions
    def create_time(self):
        return self._get_kinfo_proc()[kinfo_proc_map['ctime']]
    @wrap_exceptions
    def num_ctx_switches(self):
        vol = self._get_pidtaskinfo()[pidtaskinfo_map['volctxsw']]
        return _common.pctxsw(vol, 0)
    @wrap_exceptions
    def num_threads(self):
        return self._get_pidtaskinfo()[pidtaskinfo_map['numthreads']]
    @wrap_exceptions
    def open_files(self):
        if self.pid == 0:
            return []
        files = []
        with catch_zombie(self):
            rawlist = cext.proc_open_files(self.pid)
        for path, fd in rawlist:
            if isfile_strict(path):
                ntuple = _common.popenfile(path, fd)
                files.append(ntuple)
        return files
    @wrap_exceptions
    def connections(self, kind='inet'):
        if kind not in conn_tmap:
            raise ValueError("invalid %r kind argument; choose between %s"
                             % (kind, ', '.join([repr(x) for x in conn_tmap])))
        families, types = conn_tmap[kind]
        with catch_zombie(self):
            rawlist = cext.proc_connections(self.pid, families, types)
        ret = []
        for item in rawlist:
            fd, fam, type, laddr, raddr, status = item
            nt = conn_to_ntuple(fd, fam, type, laddr, raddr, status,
                                TCP_STATUSES)
            ret.append(nt)
        return ret
    @wrap_exceptions
    def num_fds(self):
        if self.pid == 0:
            return 0
        with catch_zombie(self):
            return cext.proc_num_fds(self.pid)
    @wrap_exceptions
    def wait(self, timeout=None):
        return _psposix.wait_pid(self.pid, timeout, self._name)
    @wrap_exceptions
    def nice_get(self):
        with catch_zombie(self):
            return cext_posix.getpriority(self.pid)
    @wrap_exceptions
    def nice_set(self, value):
        with catch_zombie(self):
            return cext_posix.setpriority(self.pid, value)
    @wrap_exceptions
    def status(self):
        code = self._get_kinfo_proc()[kinfo_proc_map['status']]
        return PROC_STATUSES.get(code, '?')
    @wrap_exceptions
    def threads(self):
        rawlist = cext.proc_threads(self.pid)
        retlist = []
        for thread_id, utime, stime in rawlist:
            ntuple = _common.pthread(thread_id, utime, stime)
            retlist.append(ntuple)
        return retlist
