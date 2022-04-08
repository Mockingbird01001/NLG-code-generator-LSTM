import unittest, gc
from ctypes import *
from ctypes.wintypes import *
class PROCESS_MEMORY_COUNTERS(Structure):
    _fields_ = [("cb", DWORD),
                ("PageFaultCount", DWORD),
                ("PeakWorkingSetSize", c_size_t),
                ("WorkingSetSize", c_size_t),
                ("QuotaPeakPagedPoolUsage", c_size_t),
                ("QuotaPagedPoolUsage", c_size_t),
                ("QuotaPeakNonPagedPoolUsage", c_size_t),
                ("QuotaNonPagedPoolUsage", c_size_t),
                ("PagefileUsage", c_size_t),
                ("PeakPagefileUsage", c_size_t)]
    def __init__(self):
        self.cb = sizeof(self)
    def dump(self):
        for n, _ in self._fields_[2:]:
            print(n, getattr(self, n)/1e6)
try:
    windll.psapi.GetProcessMemoryInfo.argtypes = (HANDLE, POINTER(PROCESS_MEMORY_COUNTERS), DWORD)
except WindowsError:
    def find_memleak(func, loops=None):
        return 0
else:
    def wss():
        pmi = PROCESS_MEMORY_COUNTERS()
        if not windll.psapi.GetProcessMemoryInfo(-1, byref(pmi), sizeof(pmi)):
            raise WinError()
        return pmi.WorkingSetSize
    LOOPS = 10, 1000
    def find_memleak(func, loops=LOOPS):
        for j in range(loops[0]):
            for k in range(loops[1]):
                func()
        gc.collect(); gc.collect(); gc.collect()
        bytes = wss()
        for j in range(loops[0]):
            for k in range(loops[1]):
                func()
        gc.collect(); gc.collect(); gc.collect()
        result = wss() - bytes
        return max(result, 0)
