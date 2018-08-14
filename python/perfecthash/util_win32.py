#===============================================================================
# Imports
#===============================================================================
import ctypes
from ctypes import wintypes

#===============================================================================
# Globals/Aliases
#===============================================================================

DWORD = ctypes.c_ulong
SIZE_T = ctypes.c_size_t

kernel32 = ctypes.windll.kernel32
kernel32.GetCurrentProcess.argtypes = []
kernel32.GetCurrentProcess.restype = wintypes.HANDLE

psapi = ctypes.windll.psapi
#===============================================================================
# Classes
#===============================================================================
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD),
        ("PageFaultCount", DWORD),
        ("PeakWorkingSetSize", SIZE_T),
        ("WorkingSetSize", SIZE_T),
        ("QuotaPeakPagedPoolUsage", SIZE_T),
        ("QuotaPagedPoolUsage", SIZE_T),
        ("QuotaPeakNonPagedPoolUsage", SIZE_T),
        ("QuotaNonPagedPoolUsage", SIZE_T),
        ("PagefileUsage", SIZE_T),
        ("PeakPagefileUsage", SIZE_T),
    ]

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", DWORD),
        ("PageFaultCount", DWORD),
        ("PeakWorkingSetSize", SIZE_T),
        ("WorkingSetSize", SIZE_T),
        ("QuotaPeakPagedPoolUsage", SIZE_T),
        ("QuotaPagedPoolUsage", SIZE_T),
        ("QuotaPeakNonPagedPoolUsage", SIZE_T),
        ("QuotaNonPagedPoolUsage", SIZE_T),
        ("PagefileUsage", SIZE_T),
        ("PeakPagefileUsage", SIZE_T),
        ("PrivateUsage", SIZE_T),
    ]

psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
    wintypes.DWORD,
]

psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

#===============================================================================
# Helpers
#===============================================================================
def GetCurrentProcess():
    return kernel32.GetCurrentProcess()

def GetProcessMemoryInfo(handle=None):
    if not handle:
        handle = GetCurrentProcess()
    psmemCounters = PROCESS_MEMORY_COUNTERS_EX()
    cb = DWORD(ctypes.sizeof(psmemCounters))

    b = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(psmemCounters),
        cb
    )

    if not b:
        psmemCounters = PROCESS_MEMORY_COUNTERS()
        cb = DWORD(ctypes.sizeof(psmemCounters))
        b = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(psmemCounters),
            cb,
        )
        if not b:
            raise ctypes.WinError()
    d = {}
    for k, t in psmemCounters._fields_:
        d[k] = getattr(psmemCounters, k)
    return d

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
