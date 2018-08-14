#===============================================================================
# Imports
#===============================================================================
import sys
import ctypes

from ctypes import *
from ctypes.wintypes import *

from .util import Constant

#===============================================================================
# Globals/Aliases
#===============================================================================
CHAR = c_char
UCHAR = c_ubyte
BYTE = c_ubyte
PBYTE = POINTER(BYTE)
BOOL = c_bool
PBOOL = POINTER(BOOL)
PBOOLEAN = POINTER(BOOL)
VOID = None
CLONG = c_long
SIZE_T = c_size_t
SSIZE_T = c_int64
ULONG_PTR = SIZE_T
LONG_PTR = SIZE_T
DWORD_PTR = SIZE_T
PSHORT = POINTER(SHORT)
PUSHORT = POINTER(USHORT)
PLONG = POINTER(LONG)
PULONG = POINTER(ULONG)
PHANDLE = POINTER(HANDLE)
LONGLONG = c_int64
PLONGLONG = POINTER(LONGLONG)
ULONGLONG = c_uint64
DWORDLONG = c_uint64
PULONGLONG = POINTER(ULONGLONG)
PLARGE_INTEGER = POINTER(LARGE_INTEGER)
PULARGE_INTEGER = POINTER(ULARGE_INTEGER)
PVOID = c_void_p
PPVOID = POINTER(PVOID)
PISID = PVOID
PSZ = c_char_p
PCSZ = c_char_p
PSTR = c_char_p
PCSTR = c_char_p
PCHAR = c_char_p
PWCHAR = c_wchar_p
PWSTR = c_wchar_p
PCWSTR = c_wchar_p
PDWORD = POINTER(DWORD)
PFILETIME = POINTER(FILETIME)
SRWLOCK = PVOID
TP_VERSION = DWORD
PTP_WORK = PVOID
PTP_POOL = PVOID
PTP_TIMER = PVOID
PTP_CLEANUP_GROUP = PVOID
PTP_CLEANUP_GROUP_CANCEL_CALLBACK = PVOID
TP_CALLBACK_PRIORITY = DWORD
PACTIVATION_CONTEXT = PVOID
SRWLOCK = PVOID
INFINITE = 0xffffffff
WAIT_FAILED = 0xffffffff
WAIT_OBJECT_0 = 0
WAIT_ABANDONED = 0x80
WAIT_TIMEOUT = 0x102
HRESULT = ULONG

#===============================================================================
# Enums
#===============================================================================
TP_CALLBACK_PRIORITY_HIGH = 0
TP_CALLBACK_PRIORITY_NORMAL = 1
TP_CALLBACK_PRIORITY_LOW = 2
TP_CALLBACK_PRIORITY_INVALID = 3
TP_CALLBACK_PRIORITY_COUNT = TP_CALLBACK_PRIORITY_INVALID

FileBasicInfo = 0
FileStandardInfo = 1
FileNameInfo = 2
FileStreamInfo = 7
FileCompressionInfo = 8
FileAttributeTagInfo = 9
FileIdBothDirectoryInfo = 0xa
FileIdBothDirectoryRestartInfo = 0xb
FileRemoteProtocolInfo = 0xd
FileFullDirectoryInfo = 0xe
FileFullDirectoryRestartInfo = 0xf
FileStorageInfo = 0x10
FileAlignmentInfo = 0x11
FileIdInfo = 0x12
FileIdExtdDirectoryInfo = 0x13
FileIdExtdDirectoryRestartInfo = 0x14

#===============================================================================
# Classes/Structures
#===============================================================================

class Structure(ctypes.Structure):
    _exclude = ('Unused', 'Padding')

    def _field_names(self):
        return (f[0] for f in self._fields_)

    def _to_dict(self):
        return {
            key: getattr(self, key)
                for key in self._field_names()
                    if not key.startswith(self._exclude)
        }

    def __repr__(self):
        q = lambda v: (v or '') if (not v or isinstance(v, int)) else '"%s"' % v
        return "<%s %s>" % (
            self.__class__.__name__,
            ', '.join(
                '%s=%s' % (k, q(v))
                    for (k, v) in (
                        (k, getattr(self, k))
                            for k in self._field_names()
                                if not k.startswith(self._exclude)
                    )
                )
        )

    @classmethod
    def _get_numpy_dtype(cls):
        from .util import ctypes_to_numpy
        import numpy as np

        try:
            type_map = ctypes_to_numpy()
            return np.dtype([
                (f[0], type_map[f[1]])
                    for f in cls._fields_
                        if f[1] in type_map
            ], align=True)
        except KeyError:
            pass

class _ULARGE_INTEGER_INNER(Structure):
    _fields_ = [
        ('LowPart', ULONG),
        ('HighPart', ULONG),
    ]

class _ULARGE_INTEGER(Union):
    _fields_ = [
        ('u', _ULARGE_INTEGER_INNER),
        ('QuadPart', ULONGLONG),
    ]
_PULARGE_INTEGER = POINTER(_ULARGE_INTEGER)

class _LARGE_INTEGER_INNER(Structure):
    _fields_ = [
        ('LowPart', LONG),
        ('HighPart', LONG),
    ]

class _LARGE_INTEGER(Union):
    _fields_ = [
        ('u', _LARGE_INTEGER_INNER),
        ('QuadPart', LONGLONG),
    ]
_PLARGE_INTEGER = POINTER(_LARGE_INTEGER)

class SYSTEMTIME(Structure):
    _fields_ = [
        ('wYear', WORD),
        ('wMonth', WORD),
        ('wDayOfWeek', WORD),
        ('wDay', WORD),
        ('wHour', WORD),
        ('wMinute', WORD),
        ('wSecond', WORD),
        ('wMillisecond', WORD),
    ]
PSYSTEMTIME = POINTER(SYSTEMTIME)

class PROCESSOR_NUMBER(Structure):
    _fields_ = [
        ('Group', WORD),
        ('Number', BYTE),
        ('Reserved', BYTE),
    ]
PPROCESSOR_NUMBER = POINTER(PROCESSOR_NUMBER)

class GUID(Structure):
    _fields_ = [
        ('Data1',   LONG),
        ('Data2',   SHORT),
        ('Data3',   SHORT),
        ('Data4',   BYTE * 8),
    ]

class FILE_STANDARD_INFO(Structure):
    _fields_ = [
        ('AllocationSize', LARGE_INTEGER),
        ('EndOfFile', LARGE_INTEGER),
        ('NumberOfLinks', DWORD),
        ('DeletePending', BOOLEAN),
        ('Directory', BOOLEAN),
    ]

class LIST_ENTRY(Structure):
    _fields_ = [
        ('Flink', PVOID),
        ('Blink', PVOID),
    ]
PLIST_ENTRY = POINTER(LIST_ENTRY)

class _SLIST_HEADER_INNER1(Structure):
    _fields_ = [
        ('Alignment', ULONGLONG),
        ('Region', ULONGLONG),
    ]

class _SLIST_HEADER_INNER2(Structure):
    _fields_ = [
        ('Depth', ULONGLONG, 16),
        ('Sequence', ULONGLONG, 48),
        ('Reserved', ULONGLONG, 4),
        ('NextEntry', ULONGLONG, 60),
    ]

class SLIST_HEADER(Union):
    _fields_ = [
        ('u1', _SLIST_HEADER_INNER1),
        ('u2', _SLIST_HEADER_INNER2),
    ]

class SLIST_ENTRY(Structure):
    pass
PSLIST_ENTRY = POINTER(SLIST_ENTRY)

SLIST_ENTRY._fields_ = [
    ('Next', PSLIST_ENTRY),
]

class CRITICAL_SECTION_DEBUG(Structure):
    _fields_ = [
        ('Type', WORD),
        ('CreatorBackTraceIndex', WORD),
        ('CriticalSection', PVOID),
        ('ProcessLocksList', LIST_ENTRY),
        ('EntryCount', DWORD),
        ('ContentionCount', DWORD),
        ('Flags', DWORD),
        ('CreatorBackTraceIndexHigh', WORD),
        ('SpareWORD', WORD),
    ]
PCRITICAL_SECTION_DEBUG = POINTER(CRITICAL_SECTION_DEBUG)

class CRITICAL_SECTION(Structure):
    _fields_ = [
        ('DebugInfo', PCRITICAL_SECTION_DEBUG),
        ('LockCount', LONG),
        ('RecursionCount', LONG),
        ('OwningThread', HANDLE),
        ('LockSemaphore', HANDLE),
        ('SpinCount', ULONG_PTR),
    ]
PCRITICAL_SECTION = POINTER(CRITICAL_SECTION)

class UNICODE_STRING(Structure):
    _fields_ = [
        ('Length', USHORT),
        ('MaximumLength', USHORT),
        ('Buffer', PWSTR),
    ]
    def __str__(self):
        return self.Buffer if self.Length > 0 else ''
    def __repr__(self):
        return repr(self.Buffer) if self.Length > 0 else "''"
PUNICODE_STRING = POINTER(UNICODE_STRING)
PPUNICODE_STRING = POINTER(PUNICODE_STRING)

class STRING(Structure):
    _fields_ = [
        ('Length', USHORT),
        ('MaximumLength', USHORT),
        ('Buffer', PSTR),
    ]
    def __str__(self):
        return self.Buffer if self.Length > 0 else ''
    def __repr__(self):
        return repr(self.Buffer) if self.Length > 0 else "''"
PSTRING = POINTER(STRING)
PPSTRING = POINTER(PSTRING)

class _OVERLAPPED_INNER_STRUCT(Structure):
    _fields_ = [
        ('Offset', DWORD),
        ('OffsetHigh', DWORD),
    ]

class _OVERLAPPED_INNER(Union):
    _fields_ = [
        ('s', _OVERLAPPED_INNER_STRUCT),
        ('Pointer', PVOID),
    ]

class OVERLAPPED(Structure):
    _fields_ = [
        ('Internal', ULONG_PTR),
        ('InternalHigh', ULONG_PTR),
        ('u', _OVERLAPPED_INNER),
        ('hEvent', HANDLE),
    ]
POVERLAPPED = POINTER(OVERLAPPED)
LPOVERLAPPED = POINTER(OVERLAPPED)

class OVERLAPPED_ENTRY(Structure):
    _fields_ = [
        ('lpCompletionKey', ULONG_PTR),
        ('lpOverlapped', LPOVERLAPPED),
        ('Internal', ULONG_PTR),
        ('dwNumberOfBytesTransferred', DWORD),
    ]
POVERLAPPED_ENTRY = POINTER(OVERLAPPED_ENTRY)
LPOVERLAPPED_ENTRY = POINTER(OVERLAPPED_ENTRY)

class TP_CALLBACK_ENVIRON_V3(Structure):
    _fields_ = [
        ('Version', TP_VERSION),
        ('Pool', PTP_POOL),
        ('CleanupGroup', PTP_CLEANUP_GROUP),
        ('CleanupGroupCancelCallback', PTP_CLEANUP_GROUP_CANCEL_CALLBACK),
        ('RaceDll', PVOID),
        ('ActivationContext', PVOID),
        ('FinalizationCallback', PVOID),
        ('Flags', DWORD),
        ('Priority', TP_CALLBACK_PRIORITY),
        ('Size', DWORD),
    ]
TP_CALLBACK_ENVIRON = TP_CALLBACK_ENVIRON_V3
PTP_CALLBACK_ENVIRON_V3 = POINTER(TP_CALLBACK_ENVIRON_V3)
PTP_CALLBACK_ENVIRON = POINTER(TP_CALLBACK_ENVIRON)

class FILE_INFO(object):
    @classmethod
    def get(cls, handle, info_class, info_buffer, buffer_size):
        success = kernel32.GetFileInformationByHandleEx(
            handle,
            info_class,
            info_buffer,
            buffer_size
        )
        assert success

class FILE_COMPRESSION_INFO(Structure):
    info_class = FileCompressionInfo
    _fields_ = [
        ('CompressedFileSize', LARGE_INTEGER),
        ('CompressionFormat', WORD),
        ('CompressionUnitShift', UCHAR),
        ('ChunkShift', UCHAR),
        ('ClusterShift', UCHAR),
        ('Reserved', UCHAR * 3),
    ]

    @classmethod
    def get(cls, handle):
        buf = cls()
        FILE_INFO.get(
            handle,
            cls.info_class,
            byref(buf),
            sizeof(cls)
        )
        return buf

PFILE_COMPRESSION_INFO = POINTER(FILE_COMPRESSION_INFO)

class MEMORY_PROTECTION(Constant):
    PAGE_NO_ACCESS          =   0x00000001
    PAGE_READONLY           =   0x00000002
    PAGE_READWRITE          =   0x00000004
    PAGE_WRITECOPY          =   0x00000008
    PAGE_EXECUTE            =   0x00000010
    PAGE_EXECUTE_READ       =   0x00000020
    PAGE_EXECUTE_READWRITE  =   0x00000040
    PAGE_EXECUTE_WRITECOPY  =   0x00000080
    PAGE_GUARD              =   0x00000100
    PAGE_NOCACHE            =   0x00000200
    PAGE_WRITECOMBINE       =   0x00000400
    PAGE_TARGETS_INVALID    =   0x40000000

    @classmethod
    def _format(cls, val):
        inverted = [
            (value, key) for (key, value) in cls.__dict__.items()
                if key.startswith('PAGE')
        ]
        inverted.sort()
        parts = [
            key for (value, key) in inverted
                if (val & value != 0)
        ]
        return ' | '.join(parts)

class _PSAPI_WORKING_SET_EX_BLOCK_VALID(Structure):
    _fields_ = [
        ('Valid', ULONGLONG, 1),
        ('ShareCount', ULONGLONG, 3),
        ('Win32Protection', ULONGLONG, 11),
        ('Shared', ULONGLONG, 1),
        ('Node', ULONGLONG, 6),
        ('Locked', ULONGLONG, 1),
        ('LargePage', ULONGLONG, 1),
        ('Reserved', ULONGLONG, 7),
        ('Bad', ULONGLONG, 1),
        ('ReservedUlong', ULONGLONG, 32),
    ]

PSAPI_WORKING_SET_EX_BLOCK = _PSAPI_WORKING_SET_EX_BLOCK_VALID
PPSAPI_WORKING_SET_EX_BLOCK = POINTER(PSAPI_WORKING_SET_EX_BLOCK)

class PSAPI_WORKING_SET_EX_INFORMATION(Structure):
    _fields_ = [
        ('VirtualAddress', PVOID),
        ('VirtualAttributes', _PSAPI_WORKING_SET_EX_BLOCK_VALID),
    ]

    @property
    def attributes(self):
        return
        if self.VirtualAttributes.v.Valid:
            return self.VirtualAttributes.v
        else:
            return self.VirtualAttributes.i
PPSAPI_WORKING_SET_EX_INFORMATION = POINTER(PSAPI_WORKING_SET_EX_INFORMATION)

class PSAPI_WS_WATCH_INFORMATION(Structure):
    _fields_ = [
        ('FaultingPc', PVOID),
        ('FaultingVa', PVOID),
    ]

class PSAPI_WS_WATCH_INFORMATION_EX(Structure):
    _fields_ = [
        ('BasicInfo', PSAPI_WS_WATCH_INFORMATION),
        ('FaultingThreadId', ULONG_PTR),
        ('Flags', ULONG_PTR),
    ]
PPSAPI_WS_WATCH_INFORMATION_EX = POINTER(PSAPI_WS_WATCH_INFORMATION_EX)

# Splay
class RTL_SPLAY_LINKS(Structure):
    pass
PRTL_SPLAY_LINKS = POINTER(RTL_SPLAY_LINKS)

RTL_SPLAY_LINKS._fields_ = [
    ('Parent', PRTL_SPLAY_LINKS),
    ('LeftChild', PRTL_SPLAY_LINKS),
    ('RightChild', PRTL_SPLAY_LINKS),
]

TableEmptyTree = 0
TableFoundNode = 1
TableInsertAsLeft = 2
TableInsertAsRight = 3
TABLE_SEARCH_RESULT = INT

GenericLessThan = 0
GenericGreaterThan = 1
GenericEqual = 2

class RTL_GENERIC_TABLE(Structure):
    pass
PRTL_GENERIC_TABLE = POINTER(RTL_GENERIC_TABLE)

RTL_GENERIC_COMPARE_ROUTINE = WINFUNCTYPE(PVOID,
    PRTL_GENERIC_TABLE, # Table
    PVOID,              # FirstStruct
    PVOID,              # SecondStruct
)
PRTL_GENERIC_COMPARE_ROUTINE = POINTER(RTL_GENERIC_COMPARE_ROUTINE)

RTL_GENERIC_ALLOCATE_ROUTINE = WINFUNCTYPE(PVOID,
    PRTL_GENERIC_TABLE, # Table
    CLONG,              # ByteSize
)
PRTL_GENERIC_ALLOCATE_ROUTINE = POINTER(RTL_GENERIC_ALLOCATE_ROUTINE)

RTL_GENERIC_FREE_ROUTINE = WINFUNCTYPE(VOID,
    PRTL_GENERIC_TABLE, # Table
    PVOID,              # Buffer
)
PRTL_GENERIC_FREE_ROUTINE = POINTER(RTL_GENERIC_FREE_ROUTINE)

RTL_GENERIC_TABLE._fields_ = (
    ('TableRoot', PRTL_SPLAY_LINKS),
    ('InsertOrderList', LIST_ENTRY),
    ('OrderedPointer', PLIST_ENTRY),
    ('WhichOrderedElement', ULONG),
    ('NumberGenericTableElements', ULONG),
    ('CompareRoutine', PRTL_GENERIC_COMPARE_ROUTINE),
    ('AllocateRoutine', PRTL_GENERIC_ALLOCATE_ROUTINE),
    ('FreeRoutine', PRTL_GENERIC_FREE_ROUTINE),
    ('TableContext', PVOID),
)
PRTL_GENERIC_TABLE = POINTER(RTL_GENERIC_TABLE)

# Hash Table

class RTL_DYNAMIC_HASH_TABLE_ENTRY(Structure):
    _fields_ = [
        ('Linkage', LIST_ENTRY),
        ('Signature', ULONG_PTR),
    ]

class RTL_DYNAMIC_HASH_TABLE_CONTEXT(Structure):
    _fields_ = [
        ('ChainHead', PLIST_ENTRY),
        ('PrevLinkage', PLIST_ENTRY),
        ('Signature', ULONG_PTR),
    ]

class _RTL_DYNAMIC_HASH_TABLE_ENUMERATOR_INNER(Union):
    _fields_ = [
        ('HashEntry', RTL_DYNAMIC_HASH_TABLE_ENTRY),
        ('CurEntry', PLIST_ENTRY),
    ]

class RTL_DYNAMIC_HASH_TABLE_ENUMERATOR(Structure):
    _fields_ = [
        ('u', _RTL_DYNAMIC_HASH_TABLE_ENUMERATOR_INNER),
        ('ChainHead', PLIST_ENTRY),
        ('BucketIndex', ULONG),
    ]

class RTL_DYNAMIC_HASH_TABLE(Structure):
    _fields_ = [
        # Initialized at creation.
        ('Flags', ULONG),
        ('Shift', ULONG),

        # Used for bucket computation.
        ('TableSize', ULONG),
        ('Pivot', ULONG),
        ('DivisorMask', ULONG),

        # Counters.
        ('NumEntries', ULONG),
        ('NonEmptyBuckets', ULONG),
        ('NumEnumerators', ULONG),

        # Directory (internal).
        ('Directory', PVOID),
    ]
PRTL_DYNAMIC_HASH_TABLE = POINTER(RTL_DYNAMIC_HASH_TABLE)

class PREFIX_TABLE_ENTRY(Structure):
    pass
PPREFIX_TABLE_ENTRY = POINTER(PREFIX_TABLE_ENTRY)

class PREFIX_TABLE(Structure):
    pass
PPREFIX_TABLE = POINTER(PREFIX_TABLE)

PREFIX_TABLE_ENTRY._fields_ = [
    ('NodeTypeCode', SHORT),
    ('NameLength', SHORT),
    ('NextPrefixTree', PPREFIX_TABLE_ENTRY),
    ('Links', RTL_SPLAY_LINKS),
    ('Prefix', PSTRING),
]

PREFIX_TABLE._fields_ = [
    ('NodeTypeCode', SHORT),
    ('NameLength', SHORT),
    ('NextPrefixTree', PPREFIX_TABLE_ENTRY),
]

class RTL_BITMAP(Structure):
    _fields_ = [
        ('SizeOfBitMap', ULONG),
        ('Buffer', PVOID),
    ]
PRTL_BITMAP = POINTER(RTL_BITMAP)

#===============================================================================
# COM
#===============================================================================

class GUID(Structure):
    _fields_ = [
        ('Data1',  ULONG),
        ('Data2',  SHORT),
        ('Data3',  SHORT),
        ('Data4',  BYTE),
        ('Data5',  BYTE),
        ('Data6',  BYTE),
        ('Data7',  BYTE),
        ('Data8',  BYTE),
        ('Data9',  BYTE),
        ('Data10', BYTE),
        ('Data11', BYTE),
        ('Data12', BYTE),
    ]
PGUID = POINTER(GUID)

CLSID = GUID
REFCLSID = PGUID
IID = GUID
REFIID = PGUID

DLL_GET_CLASS_OBJECT = WINFUNCTYPE(REFCLSID, REFIID, PPVOID)
DLL_GET_CLASS_OBJECT.restype = HRESULT
PDLL_GET_CLASS_OBJECT = POINTER(DLL_GET_CLASS_OBJECT)

# IUnknown

IID_IUNKNOWN = GUID(
    0x00000000,
    0x0000,
    0x0000,
    0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46
)
REFIID_IUNKNOWN = byref(IID_IUNKNOWN)

class IUNKNOWN(Structure):
    pass
PIUNKNOWN = POINTER(IUNKNOWN)

IUNKNOWN_QUERY_INTERFACE = WINFUNCTYPE(PIUNKNOWN, REFIID, PPVOID)
IUNKNOWN_QUERY_INTERFACE.restype = HRESULT
PIUNKNOWN_QUERY_INTERFACE = POINTER(IUNKNOWN_QUERY_INTERFACE)

IUNKNOWN_ADD_REF = WINFUNCTYPE(PIUNKNOWN)
IUNKNOWN_ADD_REF.restype = ULONG
PIUNKNOWN_ADD_REF = POINTER(IUNKNOWN_ADD_REF)

IUNKNOWN_RELEASE = WINFUNCTYPE(PIUNKNOWN)
IUNKNOWN_RELEASE.restype = ULONG
PIUNKNOWN_RELEASE = POINTER(IUNKNOWN_RELEASE)

class IUNKNOWN_VTBL(Structure):
    _fields_ = [
        ('QueryInterface', PIUNKNOWN_QUERY_INTERFACE),
        ('AddRef', PIUNKNOWN_ADD_REF),
        ('Release', PIUNKNOWN_RELEASE),
    ]
PIUNKNOWN_VTBL = POINTER(IUNKNOWN_VTBL)

IUNKNOWN._fields_ = [
    ('Vtbl', PIUNKNOWN_VTBL),
]

# IClassFactory

IID_ICLASSFACTORY = GUID(
    0x00000001,
    0x0000,
    0x0000,
    0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46
)
REFIID_ICLASSFACTORY = byref(IID_ICLASSFACTORY)

class ICLASSFACTORY(Structure):
    pass
PICLASSFACTORY = POINTER(ICLASSFACTORY)

ICLASSFACTORY_QUERY_INTERFACE = WINFUNCTYPE(PICLASSFACTORY, REFIID, PPVOID)
ICLASSFACTORY_QUERY_INTERFACE.restype = HRESULT
PICLASSFACTORY_QUERY_INTERFACE = POINTER(ICLASSFACTORY_QUERY_INTERFACE)

ICLASSFACTORY_ADD_REF = WINFUNCTYPE(PICLASSFACTORY)
ICLASSFACTORY_ADD_REF.restype = ULONG
PICLASSFACTORY_ADD_REF = POINTER(ICLASSFACTORY_ADD_REF)

ICLASSFACTORY_RELEASE = WINFUNCTYPE(PICLASSFACTORY)
ICLASSFACTORY_RELEASE.restype = ULONG
PICLASSFACTORY_RELEASE = POINTER(ICLASSFACTORY_RELEASE)

ICLASSFACTORY_CREATE_INSTANCE = WINFUNCTYPE(
    PICLASSFACTORY,
    PIUNKNOWN,
    REFIID,
    PPVOID
)
ICLASSFACTORY_CREATE_INSTANCE.restype = HRESULT
PICLASSFACTORY_CREATE_INSTANCE = POINTER(ICLASSFACTORY_CREATE_INSTANCE)

ICLASSFACTORY_LOCK_SERVER = WINFUNCTYPE(PICLASSFACTORY, BOOL)
ICLASSFACTORY_LOCK_SERVER.restype = HRESULT
PICLASSFACTORY_LOCK_SERVER = POINTER(ICLASSFACTORY_LOCK_SERVER)

class ICLASSFACTORY_VTBL(Structure):
    _fields_ = [
        ('QueryInterface', PICLASSFACTORY_QUERY_INTERFACE),
        ('AddRef', PICLASSFACTORY_ADD_REF),
        ('Release', PICLASSFACTORY_RELEASE),
        ('CreateInstance', PICLASSFACTORY_CREATE_INSTANCE),
        ('LockServer', PICLASSFACTORY_LOCK_SERVER),
    ]
PICLASSFACTORY_VTBL = POINTER(ICLASSFACTORY_VTBL)

ICLASSFACTORY._fields_ = [
    ('Vtbl', PICLASSFACTORY_VTBL),
]

#===============================================================================
# Kernel32
#===============================================================================
kernel32 = ctypes.windll.kernel32

kernel32.CreateThreadpool.restype = PTP_POOL
kernel32.CreateThreadpool.argtypes = [ PVOID, ]

kernel32.SetThreadpoolThreadMinimum.restype = BOOL
kernel32.SetThreadpoolThreadMinimum.argtypes = [ PTP_POOL, DWORD ]

kernel32.SetThreadpoolThreadMaximum.restype = VOID
kernel32.SetThreadpoolThreadMaximum.argtypes = [ PTP_POOL, DWORD ]

kernel32.CloseThreadpool.restype = VOID
kernel32.CloseThreadpool.argtypes = [ PTP_POOL, ]

kernel32.WaitForSingleObject.restype = DWORD
kernel32.WaitForSingleObject.argtypes = [ HANDLE, DWORD ]

kernel32.GetFileInformationByHandleEx.restype = BOOL
kernel32.GetFileInformationByHandleEx.argtypes = [
    HANDLE,     # hFile
    ULONG,      # FILE_INFO_BY_HANDLE_CLASS
    PVOID,      # lpFileInformation
    ULONG,      # dwBufferSize
]

#===============================================================================
# NtDll
#===============================================================================
ntdll = ctypes.windll.ntdll

#ntdll.RtlInitializeGenericTable.restype = VOID
#ntdll.RtlInitializeGenericTable.argtypes = [
#    PRTL_GENERIC_TABLE,
#    PRTL_GENERIC_COMPARE_ROUTINE,
#    PRTL_GENERIC_ALLOCATE_ROUTINE,
#    PRTL_GENERIC_FREE_ROUTINE,
#    PVOID,
#]

RTL_INITIALIZE_GENERIC_TABLE = WINFUNCTYPE(VOID,
    PRTL_GENERIC_TABLE,
    PRTL_GENERIC_COMPARE_ROUTINE,
    PRTL_GENERIC_ALLOCATE_ROUTINE,
    PRTL_GENERIC_FREE_ROUTINE,
    PVOID,
)
PRTL_INITIALIZE_GENERIC_TABLE = POINTER(RTL_INITIALIZE_GENERIC_TABLE)

RtlInitializeGenericTable = (
    RTL_INITIALIZE_GENERIC_TABLE(ntdll.RtlInitializeGenericTable)
)

#ntdll.RtlInsertElementGenericTable.restype = PVOID
#ntdll.RtlInsertElementGenericTable.argtypes = [
#    PRTL_GENERIC_TABLE, # Table
#    PVOID,              # Buffer
#    CLONG,              # BufferSize
#    PBOOLEAN,           # NewElement
#]
RTL_INSERT_ELEMENT_GENERIC_TABLE = WINFUNCTYPE(PVOID,
    PRTL_GENERIC_TABLE, # Table
    PVOID,              # Buffer
    CLONG,              # BufferSize
    #PBOOLEAN,          # NewElement
)

RtlInsertElementGenericTable = (
    RTL_INSERT_ELEMENT_GENERIC_TABLE(ntdll.RtlInsertElementGenericTable)
)


ntdll.RtlInsertElementGenericTableFull.restype = PVOID
ntdll.RtlInsertElementGenericTableFull.argtypes = [
    PRTL_GENERIC_TABLE,     # Table
    PVOID,                  # Buffer
    CLONG,                  # BufferSize
    PBOOLEAN,               # NewElement
    PVOID,                  # NodeOrParent
    TABLE_SEARCH_RESULT,    # SearchResult
]

ntdll.RtlDeleteElementGenericTable.restype = BOOLEAN
ntdll.RtlDeleteElementGenericTable.argtypes = [
    PRTL_GENERIC_TABLE, # Table
    PVOID,              # Buffer
]

ntdll.RtlLookupElementGenericTable.restype = PVOID
ntdll.RtlLookupElementGenericTable.argtypes = [
    PRTL_GENERIC_TABLE, # Table
    PVOID,              # Buffer
]

ntdll.RtlCreateHashTable.restype = BOOLEAN
ntdll.RtlCreateHashTable.argtypes = [
    PRTL_DYNAMIC_HASH_TABLE,
    ULONG,  # Shift
    ULONG,  # Flags
]


#===============================================================================
# Functions
#===============================================================================

# Provide implementations for various Rtl inlined threadpool functions.
def InitializeThreadpoolEnvironmentV3(CallbackEnviron):
    CallbackEnviron.Version = 3
    CallbackEnviron.CallbackPriority = TP_CALLBACK_PRIORITY_NORMAL
    CallbackEnviron.Size = sizeof(TP_CALLBACK_ENVIRON_V3)
InitializeThreadpoolEnvironment = InitializeThreadpoolEnvironmentV3

def SetThreadpoolCallbackCleanupGroup(CallbackEnviron,
                                      CleanupGroup,
                                      CleanupGroupCancelCallback):
    CallbackEnviron.CleanupGroup = CleanupGroup
    CallbackEnviron.CleanupGroupCancelCallback = CleanupGroupCancelCallback

def SetThreadpoolCallbackActivationContext(CallbackEnviron,
                                           ActivationContext):
    CallbackEnviron.ActivationContext = ActivationContext

def SetThreadpoolCallbackPool(CallbackEnviron, Threadpool):
    CallbackEnviron.Pool = Threadpool

#===============================================================================
# Helpers
#===============================================================================

def errcheck(result, func, args):
    if not result:
        raise RuntimeError("%s failed" % func.__name__)
    return args

def create_unicode_string(string):
    unicode_string = UNICODE_STRING()
    unicode_string.Length = len(string) << 1
    unicode_string.MaximumLength = unicode_string.Length + 2
    unicode_string.Buffer = cast(
        pointer(create_unicode_buffer(string)),
        PWSTR
    )
    return unicode_string

def create_threadpool(num_cpus=None):
    if not num_cpus:
        from multiprocessing import cpu_count
        num_cpus = cpu_count()


    threadpool = kernel32.CreateThreadpool(None)
    if not threadpool:
        raise RuntimeError("CreateThreadpool() failed")

    kernel32.SetThreadpoolThreadMinimum(threadpool, num_cpus)
    kernel32.SetThreadpoolThreadMaximum(threadpool, num_cpus)

def create_threadpool_callback_environment(num_cpus=None, threadpool=None):
    if not threadpool:
        threadpool = create_threadpool(num_cpus=num_cpus)
    threadpool_callback_environment = TP_CALLBACK_ENVIRON()
    InitializeThreadpoolEnvironment(threadpool_callback_environment)
    SetThreadpoolCallbackPool(
        threadpool_callback_environment,
        threadpool,
    )
    return threadpool_callback_environment

def is_signaled(event):
    result = kernel32.WaitForSingleObject(event, 0)
    if result == WAIT_OBJECT_0:
        return True
    elif result == WAIT_TIMEOUT:
        return False
    elif result == WAIT_ABANDONED:
        return False
    elif result == WAIT_FAILED:
        raise OSError("WaitForSingleObject: WAIT_FAILED")

def wait(event, timeout=INFINITE):
    result = kernel32.WaitForSingleObject(event, timeout)
    if result == WAIT_OBJECT_0:
        return True
    elif result == WAIT_TIMEOUT:
        return False
    elif result == WAIT_ABANDONED:
        return False
    elif result == WAIT_FAILED:
        raise OSError("WaitForSingleObject: WAIT_FAILED")


def field_to_offset_hex(struct):
    return [
        (hex(getattr(struct, name[0]).offset), name)
            for name in struct._fields_
    ]

def field_to_offset(struct):
    return [
        (getattr(struct, name[0]).offset, name)
            for name in struct._fields_
    ]

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
