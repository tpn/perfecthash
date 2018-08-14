#===============================================================================
# Imports
#===============================================================================

from ..config import (
    get_or_create_config,
)

from ..wintypes import (
    cast,
    byref,
    sizeof,
    create_unicode_string,

    WinDLL,
    OleDLL,
    Structure,

    IID,
    GUID,
    BOOL,
    BYTE,
    SHORT,
    LONG,
    CLSID,
    ULONG,
    PVOID,
    USHORT,
    PULONG,
    STRING,
    PUSHORT,
    REFIID,
    PPVOID,
    HRESULT,
    PSTRING,
    POINTER,
    REFCLSID,
    ULONGLONG,
    ULONG_PTR,
    PIUNKNOWN,
    WINFUNCTYPE,
    IID_IUNKNOWN,
    PUNICODE_STRING,
    REFIID_IUNKNOWN,
    IID_ICLASSFACTORY,
    REFIID_ICLASSFACTORY,
)


#===============================================================================
# Enums/Constants
#===============================================================================

# PERFECT_HASH_INTERFACE_ID
PerfectHashNullInterfaceId             = 0
PerfectHashUnknownInterfaceId          = 1
PerfectHashClassFactoryInterfaceId     = 2
PerfectHashKeysInterfaceId             = 3
PerfectHashContextInterfaceId          = 4
PerfectHashInterfaceId                 = 5
PerfectHashRtlInterfaceId              = 6
PerfectHashAllocatorInterfaceId        = 7
PerfectHashInvalidInterfaceId          = 8

# PERFECT_HASH_ALGORITHM_ID
PerfectHashNullAlgorithmId              = 0
PerfectHashChm01AlgorithmId             = 1
PerfectHashDefaultAlgorithmId           = 1
PerfectHashInvalidAlgorithmId           = 2

# PERFECT_HASH_HASH_FUNCTION_ID
PerfectHashNullHashFunctionId           = 0
PerfectHashHashCrc32RotateFunctionId    = 1
PerfectHashDefaultHashFunctionId        = 1
PerfectHashHashJenkinsFunctionId        = 2
PerfectHashHashRotateXorFunctionId      = 3
PerfectHashHashAddSubXorFunctionId      = 4
PerfectHashHashXorFunctionId            = 5
PerfectHashInvalidHashFunctionId        = 6

# PERFECT_HASH_MASK_FUNCTION_ID
PerfectHashNullMaskFunctionId           = 0
PerfectHashModulusMaskFunctionId        = 1
PerfectHashAndMaskFunctionId            = 2
PerfectHashDefaultMaskFunctionId        = 2
PerfectHashInvalidMaskFunctionId        = 3

# PERFECT_HASH_BENCHMARK_FUNCTION_ID
PerfectHashNullBenchmarkFunctionId      = 0
PerfectHashFastIndexBenchmarkFunctionId = 1
PerfectHashInvalidBenchmarkFunctionId   = 2

# PERFECT_HASH_BENCHMARK_TYPE
PerfectHashNullBenchmarkType            = 0
PerfectHashSingleBenchmarkType          = 1
PerfectHashAllBenchmarkType             = 2
PerfectHashInvalidBenchmarkType         = 3

#===============================================================================
# Helpers
#===============================================================================

def load_dll():
    conf = get_or_create_config()
    path = conf.perfecthash_dll_path
    dll = OleDLL(path)
    #dll = WinDLL(path)
    return dll

PerfectHashDll = load_dll()

def new_class_factory():
    obj = DllGetClassObject(REFCLSID_PERFECT_HASH, REFIID_ICLASSFACTORY)
    return cast(obj, PICLASSFACTORY)

def new_keys():
    obj = DllGetClassObject(REFCLSID_PERFECT_HASH, REFIID_PERFECT_HASH_KEYS)
    return cast(obj, PPERFECT_HASH_KEYS)


#===============================================================================
# Interfaces
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

DLL_GET_CLASS_OBJECT = WINFUNCTYPE(HRESULT, REFCLSID, REFIID, PPVOID)
PDLL_GET_CLASS_OBJECT = POINTER(DLL_GET_CLASS_OBJECT)

DllGetClassObject = DLL_GET_CLASS_OBJECT(
    ('PerfectHashDllGetClassObject', PerfectHashDll),
    (
        (1, 'ClassId'),
        (1, 'InterfaceId'),
        (2, 'Interface'),
    )
)

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

IUNKNOWN_QUERY_INTERFACE = WINFUNCTYPE(HRESULT, PIUNKNOWN, REFIID, PPVOID)
PIUNKNOWN_QUERY_INTERFACE = POINTER(IUNKNOWN_QUERY_INTERFACE)

IUNKNOWN_ADD_REF = WINFUNCTYPE(ULONG, PIUNKNOWN)
PIUNKNOWN_ADD_REF = POINTER(IUNKNOWN_ADD_REF)

IUNKNOWN_RELEASE = WINFUNCTYPE(ULONG, PIUNKNOWN)
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

ICLASSFACTORY_QUERY_INTERFACE = WINFUNCTYPE(
    HRESULT,
    PICLASSFACTORY,
    REFIID,
    PPVOID
)
PICLASSFACTORY_QUERY_INTERFACE = POINTER(ICLASSFACTORY_QUERY_INTERFACE)

ICLASSFACTORY_ADD_REF = WINFUNCTYPE(ULONG, PICLASSFACTORY)
PICLASSFACTORY_ADD_REF = POINTER(ICLASSFACTORY_ADD_REF)

ICLASSFACTORY_RELEASE = WINFUNCTYPE(ULONG, PICLASSFACTORY)
PICLASSFACTORY_RELEASE = POINTER(ICLASSFACTORY_RELEASE)

ICLASSFACTORY_CREATE_INSTANCE = WINFUNCTYPE(
    HRESULT,
    PICLASSFACTORY,
    PIUNKNOWN,
    REFIID,
    PPVOID
)
PICLASSFACTORY_CREATE_INSTANCE = POINTER(ICLASSFACTORY_CREATE_INSTANCE)

ICLASSFACTORY_LOCK_SERVER = WINFUNCTYPE(HRESULT, PICLASSFACTORY, BOOL)
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

# CLSID_PERFECT_HASH

CLSID_PERFECT_HASH = GUID(
    0x402045fd,
    0x72f4,
    0x4a05,
    0x90, 0x2e, 0xd2, 0x2b, 0x7c, 0x18, 0x77, 0xb4
)
REFCLSID_PERFECT_HASH = byref(CLSID_PERFECT_HASH)

# PERFECT_HASH_TABLE_KEYS

IID_PERFECT_HASH_KEYS = GUID(
    0x7e43ebea,
    0x8671,
    0x47ba,
    0xb8, 0x44, 0x76, 0xb, 0x7a, 0x9e, 0xa9, 0x21
)
REFIID_PERFECT_HASH_KEYS = byref(IID_PERFECT_HASH_KEYS)

class PERFECT_HASH_KEYS(Structure):
    pass
PPERFECT_HASH_KEYS = POINTER(PERFECT_HASH_KEYS)

PERFECT_HASH_KEYS_QUERY_INTERFACE = WINFUNCTYPE(
    HRESULT,
    PPERFECT_HASH_KEYS,
    REFIID,
    PPVOID
)
PPERFECT_HASH_KEYS_QUERY_INTERFACE = POINTER(
    PERFECT_HASH_KEYS_QUERY_INTERFACE
)

PERFECT_HASH_KEYS_ADD_REF = WINFUNCTYPE(ULONG, PPERFECT_HASH_KEYS)
PPERFECT_HASH_KEYS_ADD_REF = POINTER(PERFECT_HASH_KEYS_ADD_REF)

PERFECT_HASH_KEYS_RELEASE = WINFUNCTYPE(ULONG, PPERFECT_HASH_KEYS)
PPERFECT_HASH_KEYS_RELEASE = POINTER(PERFECT_HASH_KEYS_RELEASE)

PERFECT_HASH_KEYS_CREATE_INSTANCE = WINFUNCTYPE(
    HRESULT,
    PPERFECT_HASH_KEYS,
    PIUNKNOWN,
    REFIID,
    PPVOID
)
PPERFECT_HASH_KEYS_CREATE_INSTANCE = POINTER(
    PERFECT_HASH_KEYS_CREATE_INSTANCE
)

PERFECT_HASH_KEYS_LOCK_SERVER = WINFUNCTYPE(
    HRESULT,
    PPERFECT_HASH_KEYS,
    BOOL
)
PPERFECT_HASH_KEYS_LOCK_SERVER = POINTER(
    PERFECT_HASH_KEYS_LOCK_SERVER
)

PERFECT_HASH_KEYS_LOAD = WINFUNCTYPE(
    HRESULT,
    PPERFECT_HASH_KEYS,
    PUNICODE_STRING,
)
PPERFECT_HASH_KEYS_LOAD = POINTER(PERFECT_HASH_KEYS_LOAD)

PERFECT_HASH_KEYS_GET_BITMAP = WINFUNCTYPE(
    HRESULT,
    PPERFECT_HASH_KEYS,
    PULONG,
)
PPERFECT_HASH_KEYS_GET_BITMAP = POINTER(PERFECT_HASH_KEYS_GET_BITMAP)

class PERFECT_HASH_KEYS_VTBL(Structure):
    _fields_ = [
        ('QueryInterface', PPERFECT_HASH_KEYS_QUERY_INTERFACE),
        ('AddRef', PPERFECT_HASH_KEYS_ADD_REF),
        ('Release', PPERFECT_HASH_KEYS_RELEASE),
        ('Load', PPERFECT_HASH_KEYS_LOAD),
        ('GetBitmap', PPERFECT_HASH_KEYS_GET_BITMAP),
    ]
PPERFECT_HASH_KEYS_VTBL = POINTER(PERFECT_HASH_KEYS_VTBL)

PERFECT_HASH_KEYS._fields_ = [
    ('Vtbl', PPERFECT_HASH_KEYS_VTBL),
]

PerfectHashKeysLoad = PERFECT_HASH_KEYS_LOAD(
    ('PerfectHashKeysLoad', PerfectHashDll),
    (
        (1, 'Keys'),
        (1, 'Path'),
    ),
)

PerfectHashKeysGetBitmap = PERFECT_HASH_KEYS_GET_BITMAP(
    ('PerfectHashKeysGetBitmap', PerfectHashDll),
    (
        (1, 'Keys'),
        (1, 'Bitmap'),
    )
)

class Keys:
    def __init__(self, path):
        self.path = create_unicode_string(path)
        self.obj = new_keys()

        PerfectHashKeysLoad(self.obj, byref(self.path))

    def get_bitmap(self):
        bitmap = ULONG()
        PerfectHashKeysGetBitmap(self.obj, byref(bitmap))
        return bitmap.value


# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
