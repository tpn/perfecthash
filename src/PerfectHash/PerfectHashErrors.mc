;/*++
;
;Copyright (c) 2018 Trent Nelson <trent@trent.me>
;
;Module Name:
;
;    PerfectHashErrors.h
;
;Abstract:
;
;    This is the public header file for error codes used by the perfect
;    hash library.  It is automatically generated.
;
;--*/

MessageIdTypedef=HRESULT
SeverityNames=(Success=0x0:PH_SEVERITY_SUCCESS
               Informational=0x1:PH_SEVERITY_INFORMATIONAL
               Warning=0x2:PH_SEVERITY_WARNING
               Fail=0x3:PH_SEVERITY_FAIL)
FacilityNames=(ITF=0x4:PH_FACILITY_ITF)
LanguageNames=(English=0x409:English)

MessageId=0x201
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS
Language=English
A table creation operation is in progress for this context.
.

MessageId=0x202
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_KEYS
Language=English
Too many keys.
.

MessageId=0x203
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INFO_FILE_SMALLER_THAN_HEADER
Language=English
:Info file is smaller than smallest known table header size.
.

MessageId=0x204
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MAGIC_VALUES
Language=English
Invalid magic values.
.

MessageId=0x205
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_INFO_HEADER_SIZE
Language=English
Invalid :Info header size.
.

MessageId=0x206
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS
Language=English
The number of keys reported in the keys file does not match the number of keys reported in the header.
.

MessageId=0x207
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_ALGORITHM_ID
Language=English
Invalid algorithm ID.
.

MessageId=0x208
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_HASH_FUNCTION_ID
Language=English
Invalid hash function ID.
.

MessageId=0x209
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MASK_FUNCTION_ID
Language=English
Invalid mask function ID.
.

MessageId=0x20a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_HEADER_KEY_SIZE_TOO_LARGE
Language=English
The key size reported by the header is too large.
.

MessageId=0x20b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_KEYS_IS_ZERO
Language=English
The number of keys is zero.
.

MessageId=0x20c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_TABLE_ELEMENTS_IS_ZERO
Language=English
The number of table elements is zero.
.

MessageId=0x20d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS
Language=English
The number of keys exceeds the number of table elements.
.

MessageId=0x20e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH
Language=English
The expected end of file does not match the actual end of file.
.

MessageId=0x20f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE
Language=English
The keys file size is not a multiple of the key size.
.

MessageId=0x210
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH
Language=English
The number of bits set for the keys bitmap does not match the number of keys.
.

MessageId=0x211
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DUPLICATE_KEYS_DETECTED
Language=English
Duplicate keys detected.  Key files must not contain duplicate keys.
.

MessageId=0x212
Severity=Fail
Facility=ITF
SymbolicName=PH_E_HEAP_CREATE_FAILED
Language=English
A call to HeapCreate() failed.
.

MessageId=0x213
Severity=Fail
Facility=ITF
SymbolicName=PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED
Language=English
A call to RtlLoadSymbolsFromMultipleModules() failed.
.

MessageId=0x214
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS
Language=English
Invalid number of arguments for context self-test.
.

MessageId=0x215
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_MAXIMUM_CONCURRENCY
Language=English
Invalid value for maximum concurrency.
.

MessageId=0x216
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SET_MAXIMUM_CONCURRENCY_FAILED
Language=English
Setting the maximum concurrency of the context failed.
.

MessageId=0x217
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INITIALIZE_LARGE_PAGES_FAILED
Language=English
Internal error when attempting to initialize large pages.
.

MessageId=0x218
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_NOT_SORTED
Language=English
The keys file supplied was not sorted.
.

MessageId=0x219
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_NOT_LOADED
Language=English
A keys file has not been loaded yet.
.

MessageId=0x21a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS
Language=English
A key loading operation is already in progress.
.

MessageId=0x21b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_ALREADY_LOADED
Language=English
A set of keys has already been loaded.
.

MessageId=0x21c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_KEYS_LOAD_FLAGS
Language=English
Invalid key load flags.
.

MessageId=0x21d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_KEY_SIZE
Language=English
Invalid key size.
.

MessageId=0x21e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_NOT_LOADED
Language=English
No table has been loaded yet.
.

MessageId=0x21f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS
Language=English
A table loading operation is already in progress.
.

MessageId=0x220
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS
Language=English
Invalid context create table flags.
.

MessageId=0x221
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CONTEXT_SELF_TEST_FLAGS
Language=English
Invalid context self-test flags.
.

MessageId=0x222
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_LOAD_FLAGS
Language=English
Invalid table load flags.
.

MessageId=0x223
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_LOCKED
Language=English
Table is locked.
.

MessageId=0x224
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SYSTEM_CALL_FAILED
Language=English
A routine supplied by the operating system has indicated failure.
.

MessageId=0x225
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_ALREADY_CREATED
Language=English
The table instance has already been created.
.

MessageId=0x226
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_ALREADY_LOADED
Language=English
The table instance has already been loaded.
.

