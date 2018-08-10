;/*++
;
;Copyright (c) 2018 Trent Nelson <trent@trent.me>
;
;Module Name:
;
;    PerfectHashTableErrors.h
;
;Abstract:
;
;    This is the public header file for error codes used by the perfect
;    hash table library.  It is automatically generated.
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


