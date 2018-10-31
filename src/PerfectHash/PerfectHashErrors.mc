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
;    hash library.  It is automatically generated from the messages
;    messages defined in src/PerfectHash/PerfectHashErrors.mc by the
;    helper script src/PerfectHash/build-message-tables.bat (which must
;    be run whenever the .mc file changes).
;
;--*/

MessageIdTypedef=HRESULT
SeverityNames=(Success=0x0:PH_SEVERITY_SUCCESS
               Informational=0x1:PH_SEVERITY_INFORMATIONAL
               Warning=0x2:PH_SEVERITY_WARNING
               Fail=0x3:PH_SEVERITY_FAIL)
FacilityNames=(ITF=0x4:PH_FACILITY_ITF)
LanguageNames=(English=0x409:English)

MessageId=0x001
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_SOLVED
Language=English
Graph solved.
.

MessageId=0x002
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_NOT_SOLVED
Language=English
Graph not solved.
.

MessageId=0x003
Severity=Success
Facility=ITF
SymbolicName=PH_S_CONTINUE_GRAPH_SOLVING
Language=English
Continue graph solving.
.

MessageId=0x004
Severity=Success
Facility=ITF
SymbolicName=PH_S_STOP_GRAPH_SOLVING
Language=English
Stop graph solving.
.

MessageId=0x005
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_VERIFICATION_SKIPPED
Language=English
Graph verification skipped.
.

MessageId=0x006
Severity=Success
Facility=ITF
SymbolicName=PH_S_GRAPH_SOLVING_STOPPED
Language=English
Graph solving has been stopped.
.

MessageId=0x007
Severity=Success
Facility=ITF
SymbolicName=PH_S_TABLE_RESIZE_IMMINENT
Language=English
Table resize imminent.
.

MessageId=0x008
Severity=Success
Facility=ITF
SymbolicName=PH_S_USE_NEW_GRAPH_FOR_SOLVING
Language=English
Use new graph for solving.
.

;
;////////////////////////////////////////////////////////////////////////////////
;// PH_SEVERITY_FAIL
;////////////////////////////////////////////////////////////////////////////////
;

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

;//
;// Removed 2018-10-01.
;//
;// MessageId=0x220
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS
;// Language=English
;// Invalid context create table flags.
;// .
;//

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
System call failed.
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

MessageId=0x227
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_COMPILE_FLAGS
Language=English
Invalid table compile flags.
.

MessageId=0x228
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_COMPILATION_NOT_AVAILABLE
Language=English
Table compilation is not available for the current combination of architecture, algorithm ID, hash function and masking type.
.

MessageId=0x229
Severity=Fail
Facility=ITF
SymbolicName=PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
Language=English
The maximum number of table resize events was reached before a perfect hash table solution could be found.
.

MessageId=0x22a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
Language=English
The requested number of table elements was too large.
.

MessageId=0x22b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_TABLE_FILE
Language=English
Error preparing perfect hash table file.
.

MessageId=0x22c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_TABLE_FILE
Language=English
Error saving perfect hash table file.
.

;//
;// A perfect hash table solution was found, however, it did not
;// pass internal validation checks (e.g. collisions were found
;// when attempting to independently verify that the perfect hash
;// function generated no collisions).
;//

MessageId=0x22d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_VERIFICATION_FAILED
Language=English
Table verification failed.
.

MessageId=0x22e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_CROSS_COMPILATION_NOT_AVAILABLE
Language=English
Table cross-compilation is not available.
.

MessageId=0x22f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CPU_ARCH_ID
Language=English
The CPU architecture ID was invalid.
.

MessageId=0x230
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NOT_IMPLEMENTED
Language=English
Functionality not yet implemented.
.

MessageId=0x231
Severity=Fail
Facility=ITF
SymbolicName=PH_E_WORK_IN_PROGRESS
Language=English
Work in progress.
.

MessageId=0x232
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER
Language=English
Keys file base name is not a valid C identifier.
.

MessageId=0x233
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_FILE
Language=English
Error preparing C header file.
.

MessageId=0x234
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_FILE
Language=English
Error saving C header file.
.

MessageId=0x235
Severity=Fail
Facility=ITF
SymbolicName=PH_E_UNREACHABLE_CODE
Language=English
Unreachable code reached.
.

MessageId=0x236
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVARIANT_CHECK_FAILED
Language=English
Internal invariant check failed.
.

MessageId=0x237
Severity=Fail
Facility=ITF
SymbolicName=PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE
Language=English
The calculated C header file size exceeded 4GB.
.

MessageId=0x238
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_NOT_SET
Language=English
Base output directory has not been set.
.

MessageId=0x239
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_LOCKED
Language=English
The context is locked.
.

MessageId=0x23a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_RESET_FAILED
Language=English
Failed to reset context.
.

MessageId=0x23b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_SET_BASE_OUTPUT_DIRECTORY_FAILED
Language=English
Failed to set context output directory.
.

MessageId=0x23c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_TABLE_CREATED_OR_LOADED
Language=English
The table has not been created or loaded.
.

MessageId=0x23d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_PATHS_ALREADY_INITIALIZED
Language=English
Paths have already been initialized for this table instance.
.

MessageId=0x23e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_TABLE_INFO_STREAM
Language=English
Error preparing :Info stream.
.

MessageId=0x23f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_TABLE_INFO_STREAM
Language=English
Error saving :Info stream.
.

MessageId=0x240
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_FILE
Language=English
Error saving C source file.
.

MessageId=0x241
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_KEYS_FILE
Language=English
Error saving C source keys file.
.

MessageId=0x242
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_TABLE_DATA_FILE
Language=English
Error saving C source table data file.
.

MessageId=0x243
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_CLOSED
Language=English
The file has already been closed.
.

MessageId=0x244
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NOT_OPEN
Language=English
The file has not been opened yet, or has been closed.
.

MessageId=0x245
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_LOCKED
Language=English
The file is locked.
.

MessageId=0x246
Severity=Fail
Facility=ITF
SymbolicName=PH_E_KEYS_LOCKED
Language=English
The keys are locked.
.

MessageId=0x247
Severity=Fail
Facility=ITF
SymbolicName=PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE
Language=English
Mapping size is less than or equal to current file size.
.

MessageId=0x248
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_READONLY
Language=English
The file is readonly.
.

MessageId=0x249
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_VIEW_CREATED
Language=English
A file view has already been created.
.

MessageId=0x24a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_VIEW_MAPPED
Language=English
A file view has already been mapped.
.

MessageId=0x24b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_MAPPING_SIZE_IS_ZERO
Language=English
The mapping size for the file is zero.
.

MessageId=0x24c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED
Language=English
Mapping size is not aligned to the system allocation granularity.
.

MessageId=0x24d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED
Language=English
Mapping size is not aligned to the large page granularity.
.

MessageId=0x24e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_OPEN
Language=English
An existing file has already been loaded or created for this file instance.
.

MessageId=0x24f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_FILE_LOAD_FLAGS
Language=English
Invalid file load flags.
.

MessageId=0x250
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_CLOSED
Language=English
File already closed.
.

MessageId=0x251
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_EMPTY
Language=English
The file is empty.
.

MessageId=0x252
Severity=Fail
Facility=ITF
SymbolicName=PH_E_PATH_PARTS_EXTRACTION_FAILED
Language=English
Failed to extract the path into parts.
.

MessageId=0x253
Severity=Fail
Facility=ITF
SymbolicName=PH_E_PATH_LOCKED
Language=English
Path is locked.
.

MessageId=0x254
Severity=Fail
Facility=ITF
SymbolicName=PH_E_EXISTING_PATH_LOCKED
Language=English
Existing path parameter is locked.
.

MessageId=0x255
Severity=Fail
Facility=ITF
SymbolicName=PH_E_PATH_ALREADY_SET
Language=English
A path has already been set for this instance.
.

MessageId=0x256
Severity=Fail
Facility=ITF
SymbolicName=PH_E_EXISTING_PATH_NO_PATH_SET
Language=English
Existing path parameter has not had a path set.
.

MessageId=0x257
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_PATH_SET
Language=English
No path set.
.

MessageId=0x258
Severity=Fail
Facility=ITF
SymbolicName=PH_E_STRING_BUFFER_OVERFLOW
Language=English
An internal string buffer has overflowed.
.

MessageId=0x259
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SOURCE_PATH_LOCKED
Language=English
Source path parameter is locked.
.

MessageId=0x25a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SOURCE_PATH_NO_PATH_SET
Language=English
Source path parameter had no path set.
.

MessageId=0x25b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE
Language=English
Invalid table.
.

MessageId=0x25c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_FLAGS
Language=English
Invalid table create flags.
.

MessageId=0x25d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_FILE
Language=English
Error preparing C source file.
.

MessageId=0x25e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_KEYS_FILE
Language=English
Error preparing C source keys file.
.

MessageId=0x25f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_TABLE_DATA_FILE
Language=English
Error preparing C source table data file..
.

MessageId=0x260
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NEVER_OPENED
Language=English
The file has never been opened.
.

MessageId=0x261
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NO_RENAME_SCHEDULED
Language=English
The file has not had a rename operation scheduled.
.

MessageId=0x262
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NOT_CLOSED
Language=English
The file has not yet been closed.
.

MessageId=0x263
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_FILE_WORK_ID
Language=English
Invalid file work ID.
.

MessageId=0x264
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_END_OF_FILE
Language=English
Invalid end of file (less than or equal to 0).
.

MessageId=0x265
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF
Language=English
New end-of-file is less than or equal to current end-of-file.
.

MessageId=0x266
Severity=Fail
Facility=ITF
SymbolicName=PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH
Language=English
The rename path equivalent to the existing path.
.

MessageId=0x267
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_FILE_CREATE_FLAGS
Language=English
Invalid file create flags.
.

MessageId=0x268
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_INTERFACE_ID
Language=English
Invalid interface ID.
.

MessageId=0x269
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_TLS_CONTEXT_SET
Language=English
PerfectHashTlsEnsureContext() was called but no TLS context was set.
.

MessageId=0x26a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NOT_GLOBAL_INTERFACE_ID
Language=English
The interface ID provided is not a global interface.
.

MessageId=0x26b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_VALUE_SIZE
Language=English
Invalid value size.
.

MessageId=0x26c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_BEING_EXTENDED
Language=English
A file extension operation is already in progress.
.

MessageId=0x26d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION
Language=English
Collisions encountered during graph verification.
.

MessageId=0x26e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION
Language=English
The number of value assignments did not equal the number of keys during graph verification.
.

MessageId=0x26f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_NUMBER_OF_SEEDS
Language=English
Invalid number of seeds.
.

MessageId=0x270
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_BASE_OUTPUT_DIRECTORY_ALREADY_SET
Language=English
Base output directory already set.
.

MessageId=0x271
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_ALREADY_CLOSED
Language=English
The directory has already been closed.
.

MessageId=0x272
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NOT_SET
Language=English
The directory has not been opened yet, or has been closed.
.

MessageId=0x273
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_LOCKED
Language=English
The directory is locked.
.

MessageId=0x274
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_ALREADY_SET
Language=English
The directory is already set.
.

MessageId=0x275
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_DOES_NOT_EXIST
Language=English
Directory does not exist.
.

MessageId=0x276
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_DIRECTORY_OPEN_FLAGS
Language=English
Invalid directory open flags.
.

MessageId=0x277
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_DIRECTORY_CREATE_FLAGS
Language=English
Invalid directory create flags.
.

MessageId=0x278
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NEVER_SET
Language=English
The directory was never set.
.

MessageId=0x279
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_READONLY
Language=English
The directory is readonly.
.

MessageId=0x27a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NO_RENAME_SCHEDULED
Language=English
The directory has not had a rename operation scheduled.
.

MessageId=0x27b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_NOT_CLOSED
Language=English
Directory is not closed.
.

MessageId=0x27c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ALREADY_ADDED_TO_A_DIRECTORY
Language=English
The file has already been added to a directory.
.

MessageId=0x27d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_ADDED_TO_DIFFERENT_DIRECTORY
Language=English
The file was added to a different directory.
.

MessageId=0x27e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_RENAME_ALREADY_SCHEDULED
Language=English
Directory rename already scheduled.
.

MessageId=0x27f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_FILE_NOT_ADDED_TO_DIRECTORY
Language=English
The file has not been added to a directory.
.

MessageId=0x280
Severity=Fail
Facility=ITF
SymbolicName=PH_E_DIRECTORY_CLOSED
Language=English
Directory is closed.
.

MessageId=0x281
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CREATE_RANDOM_OBJECT_NAMES_LENGTH_OF_NAME_TOO_SHORT
Language=English
LengthOfNameInChars parameter too short.
.

MessageId=0x282
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_DLL_FILE
Language=English
Error preparing Dll.vcxproj file.
.

MessageId=0x283
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_SUPPORT_FILE
Language=English
Error preparing C source support file.
.

MessageId=0x284
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_FILE
Language=English
Error preparing C source test file.
.

MessageId=0x285
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_TEST_EXE_FILE
Language=English
Error preparing C source test exe file.
.

MessageId=0x286
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_TEST_EXE_FILE
Language=English
Error preparing TestExe.vcxproj file.
.

MessageId=0x287
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_FILE
Language=English
Error preparing C source benchmark full file.
.

MessageId=0x288
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
Language=English
Error preparing C source benchmark full exe file.
.

MessageId=0x289
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
Language=English
Error preparing BenchmarkFullExe.vcxproj file.
.

MessageId=0x28a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_FILE
Language=English
Error preparing C source benchmark index file.
.

MessageId=0x28b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
Language=English
Error preparing C source benchmark index exe file.
.

MessageId=0x28c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
Language=English
Error preparing BenchmarkIndexExe.vcxproj file.
.

MessageId=0x28d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_TABLE_STATS_TEXT_FILE
Language=English
Error preparing table stats text file.
.

MessageId=0x28e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_FILE
Language=English
Error preparing C header CompiledPerfectHash.h file.
.

MessageId=0x28f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VCPROPS_COMPILED_PERFECT_HASH_FILE
Language=English
Error preparing CompiledPerfectHash.props file.
.

;//
;// Spare IDs: 0x290, 0x291.
;//

MessageId=0x292
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_DLL_FILE
Language=English
Error saving Dll.vcxproj file.
.

MessageId=0x293
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_SUPPORT_FILE
Language=English
Error saving C source support file.
.

MessageId=0x294
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_FILE
Language=English
Error saving C source test file.
.

MessageId=0x295
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_TEST_EXE_FILE
Language=English
Error saving C source test exe file.
.

MessageId=0x296
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_TEST_EXE_FILE
Language=English
Error saving TestExe.vcxproj file.
.

MessageId=0x297
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_FILE
Language=English
Error saving C source benchmark full file.
.

MessageId=0x298
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_FULL_EXE_FILE
Language=English
Error saving C source benchmark full exe file.
.

MessageId=0x299
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_FULL_EXE_FILE
Language=English
Error saving BenchmarkFullExe.vcxproj file.
.

MessageId=0x29a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_FILE
Language=English
Error saving C source benchmark index file.
.

MessageId=0x29b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_BENCHMARK_INDEX_EXE_FILE
Language=English
Error saving C source benchmark index exe file.
.

MessageId=0x29c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROJECT_BENCHMARK_INDEX_EXE_FILE
Language=English
Error saving BenchmarkIndexExe.vcxproj file.
.

MessageId=0x29d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_TABLE_STATS_TEXT_FILE
Language=English
Error saving table stats text file.
.

MessageId=0x29e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_FILE
Language=English
Error saving C header CompiledPerfectHash.h file.
.

MessageId=0x29f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VCPROPS_COMPILED_PERFECT_HASH_FILE
Language=English
Error saving CompiledPerfectHash.props file.
.

MessageId=0x300
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_MAIN_WORK_LIST_EMPTY
Language=English
SubmitThreadpoolWork() was called against the main work pool, but no corresponding work item was present on the main work list.
.

MessageId=0x301
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_FILE_WORK_LIST_EMPTY
Language=English
SubmitThreadpoolWork() was called against the file work pool, but no corresponding work item was present on the file work list.
.

MessageId=0x302
Severity=Fail
Facility=ITF
SymbolicName=PH_E_GUARDED_LIST_EMPTY
Language=English
The guarded list is empty.
.

MessageId=0x303
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CHUNK_OP
Language=English
Invalid chunk op.
.

MessageId=0x304
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_CHUNK_STRING
Language=English
Invalid chunk string.
.

MessageId=0x305
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_STDAFX_FILE
Language=English
Error preparing C header stdafx.h file.
.

MessageId=0x306
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_STDAFX_FILE
Language=English
Error saving C header stdafx.h file.
.

MessageId=0x307
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_SOURCE_STDAFX_FILE
Language=English
Error preparing C source stdafx.c file.
.

MessageId=0x308
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_SOURCE_STDAFX_FILE
Language=English
Error saving C source stdafx.c file.
.

MessageId=0x309
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CONTEXT_FILE_ALREADY_PREPARED
Language=English
Context file already prepared.
.

MessageId=0x30a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_VSSOLUTION_FILE
Language=English
Error preparing VS Solution .sln file.
.

MessageId=0x30b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_VSSOLUTION_FILE
Language=English
Error saving VS Solution .sln file.
.

MessageId=0x30c
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_UUID_STRING
Language=English
Invalid UUID string.
.

MessageId=0x30d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_INDEX_IMPL_C_STRING_FOUND
Language=English
No Index() routine raw C string found for the current algorithm, hash function and masking type..
.

MessageId=0x30e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_SUPPORT_FILE
Language=English
Error preparing C header support file.
.

MessageId=0x30f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_SUPPORT_FILE
Language=English
Error saving C header support file.
.

MessageId=0x310
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_PREPARE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
Language=English
Error preparing C header CompiledPerfectHashMacroGlue.h file.
.

MessageId=0x311
Severity=Fail
Facility=ITF
SymbolicName=PH_E_ERROR_DURING_SAVE_C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE
Language=English
Error saving C header CompiledPerfectHashMacroGlue.h file.
.

MessageId=0x312
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_COMPILATION_FAILED
Language=English
Table compilation failed.
.

MessageId=0x313
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_NOT_CREATED
Language=English
Table not created.
.

MessageId=0x314
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_EDGES
Language=English
Too many edges.
.

MessageId=0x315
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_VERTICES
Language=English
Too many vertices.
.

MessageId=0x316
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_BITS_FOR_BITMAP
Language=English
Too many bits for bitmap.
.

MessageId=0x317
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TOO_MANY_TOTAL_EDGES
Language=English
Too many total edges.
.

MessageId=0x318
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_VERTICES_LESS_THAN_OR_EQUAL_NUM_EDGES
Language=English
Number of vertices is less than or equal to the number of edges.
.

MessageId=0x319
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT
Language=English
Create table routine received shutdown event.
.

MessageId=0x31a
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NO_MORE_SEEDS
Language=English
No more seed data available.
.

MessageId=0x31b
Severity=Fail
Facility=ITF
SymbolicName=PH_E_GRAPH_NO_INFO_SET
Language=English
No graph information has been set for graph.
.

;//
;// Disabled 30th Oct 2018: changed to PH_S_TABLE_RESIZE_IMMINENT.
;// MessageId=0x31c
;// Severity=Fail
;// Facility=ITF
;// SymbolicName=PH_E_TABLE_RESIZE_IMMINENT
;// Language=English
;// Table resize imminent.
;// .

MessageId=0x31d
Severity=Fail
Facility=ITF
SymbolicName=PH_E_NUM_TABLE_CREATE_PARAMS_IS_ZERO_BUT_PARAMS_POINTER_NOT_NULL
Language=English
The number of table create parameters is zero, but table create parameters pointer is not null.
.

MessageId=0x31e
Severity=Fail
Facility=ITF
SymbolicName=PH_E_TABLE_CREATE_PARAMETER_VALIDATION_FAILED
Language=English
Failed to validate one or more table create parameters.
.

MessageId=0x31f
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_PARAMETER_ID
Language=English
Invalid table create parameter ID.
.

MessageId=0x320
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_BEST_COVERAGE_TYPE
Language=English
Invalid best coverage type.
.

MessageId=0x321
Severity=Fail
Facility=ITF
SymbolicName=PH_E_SPARE_GRAPH
Language=English
Operation invalid on spare graph.
.

MessageId=0x322
Severity=Fail
Facility=ITF
SymbolicName=PH_E_GRAPH_INFO_ALREADY_LOADED
Language=English
Graph information already loaded.
.

MessageId=0x323
Severity=Fail
Facility=ITF
SymbolicName=PH_E_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION
Language=English
Create table routine failed to find perfect hash solution.
.

MessageId=0x324
Severity=Fail
Facility=ITF
SymbolicName=PH_E_INVALID_TABLE_CREATE_PARAMETERS_FOR_FIND_BEST_GRAPH
Language=English
Find best graph was requested but one or more mandatory table create parameters were missing or invalid.
.

