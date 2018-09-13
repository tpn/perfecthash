/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashErrors.h

Abstract:

    This is the public header file for error codes used by the perfect
    hash library.  It is automatically generated.

--*/
//
//  Values are 32 bit values laid out as follows:
//
//   3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
//   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
//  +-+-+-+-+-+---------------------+-------------------------------+
//  |S|R|C|N|r|    Facility         |               Code            |
//  +-+-+-+-+-+---------------------+-------------------------------+
//
//  where
//
//      S - Severity - indicates success/fail
//
//          0 - Success
//          1 - Fail (COERROR)
//
//      R - reserved portion of the facility code, corresponds to NT's
//              second severity bit.
//
//      C - reserved portion of the facility code, corresponds to NT's
//              C field.
//
//      N - reserved portion of the facility code. Used to indicate a
//              mapped NT status value.
//
//      r - reserved portion of the facility code. Reserved for internal
//              use. Used to indicate HRESULT values that are not status
//              values, but are instead message ids for display strings.
//
//      Facility - is the facility code
//
//      Code - is the facility's status code
//
//
// Define the facility codes
//
#define PH_FACILITY_ITF                  0x4


//
// Define the severity codes
//
#define PH_SEVERITY_WARNING              0x2
#define PH_SEVERITY_SUCCESS              0x0
#define PH_SEVERITY_INFORMATIONAL        0x1
#define PH_SEVERITY_FAIL                 0x3


//
// MessageId: PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS
//
// MessageText:
//
// A table creation operation is in progress for this context.
//
#define PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS ((HRESULT)0xE0040201L)

//
// MessageId: PH_E_TOO_MANY_KEYS
//
// MessageText:
//
// Too many keys.
//
#define PH_E_TOO_MANY_KEYS               ((HRESULT)0xE0040202L)

//
// MessageId: PH_E_INFO_FILE_SMALLER_THAN_HEADER
//
// MessageText:
//
// :Info file is smaller than smallest known table header size.
//
#define PH_E_INFO_FILE_SMALLER_THAN_HEADER ((HRESULT)0xE0040203L)

//
// MessageId: PH_E_INVALID_MAGIC_VALUES
//
// MessageText:
//
// Invalid magic values.
//
#define PH_E_INVALID_MAGIC_VALUES        ((HRESULT)0xE0040204L)

//
// MessageId: PH_E_INVALID_INFO_HEADER_SIZE
//
// MessageText:
//
// Invalid :Info header size.
//
#define PH_E_INVALID_INFO_HEADER_SIZE    ((HRESULT)0xE0040205L)

//
// MessageId: PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS
//
// MessageText:
//
// The number of keys reported in the keys file does not match the number of keys reported in the header.
//
#define PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS ((HRESULT)0xE0040206L)

//
// MessageId: PH_E_INVALID_ALGORITHM_ID
//
// MessageText:
//
// Invalid algorithm ID.
//
#define PH_E_INVALID_ALGORITHM_ID        ((HRESULT)0xE0040207L)

//
// MessageId: PH_E_INVALID_HASH_FUNCTION_ID
//
// MessageText:
//
// Invalid hash function ID.
//
#define PH_E_INVALID_HASH_FUNCTION_ID    ((HRESULT)0xE0040208L)

//
// MessageId: PH_E_INVALID_MASK_FUNCTION_ID
//
// MessageText:
//
// Invalid mask function ID.
//
#define PH_E_INVALID_MASK_FUNCTION_ID    ((HRESULT)0xE0040209L)

//
// MessageId: PH_E_HEADER_KEY_SIZE_TOO_LARGE
//
// MessageText:
//
// The key size reported by the header is too large.
//
#define PH_E_HEADER_KEY_SIZE_TOO_LARGE   ((HRESULT)0xE004020AL)

//
// MessageId: PH_E_NUM_KEYS_IS_ZERO
//
// MessageText:
//
// The number of keys is zero.
//
#define PH_E_NUM_KEYS_IS_ZERO            ((HRESULT)0xE004020BL)

//
// MessageId: PH_E_NUM_TABLE_ELEMENTS_IS_ZERO
//
// MessageText:
//
// The number of table elements is zero.
//
#define PH_E_NUM_TABLE_ELEMENTS_IS_ZERO  ((HRESULT)0xE004020CL)

//
// MessageId: PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS
//
// MessageText:
//
// The number of keys exceeds the number of table elements.
//
#define PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS ((HRESULT)0xE004020DL)

//
// MessageId: PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH
//
// MessageText:
//
// The expected end of file does not match the actual end of file.
//
#define PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH ((HRESULT)0xE004020EL)

//
// MessageId: PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE
//
// MessageText:
//
// The keys file size is not a multiple of the key size.
//
#define PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE ((HRESULT)0xE004020FL)

//
// MessageId: PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH
//
// MessageText:
//
// The number of bits set for the keys bitmap does not match the number of keys.
//
#define PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH ((HRESULT)0xE0040210L)

//
// MessageId: PH_E_DUPLICATE_KEYS_DETECTED
//
// MessageText:
//
// Duplicate keys detected.  Key files must not contain duplicate keys.
//
#define PH_E_DUPLICATE_KEYS_DETECTED     ((HRESULT)0xE0040211L)

//
// MessageId: PH_E_HEAP_CREATE_FAILED
//
// MessageText:
//
// A call to HeapCreate() failed.
//
#define PH_E_HEAP_CREATE_FAILED          ((HRESULT)0xE0040212L)

//
// MessageId: PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED
//
// MessageText:
//
// A call to RtlLoadSymbolsFromMultipleModules() failed.
//
#define PH_E_RTL_LOAD_SYMBOLS_FROM_MULTIPLE_MODULES_FAILED ((HRESULT)0xE0040213L)

//
// MessageId: PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS
//
// MessageText:
//
// Invalid number of arguments for context self-test.
//
#define PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS ((HRESULT)0xE0040214L)

//
// MessageId: PH_E_INVALID_MAXIMUM_CONCURRENCY
//
// MessageText:
//
// Invalid value for maximum concurrency.
//
#define PH_E_INVALID_MAXIMUM_CONCURRENCY ((HRESULT)0xE0040215L)

//
// MessageId: PH_E_SET_MAXIMUM_CONCURRENCY_FAILED
//
// MessageText:
//
// Setting the maximum concurrency of the context failed.
//
#define PH_E_SET_MAXIMUM_CONCURRENCY_FAILED ((HRESULT)0xE0040216L)

//
// MessageId: PH_E_INITIALIZE_LARGE_PAGES_FAILED
//
// MessageText:
//
// Internal error when attempting to initialize large pages.
//
#define PH_E_INITIALIZE_LARGE_PAGES_FAILED ((HRESULT)0xE0040217L)

//
// MessageId: PH_E_KEYS_NOT_SORTED
//
// MessageText:
//
// The keys file supplied was not sorted.
//
#define PH_E_KEYS_NOT_SORTED             ((HRESULT)0xE0040218L)

//
// MessageId: PH_E_KEYS_NOT_LOADED
//
// MessageText:
//
// A keys file has not been loaded yet.
//
#define PH_E_KEYS_NOT_LOADED             ((HRESULT)0xE0040219L)

//
// MessageId: PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS
//
// MessageText:
//
// A key loading operation is already in progress.
//
#define PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS ((HRESULT)0xE004021AL)

//
// MessageId: PH_E_KEYS_ALREADY_LOADED
//
// MessageText:
//
// A set of keys has already been loaded.
//
#define PH_E_KEYS_ALREADY_LOADED         ((HRESULT)0xE004021BL)

//
// MessageId: PH_E_INVALID_KEYS_LOAD_FLAGS
//
// MessageText:
//
// Invalid key load flags.
//
#define PH_E_INVALID_KEYS_LOAD_FLAGS     ((HRESULT)0xE004021CL)

//
// MessageId: PH_E_INVALID_KEY_SIZE
//
// MessageText:
//
// Invalid key size.
//
#define PH_E_INVALID_KEY_SIZE            ((HRESULT)0xE004021DL)

//
// MessageId: PH_E_TABLE_NOT_LOADED
//
// MessageText:
//
// No table has been loaded yet.
//
#define PH_E_TABLE_NOT_LOADED            ((HRESULT)0xE004021EL)

//
// MessageId: PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS
//
// MessageText:
//
// A table loading operation is already in progress.
//
#define PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS ((HRESULT)0xE004021FL)

//
// MessageId: PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS
//
// MessageText:
//
// Invalid context create table flags.
//
#define PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS ((HRESULT)0xE0040220L)

//
// MessageId: PH_E_INVALID_CONTEXT_SELF_TEST_FLAGS
//
// MessageText:
//
// Invalid context self-test flags.
//
#define PH_E_INVALID_CONTEXT_SELF_TEST_FLAGS ((HRESULT)0xE0040221L)

//
// MessageId: PH_E_INVALID_TABLE_LOAD_FLAGS
//
// MessageText:
//
// Invalid table load flags.
//
#define PH_E_INVALID_TABLE_LOAD_FLAGS    ((HRESULT)0xE0040222L)

//
// MessageId: PH_E_TABLE_LOCKED
//
// MessageText:
//
// Table is locked.
//
#define PH_E_TABLE_LOCKED                ((HRESULT)0xE0040223L)

//
// MessageId: PH_E_SYSTEM_CALL_FAILED
//
// MessageText:
//
// A routine supplied by the operating system has indicated failure.
//
#define PH_E_SYSTEM_CALL_FAILED          ((HRESULT)0xE0040224L)

//
// MessageId: PH_E_TABLE_ALREADY_CREATED
//
// MessageText:
//
// The table instance has already been created.
//
#define PH_E_TABLE_ALREADY_CREATED       ((HRESULT)0xE0040225L)

//
// MessageId: PH_E_TABLE_ALREADY_LOADED
//
// MessageText:
//
// The table instance has already been loaded.
//
#define PH_E_TABLE_ALREADY_LOADED        ((HRESULT)0xE0040226L)

//
// MessageId: PH_E_INVALID_TABLE_COMPILE_FLAGS
//
// MessageText:
//
// Invalid table compile flags.
//
#define PH_E_INVALID_TABLE_COMPILE_FLAGS ((HRESULT)0xE0040227L)

//
// MessageId: PH_E_TABLE_COMPILATION_NOT_AVAILABLE
//
// MessageText:
//
// Table compilation is not available for the current combination of
// architecture, algorithm ID, hash function and masking type.
//
#define PH_E_TABLE_COMPILATION_NOT_AVAILABLE ((HRESULT)0xE0040228L)

