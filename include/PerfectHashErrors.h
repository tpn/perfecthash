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
// System call failed.
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

//
// MessageId: PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED
//
// MessageText:
//
// The maximum number of table resize events was reached before a perfect hash
// table solution could be found.
//
#define PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED ((HRESULT)0xE0040229L)

//
// MessageId: PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE
//
// MessageText:
//
// The requested number of table elements was too large.
//
#define PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE ((HRESULT)0xE004022AL)

//
// MessageId: PH_E_ERROR_PREPARING_TABLE_FILE
//
// MessageText:
//
// An error occurred whilst preparing a table file to use for saving the
// perfect hash table solution.
//
#define PH_E_ERROR_PREPARING_TABLE_FILE  ((HRESULT)0xE004022BL)

//
// MessageId: PH_E_ERROR_SAVING_TABLE_FILE
//
// MessageText:
//
// An error occurred whilst trying to save a perfect hash table to
// the previously prepared table file.
//
#define PH_E_ERROR_SAVING_TABLE_FILE     ((HRESULT)0xE004022CL)

//
// MessageId: PH_E_TABLE_VERIFICATION_FAILED
//
// MessageText:
//
// A perfect hash table solution was found, however, it did not pass internal
// validation checks (e.g. collisions were found when attempting to independently
// verify that the perfect hash function generated no collisions).
//
#define PH_E_TABLE_VERIFICATION_FAILED   ((HRESULT)0xE004022DL)

//
// MessageId: PH_E_TABLE_CROSS_COMPILATION_NOT_AVAILABLE
//
// MessageText:
//
// Table cross-compilation is not available between the current architecture
// and requested architecture.
//
#define PH_E_TABLE_CROSS_COMPILATION_NOT_AVAILABLE ((HRESULT)0xE004022EL)

//
// MessageId: PH_E_INVALID_CPU_ARCH_ID
//
// MessageText:
//
// The CPU architecture ID was invalid.
//
#define PH_E_INVALID_CPU_ARCH_ID         ((HRESULT)0xE004022FL)

//
// MessageId: PH_E_NOT_IMPLEMENTED
//
// MessageText:
//
// Functionality not yet implemented.
//
#define PH_E_NOT_IMPLEMENTED             ((HRESULT)0xE0040230L)

//
// MessageId: PH_E_WORK_IN_PROGRESS
//
// MessageText:
//
// This functionality is actively undergoing development and is not yet working.
//
#define PH_E_WORK_IN_PROGRESS            ((HRESULT)0xE0040231L)

//
// MessageId: PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER
//
// MessageText:
//
// The base component of the keys file name (name excluding extension)
// is not a valid C identifier.  (This is caused by file names containing
// characters other than '0-9', 'A-Z', 'a-z' and '_'.)
//
#define PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER ((HRESULT)0xE0040232L)

//
// MessageId: PH_E_ERROR_PREPARING_C_HEADER_FILE
//
// MessageText:
//
// An error occurred whilst preparing a C header file to use for saving the
// perfect hash table solution.
//
#define PH_E_ERROR_PREPARING_C_HEADER_FILE ((HRESULT)0xE0040233L)

//
// MessageId: PH_E_ERROR_SAVING_C_HEADER_FILE
//
// MessageText:
//
// An error occurred whilst trying to save a perfect hash table to
// the previously prepared C header file.
//
#define PH_E_ERROR_SAVING_C_HEADER_FILE  ((HRESULT)0xE0040234L)

//
// MessageId: PH_E_UNREACHABLE_CODE
//
// MessageText:
//
// An internal error has occurred; code marked as unreachable has been reached.
//
#define PH_E_UNREACHABLE_CODE            ((HRESULT)0xE0040235L)

//
// MessageId: PH_E_INVARIANT_CHECK_FAILED
//
// MessageText:
//
// An internal error has occurred; an invariant check has failed.
//
#define PH_E_INVARIANT_CHECK_FAILED      ((HRESULT)0xE0040236L)

//
// MessageId: PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE
//
// MessageText:
//
// The calculated C header file size exceeded 4GB.
//
#define PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE ((HRESULT)0xE0040237L)

//
// MessageId: PH_E_CONTEXT_OUTPUT_DIRECTORY_NOT_SET
//
// MessageText:
//
// No output directory has been set for the perfect hash context.
//
#define PH_E_CONTEXT_OUTPUT_DIRECTORY_NOT_SET ((HRESULT)0xE0040238L)

//
// MessageId: PH_E_CONTEXT_LOCKED
//
// MessageText:
//
// The context is locked.
//
#define PH_E_CONTEXT_LOCKED              ((HRESULT)0xE0040239L)

//
// MessageId: PH_E_CONTEXT_RESET_FAILED
//
// MessageText:
//
// Failed to reset context.
//
#define PH_E_CONTEXT_RESET_FAILED        ((HRESULT)0xE004023AL)

//
// MessageId: PH_E_CONTEXT_SET_OUTPUT_DIRECTORY_FAILED
//
// MessageText:
//
// Failed to set context output directory.
//
#define PH_E_CONTEXT_SET_OUTPUT_DIRECTORY_FAILED ((HRESULT)0xE004023BL)

//
// MessageId: PH_E_NO_TABLE_CREATED_OR_LOADED
//
// MessageText:
//
// The table has not been created or loaded.
//
#define PH_E_NO_TABLE_CREATED_OR_LOADED  ((HRESULT)0xE004023CL)

//
// MessageId: PH_E_TABLE_PATHS_ALREADY_INITIALIZED
//
// MessageText:
//
// Paths have already been initialized for this table instance.
//
#define PH_E_TABLE_PATHS_ALREADY_INITIALIZED ((HRESULT)0xE004023DL)

//
// MessageId: PH_E_ERROR_PREPARING_TABLE_INFO_STREAM
//
// MessageText:
//
// An error occurred whilst trying to prepare the perfect hash table's :Info
// stream.
//
#define PH_E_ERROR_PREPARING_TABLE_INFO_STREAM ((HRESULT)0xE004023EL)

//
// MessageId: PH_E_ERROR_SAVING_TABLE_INFO_STREAM
//
// MessageText:
//
// An error occurred whilst trying to save the perfect hash table's :Info
// stream.
//
#define PH_E_ERROR_SAVING_TABLE_INFO_STREAM ((HRESULT)0xE004023FL)

//
// MessageId: PH_E_ERROR_SAVING_C_SOURCE_FILE
//
// MessageText:
//
// An error occurred whilst trying to save the C source file for the perfect
// hash table solution.
//
#define PH_E_ERROR_SAVING_C_SOURCE_FILE  ((HRESULT)0xE0040240L)

//
// MessageId: PH_E_ERROR_SAVING_C_SOURCE_KEYS_FILE
//
// MessageText:
//
// An error occurred whilst trying to save the C source keys file for the perfect
// hash table solution.
//
#define PH_E_ERROR_SAVING_C_SOURCE_KEYS_FILE ((HRESULT)0xE0040241L)

//
// MessageId: PH_E_ERROR_SAVING_C_SOURCE_TABLE_DATA_FILE
//
// MessageText:
//
// An error occurred whilst trying to save the C source table data file for the
// perfect hash table solution.
//
#define PH_E_ERROR_SAVING_C_SOURCE_TABLE_DATA_FILE ((HRESULT)0xE0040242L)

//
// MessageId: PH_E_FILE_CLOSED
//
// MessageText:
//
// The file has already been closed.
//
#define PH_E_FILE_CLOSED                 ((HRESULT)0xE0040243L)

//
// MessageId: PH_E_FILE_NOT_OPEN
//
// MessageText:
//
// The file has not been opened yet, or has been closed.
//
#define PH_E_FILE_NOT_OPEN               ((HRESULT)0xE0040244L)

//
// MessageId: PH_E_FILE_LOCKED
//
// MessageText:
//
// The file is locked.
//
#define PH_E_FILE_LOCKED                 ((HRESULT)0xE0040245L)

//
// MessageId: PH_E_KEYS_LOCKED
//
// MessageText:
//
// The keys are locked.
//
#define PH_E_KEYS_LOCKED                 ((HRESULT)0xE0040246L)

//
// MessageId: PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE
//
// MessageText:
//
// The mapping size provided is less than or equal to the current file size.
//
#define PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE ((HRESULT)0xE0040247L)

//
// MessageId: PH_E_FILE_READONLY
//
// MessageText:
//
// The file is readonly.
//
#define PH_E_FILE_READONLY               ((HRESULT)0xE0040248L)

//
// MessageId: PH_E_FILE_VIEW_CREATED
//
// MessageText:
//
// A file view has already been created.
//
#define PH_E_FILE_VIEW_CREATED           ((HRESULT)0xE0040249L)

//
// MessageId: PH_E_FILE_VIEW_MAPPED
//
// MessageText:
//
// A file view has already been mapped.
//
#define PH_E_FILE_VIEW_MAPPED            ((HRESULT)0xE004024AL)

//
// MessageId: PH_E_FILE_MAPPING_SIZE_IS_ZERO
//
// MessageText:
//
// The mapping size for the file is zero.
//
#define PH_E_FILE_MAPPING_SIZE_IS_ZERO   ((HRESULT)0xE004024BL)

//
// MessageId: PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED
//
// MessageText:
//
// The mapping size for the file is not aligned to the system allocation
// granularity.
//
#define PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED ((HRESULT)0xE004024CL)

//
// MessageId: PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED
//
// MessageText:
//
// The mapping size for the file is not aligned to the large page granularity,
// and large pages have been requested.
//
#define PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED ((HRESULT)0xE004024DL)

//
// MessageId: PH_E_FILE_ALREADY_OPEN
//
// MessageText:
//
// An existing file has already been loaded or created for this file instance.
//
#define PH_E_FILE_ALREADY_OPEN           ((HRESULT)0xE004024EL)

//
// MessageId: PH_E_INVALID_FILE_LOAD_FLAGS
//
// MessageText:
//
// Invalid file load flags.
//
#define PH_E_INVALID_FILE_LOAD_FLAGS     ((HRESULT)0xE004024FL)

//
// MessageId: PH_E_FILE_ALREADY_CLOSED
//
// MessageText:
//
// An existing file was already loaded or created and then subsequently closed
// for this file instance.
//
#define PH_E_FILE_ALREADY_CLOSED         ((HRESULT)0xE0040250L)

//
// MessageId: PH_E_FILE_EMPTY
//
// MessageText:
//
// The file is empty.
//
#define PH_E_FILE_EMPTY                  ((HRESULT)0xE0040251L)

//
// MessageId: PH_E_PATH_PARTS_EXTRACTION_FAILED
//
// MessageText:
//
// Failed to extract the path into parts.
//
#define PH_E_PATH_PARTS_EXTRACTION_FAILED ((HRESULT)0xE0040252L)

//
// MessageId: PH_E_PATH_LOCKED
//
// MessageText:
//
// Path is locked.
//
#define PH_E_PATH_LOCKED                 ((HRESULT)0xE0040253L)

//
// MessageId: PH_E_EXISTING_PATH_LOCKED
//
// MessageText:
//
// Existing path parameter is locked.
//
#define PH_E_EXISTING_PATH_LOCKED        ((HRESULT)0xE0040254L)

//
// MessageId: PH_E_PATH_ALREADY_SET
//
// MessageText:
//
// A path has already been set for this instance.
//
#define PH_E_PATH_ALREADY_SET            ((HRESULT)0xE0040255L)

//
// MessageId: PH_E_EXISTING_PATH_NO_PATH_SET
//
// MessageText:
//
// Existing path parameter has not had a path set.
//
#define PH_E_EXISTING_PATH_NO_PATH_SET   ((HRESULT)0xE0040256L)

//
// MessageId: PH_E_NO_PATH_SET
//
// MessageText:
//
// No path set.
//
#define PH_E_NO_PATH_SET                 ((HRESULT)0xE0040257L)

//
// MessageId: PH_E_STRING_BUFFER_OVERFLOW
//
// MessageText:
//
// An internal string buffer has overflowed.
//
#define PH_E_STRING_BUFFER_OVERFLOW      ((HRESULT)0xE0040258L)

//
// MessageId: PH_E_SOURCE_PATH_LOCKED
//
// MessageText:
//
// Source path parameter is locked.
//
#define PH_E_SOURCE_PATH_LOCKED          ((HRESULT)0xE0040259L)

//
// MessageId: PH_E_SOURCE_PATH_NO_PATH_SET
//
// MessageText:
//
// Source path parameter had no path set.
//
#define PH_E_SOURCE_PATH_NO_PATH_SET     ((HRESULT)0xE004025AL)

//
// MessageId: PH_E_INVALID_TABLE
//
// MessageText:
//
// Invalid table.
//
#define PH_E_INVALID_TABLE               ((HRESULT)0xE004025BL)

//
// MessageId: PH_E_INVALID_TABLE_CREATE_FLAGS
//
// MessageText:
//
// Invalid table create flags.
//
#define PH_E_INVALID_TABLE_CREATE_FLAGS  ((HRESULT)0xE004025CL)

//
// MessageId: PH_E_ERROR_PREPARING_C_SOURCE_FILE
//
// MessageText:
//
// An error occurred whilst preparing the C source file.
//
#define PH_E_ERROR_PREPARING_C_SOURCE_FILE ((HRESULT)0xE004025DL)

//
// MessageId: PH_E_ERROR_PREPARING_C_SOURCE_KEYS_FILE
//
// MessageText:
//
// An error occurred whilst preparing the C source keys file.
//
#define PH_E_ERROR_PREPARING_C_SOURCE_KEYS_FILE ((HRESULT)0xE004025EL)

//
// MessageId: PH_E_ERROR_PREPARING_C_SOURCE_TABLE_DATA_FILE
//
// MessageText:
//
// An error occurred whilst preparing the C source table data file.
//
#define PH_E_ERROR_PREPARING_C_SOURCE_TABLE_DATA_FILE ((HRESULT)0xE004025FL)

//
// MessageId: PH_E_FILE_NEVER_OPENED
//
// MessageText:
//
// The file has never been opened.
//
#define PH_E_FILE_NEVER_OPENED           ((HRESULT)0xE0040260L)

//
// MessageId: PH_E_FILE_NO_RENAME_SCHEDULED
//
// MessageText:
//
// The file has not had a rename operation scheduled.
//
#define PH_E_FILE_NO_RENAME_SCHEDULED    ((HRESULT)0xE0040261L)

//
// MessageId: PH_E_FILE_NOT_CLOSED
//
// MessageText:
//
// The file has not yet been closed.
//
#define PH_E_FILE_NOT_CLOSED             ((HRESULT)0xE0040262L)

//
// MessageId: PH_E_INVALID_FILE_WORK_ID
//
// MessageText:
//
// Invalid file work ID.
//
#define PH_E_INVALID_FILE_WORK_ID        ((HRESULT)0xE0040263L)

//
// MessageId: PH_E_INVALID_END_OF_FILE
//
// MessageText:
//
// Invalid end of file (less than or equal to 0).
//
#define PH_E_INVALID_END_OF_FILE         ((HRESULT)0xE0040264L)

//
// MessageId: PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF
//
// MessageText:
//
// New end-of-file is less than or equal to current end-of-file.
//
#define PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF ((HRESULT)0xE0040265L)

//
// MessageId: PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH
//
// MessageText:
//
// The rename path equivalent to the existing path.
//
#define PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH ((HRESULT)0xE0040266L)

//
// MessageId: PH_E_INVALID_FILE_CREATE_FLAGS
//
// MessageText:
//
// Invalid file create flags.
//
#define PH_E_INVALID_FILE_CREATE_FLAGS   ((HRESULT)0xE0040267L)

//
// MessageId: PH_E_INVALID_INTERFACE_ID
//
// MessageText:
//
// Invalid interface ID.
//
#define PH_E_INVALID_INTERFACE_ID        ((HRESULT)0xE0040268L)

//
// MessageId: PH_E_NO_TLS_CONTEXT_SET
//
// MessageText:
//
// PerfectHashTlsEnsureContext() was called but no TLS context was set.
//
#define PH_E_NO_TLS_CONTEXT_SET          ((HRESULT)0xE0040269L)

//
// MessageId: PH_E_NOT_GLOBAL_INTERFACE_ID
//
// MessageText:
//
// The interface ID provided is not a global interface.
//
#define PH_E_NOT_GLOBAL_INTERFACE_ID     ((HRESULT)0xE004026AL)

//
// MessageId: PH_E_INVALID_VALUE_SIZE
//
// MessageText:
//
// Invalid value size.
//
#define PH_E_INVALID_VALUE_SIZE          ((HRESULT)0xE004026BL)

//
// MessageId: PH_E_FILE_ALREADY_BEING_EXTENDED
//
// MessageText:
//
// A file extension operation is already in progress.
//
#define PH_E_FILE_ALREADY_BEING_EXTENDED ((HRESULT)0xE004026CL)

