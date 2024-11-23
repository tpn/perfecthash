#===============================================================================
# Imports
#===============================================================================
from .path import (
    dirname,
    basename,
    join_path,
)

from .util import (
    filetime_utc_to_local_tz,
    datetime_to_perfecthash_time,
)

#===============================================================================
# Globals
#===============================================================================

HASH_FUNCTIONS = (
    'Jenkins',
    'MultiplyRotateLR',
    'MultiplyRotateR',
    'RotateRMultiply',
    'MultiplyRotateR2',
    'MultiplyRotateRMultiply',
    'RotateMultiplyXorRotate',
    'RotateMultiplyXorRotate2',
    'RotateRMultiplyRotateR',
    'MultiplyShiftR',
    'MultiplyShift643R',
    'MultiplyShift644R',
    'MultiplyShiftR2',
    'MultiplyShiftRMultiply',
    'ShiftMultiplyXorShift',
    'ShiftMultiplyXorShift2',
    'Crc32RotateX',
    'Crc32RotateXY',
    'Crc32RotateWXYZ',
    'Fnv',
)

BEST_COVERAGE_TYPES = (
    'HighestNumberOfEmptyPages',
    'LowestNumberOfEmptyPages',
    'HighestNumberOfEmptyLargePages',
    'LowestNumberOfEmptyLargePages',
    'HighestNumberOfEmptyCacheLines',
    'LowestNumberOfEmptyCacheLines',
    'HighestNumberOfUsedPages',
    'LowestNumberOfUsedPages',
    'HighestNumberOfUsedLargePages',
    'LowestNumberOfUsedLargePages',
    'HighestNumberOfUsedCacheLines',
    'LowestNumberOfUsedCacheLines',
    'HighestMaxGraphTraversalDepth',
    'LowestMaxGraphTraversalDepth',
    'HighestTotalGraphTraversals',
    'LowestTotalGraphTraversals',
    'HighestNumberOfEmptyVertices',
    'LowestNumberOfEmptyVertices',
    'HighestNumberOfCollisionsDuringAssignment',
    'LowestNumberOfCollisionsDuringAssignment',
    'HighestMaxAssignedPerCacheLineCount',
    'LowestMaxAssignedPerCacheLineCount',
    'HighestPredictedNumberOfFilledCacheLines',
    'LowestPredictedNumberOfFilledCacheLines',
    'HighestSlope',
    'LowestSlope',
    'HighestScore',
    'LowestScore',
    'HighestRank',
    'LowestRank',
)

BEST_COVERAGE_TYPES_KEYS_SUBSET = (
    'HighestMaxAssignedPerCacheLineCountForKeysSubset',
    'LowestMaxAssignedPerCacheLineCountForKeysSubset',
    'HighestNumberOfCacheLinesUsedByKeysSubset',
    'LowestNumberOfCacheLinesUsedByKeysSubset',
    'HighestNumberOfLargePagesUsedByKeysSubset',
    'LowestNumberOfLargePagesUsedByKeysSubset',
    'HighestNumberOfPagesUsedByKeysSubset',
    'LowestNumberOfPagesUsedByKeysSubset',
)

KEYS_SLIM_1 = [
    'KeysName',
    'HashFunction',
    'RngName',
    'BestCoverageType',
    'KeysToEdgesRatio',
    'KeysToVerticesRatio',
    'SolutionsFoundRatio',
    'NumberOfKeys',
    'NumberOfEdges',
    'NumberOfVertices',
    'NumberOfTableResizeEvents',
    'Attempts',
    'FailedAttempts',
    'VertexCollisionFailures',
    'CyclicGraphFailures',
    'BestCoverageAttempts',
    'ClampNumberOfEdges',
    'HasSeedMaskCounts',
    'SolutionFound',
    'NumberOfSolutionsFound',
    'NewBestGraphCount',
    'AttemptThatFoundBestGraph',
    'ComputerName',
    'CpuBrand',
    'ContextTimestamp',
    'TableTimestamp',
    'Version',
    'SolveMicroseconds',
    'VerifyMicroseconds',
    'DeltaHashMinimumCycles',
    'AddKeysElapsedCycles',
    'HashKeysElapsedCycles',
    'AddHashedKeysElapsedCycles',
    'AddKeysElapsedMicroseconds',
    'HashKeysElapsedMicroseconds',
    'AddHashedKeysElapsedMicroseconds',
    'RngId',
    'RngFlags',
    'RngStartSeed',
    'RngStartSubsequence',
    'RngStartOffset',
    'RngWinningSeed',
    'RngWinningSubsequence',
    'RngWinningOffset',
    'CommandLineW',
]

KEYS_BEST_COVERAGE_1 = [
    'BestCoverageSlope',
    'BestCoverageIntercept',
    'BestCoverageRValue',
    'BestCoveragePValue',
    'BestCoverageStdErr',
    'BestCoverageSlope',
    'BestCoverageValue',
    'BestCoverageEqualCount',
]

KEYS_SEEDS = [
    'NumberOfSeeds',
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed3_Byte1',
    'Seed3_Byte2',
    'Seed3_Byte3',
    'Seed3_Byte4',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed6_Byte1',
    'Seed6_Byte2',
    'Seed6_Byte3',
    'Seed6_Byte4',
    'Seed7',
    'Seed8',
    'NumberOfUserSeeds',
    'UserSeed1',
    'UserSeed2',
    'UserSeed3',
    'UserSeed4',
    'UserSeed5',
    'UserSeed6',
    'UserSeed7',
    'UserSeed8',
    'SeedMask1',
    'SeedMask2',
    'SeedMask3',
    'SeedMask4',
    'SeedMask5',
    'SeedMask6',
    'SeedMask7',
    'SeedMask8',
    'Seed3Byte1MaskCounts',
    'Seed3Byte2MaskCounts',
]

KEYS_SEED_N = [
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed3_Byte1',
    'Seed3_Byte2',
    'Seed3_Byte3',
    'Seed3_Byte4',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed6_Byte1',
    'Seed6_Byte2',
    'Seed6_Byte3',
    'Seed6_Byte4',
    'Seed7',
    'Seed8',
]

KEYS_USER_SEED_N = [
    'UserSeed1',
    'UserSeed2',
    'UserSeed3',
    'UserSeed4',
    'UserSeed5',
    'UserSeed6',
    'UserSeed7',
    'UserSeed8',
]

KEYS_SEED_MASK_N = [
    'SeedMask1',
    'SeedMask2',
    'SeedMask3',
    'SeedMask4',
    'SeedMask5',
    'SeedMask6',
    'SeedMask7',
    'SeedMask8',
]

BOKEH_GLYPHS = [
    'circle',
    'diamond',
    'asterisk',
    'cross',
    'triangle',
    'x',
]


#===============================================================================
# ETW-related Globals (System Events)
#===============================================================================

# Context Switch

CSWITCH = '                CSwitch'
CSWITCH_ETW_HEADER = (
    '                CSwitch,'
    '  TimeStamp,'
    ' New Process Name ( PID),'
    '    New TID,'
    ' NPri,'
    ' NQnt,'
    ' TmSinceLast,'
    ' WaitTime,'
    ' Old Process Name ( PID),'
    '    Old TID,'
    ' OPri,'
    ' OQnt,'
    '        OldState,'
    '      Wait Reason,'
    ' Swapable,'
    ' InSwitchTime,'
    ' CPU,'
    ' IdealProc,'
    '  OldRemQnt,'
    ' NewPriDecr,'
    ' PrevCState,'
    ' OldThrdBamQosLevel,'
    ' NewThrdBamQosLevel'
)

CSWITCH_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'NewProcessName(PID)',
    'NewTID',
    'NPri',
    'NQnt',
    'TmSinceLast',
    'WaitTime',
    'OldProcessName(PID)',
    'OldTID',
    'OPri',
    'OQnt',
    'OldState',
    'WaitReason',
    'Swapable',
    'InSwitchTime',
    'CPU',
    'IdealProc',
    'OldRemQnt',
    'NewPriDecr',
    'PrevCState',
    'OldThrdBamQosLevel',
    'NewThrdBamQosLevel',
)

CSWITCH_CSV_HEADER_SLIM = (
    'LineNumber',
    'TimeStamp',
    'NewProcessName(PID)',
    'NewTID',
    'NPri',
    'NQnt',
    'TmSinceLast',
    'WaitTime',
    'OldProcessName(PID)',
    'OldTID',
    'OPri',
    'OQnt',
    'OldState',
    'WaitReason',
    'Swapable',
    'InSwitchTime',
    'CPU',
    'IdealProc',
    'OldRemQnt',
    'NewPriDecr',
    'PrevCState',
    'OldThrdBamQosLevel',
    'NewThrdBamQosLevel',
)

# Performance Counters

PMC = '                    Pmc'
PMC_ETW_HEADER_STARTSWITH = '                    Pmc,  TimeStamp,   ThreadID,'

#===============================================================================
# ETW-related Globals (Our Events)
#===============================================================================

# AddKeys

ADD_KEYS = 'PerfectHash/AddKeys/win:Info'

ADD_KEYS_ETW_HEADER = (
    'PerfectHash/AddKeys/win:Info,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' KeysProcessed,'
    ' NumberOfKeys,'
    ' LastKey,'
    ' Result,'
    ' Cycles,'
    ' Microseconds,'
    ' Seed1,'
    ' Seed2,'
    ' Seed3,'
    ' Seed4,'
    ' Seed5,'
    ' Seed6,'
    ' Seed7,'
    ' Seed8'
)

ADD_KEYS_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'KeysProcessed',
    'NumberOfKeys',
    'LastKey',
    'Result',
    'Cycles',
    'Microseconds',
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed7',
    'Seed8',
)

ADD_KEYS_CSV_HEADER_SLIM = (
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'KeysProcessed',
    'NumberOfKeys',
    'LastKey',
    'Success',
    'Cycles',
    'Microseconds',
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed7',
    'Seed8',
)

# HashKeys

HASH_KEYS = 'PerfectHash/HashKeys/win:Info'

HASH_KEYS_ETW_HEADER = (
    'PerfectHash/HashKeys/win:Info,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' KeysProcessed,'
    ' NumberOfKeys,'
    ' LastKey,'
    ' Result,'
    ' Cycles,'
    ' Microseconds,'
    ' Seed1,'
    ' Seed2,'
    ' Seed3,'
    ' Seed4,'
    ' Seed5,'
    ' Seed6,'
    ' Seed7,'
    ' Seed8'
)

HASH_KEYS_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'KeysProcessed',
    'NumberOfKeys',
    'LastKey',
    'Result',
    'Cycles',
    'Microseconds',
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed7',
    'Seed8',
)

HASH_KEYS_CSV_HEADER_SLIM = (
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'KeysProcessed',
    'NumberOfKeys',
    'LastKey',
    'Success',
    'Cycles',
    'Microseconds',
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed7',
    'Seed8',
)

# AddHashedKeys

ADD_HASHED_KEYS = 'PerfectHash/AddHashedKeys/win:Info'

ADD_HASHED_KEYS_ETW_HEADER = (
    'PerfectHash/AddHashedKeys/win:Info,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' NumberOfKeys,'
    ' Cycles,'
    ' Microseconds'
)

ADD_HASHED_KEYS_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'NumberOfKeys',
    'Cycles',
    'Microseconds',
)

ADD_HASHED_KEYS_CSV_HEADER_SLIM = (
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'NumberOfKeys',
    'Cycles',
    'Microseconds',
)

# FoundNewBestGraph, FoundEqualBestGraph, and FoundGraph all share the same
# template.

FOUND_GRAPH_ETW_HEADER_SHARED = (
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' KeysFileName,'
    ' Attempt,'
    ' SolutionNumber,'
    ' ElapsedMilliseconds,'
    ' CoverageType,'
    ' CoverageValue,'
    ' CoverageValueAsDouble,'
    ' StopGraphSolving,'
    ' IsBest,'
    ' IsEqual,'
    ' IsCoverageValueDouble,'
    ' EqualCount,'
    ' TotalNumberOfPages,'
    ' TotalNumberOfLargePages,'
    ' TotalNumberOfCacheLines,'
    ' NumberOfUsedPages,'
    ' NumberOfUsedLargePages,'
    ' NumberOfUsedCacheLines,'
    ' NumberOfEmptyPages,'
    ' NumberOfEmptyLargePages,'
    ' NumberOfEmptyCacheLines,'
    ' FirstPageUsed,'
    ' FirstLargePageUsed,'
    ' FirstCacheLineUsed,'
    ' LastPageUsed,'
    ' LastLargePageUsed,'
    ' LastCacheLineUsed,'
    ' TotalNumberOfAssigned,'
    ' NumberOfKeysWithVerticesMappingToSamePage,'
    ' NumberOfKeysWithVerticesMappingToSameLargePage,'
    ' NumberOfKeysWithVerticesMappingToSameCacheLine,'
    ' MaxGraphTraversalDepth,'
    ' TotalGraphTraversals,'
    ' Seed1,'
    ' Seed2,'
    ' Seed3,'
    ' Seed4,'
    ' Seed5,'
    ' Seed6,'
    ' Seed7,'
    ' Seed8,'
    ' NumberOfAssignedPerCacheLineCounts_0,'
    ' NumberOfAssignedPerCacheLineCounts_1,'
    ' NumberOfAssignedPerCacheLineCounts_2,'
    ' NumberOfAssignedPerCacheLineCounts_3,'
    ' NumberOfAssignedPerCacheLineCounts_4,'
    ' NumberOfAssignedPerCacheLineCounts_5,'
    ' NumberOfAssignedPerCacheLineCounts_6,'
    ' NumberOfAssignedPerCacheLineCounts_7,'
    ' NumberOfAssignedPerCacheLineCounts_8,'
    ' NumberOfAssignedPerCacheLineCounts_9,'
    ' NumberOfAssignedPerCacheLineCounts_10,'
    ' NumberOfAssignedPerCacheLineCounts_11,'
    ' NumberOfAssignedPerCacheLineCounts_12,'
    ' NumberOfAssignedPerCacheLineCounts_13,'
    ' NumberOfAssignedPerCacheLineCounts_14,'
    ' NumberOfAssignedPerCacheLineCounts_15,'
    ' NumberOfAssignedPerCacheLineCounts_16,'
    ' Slope,'
    ' Intercept,'
    ' CorrelationCoefficient,'
    ' Score,'
    ' Rank'
)

FOUND_GRAPH_CSV_HEADER_SHARED = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'KeysFileName',
    'Attempt',
    'SolutionNumber',
    'ElapsedMilliseconds',
    'CoverageType',
    'CoverageValue',
    'CoverageValueAsDouble',
    'StopGraphSolving',
    'IsBest',
    'IsEqual',
    'IsCoverageValueDouble',
    'EqualCount',
    'TotalNumberOfPages',
    'TotalNumberOfLargePages',
    'TotalNumberOfCacheLines',
    'NumberOfUsedPages',
    'NumberOfUsedLargePages',
    'NumberOfUsedCacheLines',
    'NumberOfEmptyPages',
    'NumberOfEmptyLargePages',
    'NumberOfEmptyCacheLines',
    'FirstPageUsed',
    'FirstLargePageUsed',
    'FirstCacheLineUsed',
    'LastPageUsed',
    'LastLargePageUsed',
    'LastCacheLineUsed',
    'TotalNumberOfAssigned',
    'NumberOfKeysWithVerticesMappingToSamePage',
    'NumberOfKeysWithVerticesMappingToSameLargePage',
    'NumberOfKeysWithVerticesMappingToSameCacheLine',
    'MaxGraphTraversalDepth',
    'TotalGraphTraversals',
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed7',
    'Seed8',
    'NumberOfAssignedPerCacheLineCounts_0',
    'NumberOfAssignedPerCacheLineCounts_1',
    'NumberOfAssignedPerCacheLineCounts_2',
    'NumberOfAssignedPerCacheLineCounts_3',
    'NumberOfAssignedPerCacheLineCounts_4',
    'NumberOfAssignedPerCacheLineCounts_5',
    'NumberOfAssignedPerCacheLineCounts_6',
    'NumberOfAssignedPerCacheLineCounts_7',
    'NumberOfAssignedPerCacheLineCounts_8',
    'NumberOfAssignedPerCacheLineCounts_9',
    'NumberOfAssignedPerCacheLineCounts_10',
    'NumberOfAssignedPerCacheLineCounts_11',
    'NumberOfAssignedPerCacheLineCounts_12',
    'NumberOfAssignedPerCacheLineCounts_13',
    'NumberOfAssignedPerCacheLineCounts_14',
    'NumberOfAssignedPerCacheLineCounts_15',
    'NumberOfAssignedPerCacheLineCounts_16',
    'Slope',
    'Intercept',
    'CorrelationCoefficient',
    'Score',
    'Rank',
)

FOUND_GRAPH_CSV_HEADER_SLIM_SHARED = (
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'KeysFileName',
    'Attempt',
    'SolutionNumber',
    'ElapsedMilliseconds',
    'CoverageType',
    'CoverageValue',
    'CoverageValueAsDouble',
    'StopGraphSolving',
    'IsBest',
    'IsEqual',
    'IsCoverageValueDouble',
    'EqualCount',
    'TotalNumberOfPages',
    'TotalNumberOfLargePages',
    'TotalNumberOfCacheLines',
    'NumberOfUsedPages',
    'NumberOfUsedLargePages',
    'NumberOfUsedCacheLines',
    'NumberOfEmptyPages',
    'NumberOfEmptyLargePages',
    'NumberOfEmptyCacheLines',
    'FirstPageUsed',
    'FirstLargePageUsed',
    'FirstCacheLineUsed',
    'LastPageUsed',
    'LastLargePageUsed',
    'LastCacheLineUsed',
    'TotalNumberOfAssigned',
    'NumberOfKeysWithVerticesMappingToSamePage',
    'NumberOfKeysWithVerticesMappingToSameLargePage',
    'NumberOfKeysWithVerticesMappingToSameCacheLine',
    'MaxGraphTraversalDepth',
    'TotalGraphTraversals',
    'Seed1',
    'Seed2',
    'Seed3',
    'Seed4',
    'Seed5',
    'Seed6',
    'Seed7',
    'Seed8',
    'NumberOfAssignedPerCacheLineCounts_0',
    'NumberOfAssignedPerCacheLineCounts_1',
    'NumberOfAssignedPerCacheLineCounts_2',
    'NumberOfAssignedPerCacheLineCounts_3',
    'NumberOfAssignedPerCacheLineCounts_4',
    'NumberOfAssignedPerCacheLineCounts_5',
    'NumberOfAssignedPerCacheLineCounts_6',
    'NumberOfAssignedPerCacheLineCounts_7',
    'NumberOfAssignedPerCacheLineCounts_8',
    'NumberOfAssignedPerCacheLineCounts_9',
    'NumberOfAssignedPerCacheLineCounts_10',
    'NumberOfAssignedPerCacheLineCounts_11',
    'NumberOfAssignedPerCacheLineCounts_12',
    'NumberOfAssignedPerCacheLineCounts_13',
    'NumberOfAssignedPerCacheLineCounts_14',
    'NumberOfAssignedPerCacheLineCounts_15',
    'NumberOfAssignedPerCacheLineCounts_16',
    'Slope',
    'Intercept',
    'CorrelationCoefficient',
    'Score',
    'Rank',
)

# FoundNewBestGraph

FOUND_NEW_BEST_GRAPH = 'PerfectHash/FoundNewBestGraph/win:Info'

FOUND_NEW_BEST_GRAPH_ETW_HEADER = (
    'PerfectHash/FoundNewBestGraph/win:Info,' +
    FOUND_GRAPH_ETW_HEADER_SHARED
)

FOUND_NEW_BEST_GRAPH_CSV_HEADER = FOUND_GRAPH_CSV_HEADER_SHARED
FOUND_NEW_BEST_GRAPH_CSV_HEADER_SLIM = FOUND_GRAPH_CSV_HEADER_SLIM_SHARED

# FoundEqualBestGraph

FOUND_EQUAL_BEST_GRAPH = 'PerfectHash/FoundEqualBestGraph/win:Info'

FOUND_EQUAL_BEST_GRAPH_ETW_HEADER = (
    'PerfectHash/FoundEqualBestGraph/win:Info,' +
    FOUND_GRAPH_ETW_HEADER_SHARED
)

FOUND_EQUAL_BEST_GRAPH_CSV_HEADER = FOUND_GRAPH_CSV_HEADER_SHARED
FOUND_EQUAL_BEST_GRAPH_CSV_HEADER_SLIM = FOUND_GRAPH_CSV_HEADER_SLIM_SHARED

# FoundGraph

FOUND_GRAPH = 'PerfectHash/FoundGraph/win:Info'

FOUND_GRAPH_ETW_HEADER = (
    'PerfectHash/FoundGraph/win:Info,' +
    FOUND_GRAPH_ETW_HEADER_SHARED
)

FOUND_GRAPH_CSV_HEADER = FOUND_GRAPH_CSV_HEADER_SHARED
FOUND_GRAPH_CSV_HEADER_SLIM = FOUND_GRAPH_CSV_HEADER_SLIM_SHARED

# GraphAssignStart

ASSIGN_START = 'PerfectHash/Assign/win:Start'

ASSIGN_START_ETW_HEADER = (
    'PerfectHash/Assign/win:Start,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' KeysFileName,'
    ' Attempt,'
    ' NumberOfKeys,'
    ' NumberOfVertices'
)

ASSIGN_START_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'KeysFileName',
    'Attempt',
    'NumberOfKeys',
    'NumberOfVertices',
)

ASSIGN_START_CSV_HEADER_SLIM = (
    'EventName',
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'KeysFileName',
    'Attempt',
    'NumberOfKeys',
    'NumberOfVertices',
)

# GraphAssignStop

ASSIGN_STOP = 'PerfectHash/Assign/win:Stop'

ASSIGN_STOP_ETW_HEADER = (
    'PerfectHash/Assign/win:Start,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' KeysFileName,'
    ' Attempt,'
    ' NumberOfKeys,'
    ' NumberOfVertices,'
    ' NumberOfEmptyVertices,'
    ' MaxTraversalDepth,'
    ' TotalTraversals'
)

ASSIGN_STOP_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'KeysFileName',
    'Attempt',
    'NumberOfKeys',
    'NumberOfVertices',
    'NumberOfEmptyVertices',
    'MaxTraversalDepth',
    'TotalTraversals',
)

ASSIGN_STOP_CSV_HEADER_SLIM = (
    'EventName',
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'KeysFileName',
    'Attempt',
    'NumberOfKeys',
    'NumberOfVertices',
    'NumberOfEmptyVertices',
    'MaxTraversalDepth',
    'TotalTraversals',
)

# RandomBytesStart

GENERATE_RANDOM_BYTES_START = 'PerfectHash/GenerateRandomBytes/win:Start'

GENERATE_RANDOM_BYTES_START_ETW_HEADER = (
    'PerfectHash/GenerateRandomBytes/win:Start,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' BytesRequested'
)

GENERATE_RANDOM_BYTES_START_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'BytesRequested',
)

GENERATE_RANDOM_BYTES_START_CSV_HEADER_SLIM = (
    'EventName',
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'BytesRequested',
)

# RandomBytesStop

GENERATE_RANDOM_BYTES_STOP = 'PerfectHash/GenerateRandomBytes/win:Stop'

GENERATE_RANDOM_BYTES_STOP_ETW_HEADER = (
    'PerfectHash/GenerateRandomBytes/win:Stop,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' BytesRequested,'
    ' Result'
)

GENERATE_RANDOM_BYTES_STOP_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'BytesRequested',
    'Result',
)

GENERATE_RANDOM_BYTES_STOP_CSV_HEADER_SLIM = (
    'EventName',
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'BytesRequested',
    'Result',
)

# GraphIsAcyclic

IS_ACYCLIC = 'PerfectHash/IsAcyclic/win:Info'

IS_ACYCLIC_ETW_HEADER = (
    'PerfectHash/IsAcyclic/win:Info,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' KeysFileName,'
    ' Attempt,'
    ' FunctionVersion,'
    ' Cycles,'
    ' Microseconds,'
    ' NumberOfKeys,'
    ' NumberOfVertices,'
    ' IsAcyclic'
)

IS_ACYCLIC_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'KeysFileName',
    'Attempt',
    'FunctionVersion',
    'Cycles',
    'Microseconds',
    'NumberOfKeys',
    'NumberOfVertices',
    'IsAcyclic',
)

IS_ACYCLIC_CSV_HEADER_SLIM = (
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'KeysFileName',
    'Attempt',
    'FunctionVersion',
    'Cycles',
    'Microseconds',
    'NumberOfKeys',
    'NumberOfVertices',
    'IsAcyclic',
)

# MemoryCoverageCacheLineCounts

MEMORY_COVERAGE_CACHE_LINE_COUNTS = (
    'PerfectHash/MemoryCoverageCacheLineCounts/win:Info'
)

MEMORY_COVERAGE_CACHE_LINE_COUNTS_ETW_HEADER = (
    'PerfectHash/MemoryCoverageCacheLineCounts/win:Info,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' KeysFileName,'
    ' Attempt,'
    ' PageIndex,'
    ' TotalNumberOfPages,'
    ' CacheLine_01,'
    ' CacheLine_02,'
    ' CacheLine_03,'
    ' CacheLine_04,'
    ' CacheLine_05,'
    ' CacheLine_06,'
    ' CacheLine_07,'
    ' CacheLine_08,'
    ' CacheLine_09,'
    ' CacheLine_10,'
    ' CacheLine_11,'
    ' CacheLine_12,'
    ' CacheLine_13,'
    ' CacheLine_14,'
    ' CacheLine_15,'
    ' CacheLine_16,'
    ' CacheLine_17,'
    ' CacheLine_18,'
    ' CacheLine_19,'
    ' CacheLine_20,'
    ' CacheLine_21,'
    ' CacheLine_22,'
    ' CacheLine_23,'
    ' CacheLine_24,'
    ' CacheLine_25,'
    ' CacheLine_26,'
    ' CacheLine_27,'
    ' CacheLine_28,'
    ' CacheLine_29,'
    ' CacheLine_30,'
    ' CacheLine_31,'
    ' CacheLine_32,'
    ' CacheLine_33,'
    ' CacheLine_34,'
    ' CacheLine_35,'
    ' CacheLine_36,'
    ' CacheLine_37,'
    ' CacheLine_38,'
    ' CacheLine_39,'
    ' CacheLine_40,'
    ' CacheLine_41,'
    ' CacheLine_42,'
    ' CacheLine_43,'
    ' CacheLine_44,'
    ' CacheLine_45,'
    ' CacheLine_46,'
    ' CacheLine_47,'
    ' CacheLine_48,'
    ' CacheLine_49,'
    ' CacheLine_50,'
    ' CacheLine_51,'
    ' CacheLine_52,'
    ' CacheLine_53,'
    ' CacheLine_54,'
    ' CacheLine_55,'
    ' CacheLine_56,'
    ' CacheLine_57,'
    ' CacheLine_58,'
    ' CacheLine_59,'
    ' CacheLine_60,'
    ' CacheLine_61,'
    ' CacheLine_62,'
    ' CacheLine_63,'
    ' CacheLine_64'
)

MEMORY_COVERAGE_CACHE_LINE_COUNTS_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'KeysFileName',
    'Attempt',
    'PageIndex',
    'TotalNumberOfPages',
    'CacheLine_01',
    'CacheLine_02',
    'CacheLine_03',
    'CacheLine_04',
    'CacheLine_05',
    'CacheLine_06',
    'CacheLine_07',
    'CacheLine_08',
    'CacheLine_09',
    'CacheLine_10',
    'CacheLine_11',
    'CacheLine_12',
    'CacheLine_13',
    'CacheLine_14',
    'CacheLine_15',
    'CacheLine_16',
    'CacheLine_17',
    'CacheLine_18',
    'CacheLine_19',
    'CacheLine_20',
    'CacheLine_21',
    'CacheLine_22',
    'CacheLine_23',
    'CacheLine_24',
    'CacheLine_25',
    'CacheLine_26',
    'CacheLine_27',
    'CacheLine_28',
    'CacheLine_29',
    'CacheLine_30',
    'CacheLine_31',
    'CacheLine_32',
    'CacheLine_33',
    'CacheLine_34',
    'CacheLine_35',
    'CacheLine_36',
    'CacheLine_37',
    'CacheLine_38',
    'CacheLine_39',
    'CacheLine_40',
    'CacheLine_41',
    'CacheLine_42',
    'CacheLine_43',
    'CacheLine_44',
    'CacheLine_45',
    'CacheLine_46',
    'CacheLine_47',
    'CacheLine_48',
    'CacheLine_49',
    'CacheLine_50',
    'CacheLine_51',
    'CacheLine_52',
    'CacheLine_53',
    'CacheLine_54',
    'CacheLine_55',
    'CacheLine_56',
    'CacheLine_57',
    'CacheLine_58',
    'CacheLine_59',
    'CacheLine_60',
    'CacheLine_61',
    'CacheLine_62',
    'CacheLine_63',
    'CacheLine_64',
)

MEMORY_COVERAGE_CACHE_LINE_COUNTS_CSV_HEADER_SLIM = (
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'KeysFileName',
    'Attempt',
    'PageIndex',
    'TotalNumberOfPages',
    'CacheLine_01',
    'CacheLine_02',
    'CacheLine_03',
    'CacheLine_04',
    'CacheLine_05',
    'CacheLine_06',
    'CacheLine_07',
    'CacheLine_08',
    'CacheLine_09',
    'CacheLine_10',
    'CacheLine_11',
    'CacheLine_12',
    'CacheLine_13',
    'CacheLine_14',
    'CacheLine_15',
    'CacheLine_16',
    'CacheLine_17',
    'CacheLine_18',
    'CacheLine_19',
    'CacheLine_20',
    'CacheLine_21',
    'CacheLine_22',
    'CacheLine_23',
    'CacheLine_24',
    'CacheLine_25',
    'CacheLine_26',
    'CacheLine_27',
    'CacheLine_28',
    'CacheLine_29',
    'CacheLine_30',
    'CacheLine_31',
    'CacheLine_32',
    'CacheLine_33',
    'CacheLine_34',
    'CacheLine_35',
    'CacheLine_36',
    'CacheLine_37',
    'CacheLine_38',
    'CacheLine_39',
    'CacheLine_40',
    'CacheLine_41',
    'CacheLine_42',
    'CacheLine_43',
    'CacheLine_44',
    'CacheLine_45',
    'CacheLine_46',
    'CacheLine_47',
    'CacheLine_48',
    'CacheLine_49',
    'CacheLine_50',
    'CacheLine_51',
    'CacheLine_52',
    'CacheLine_53',
    'CacheLine_54',
    'CacheLine_55',
    'CacheLine_56',
    'CacheLine_57',
    'CacheLine_58',
    'CacheLine_59',
    'CacheLine_60',
    'CacheLine_61',
    'CacheLine_62',
    'CacheLine_63',
    'CacheLine_64',
)

# FunctionEntry

FUNCTION_ENTRY = (
    'PerfectHash/FunctionEntry/win:Info'
)

FUNCTION_ENTRY_ETW_HEADER = (
    'PerfectHash/FunctionEntry/win:Info,'
    '  TimeStamp,'
    '     Process Name ( PID),'
    '   ThreadID,'
    ' CPU,'
    ' etw:ActivityId,'
    ' etw:Related ActivityId,'
    ' etw:UserSid,'
    ' etw:SessionId,'
    ' Rsp'
)

FUNCTION_ENTRY_CSV_HEADER = (
    'EventName',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'RelatedActivityId',
    'UserSid',
    'SessionId',
    'Rsp',
)

FUNCTION_ENTRY_CSV_HEADER_SLIM = (
    'LineNumber',
    'TimeStamp',
    'ProcessID',
    'ThreadID',
    'CPU',
    'ActivityId',
    'Rsp',
)

# Maps

EVENT_NAME_TO_ETW_HEADER = {
    ADD_KEYS: ADD_KEYS_ETW_HEADER,
    HASH_KEYS: HASH_KEYS_ETW_HEADER,
    ADD_HASHED_KEYS: ADD_HASHED_KEYS_ETW_HEADER,
    CSWITCH: CSWITCH_ETW_HEADER,
    PMC: None,
    FOUND_NEW_BEST_GRAPH: FOUND_NEW_BEST_GRAPH_ETW_HEADER,
    FOUND_EQUAL_BEST_GRAPH: FOUND_EQUAL_BEST_GRAPH_ETW_HEADER,
    FOUND_GRAPH: FOUND_GRAPH_ETW_HEADER,
    IS_ACYCLIC: IS_ACYCLIC_ETW_HEADER,
    MEMORY_COVERAGE_CACHE_LINE_COUNTS:
        MEMORY_COVERAGE_CACHE_LINE_COUNTS_ETW_HEADER,
    FUNCTION_ENTRY: FUNCTION_ENTRY_ETW_HEADER,
}

EVENT_NAME_TO_CSV_HEADER = {
    ADD_KEYS: ADD_KEYS_CSV_HEADER,
    HASH_KEYS: HASH_KEYS_CSV_HEADER,
    ADD_HASHED_KEYS: ADD_HASHED_KEYS_CSV_HEADER,
    FOUND_NEW_BEST_GRAPH: FOUND_NEW_BEST_GRAPH_CSV_HEADER,
    ASSIGN_START: ASSIGN_START_CSV_HEADER,
    ASSIGN_STOP: ASSIGN_STOP_CSV_HEADER,
    CSWITCH: CSWITCH_CSV_HEADER,
    PMC: None,
    GENERATE_RANDOM_BYTES_START: GENERATE_RANDOM_BYTES_START_CSV_HEADER,
    GENERATE_RANDOM_BYTES_STOP: GENERATE_RANDOM_BYTES_STOP_CSV_HEADER,
    FOUND_NEW_BEST_GRAPH: FOUND_NEW_BEST_GRAPH_CSV_HEADER,
    FOUND_EQUAL_BEST_GRAPH: FOUND_EQUAL_BEST_GRAPH_CSV_HEADER,
    FOUND_GRAPH: FOUND_GRAPH_CSV_HEADER,
    IS_ACYCLIC: IS_ACYCLIC_CSV_HEADER,
    MEMORY_COVERAGE_CACHE_LINE_COUNTS:
        MEMORY_COVERAGE_CACHE_LINE_COUNTS_CSV_HEADER,
    FUNCTION_ENTRY: FUNCTION_ENTRY_CSV_HEADER,
}

EVENT_NAME_TO_CSV_HEADER_SLIM = {
    ADD_KEYS: ADD_KEYS_CSV_HEADER_SLIM,
    HASH_KEYS: HASH_KEYS_CSV_HEADER_SLIM,
    ADD_HASHED_KEYS: ADD_HASHED_KEYS_CSV_HEADER_SLIM,
    FOUND_NEW_BEST_GRAPH: FOUND_NEW_BEST_GRAPH_CSV_HEADER_SLIM,
    ASSIGN_START: ASSIGN_START_CSV_HEADER_SLIM,
    ASSIGN_STOP: ASSIGN_STOP_CSV_HEADER_SLIM,
    CSWITCH: CSWITCH_CSV_HEADER_SLIM,
    GENERATE_RANDOM_BYTES_START: GENERATE_RANDOM_BYTES_START_CSV_HEADER_SLIM,
    GENERATE_RANDOM_BYTES_STOP: GENERATE_RANDOM_BYTES_STOP_CSV_HEADER_SLIM,
    PMC: None,
    FOUND_NEW_BEST_GRAPH: FOUND_NEW_BEST_GRAPH_CSV_HEADER_SLIM,
    FOUND_EQUAL_BEST_GRAPH: FOUND_EQUAL_BEST_GRAPH_CSV_HEADER_SLIM,
    FOUND_GRAPH: FOUND_GRAPH_CSV_HEADER_SLIM,
    IS_ACYCLIC: IS_ACYCLIC_CSV_HEADER_SLIM,
    MEMORY_COVERAGE_CACHE_LINE_COUNTS:
        MEMORY_COVERAGE_CACHE_LINE_COUNTS_CSV_HEADER_SLIM,
    FUNCTION_ENTRY: FUNCTION_ENTRY_CSV_HEADER_SLIM,
}

HAS_SEED_DATA = {
    ADD_KEYS,
    HASH_KEYS,
    FOUND_NEW_BEST_GRAPH,
    FOUND_EQUAL_BEST_GRAPH,
    FOUND_GRAPH,
}

#===============================================================================
# Helper Functions
#===============================================================================

def get_num_zeros_fraction(p):
    # I'm sure there's a more efficient way of doing this.
    p = f'{p:.9f}'
    ix = p.find('.') + 1
    zeros = (len(p) - ix) - len(p[ix:].lstrip('0'))
    return zeros

def predict_attempts(probability, target=0.5):
    import numpy as np
    from scipy.stats import geom
    zeros = get_num_zeros_fraction(probability) or 1
    max_attempts = 10 ** (zeros + 1)
    try:
        while True:
            x = np.arange(0, max_attempts)
            y = geom.cdf(x, probability)
            i = y.searchsorted(target)
            if i == max_attempts:
                max_attempts *= max_attempts
            else:
                return i
    except FloatingPointError:
        return np.inf
    except MemoryError:
        return np.inf

def predict_targets(probability, targets=(0.5, 0.75, 0.99, 0.999)):
    return {
        target: predict_attempts(probability, target=target)
            for target in targets
    }

def calculate_predicted_attempts(solutions_found_ratio):
    # Equivalent to C routine CalculatePredictedAttempts().
    if solutions_found_ratio == 1.0:
        return 1

    if not (0.0 < solutions_found_ratio < 1.0):
        raise ValueError(
            "SolutionsFoundRatio must be greater than 0.0 and "
            f"less than or equal to 1.0; actual: {solutions_found_ratio}"
        )

    last_cumulative = 0.0
    success = solutions_found_ratio
    failure = 1 - solutions_found_ratio
    attempt = 1

    while True:
        probability = success * (failure ** (attempt - 1))
        cumulative = attempt * probability
        delta = last_cumulative - cumulative
        if delta > 0.0:
            break
        last_cumulative = cumulative
        attempt += 1

    return attempt

def df_from_csv_with_sys_and_group(path):
    d = dirname(path)
    (sys, group) = d.split('/')
    df = pd.read_csv(path)
    df['System'] = sys
    df['Group'] = group
    return df

def df_from_csv(path):
    df = pd.read_csv(path)
    update_df(df)
    return df

def update_df_old(df):
    from tqdm import tqdm
    import numpy as np
    from scipy.stats import linregress
    df['BestCoverageValue'] = np.int(0)
    df['BestCoverageEqualCount'] = np.int(0)
    df['BestCoverageSlope'] = float(0)
    df['BestCoverageIntercept'] = float(0)
    df['BestCoverageRValue'] = float(0)
    df['BestCoverageR2'] = float(0)
    df['BestCoveragePValue'] = float(0)
    df['BestCoverageStdErr'] = float(0)
    #df['BestCoverageCovariance'] = float(0)
    df['BestCoveragePositiveSlopeNumber'] = np.int(0)
    df['BestCoveragePositiveSlopeAttempt'] = np.int(0)
    df['KeysToEdgesRatio'] = df.NumberOfKeys / df.NumberOfEdges
    df['SolutionsFoundRatio'] = df.NumberOfSolutionsFound / df.Attempts
    x = np.array(list(range(0, 17)))[:, np.newaxis]
    x_flat = np.array(list(range(0, 17)))
    count_keys = [
        f'CountOfCacheLinesWithNumberOfAssigned_{n}'
            for n in range(0, 17)
    ]
    for (i, row) in tqdm(df.iterrows(), total=len(df)):
        best_count = row['NewBestGraphCount']
        if best_count == 0:
            # No solution was found.
            continue
        if best_count > 32:
            # We only capture up to 32 best graph attempts, in practice, it's
            # rare to get anything higher than 32 here.
            best_count = 32

        best_value = row[f'BestGraph{best_count}_Value']
        best_equal_count = row[f'BestGraph{best_count}_EqualCount']
        df.at[i, 'BestCoverageValue'] = np.int(best_value)
        df.at[i, 'BestCoverageEqualCount'] = np.int(best_equal_count)

        y = df.iloc[i][count_keys].values.astype(np.int)

        r = linregress(x_flat, y)

        df.at[i, 'BestCoverageSlope'] = r.slope
        df.at[i, 'BestCoverageIntercept'] = r.intercept
        df.at[i, 'BestCoverageRValue'] = r.rvalue
        df.at[i, 'BestCoverageR2'] = r.rvalue**2
        df.at[i, 'BestCoveragePValue'] = r.pvalue
        df.at[i, 'BestCoverageStdErr'] = r.stderr
        #df.at[i, 'BestCoverageCovariance'] = np.cov(x_flat, y)

        version = int(row['Version'][1:].split('.')[1])
        if version < 38:
            continue

        positive_slope = None
        attempt = None

        for j in range(1, best_count+1):
            best_keys = [
                f'BestGraph{j}_CountOfCacheLinesWithNumberOfAssigned_{n}'
                    for n in range(0, 17)
            ]

            y = df.iloc[i][best_keys].values.astype(np.int)
            r = linregress(x_flat, y)
            r2 = r.rvalue**2

            df.at[i, f'BestGraph{j}_Slope'] = r.slope
            df.at[i, f'BestGraph{j}_Intercept'] = r.intercept
            df.at[i, f'BestGraph{j}_RValue'] = r.rvalue
            df.at[i, f'BestGraph{j}_R2'] = r2
            df.at[i, f'BestGraph{j}_PValue'] = r.pvalue
            df.at[i, f'BestGraph{j}_StdErr'] = r.stderr
            #df.at[i, f'BestGraph{j}_Covariance'] = np.cov(x_flat, y)

            if j == 1 or positive_slope is not None:
                continue

            if r.slope > 0 and r2 > 70.0:
                positive_slope = j
                attempt = df.iloc[i]['BestGraph{j}_Attempt'].values[0]

        if positive_slope is not None:
            df[i, 'BestCoveragePositiveSlopeNumber'] = j
            df[i, 'BestCoveragePositiveSlopeAttempt'] = attempt
        else:
            df[i, 'BestCoveragePositiveSlopeNumber'] = 0
            df[i, 'BestCoveragePositiveSlopeAttempt'] = 0

def update_df_with_seed_bytes(df):

    import numpy as np

    # Only seeds 3 and 6 are used for masking currently.  Assume that if the
    # first byte isn't present as a column name, none of them will be.

    if 'Seed3_Byte1' not in df.columns:
        df['Seed3_Byte1'] = (df.Seed3 & 0x0000001f)
        df['Seed3_Byte2'] = np.right_shift((df.Seed3 & 0x00001f00), 8)
        df['Seed3_Byte3'] = np.right_shift((df.Seed3 & 0x001f0000), 16)
        df['Seed3_Byte4'] = np.right_shift((df.Seed3 & 0x1f000000), 24)

    if 'Seed6_Byte1' not in df.columns:
        df['Seed6_Byte1'] = (df.Seed6 & 0x0000001f)
        df['Seed6_Byte2'] = np.right_shift((df.Seed6 & 0x00001f00), 8)
        df['Seed6_Byte3'] = np.right_shift((df.Seed6 & 0x001f0000), 16)
        df['Seed6_Byte4'] = np.right_shift((df.Seed6 & 0x1f000000), 24)

    return df

def update_df_with_solve_duration(df):
    import pandas as pd

    df['SolveDuration'] = (
        df['SolveMicroseconds'].apply(
            lambda v: pd.Timedelta(v, unit='micro')
        )
    )

    return df

def update_df(df, source_csv_file):
    df['SourceCsvFile'] = source_csv_file
    return df

def update_df_best_coverage(df, source_csv_file):
    from tqdm import tqdm
    import numpy as np
    import pandas as pd
    from scipy.stats import linregress
    df['SourceCsvFile'] = source_csv_file
    df['BestCoverageValue'] = np.int(0)
    df['BestCoverageEqualCount'] = np.int(0)
    df['BestCoverageSlope'] = float(0)
    df['BestCoverageIntercept'] = float(0)
    df['BestCoverageRValue'] = float(0)
    df['BestCoverageR2'] = float(0)
    df['BestCoveragePValue'] = float(0)
    df['BestCoverageStdErr'] = float(0)
    #df['BestCoverageCovariance'] = float(0)
    df['BestCoveragePositiveSlopeNumber'] = np.int(0)
    df['BestCoveragePositiveSlopeAttempt'] = np.int(0)
    df['KeysToEdgesRatio'] = df.NumberOfKeys / df.NumberOfEdges
    df['KeysToVerticesRatio'] = df.NumberOfKeys / df.NumberOfVertices
    df['SolutionsFoundRatio'] = df.NumberOfSolutionsFound / df.Attempts

    if 0:
        # This currently causes pyarrow to bomb out downstream when attempting
        # to convert into a parquet file:
        #   pyarrow.lib.ArrowNotImplementedError:
        #       Unhandled type for Arrow to Parquet schema conversion:
        #       duration[ns]
        df['SolveDuration'] = (
            df['SolveMicroseconds'].apply(
                lambda v: pd.Timedelta(v, unit='micro')
            )
        )

    x = np.array(list(range(0, 17)))[:, np.newaxis]
    x_flat = np.array(list(range(0, 17)))
    count_keys = [
        f'CountOfCacheLinesWithNumberOfAssigned_{n}'
            for n in range(0, 17)
    ]
    if 'InitialNumberOfTableResizes' not in df.columns:
        df['InitialNumberOfTableResizes'] = np.int(0)

    # Only seeds 3 and 6 are used for masking currently.  Isolate the four bytes
    # being used for each ULONG mask.
    df['Seed3_Byte1'] = (df.Seed3 & 0x0000001f)
    df['Seed3_Byte2'] = np.right_shift((df.Seed3 & 0x00001f00), 8)
    df['Seed3_Byte3'] = np.right_shift((df.Seed3 & 0x001f0000), 16)
    df['Seed3_Byte4'] = np.right_shift((df.Seed3 & 0x1f000000), 24)

    df['Seed6_Byte1'] = (df.Seed6 & 0x0000001f)
    df['Seed6_Byte2'] = np.right_shift((df.Seed6 & 0x00001f00), 8)
    df['Seed6_Byte3'] = np.right_shift((df.Seed6 & 0x001f0000), 16)
    df['Seed6_Byte4'] = np.right_shift((df.Seed6 & 0x1f000000), 24)

    for (i, row) in tqdm(df.iterrows(), total=len(df)):
        best_count = row['NewBestGraphCount']
        if best_count == 0:
            # No solution was found.
            continue
        if best_count > 32:
            # We only capture up to 32 best graph attempts, in practice, it's
            # rare to get anything higher than 32 here.
            best_count = 32

        version = int(row['Version'][1:].split('.')[1])
        if version < 38:
            continue

        best_value = row[f'BestGraph{best_count}_Value']
        best_equal_count = row[f'BestGraph{best_count}_EqualCount']
        df.at[i, 'BestCoverageValue'] = np.int(best_value)
        df.at[i, 'BestCoverageEqualCount'] = np.int(best_equal_count)

        y = df.iloc[i][count_keys].values.astype(np.int)

        r = linregress(x_flat, y)

        df.at[i, 'BestCoverageSlope'] = r.slope
        df.at[i, 'BestCoverageIntercept'] = r.intercept
        df.at[i, 'BestCoverageRValue'] = r.rvalue
        df.at[i, 'BestCoverageR2'] = r.rvalue**2
        df.at[i, 'BestCoveragePValue'] = r.pvalue
        df.at[i, 'BestCoverageStdErr'] = r.stderr
        #df.at[i, 'BestCoverageCovariance'] = np.cov(x_flat, y)

def linregress_hash_func_by_number_of_edges(df):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_edges = (
        df.NumberOfEdges
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'NumberOfEdges',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, ne) for (hf, ne) in product(hash_funcs, num_edges) ]

    results = []

    np.seterr('raise')

    for (hash_func, num_edges) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfEdges == {num_edges}'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        keys_to_edges_ratio = df.KeysToEdgesRatio.values
        solutions_found_ratio = df.SolutionsFoundRatio.values

        count = len(keys_to_edges_ratio)

        # linregress won't work if all the x values are the same.
        if len(np.unique(keys_to_edges_ratio)) == 1:
            continue

        lr = linregress(keys_to_edges_ratio, solutions_found_ratio)

        #cycles = df.DeltaHashMinimumCycles.describe(percentiles=[0.95])['95%']
        #cycles = int(cycles)

        label = (
            f'{hash_func} ({count}): '
            f'y={lr.slope:.2f}x + {lr.intercept:.02f} '
            f'[r: {lr.rvalue:.3f}, p: {lr.pvalue:.3f}, '
            f'stderr: {lr.stderr:.3f}]'
        )

        result = (
            hash_func,
            num_edges,
            lr.slope,
            lr.intercept,
            lr.rvalue,
            lr.pvalue,
            lr.stderr,
            count,
            label,
            keys_to_edges_ratio,
            solutions_found_ratio,
        )

        results.append(result)

    columns = [
        'HashFunction',
        'NumberOfEdges',
        'Slope',
        'Intercept',
        'RValue',
        'PValue',
        'StdErr',
        'Count',
        'Label',
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
    ]

    return pd.DataFrame(results, columns=columns)

def linregress_hash_func_by_number_of_vertices(df):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_vertices = (
        df.NumberOfVertices
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'NumberOfVertices',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, ne) for (hf, ne) in product(hash_funcs, num_vertices) ]

    results = []

    np.seterr('raise')

    for (hash_func, num_vertices) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfVertices == {num_vertices}'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        keys_to_vertices_ratio = df.KeysToVerticesRatio.values
        solutions_found_ratio = df.SolutionsFoundRatio.values

        count = len(keys_to_vertices_ratio)

        # linregress won't work if all the x values are the same.
        if len(np.unique(keys_to_vertices_ratio)) == 1:
            continue

        lr = linregress(keys_to_vertices_ratio, solutions_found_ratio)

        label = (
            f'{hash_func} ({count}): '
            f'y={lr.slope:.2f}x + {lr.intercept:.02f} '
            f'[r: {lr.rvalue:.3f}, p: {lr.pvalue:.3f}, '
            f'stderr: {lr.stderr:.3f}]'
        )

        result = (
            hash_func,
            num_vertices,
            lr.slope,
            lr.intercept,
            lr.rvalue,
            lr.pvalue,
            lr.stderr,
            count,
            label,
            keys_to_vertices_ratio,
            solutions_found_ratio,
        )

        results.append(result)

    columns = [
        'HashFunction',
        'NumberOfVertices',
        'Slope',
        'Intercept',
        'RValue',
        'PValue',
        'StdErr',
        'Count',
        'Label',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    return pd.DataFrame(results, columns=columns)

def linregress_2020_03_25(df):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_vertices = (
        df.NumberOfVertices
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_resizes = (
        df.NumberOfTableResizeEvents
            .value_counts()
            .sort_index()
            .index
            .values
    )

    clamp_edges = ('Y', 'N')

    keys_subset = [
        'HashFunction',
        'NumberOfVertices',
        'NumberOfTableResizeEvents',
        'ClampNumberOfEdges',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = list(
        product(
            hash_funcs,
            num_vertices,
            num_resizes,
            clamp_edges,
        )
    )

    results = []

    np.seterr('raise')

    for target in tqdm(targets):

        (hash_func, num_vertex, num_resize, clamp) = target

        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfVertices == {num_vertex} and '
            f'NumberOfTableResizeEvents == {num_resize} and '
            f'ClampNumberOfEdges == "{clamp}"'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        keys_to_vertices_ratio = df.KeysToVerticesRatio.values
        solutions_found_ratio = df.SolutionsFoundRatio.values

        count = len(keys_to_vertices_ratio)

        # linregress won't work if all the x values are the same.
        if len(np.unique(keys_to_vertices_ratio)) == 1:
            print(f'No unique keys-to-vertices for query: {query}')
            continue

        lr = linregress(keys_to_vertices_ratio, solutions_found_ratio)

        label = (
            f'{hash_func}/{num_vertex}/{num_resize}/{clamp} ({count}): '
            f'y={lr.slope:.2f}x + {lr.intercept:.02f} '
            f'[r: {lr.rvalue:.3f}, p: {lr.pvalue:.3f}, '
            f'stderr: {lr.stderr:.3f}]'
        )

        result = (
            hash_func,
            num_vertex,
            num_resize,
            clamp,
            lr.slope,
            lr.intercept,
            lr.rvalue,
            lr.pvalue,
            lr.stderr,
            count,
            label,
            keys_to_vertices_ratio,
            solutions_found_ratio,
        )

        results.append(result)

    columns = [
        'HashFunction',
        'NumberOfVertices',
        'NumberOfTableResizeEvents',
        'ClampNumberOfEdges',
        'Slope',
        'Intercept',
        'RValue',
        'PValue',
        'StdErr',
        'Count',
        'Label',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    return pd.DataFrame(results, columns=columns)


def find_non_identical_solving_ratios_per_hash_func_and_keys(df):

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_names = (
        df.KeysName
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'KeysName',
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, kn) for (hf, kn) in product(hash_funcs, keys_names) ]

    results = []

    np.seterr('raise')

    #import ipdb
    #ipdb.set_trace()

    for (hash_func, keys_name) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'KeysName == "{keys_name}"'
        )
        df = dfa.query(query)

        # Value count of solutions found... if not more than 1, continue,
        # otherwise... print.
        vc = df.SolutionsFoundRatio.value_counts()
        if len(vc) <= 1:
            continue

        results.append((hash_func, keys_name, vc))

    return results

def extract_mean_solving_data(df):

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_names = (
        df.KeysName
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_edges = (
        df.NumberOfEdges
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_resizes = (
        df.NumberOfTableResizeEvents
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'KeysName',
        'HashFunction',
        'NumberOfTableResizeEvents',
        'NumberOfKeys',
        'NumberOfEdges',
        'NumberOfVertices',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
        'ClampNumberOfEdges',
    ]

    dfa = df[keys_subset]

    targets = [
        (hf, kn, ne, nr) for (hf, kn, ne, nr) in (
            product(hash_funcs, keys_names, num_edges, num_resizes)
        )
    ]

    results = []

    np.seterr('raise')

    for (hash_func, keys_name, num_edges, num_resizes) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'KeysName == "{keys_name}" and '
            f'NumberOfEdges == {num_edges} and '
            f'NumberOfTableResizeEvents == {num_resizes}'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        result = [
            keys_name,
            hash_func,
            df.NumberOfTableResizeEvents.values[0],
            df.NumberOfKeys.values[0],
            df.NumberOfEdges.values[0],
            df.NumberOfVertices.values[0],
            df.KeysToEdgesRatio.values[0],
            df.KeysToVerticesRatio.values[0],
            df.ClampNumberOfEdges.values[0],
        ]

        result += list(df.SolutionsFoundRatio.describe().values)

        results.append(result)

    suffixes = [
        '_Count',
        '', # Mean
        '_StdDev',
        '_Min',
        '_25%',
        '_50%',
        '_75%',
        '_Max',
    ]

    columns = [
        'KeysName',
        'HashFunction',
        'NumberOfTableResizeEvents',
        'NumberOfKeys',
        'NumberOfEdges',
        'NumberOfVertices',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
        'ClampNumberOfEdges',
    ]

    columns += [
        f'SolutionsFoundRatio{suffix}' for suffix in suffixes
    ]

    new_df = pd.DataFrame(results, columns=columns)
    return new_df

def extract_mean_solving_data_no_clamp_or_resize(df):

    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_names = (
        df.KeysName
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_edges = (
        df.NumberOfEdges
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'KeysName',
        'HashFunction',
        'NumberOfKeys',
        'NumberOfEdges',
        'NumberOfVertices',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
        'SolutionsFoundRatio',
    ]

    dfa = df[keys_subset]

    targets = [
        (hf, kn, ne) for (hf, kn, ne) in (
            product(hash_funcs, keys_names, num_edges)
        )
    ]

    results = []

    np.seterr('raise')

    for (hash_func, keys_name, num_edges) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'KeysName == "{keys_name}" and '
            f'NumberOfEdges == {num_edges}'
        )
        df = dfa.query(query)

        # Skip empties.
        if df.empty:
            continue

        result = [
            keys_name,
            hash_func,
            df.NumberOfKeys.values[0],
            df.NumberOfEdges.values[0],
            df.NumberOfVertices.values[0],
            df.KeysToEdgesRatio.values[0],
            df.KeysToVerticesRatio.values[0],
        ]

        result += list(df.SolutionsFoundRatio.describe().values)

        results.append(result)

    suffixes = [
        '_Count',
        '', # Mean
        '_StdDev',
        '_Min',
        '_25%',
        '_50%',
        '_75%',
        '_Max',
    ]

    columns = [
        'KeysName',
        'HashFunction',
        'NumberOfKeys',
        'NumberOfEdges',
        'NumberOfVertices',
        'KeysToEdgesRatio',
        'KeysToVerticesRatio',
    ]

    columns += [
        f'SolutionsFoundRatio{suffix}' for suffix in suffixes
    ]

    new_df = pd.DataFrame(results, columns=columns)
    return new_df


def perf_details_by_hash_func(df):
    import pandas as pd
    from tqdm import tqdm
    from itertools import product
    from scipy.stats import linregress

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    num_edges = (
        df.NumberOfEdges
            .value_counts()
            .sort_index()
            .index
            .values
    )

    keys_subset = [
        'HashFunction',
        'NumberOfKeys',
        'NumberOfEdges',
        'VerifyMicroseconds',
        'DeltaHashMinimumCycles',
    ]

    dfa = df[keys_subset]

    targets = [ (hf, ne) for (hf, ne) in product(hash_funcs, num_edges) ]

    results = []

    for (hash_func, num_edges) in tqdm(targets):

        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfEdges == {num_edges}'
        )
        df = dfa.query(query)

        x = df.NumberOfKeys.values
        y = df.VerifyMicroseconds.values

        lr = linregress(x, y)

        result = (
            hash_func,
            num_edges,
            lr.slope,
            lr.intercept,
            lr.rvalue,
            lr.pvalue,
            lr.stderr,
            len(x),
            x,
            y,
            y / x,
        )

        results.append(result)

    columns = [
        'HashFunction',
        'NumberOfEdges',
        'Slope',
        'Intercept',
        'RValue',
        'PValue',
        'StdErr',
        'Count',
        'NumberOfKeys',
        'VerifyMicroseconds',
        'VerifyMicrosecondsPerKey',
    ]

    return pd.DataFrame(results, columns=columns)

def extract_seed_mask_byte_counts(source_df, seed_number, which_bytes):
    assert seed_number in range(1, 9), seed_number
    assert num_bytes in (1, 2, 3, 4), num_bytes
    mask_byte_range = tuple(range(0, 0x1f+1))

    key = f'Seed{seed_number}_Byte{byte_number}'
    return (
        source_df.groupby([key])
            .size()
            .reset_index()
            .rename(columns={ 0: 'Count'})
    )

def extract_solutions_found_count(df, hash_funcs):

    import pandas as pd

    num_resizes = range(0, df.NumberOfTableResizeEvents.max() + 1)
    clamps = ('N', 'Y')
    solutions_found = ('N', 'Y')

    keys = [
        'HashFunction',
        'NumberOfTableResizeEvents',
        'ClampNumberOfEdges',
        'SolutionFound',
    ]

    multi = (
        hash_funcs,
        num_resizes,
        clamps,
        solution_found,
    )

    # The .reindex() zero-fills missing rows of count 0 for hash functions that
    # had no failures (that is, solution found == 'N').
    return (
        df.groupby(keys)
            .size()
            .reindex(
                pd.MultiIndex.from_product(multi),
                fill_value=0,
            )
            .reset_index()
            .rename(
                columns={
                    'level_0': 'HashFunction',
                    'level_1': 'NumberOfTableResizeEvents',
                    'level_2': 'ClampNumberOfEdges',
                    'level_3': 'SolutionFound',
                    0: 'Count',
                }
            )
    )

def get_dfs_from_csv(csv):
    dname = dirname(csv)
    convert_csv_to_parquet(csv, '.')
    parquet_path = csv.replace('.csv', '.parquet')
    # Load the df, then post-process.
    df = df_from_parquet(parquet_path)
    df.KeysToEdgesRatio = df.NumberOfKeys / df.NumberOfEdges
    df.AddKeysElapsedCycles = (
        df.HashKeysElapsedCycles -
        df.AddHashedKeysElapsedCycles
    )
    df['AddKeysElapsedCyclesPerKey'] = (
        df.AddKeysElapsedCycles / df.NumberOfKeys
    )
    df['HashKeysElapsedCyclesPerKey'] = (
        df.HashKeysElapsedCycles / df.NumberOfKeys
    )
    df['VerifyMicrosecondsPerKey'] = (
        df.VerifyMicroseconds / df.NumberOfKeys
    )
    df.rename(columns={'DeltaHashMinimumCycles': 'Cycles'}, inplace=True)

    try:
        path = join_path(dname, 'events_FoundNewBestGraph.csv')
        nbdf = pd.read_csv(path)
    except:
        nbdf = None
    try:
        path = join_path(dname, 'events_MemoryCoverageCacheLineCounts.csv')
        mcdf = pd.read_csv(path)
    except:
        mcdf = None

    return (df, nbdf, mcdf)

def get_mcdf(keysname, attempt, mcdf):
    df = mcdf[(mcdf.Attempt == attempt) & (mcdf.KeysFileName == keysname)]
    cols = [ c for c in df.columns if c.startswith('CacheLine') ]
    counts = df[cols]
    values = counts.values
    return values

def get_mcdf_for_df_row(df, mcdf, attempt=None):
    assert len(df) == 1
    keys_filename = basename(df.KeysFullPath.values[0])
    if attempt is None:
        attempt = df.AttemptThatFoundBestGraph.values[0]
    d = mcdf[(mcdf.Attempt == attempt) & (mcdf.KeysFileName == keys_filename)]
    return d

def get_vals_from_mcdf(mcdf):
    cols = [ c for c in mcdf.columns if c.startswith('CacheLine') ]
    counts = mcdf[cols]
    values = counts.values
    return values

#===============================================================================
# Format Conversion
#===============================================================================

def df_to_parquet(df, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression='gzip')
    return table

def df_to_feather(df, path):
    import pyarrow.feather
    return pyarrow.feather.write_feather(df, path)

def df_from_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas()

def get_yyyy_mm_dd_subdirs(dirname):
    import re
    import glob
    from os.path import isdir
    pattern = re.compile(r'\d{4}[-/]\d{2}[-/]\d{2}')
    subdirs = [
        subdir for subdir in glob.iglob('*') if (
            isdir(subdir) and
            pattern.match(subdir)
        )
    ]
    return subdirs

def get_csv_files(directory):
    import glob
    return set(
        f for f in glob.iglob(
            f'{directory}/**/PerfectHashBulkCreate*.csv',
            recursive=True
        )
    )

def get_all_bulk_create_parquet_files(directory):
    import glob
    return set(
        f for f in glob.iglob(
            f'{directory}/**/PerfectHashBulkCreate*.parquet',
            recursive=True
        ) if 'failed' not in f
    ).union(set(
        f for f in glob.iglob(
            f'{directory}/PerfectHashBulkCreate*.parquet',
            recursive=False
        ) if 'failed' not in f
    ))

def get_best_bulk_create_parquet_files(directory):
    import glob
    return set(
        f for f in glob.iglob(
            f'{directory}/**/PerfectHashBulkCreateBest*.parquet',
            recursive=True
        ) if 'failed' not in f
    )

def convert_csv_to_parquet(path, base_research_dir, out=None):
    if not out:
        out = lambda _: None

    from os import sep
    from os.path import exists
    import pandas as pd

    df = None
    sdf = None

    parquet_path = path.replace('.csv', '.parquet')
    if 'failed' in parquet_path:
        return

    if not exists(parquet_path):
        out(f'Processing {path} -> {parquet_path}...')

        df = pd.read_csv(path)
        source_csv_file = path.replace(base_research_dir + sep, '')
        update_df(df, source_csv_file)

        df_to_parquet(df, parquet_path)
        out(f'Wrote {parquet_path}.')

def concat_subdir_parquets(base, subdir, out=None):
    if not out:
        out = lambda _: None

    from os import sep
    from os.path import exists
    import pandas as pd
    from tqdm import tqdm

    subdir_path = f'{base}{sep}{subdir}'
    results_path = f'{base}{sep}{subdir}{sep}results.parquet'

    if not exists(results_path):
        names = get_all_bulk_create_parquet_files(subdir_path)
        if not names:
            out('No .parquet files found to concatenate.')
            return
        out(f'Processing all {len(names)} file(s) in {subdir}:')
        out('\n'.join(f'    {name}' for name in names))
        dfs = [ df_from_parquet(n) for n in tqdm(names) ]
        df = pd.concat(dfs, sort=False, ignore_index=True)
        df_to_parquet(df, results_path)
        out(f'Wrote {results_path}.')
        return

    df = df_from_parquet(results_path)
    processed_source_csv_files = set(
        df.SourceCsvFile
            .value_counts()
            .index
            .values
    )

    files = get_best_bulk_create_parquet_files(subdir_path)

    found_source_csv_files = set(
        f.replace(base + sep, '')
         .replace('.parquet', '.csv') for f in files
    )

    missing = found_source_csv_files - processed_source_csv_files

    if not missing:
        out(f'Up-to-date: {subdir_path}.')
        return

    out(f'Adding {len(missing)} new file(s) to {results_path}:')
    out('\n'.join(f'    {m}' for m in missing))

    names = [
        f'{base}{sep}{subpath}'.replace('.csv', '.parquet')
            for subpath in missing
    ]

    dfs = [ df, ] + [ df_from_parquet(n) for n in tqdm(names) ]

    df = pd.concat(dfs, sort=False, ignore_index=True)
    df_to_parquet(df, results_path)
    out(f'Wrote {results_path}.')
    return


def post_process_results_parquet(base, subdir, out=None):
    if not out:
        out = lambda _: None

    from os import sep
    from os.path import exists
    import pandas as pd

    df = None
    sdf = None

    parquet_path = f'{base}{sep}{subdir}{sep}results.parquet'
    if not exists(parquet_path):
        out(f'{parquet_path} does not exist, run concat-parquet-to-results-'
            f'parquet command first.')
        return

    # Solutions found ratio.
    sfr_path = parquet_path.replace('.parquet', '-sfr.parquet')
    if not exists(sfr_path):
        out(f'Processing {parquet_path} -> {sfr_path}...')

        if df is None:
            df = df_from_parquet(parquet_path)

        sdf = extract_mean_solving_data(df)
        df_to_parquet(sdf, sfr_path)
        out(f'Wrote {sfr_path}.')

    # Linear regressions.
    lr_path = parquet_path.replace('.parquet', '-lr.parquet')
    if not exists(lr_path):
        out(f'Processing {sfr_path} -> {lr_path}...')

        if sdf is None:
            sdf = df_from_parquet(sfr_path)

        #ldf = linregress_hash_func_by_number_of_vertices(sdf)
        ldf = linregress_2020_03_25(sdf)
        df_to_parquet(ldf, lr_path)
        out(f'Wrote {lr_path}.')

def process_xperf_perfecthash_csv(path, out=None):
    if not out:
        out = lambda _: None

    import io
    import json
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from os.path import splitext
    from datetime import timedelta

    (prefix, ext) = splitext(path)
    assert ext == '.csv'

    etw_headers = EVENT_NAME_TO_ETW_HEADER
    csv_headers = EVENT_NAME_TO_CSV_HEADER
    slim_csv_headers = EVENT_NAME_TO_CSV_HEADER_SLIM

    add_keys = ADD_KEYS
    hash_keys = HASH_KEYS
    add_hashed_keys = ADD_HASHED_KEYS
    found_new_best_graph = FOUND_NEW_BEST_GRAPH
    found_equal_best_graph = FOUND_EQUAL_BEST_GRAPH
    found_graph = FOUND_GRAPH
    assign_start = ASSIGN_START
    assign_stop = ASSIGN_STOP
    generate_random_bytes_start = GENERATE_RANDOM_BYTES_START
    generate_random_bytes_stop = GENERATE_RANDOM_BYTES_STOP
    cswitch = CSWITCH
    pmc = PMC
    is_acyclic = IS_ACYCLIC
    memory_coverage_cache_line_counts = MEMORY_COVERAGE_CACHE_LINE_COUNTS
    function_entry = FUNCTION_ENTRY

    assign_io = io.StringIO()
    generate_random_bytes_io = io.StringIO()

    io = {
        add_keys: io.StringIO(),
        hash_keys: io.StringIO(),
        add_hashed_keys: io.StringIO(),
        found_new_best_graph: io.StringIO(),
        found_equal_best_graph: io.StringIO(),
        found_graph: io.StringIO(),
        assign_start: assign_io,
        assign_stop: assign_io,
        generate_random_bytes_start: generate_random_bytes_io,
        generate_random_bytes_stop: generate_random_bytes_io,
        cswitch: io.StringIO(),
        pmc: io.StringIO(),
        is_acyclic: io.StringIO(),
        memory_coverage_cache_line_counts: io.StringIO(),
        function_entry: io.StringIO(),
    }

    paths = {
        add_keys: f'{prefix}_AddKeys.csv',
        hash_keys: f'{prefix}_HashKeys.csv',
        add_hashed_keys: f'{prefix}_AddHashedKeys.csv',
        found_new_best_graph: f'{prefix}_FoundNewBestGraph.csv',
        found_equal_best_graph: f'{prefix}_FoundEqualBestGraph.csv',
        found_graph: f'{prefix}_FoundGraph.csv',
        assign_start: f'{prefix}_AssignRaw.csv',
        assign_stop: f'{prefix}_Assign.csv',
        generate_random_bytes_start: f'{prefix}_GenerateRandomBytesRaw.csv',
        generate_random_bytes_stop: f'{prefix}_GenerateRandomBytes.csv',
        cswitch: f'{prefix}_ContextSwitch.csv',
        pmc: f'{prefix}_Pmc.csv',
        is_acyclic: f'{prefix}_IsAcyclic.csv',
        memory_coverage_cache_line_counts:
            f'{prefix}_MemoryCoverageCacheLineCounts.csv',
        function_entry: f'{prefix}_FunctionEntry.csv',
    }

    counts = {
        add_keys: 0,
        hash_keys: 0,
        add_hashed_keys: 0,
        found_new_best_graph: 0,
        found_equal_best_graph: 0,
        found_graph: 0,
        assign_start: 0,
        assign_stop: 0,
        generate_random_bytes_start: 0,
        generate_random_bytes_stop: 0,
        cswitch: 0,
        pmc: 0,
        is_acyclic: 0,
        memory_coverage_cache_line_counts: 0,
        function_entry: 0,
    }

    names = set(counts.keys())

    with open(path, 'r') as f:
        text = f.read()

    lines = text.splitlines()

    assert lines[0] == 'BeginHeader', lines[0]

    end_header = None
    pmc_header = None
    for (i, line) in enumerate(lines):
        if line == 'EndHeader':
            end_header = i
            break
        ix = line.find(',')
        if ix == -1:
            continue
        name = line[:ix]
        if name.startswith(PMC):
            assert line.startswith(PMC_ETW_HEADER_STARTSWITH), line
            pmc_etw_header = line
            pmc_csv_header = line.replace(' ', '').split(',')
            pmc_csv_header_slim = [ 'LineNumber' ] + pmc_csv_header[1:]
            etw_headers[PMC] = pmc_etw_header
            csv_headers[PMC] = pmc_csv_header
            slim_csv_headers[PMC] = pmc_csv_header_slim
        else:
            expected = etw_headers.get(name)
            if expected:
                assert line == expected, (line, expected)

    assert end_header
    trace_info_line = lines[end_header+1]
    assert 'Trace Start: ' in trace_info_line
    trace_info = dict([ p.split(': ') for p in trace_info_line.split(', ') ])
    json_path = f'{prefix}_TraceInfo.json'
    with open(json_path, 'w') as f:
        json.dump(trace_info, f)
    out(f'Wrote {json_path}.')

    start_dt = filetime_utc_to_local_tz(int(trace_info['Trace Start']))

    # Write the CSV headers to all StringIO buffers, excluding the ones used for
    # 'Start' events (as they share the 'Stop' event buffer).
    for (k, v) in slim_csv_headers.items():
        if v and not k.endswith('Start'):
            io[k].write(','.join(v))
            io[k].write('\n')

    for (lineno, line) in enumerate(tqdm(lines[end_header:])):
        ix = line.find(',')
        if ix == -1:
            continue
        name = line[:ix]

        if name not in names:
            continue

        counts[name] += 1

        header = csv_headers[name]
        slim_header = slim_csv_headers[name]

        ld = {
            k: v for (k, v) in zip(header, line.replace(' ', '').split(','))
        }

        ld['LineNumber'] = str(lineno)

        if name == ASSIGN_START:
            ld['NumberOfEmptyVertices'] = '0'
            ld['MaxTraversalDepth'] = '0'
            ld['TotalTraversals'] = '0'
            ld['Attempt'] = '0'
            slim_header = slim_csv_headers[ASSIGN_STOP]
        elif name == GENERATE_RANDOM_BYTES_START:
            ld['Result'] = '0'
            slim_header = slim_csv_headers[GENERATE_RANDOM_BYTES_STOP]

        ts = start_dt + timedelta(microseconds=int(ld['TimeStamp']))
        ld['TimeStamp'] = datetime_to_perfecthash_time(ts)

        if 'Result' in ld:
            ld['Success'] = 'Y' if ld['Result'] == '0x00000000' else 'N'

        if 'ProcessID' in ld:
            pid = ld['ProcessID']
            ix = pid.find('(')
            assert ix != -1
            ld['ProcessID'] = pid[ix+1:pid.find(')')]

        f = io[name]
        try:
            new_line = ','.join(ld[k] for k in slim_header)
        except KeyError:
            import ipdb
            ipdb.set_trace()
        f.write(new_line)
        f.write('\n')

    out('Post-processing results.')

    start_names = (
        ASSIGN_START,
        GENERATE_RANDOM_BYTES_START,
    )

    for (name, count) in counts.items():
        if count == 0:
            continue

        path = paths[name]
        if not path:
            continue

        f = io[name]
        f.seek(0)

        if name in start_names:
            with open(path, 'w') as raw_f:
                raw_f.write(f.getvalue())
            f.seek(0)
            out(f'Wrote {count} records to {path}.')
            continue

        df = pd.read_csv(f)

        if name in HAS_SEED_DATA:
            df['Seed3_Byte1'] = (df.Seed3 & 0x0000001f)
            df['Seed3_Byte2'] = np.right_shift((df.Seed3 & 0x00001f00), 8)
            df['Seed3_Byte3'] = np.right_shift((df.Seed3 & 0x001f0000), 16)
            df['Seed3_Byte4'] = np.right_shift((df.Seed3 & 0x1f000000), 24)

            df['Seed6_Byte1'] = (df.Seed6 & 0x0000001f)
            df['Seed6_Byte2'] = np.right_shift((df.Seed6 & 0x00001f00), 8)
            df['Seed6_Byte3'] = np.right_shift((df.Seed6 & 0x001f0000), 16)
            df['Seed6_Byte4'] = np.right_shift((df.Seed6 & 0x1f000000), 24)

        if name == ASSIGN_STOP:
            df['TimeStamp'] = df.TimeStamp.astype(np.datetime64)
            df = df.sort_values([
                'ThreadID',
                'TimeStamp',
                'Attempt',
            ], ignore_index=True)
            df['Elapsed'] = df.TimeStamp - df.TimeStamp.shift(1)
            df = df[df.EventName == ASSIGN_STOP][[
                'Elapsed',
                'LineNumber',
                'TimeStamp',
                'ProcessID',
                'ThreadID',
                'CPU',
                'ActivityId',
                'NumberOfKeys',
                'NumberOfVertices',
                'NumberOfEmptyVertices',
                'MaxTraversalDepth',
                'TotalTraversals',
            ]]
        elif name == GENERATE_RANDOM_BYTES_STOP:
            df['TimeStamp'] = df.TimeStamp.astype(np.datetime64)
            df = df.sort_values([
                'ThreadID',
                'TimeStamp',
                'BytesRequested',
            ], ignore_index=True)
            df['Elapsed'] = df.TimeStamp - df.TimeStamp.shift(1)
            df = df[df.EventName == GENERATE_RANDOM_BYTES_STOP][[
                'Elapsed',
                'LineNumber',
                'TimeStamp',
                'ProcessID',
                'ThreadID',
                'CPU',
                'ActivityId',
                'BytesRequested',
                'Result',
            ]]

        df.to_csv(path)
        out(f'Wrote {count} records to {path}.')

#===============================================================================
# Plotting
#===============================================================================

def get_cache_line_coverage(df):
    import numpy as np
    count = df.NewBestGraphCount.values[0]
    keys = [
        f'BestGraph{i}_CountOfCacheLinesWithNumberOfAssigned_{n}'
            for i in range(1, count+1)
                for n in range(0, 17)
    ]
    values = df[keys].values.astype(int)
    values = values.reshape(count, 17)
    keys = [ f'BestGraph{i}_Attempt' for i in range(1, count+1) ]
    attempts = df[keys].values[0]
    columns = [ f'{i}' for i in range(0, 17) ]
    return (keys, values, attempts, columns)

def ridgeline_plot(df):
    import joypy
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    #plt.ioff()
    keys_name = df.KeysName.values[0]
    hash_func = df.HashFunction.values[0]
    best_coverage_type = df.BestCoverageType.values[0]
    table_timestamp = df.TableTimestamp.values[0]
    (keys, values, attempts, columns) = get_cache_line_coverage(df)
    dfc = pd.DataFrame(values, columns=columns, index=attempts).sort_index()
    dft = dfc.T
    dft = dft[[ c for c in reversed(sorted(dft.columns)) ]]
    x_range = list(range(0, 17))
    (figure, axes) = joypy.joyplot(
        dft,
        kind="values",
        x_range=x_range,
        #labels=attempts,
        colormap=cm.rainbow,
        title=f"{keys_name} {hash_func}\n{best_coverage_type}\n{table_timestamp}",
        alpha=0.5,
        overlap=0.5,
        #ylim=True,
        grid=True,
        figsize=(10, 10),
    )
    axes[-1].set_xticks(x_range)
    axes[-1].set_yticks(dfc.index)
    #plt.draw()
    #plt.cla()
    #plt.clf()
    return (figure, axes)


def ridgeline_plot_all(df, keys_name=None):
    df = df.copy()
    #df = df[df.HeaderHash == 'C8D67583']
    df = df.sort_values(by=['NumberOfKeys', 'BestCoverageType', 'HashFunction'])
    if keys_name:
        if isinstance(keys_name, list):
            s = ', '.join(f'"{k}"' for k in keys_name)
            df = df.query(f'KeysName in ({s})')
        else:
            df = df.query(f'KeysName == "{keys_name}"')
    timestamps = df.TableTimestamp.values
    results = []
    for t in timestamps:
        d = df.query(f'TableTimestamp == "{t}"')
        assert(len(d) == 1)
        #(figure, axes) = ridgeline_plot(d)
        #results.append((figure, axes))
        #if d.NumberOfKeys.values[0] <= 5000:
        #    continue
        #print(d[['NumberOfKeys', 'BestCoverageType', 'HashFunction', 'TableTimestamp']].values[0])
        #ridgeline_plot(d)
        (figure, axes) = ridgeline_plot(d)
        #results.append((figure, axes))
    return results

def regplot_solutions_found(df):
    import matplotlib.pyplot as plt
    #plt.ioff()
    import seaborn as sns
    import numpy as np

    edges = [ 1 << i for i in range(12, 17) ]
    colors_array = plt.cm.tab20(np.arange(20).astype(int))
    np.random.shuffle(colors_array)
    color_map = { k: colors_array[i] for (i, k) in enumerate(edges) }

    fig = plt.figure(figsize=(10, 7))
    #edges = [ 1 << i for i in range(3, 17) ]
    keys = [
        'SolutionsFoundRatio',
        'KeysToEdgesRatio',
        'NumberOfEdges',
        'NumberOfKeys',
    ]
    for edge in edges:
        color = color_map[edge]
        #print((edge, color))
        data = df[df.NumberOfEdges == edge][keys]
        sns.regplot(
            x='KeysToEdgesRatio',
            y='SolutionsFoundRatio',
            #hue='NumberOfEdges',
            data=data,
            color=color,
            scatter_kws={
                's': 1,
                'alpha': 1.0,
            },
            line_kws={
                'linewidth': 0.1,
                'alpha': 1.0,
            },
        )

    plt.legend(labels=edges)
    plt.xlabel("Keys To Edges Ratio")
    plt.ylabel("Solutions Found Ratio")
    return (plt.gcf(), plt.gca())

def lmplot_solutions_found(df):
    import matplotlib.pyplot as plt
    #plt.ioff()
    import seaborn as sns
    import numpy as np

    edges = [ 1 << i for i in range(4, 17) ]
    colors_array = plt.cm.tab20(np.arange(20).astype(int))
    np.random.shuffle(colors_array)
    color_map = { k: colors_array[i] for (i, k) in enumerate(edges) }

    fig = plt.figure(figsize=(10, 7))
    edges = [ 1 << i for i in range(3, 17) ]
    keys = [
        'SolutionsFoundRatio',
        'KeysToEdgesRatio',
        'NumberOfEdges',
        'NumberOfKeys',
    ]
    sns.set(color_codes=True)
    #fig = plt.figure(figsize=(10, 7))
    keys = [
        'SolutionsFoundRatio',
        'KeysToEdgesRatio',
        'NumberOfEdges',
    ]
    sns.lmplot(
        x='KeysToEdgesRatio',
        y='SolutionsFoundRatio',
        hue='NumberOfEdges',
        col='NumberOfEdges',
        data=df[df.NumberOfEdges >= 16][keys],
        col_wrap=3,
        scatter_kws={'s': 1},
    )

    plt.legend(labels=edges)
    plt.xlabel("Keys To Edges Ratio")
    plt.ylabel("Solutions Found Ratio")
    return (plt.gcf(), plt.gca())

#===============================================================================
# Bokeh
#===============================================================================

def scatter6(df, min_num_edges=None, max_num_edges=None, p=None,
             show_plot=True, figure_kwds=None, scatter_kwds=None):

    import numpy as np

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        TapTool,
        ColorBar,
        CustomJS,
        ColumnDataSource,
    )

    from bokeh.plotting import (
        figure,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {
            'tooltips': [
                ("Index", "@index"),
                ("Keys", "@KeysName"),
                ("Number of Keys", "@NumberOfKeys"),
                ("Number of Edges", "@NumberOfEdges"),
                ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
                ("Solutions Found", "@SolutionsFoundRatio{(.000)}"),
            ],
        }

    if scatter_kwds is None:
        scatter_kwds = {}

    if not p:
        p = figure(
            width=1000,
            height=1000,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )


    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        #max_num_edges = df.NumberOfEdges.max()
        max_num_edges = 65536

    df = df.query(f'NumberOfEdges >= {min_num_edges} and '
                  f'NumberOfEdges <= {max_num_edges}').copy()

    df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(str)

    num_edges = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]

    source = ColumnDataSource(df)

    cr = p.scatter(
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
        color='Color',
        line_color='Color',
        source=source,
        #fill_color='grey',
        #fill_alpha=0.05,
        #hover_fill_color=mapper,
        #hover_alpha=0.3,
        #line_color=None,
        #hover_line_color=mapper,
        #line_color=color_mapper,
        #color=color_mapper,
        legend_field='NumberOfEdgesStr',
        **scatter_kwds,
    )

    #color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    #p.add_layout(color_bar, 'right')

    #p.background_fill_color = "#eeeeee"
    #p.grid.grid_line_color = "white"

    #ht = HoverTool(tooltips=)
    #p.add_tools(HoverTool())

    code = """
        //console.log(cb_data);
        //console.log(cb_obj);
        //console.log(s);
        //console.log(greys_map);
        //console.log(colors_map);
        const d = s.data;
        const selected_index = s.selected.indices[0];
        const selected_num_edges = d['NumberOfEdges'][selected_index];
        const selected_color = colors_map[selected_num_edges];
        var num_edges;
        var color;

        console.log("Tap!");
        console.log(selected_index);
        //console.log(selected_num_edges);
        //console.log(selected_color);
        //console.log(s);
        //console.log(d.length);

        for (var i = 0; i < d['index'].length; i++) {

            num_edges = d['NumberOfEdges'][i];
            if (num_edges == selected_num_edges) {
                color = selected_color;
            } else {
                color = greys_map[selected_num_edges];
            }

            //console.log(i, color);
            d['Color'][i] = color;
        }

        s.change.emit();
    """

    args = {
        's': source,
        'greys_map': greys_map,
        'colors_map': colors_map,
    }

    callback = CustomJS(args=args, code=code)

    taptool = p.select(type=TapTool)
    taptool.callback = callback

    if show_plot:
        show(p)

    return p

def scatter7(df, min_num_edges=None, max_num_edges=None, p=None,
             show_plot=True, figure_kwds=None, scatter_kwds=None):

    import numpy as np

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        TapTool,
        ColorBar,
        CustomJS,
        ColumnDataSource,
    )

    from bokeh.plotting import (
        figure,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {
            'tooltips': [
                ("Index", "@index"),
                ("Keys", "@KeysName"),
                ("Number of Keys", "@NumberOfKeys"),
                ("Number of Edges", "@NumberOfEdges"),
                ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
                ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
            ],
        }

    if scatter_kwds is None:
        scatter_kwds = {}

    if not p:
        p = figure(
            width=1000,
            height=1000,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )


    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        #max_num_edges = df.NumberOfEdges.max()
        max_num_edges = 65536

    df = df.query(f'NumberOfEdges >= {min_num_edges} and '
                  f'NumberOfEdges <= {max_num_edges}').copy()

    df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(str)

    num_edges = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    colors_map[8192] = "#ffff00"

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]

    source = ColumnDataSource(df)

    cr = p.scatter(
        'KeysToEdgesRatio',
        'SolutionsFoundRatio',
        color='Color',
        line_color='Color',
        source=source,
        #fill_color='grey',
        #fill_alpha=0.05,
        #hover_fill_color=mapper,
        #hover_alpha=0.3,
        #line_color=None,
        #hover_line_color=mapper,
        #line_color=color_mapper,
        #color=color_mapper,
        legend_field='NumberOfEdgesStr',
        **scatter_kwds,
    )

    #color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
    #p.add_layout(color_bar, 'right')

    p.background_fill_color = "#eeeeee"
    p.grid.grid_line_color = "white"

    #ht = HoverTool(tooltips=)
    #p.add_tools(HoverTool())

    code = """
        //console.log(cb_data);
        //console.log(cb_obj);
        //console.log(s);
        //console.log(greys_map);
        //console.log(colors_map);
        const d = s.data;
        const selected_index = s.selected.indices[0];
        const selected_num_edges = d['NumberOfEdges'][selected_index];
        const selected_color = colors_map[selected_num_edges];
        var num_edges;
        var color;
        var new_selection = [];

        //console.log("Tap!");
        //console.log(selected_index);
        //console.log(selected_num_edges);
        //console.log(selected_color);
        //console.log(s);
        //console.log(d.length);

        for (var i = 0; i < d['index'].length; i++) {

            num_edges = d['NumberOfEdges'][i];
            if (num_edges == selected_num_edges) {
                new_selection.push(i);
            }

            //console.log(i, color);
            //d['Color'][i] = color;
        }

        //console.log(new_selection);
        s.selected.indices = new_selection;

        s.change.emit();
    """

    args = {
        's': source,
        'greys_map': greys_map,
        'colors_map': colors_map,
    }

    callback = CustomJS(args=args, code=code)

    taptool = p.select(type=TapTool)
    taptool.callback = callback

    if show_plot:
        show(p)

    return p

def panel1(df, min_num_edges=None, max_num_edges=None,
           show_plot=True, figure_kwds=None, scatter_kwds=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    try:
        from bokeh.models import TabPanel
    except ImportError:
        from bokeh.models import Panel as TabPanel

    from bokeh.plotting import (
        figure,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        #max_num_edges = df.NumberOfEdges.max()
        max_num_edges = 65536

    # Prep colors.
    num_edges = [ 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    # Change the light yellow color to a darker color.
    colors_map[8192] = "#f4d03f"

    line_colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    line_colors_map[8192] = "#f1c40f"

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    panels = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(str)

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]
        df['LineColor'] = [
            line_colors_map[e] for e in df.NumberOfEdges.values
        ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        source = ColumnDataSource(df)

        p = figure(
            width=1000,
            height=1000,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.y_range = y_range

        cr = p.scatter(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            line_color='LineColor',
            source=source,
            legend_field='NumberOfEdgesStr',
            **scatter_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            const selected_color = colors_map[selected_num_edges];
            var num_edges;
            var color;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'greys_map': greys_map,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        panel = TabPanel(child=p, title=hash_func)
        panels.append(panel)

    tabs = Tabs(tabs=panels)

    if show_plot:
        show(tabs)

    return p

def grid1(df, min_num_edges=None, max_num_edges=None,
          width=500, height=500, ncols=4,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          log_size_multiplier=None, size_multiplier=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        TapTool,
        Range1d,
        CustomJS,
        ColumnDataSource,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()
        #max_num_edges = 65536

    if log_size_multiplier is None:
        log_size_multiplier = 1.5

    if size_multiplier is None:
        size_multiplier = 2

    # Prep colors.
    num_edges = [
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    ]
    #num_edges = [ 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    # Change the light yellow color to a darker color.
    colors_map[8192] = "#f4d03f"

    line_colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    line_colors_map[8192] = "#f1c40f"

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    hash_funcs = (
        df[['HashFunction', 'SolutionsFoundRatio']]
            .groupby('HashFunction')
            .agg('mean')
            .sort_values('SolutionsFoundRatio', ascending=False)
            .index
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(str)

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]
        df['LineColor'] = [
            line_colors_map[e] for e in df.NumberOfEdges.values
        ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 6)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * log_size_multiplier)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * size_multiplier
            df.at[i, 'LogSize'] = log_size

        source = ColumnDataSource(df)

        p = figure(
            width=width,
            height=height,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = hash_func
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Edges Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'
        p.title.text_font = "Garamond"
        p.title.text_font_size = "1.25em"
        p.xaxis.axis_label_text_font = "Garamond"
        p.yaxis.axis_label_text_font = "Garamond"
        p.xaxis.axis_label_text_font_style = "italic"
        p.yaxis.axis_label_text_font_style = "italic"
        p.xaxis.axis_label_text_font_size = "1.25em"
        p.yaxis.axis_label_text_font_size = "1.25em"

        p.scatter(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            line_color='LineColor',
            muted_color='grey',
            muted_alpha=0.2,
            source=source,
            legend_field='NumberOfEdgesStr',
            **scatter_kwds,
        )

        p.legend.title = 'Number of Edges'
        p.legend.title_text_font = "Garamond"
        p.legend.title_text_font_size = "0.95em"
        p.legend.location = "top_right"
        #p.legend.click_policy = "mute"
        p.legend.label_text_font = "Comic Mono"
        p.legend.label_text_font_size = "0.75em"

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            const selected_color = colors_map[selected_num_edges];
            var num_edges;
            var color;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'greys_map': greys_map,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return p

def grid1_hf(df, min_num_edges=None, max_num_edges=None,
             width=500, height=500, ncols=4,
             show_plot=True, figure_kwds=None, scatter_kwds=None,
             log_size_multiplier=None, size_multiplier=None):
    """
    Displays a grid of scatter plots for each edge size, where the x-axis is
    the keys to edges ratio and the y-axis is the probability of finding a
    solution.  Coloured scatter glyphs are added for each hash function.
    """

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        TapTool,
        Range1d,
        CustomJS,
        ColumnDataSource,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Hash Function", "@HashFunction"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.0000000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()
        #max_num_edges = 65536

    if log_size_multiplier is None:
        log_size_multiplier = 1.5

    if size_multiplier is None:
        size_multiplier = 2

    # Prep colors.
    num_edges = [
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    ]
    #num_edges = [ 2048, 4096, 8192, 16384, 32768, 65536 ]
    num_edges_str = [ str(e) for e in num_edges ]

    colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    # Change the light yellow color to a darker color.
    colors_map[8192] = "#f4d03f"

    line_colors_map = { e: bp.Spectral11[i] for (i, e) in enumerate(num_edges) }
    line_colors_map[8192] = "#f1c40f"

    color_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=bp.Spectral11,
        factors=num_edges_str,
    )

    greys = [ bp.Greys256[i] for i in range(-50, 60, 10) ]
    greys_map = { e: greys[i] for (i, e) in enumerate(num_edges) }

    grey_mapper = bt.factor_cmap(
        field_name='NumberOfEdgesStr',
        palette=greys,
        factors=num_edges_str,
    )

    hash_funcs = (
        df[['HashFunction', 'SolutionsFoundRatio']]
            .groupby('HashFunction')
            .agg('mean')
            .sort_values('SolutionsFoundRatio', ascending=False)
            .index
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for edges in num_edges:

        df = source_df.query(f'NumberOfEdges == {edges}').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(str)

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdges.values ]
        df['LineColor'] = [
            line_colors_map[e] for e in df.NumberOfEdges.values
        ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 6)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * log_size_multiplier)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * size_multiplier
            df.at[i, 'LogSize'] = log_size

        source = ColumnDataSource(df)

        p = figure(
            width=width,
            height=height,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = f'Number of Edges: {edges}'
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Edges Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'
        p.title.text_font = "Garamond"
        p.title.text_font_size = "1.25em"
        p.xaxis.axis_label_text_font = "Garamond"
        p.yaxis.axis_label_text_font = "Garamond"
        p.xaxis.axis_label_text_font_style = "italic"
        p.yaxis.axis_label_text_font_style = "italic"
        p.xaxis.axis_label_text_font_size = "1.25em"
        p.yaxis.axis_label_text_font_size = "1.25em"

        p.scatter(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            line_color='LineColor',
            muted_color='grey',
            muted_alpha=0.2,
            source=source,
            legend_field='HashFunction',
            **scatter_kwds,
        )

        p.legend.title = 'Hash Function'
        p.legend.title_text_font = "Garamond"
        p.legend.title_text_font_size = "0.95em"
        p.legend.location = "top_right"
        #p.legend.click_policy = "mute"
        p.legend.label_text_font = "Comic Mono"
        p.legend.label_text_font_size = "0.75em"

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            const selected_color = colors_map[selected_num_edges];
            var num_edges;
            var color;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'greys_map': greys_map,
            'colors_map': colors_map,
        }

        #callback = CustomJS(args=args, code=code)

        #taptool = p.select(type=TapTool)
        #taptool.callback = callback

        figures.append(p)

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return grid


def grid2(df, min_num_edges=None, max_num_edges=None,
           show_plot=True, figure_kwds=None, scatter_kwds=None,
           color_category=None, ncols=None, log_size_multiplier=None,
           size_multiplier=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = bp.Category20

    if ncols is None:
        ncols = 4

    if log_size_multiplier is None:
        log_size_multiplier = 1.5

    if size_multiplier is None:
        size_multiplier = 2

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(str)

        num_edges = (
            df.NumberOfEdges
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_edges_str = [ str(e) for e in num_edges ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_edges)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_edges_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdgesStr.values ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * log_size_multiplier)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * size_multiplier
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfEdges'])

        source = ColumnDataSource(df)

        p = figure(
            width=500,
            height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = hash_func
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Edges Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'

        cr = p.scatter(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_field='NumberOfEdgesStr',
            **scatter_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            var num_edges;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return p

def grid3(df, lrdf=None, min_num_vertices=None, max_num_vertices=None,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          color_category=None, ncols=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Vertices", "@NumberOfVertices"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Keys to Vertices Ratio", "@KeysToVerticesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_vertices is None:
        min_num_vertices = 256

    if max_num_vertices is None:
        max_num_vertices = df.NumberOfVertices.max()

    if color_category is None:
        color_category = bp.Category20

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfVertices >= {min_num_vertices} and '
                             f'NumberOfVertices <= {max_num_vertices} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfVerticesStr'] = df.NumberOfVertices.values.astype(str)

        num_vertices = (
            df.NumberOfVertices
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_vertices_str = [ str(e) for e in num_vertices ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_vertices)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_vertices_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfVerticesStr.values ]

        df['KeysToVerticesRatio'] = np.around(df.KeysToVerticesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToVerticesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfVertices'])

        source = ColumnDataSource(df)

        p = figure(
            width=500,
            height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = hash_func
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Vertices Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'

        cr = p.scatter(
            'KeysToVerticesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_group='NumberOfVertices',
            **scatter_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_vertices = d['NumberOfVertices'][selected_index];
            var num_vertices;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_vertices = d['NumberOfVertices'][i];
                if (num_vertices == selected_num_vertices) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 2

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return p

def grid4(df, lrdf=None, min_num_edges=None, max_num_edges=None,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          color_category=None, ncols=None):

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Number of Vertices", "@NumberOfVertices"),
        ("Number Of Resizes", "@NumberOfTableResizeEvents"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio"),
        ("Keys to Vertices Ratio", "@KeysToVerticesRatio"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = bp.Spectral11

    hash_funcs = (
        df.HashFunction
            .value_counts()
            .sort_index()
            .index
            .values
    )

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    for hash_func in hash_funcs:

        df = source_df.query(f'NumberOfEdges >= {min_num_edges} and '
                             f'NumberOfEdges <= {max_num_edges} and '
                             f'HashFunction == "{hash_func}"').copy()

        df['NumberOfEdgesStr'] = df.NumberOfEdges.values.astype(str)

        num_edges = (
            df.NumberOfEdges
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_edges_str = [ str(e) for e in num_edges ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_edges)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_edges_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfEdgesStr.values ]

        df['KeysToEdgesRatio'] = np.around(df.KeysToEdgesRatio.values, 5)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 5)

        sdf = (
            df.groupby(['KeysToEdgesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToEdgesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfEdges'])

        p = figure(
            width=500,
            height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = hash_func
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Edges Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'

        source = ColumnDataSource(df)

        cr = p.scatter(
            'KeysToEdgesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_group='NumberOfEdges',
            **scatter_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_edges = d['NumberOfEdges'][selected_index];
            var num_edges;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_edges = d['NumberOfEdges'][i];
                if (num_edges == selected_num_edges) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 2

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return p

def grid5(df, lrdf=None, min_num_edges=None, max_num_edges=None,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          color_category=None, ncols=None, hash_funcs=None,
          clamp_edges=None, num_resizes=None, clamp_glyph_map=None):

    from tqdm import tqdm
    from itertools import product

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Legend,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        LegendItem,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.core.enums import (
        LegendLocation,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Number of Vertices", "@NumberOfVertices"),
        ("Number Of Resizes", "@NumberOfTableResizeEvents"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio{(0.000)}"),
        ("Keys to Vertices Ratio", "@KeysToVerticesRatio{(0.000)}"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
        ("Clamp Edges?", "@ClampNumberOfEdges"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = list(bp.Spectral11) + list(bp.Category20[20])

    figures = []

    source_df = df

    x_range = Range1d(0, 1.0)
    y_range = Range1d(0, 1.0)

    if clamp_edges is None:
        clamp_edges = ('Y', 'N')

    if hash_funcs is None:
        hash_funcs = (
            df.HashFunction
                .value_counts()
                .sort_index()
                .index
                .values
        )

    if num_resizes is None:
        num_resizes = (
            df.NumberOfTableResizeEvents
                .value_counts()
                .sort_index()
                .index
                .values
        )

    if clamp_glyph_map is None:
        clamp_glyph_map = {
            'Y': 'scatter',
            'N': 'x',
        }

    targets = [
        (hf, ne) for (hf, ne) in product(hash_funcs, num_resizes)
    ]

    for (hash_func, resizes) in targets:

        df = source_df.query(
            f'NumberOfEdges >= {min_num_edges} and '
            f'NumberOfEdges <= {max_num_edges} and '
            f'HashFunction == "{hash_func}" and '
            f'NumberOfTableResizeEvents == {resizes} '
        ).copy()

        df['NumberOfVerticesStr'] = df.NumberOfVertices.values.astype(str)

        num_vertices = (
            df.NumberOfVertices
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_vertices_str = [ str(e) for e in num_vertices ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_vertices)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_vertices_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfVerticesStr.values ]

        df['KeysToVerticesRatio'] = np.around(df.KeysToVerticesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToVerticesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfVertices'])

        p = figure(
            width=500,
            height=500,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = f'{hash_func}, Resizes: {resizes}'
        p.x_range = x_range
        p.y_range = y_range
        p.xaxis.axis_label = 'Keys to Vertices Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'

        renderers = {}
        legend_items = []

        for clamp in clamp_edges:
            clamped_df = df[df.ClampNumberOfEdges == clamp]
            source = ColumnDataSource(clamped_df)
            glyph = getattr(p, clamp_glyph_map[clamp])
            g = glyph(
                'KeysToVerticesRatio',
                'SolutionsFoundRatio',
                color='Color',
                size='Size',
                fill_alpha=0.5,
                line_alpha=1.0,
                source=source,
                legend_group='NumberOfVertices',
                **scatter_kwds,
            )
            legend_items.append((clamp, [g]))
            renderers[clamp] = g

        #p.legend.title = 'Number of Edges'
        p.legend.click_policy = 'hide'

        legend = Legend(
            title='Clamp Number of Edges?',
            location=LegendLocation.top_left,
            items=legend_items,
        )

        p.add_layout(legend)

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_vertices = d['NumberOfVertices'][selected_index];
            var num_vertices;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_vertices = d['NumberOfVertices'][i];
                if (num_vertices == selected_num_vertices) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 6

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if show_plot:
        show(grid)

    return p

def grid6(df, lrdf=None, min_num_edges=None, max_num_edges=None,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          color_category=None, ncols=None, hash_funcs=None,
          clamp_edges=None, num_resizes=None, clamp_glyph_map=None):

    from tqdm import tqdm
    from itertools import product

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Legend,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        LegendItem,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.core.enums import (
        LegendLocation,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    tooltips = [
        ("Index", "@index"),
        ("Keys", "@KeysName"),
        ("Number of Keys", "@NumberOfKeys"),
        ("Number of Edges", "@NumberOfEdges"),
        ("Number of Vertices", "@NumberOfVertices"),
        ("Number Of Resizes", "@NumberOfTableResizeEvents"),
        ("Keys to Edges Ratio", "@KeysToEdgesRatio{(0.000)}"),
        ("Keys to Vertices Ratio", "@KeysToVerticesRatio{(0.000)}"),
        ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
        ("Clamp Edges?", "@ClampNumberOfEdges"),
    ]

    if figure_kwds is None:
        figure_kwds = {}

    if 'tooltips' not in figure_kwds:
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = list(bp.Spectral11) + list(bp.Category20[20])

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    if clamp_edges is None:
        clamp_edges = ('Y', 'N')

    if hash_funcs is None:
        hash_funcs = (
            df.HashFunction
                .value_counts()
                .sort_index()
                .index
                .values
        )

    if num_resizes is None:
        num_resizes = (
            df.NumberOfTableResizeEvents
                .value_counts()
                .sort_index()
                .index
                .values
        )

    if clamp_glyph_map is None:
        clamp_glyph_map = {
            'Y': 'scatter',
            'N': 'x',
        }

    targets = [
        (hf, ce) for (hf, ce) in product(hash_funcs, clamp_edges)
    ]

    for (hash_func, clamp) in targets:

        df = source_df.query(
            f'NumberOfEdges >= {min_num_edges} and '
            f'NumberOfEdges <= {max_num_edges} and '
            f'HashFunction == "{hash_func}" and '
            f'ClampNumberOfEdges == "{clamp}"'
        ).copy()

        df['NumberOfVerticesStr'] = df.NumberOfVertices.values.astype(str)

        num_vertices = (
            df.NumberOfVertices
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_vertices_str = [ str(e) for e in num_vertices ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_vertices)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_vertices_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfVerticesStr.values ]

        df['KeysToVerticesRatio'] = np.around(df.KeysToVerticesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToVerticesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfVertices'])

        p = figure(
            width=900,
            height=900,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = f'{hash_func}, Clamp: {clamp}'
        p.xaxis.axis_label = 'Keys to Vertices Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'
        p.y_range = y_range
        p.x_range = Range1d(0, 0.6)
        if None:
            p.x_range = Range1d(
                df.KeysToVerticesRatio.min(),
                df.KeysToVerticesRatio.max()
            )

        renderers = []
        resize_legend_items = []

        for resize in num_resizes:
            target_df = df[df.NumberOfTableResizeEvents == resize]
            source = ColumnDataSource(target_df)
            glyph = getattr(p, BOKEH_GLYPHS[resize])
            g = glyph(
                'KeysToVerticesRatio',
                'SolutionsFoundRatio',
                color='Color',
                size='Size',
                fill_alpha=0.5,
                line_alpha=1.0,
                source=source,
                #legend_group='NumberOfVertices',
                **scatter_kwds,
            )
            resize_legend_items.append((str(resize), [g]))
            renderers.append(g)
            #legend_items.append((clamp, [g]))
            #renderers[clamp] = g

        main_legend_items = []
        for (i, num_vert) in enumerate(num_vertices_str):
            main_legend_items.append(
                LegendItem(
                    label=num_vert,
                    renderers=renderers,
                    index=i,
                )
            )

        legend = Legend(
            title='Number of Vertices',
            location=LegendLocation.top_right,
            items=main_legend_items,
        )

        if None:
            legend = Legend(
                title='Clamp Number of Edges?',
                location=LegendLocation.top_left,
                items=legend_items,
            )

            p.add_layout(legend)
        else:
            legend = Legend(
                title='Number Of Table Resizes',
                location=LegendLocation.top_left,
                items=resize_legend_items,
            )

            p.add_layout(legend)


        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_vertices = d['NumberOfVertices'][selected_index];
            var num_vertices;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_vertices = d['NumberOfVertices'][i];
                if (num_vertices == selected_num_vertices) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 2

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if 'sizing_mode' in figure_kwds:
        setattr(p, 'sizing_mode', figure_kwds['sizing_mode'])

    if show_plot:
        show(grid)

    return p

def grid7(df, lrdf=None, min_num_edges=None, max_num_edges=None,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          color_category=None, ncols=None, hash_funcs=None,
          clamp_edges=None, min_num_resizes=None, max_num_resizes=None,
          sample_frac=None, use_tooltips=False):

    from tqdm import tqdm
    from itertools import product

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Legend,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        LegendItem,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.core.enums import (
        LegendLocation,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {}

    if use_tooltips and 'tooltips' not in figure_kwds:
        tooltips = [
            ("Index", "@index"),
            ("Keys", "@KeysName"),
            ("Number of Keys", "@NumberOfKeys"),
            ("Number of Edges", "@NumberOfEdges"),
            ("Number of Vertices", "@NumberOfVertices"),
            ("Number Of Resizes", "@NumberOfTableResizeEvents"),
            ("Keys to Edges Ratio", "@KeysToEdgesRatio{(0.000)}"),
            ("Keys to Vertices Ratio", "@KeysToVerticesRatio{(0.000)}"),
            ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
            ("Clamp Edges?", "@ClampNumberOfEdges"),
        ]
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = list(bp.Spectral11) + list(bp.Category20[20])

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    if clamp_edges is None:
        clamp_edges = ('Y', 'N')

    if hash_funcs is None:
        hash_funcs = (
            df.HashFunction
                .value_counts()
                .sort_index()
                .index
                .values
        )

    num_resizes = (
        df.NumberOfTableResizeEvents
            .value_counts()
            .sort_index()
            .index
            .values
    )

    if min_num_resizes is None:
        min_num_resizes = num_resizes.min()

    if max_num_resizes is None:
        max_num_resizes = num_resizes.max()

    if sample_frac is None:
        sample_frac = 0.1

    targets = [
        (hf, ce) for (hf, ce) in product(hash_funcs, clamp_edges)
    ]

    for (hash_func, clamp) in targets:

        sample = source_df.sample(frac=sample_frac, replace=False)
        df = sample.query(
            f'NumberOfEdges >= {min_num_edges} and '
            f'NumberOfEdges <= {max_num_edges} and '
            f'NumberOfTableResizeEvents >= {min_num_resizes} and '
            f'NumberOfTableResizeEvents <= {max_num_resizes} and '
            f'HashFunction == "{hash_func}" and '
            f'ClampNumberOfEdges == "{clamp}"'
        ).copy()

        df['NumberOfVerticesStr'] = df.NumberOfVertices.values.astype(str)

        num_vertices = (
            df.NumberOfVertices
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_vertices_str = [ str(e) for e in num_vertices ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_vertices)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_vertices_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfVerticesStr.values ]

        df['KeysToVerticesRatio'] = np.around(df.KeysToVerticesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToVerticesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfVertices'])

        p = figure(
            width=900,
            height=900,
            tools='pan,wheel_zoom,box_select,lasso_select,reset,tap,hover',
            **figure_kwds,
        )

        p.title.text = f'{hash_func}, Clamp: {clamp}'
        p.xaxis.axis_label = 'Keys to Vertices Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'
        p.y_range = y_range
        p.x_range = Range1d(0, 0.6)
        if None:
            p.x_range = Range1d(
                df.KeysToVerticesRatio.min(),
                df.KeysToVerticesRatio.max()
            )

        renderers = []
        resize_legend_items = []

        target_df = df
        source = ColumnDataSource(target_df)
        #glyph = getattr(p, BOKEH_GLYPHS[resize])
        glyph = getattr(p, 'scatter')
        g = glyph(
            'KeysToVerticesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_group='NumberOfVertices',
            **scatter_kwds,
        )

        if 0:
            main_legend_items = []
            for (i, num_vert) in enumerate(num_vertices_str):
                main_legend_items.append(
                    LegendItem(
                        label=num_vert,
                        renderers=renderers,
                        index=i,
                    )
                )

            legend = Legend(
                title='Number of Vertices',
                location=LegendLocation.top_right,
                items=main_legend_items,
            )

        if None:
            legend = Legend(
                title='Clamp Number of Edges?',
                location=LegendLocation.top_left,
                items=legend_items,
            )

            p.add_layout(legend)
        elif 0:
            legend = Legend(
                title='Number Of Table Resizes',
                location=LegendLocation.top_left,
                items=resize_legend_items,
            )

            p.add_layout(legend)


        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_vertices = d['NumberOfVertices'][selected_index];
            var num_vertices;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_vertices = d['NumberOfVertices'][i];
                if (num_vertices == selected_num_vertices) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 2

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if 'sizing_mode' in figure_kwds:
        setattr(p, 'sizing_mode', figure_kwds['sizing_mode'])

    if show_plot:
        show(grid)

    return p

def grid8(df, lrdf, min_num_edges=None, max_num_edges=None,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          color_category=None, ncols=None, hash_funcs=None,
          clamp_edges=None, min_num_resizes=None, max_num_resizes=None,
          sample_frac=None, use_tooltips=False):
    """
    Like grid7() but colors are by number of table resizes instead of number
    of edges.
    """

    from tqdm import tqdm
    from itertools import product

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Legend,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        LegendItem,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.core.enums import (
        LegendLocation,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {}

    if use_tooltips and 'tooltips' not in figure_kwds:
        tooltips = [
            ("Index", "@index"),
            ("Keys", "@KeysName"),
            ("Number of Keys", "@NumberOfKeys"),
            ("Number of Edges", "@NumberOfEdges"),
            ("Number of Vertices", "@NumberOfVertices"),
            ("Number Of Resizes", "@NumberOfTableResizeEvents"),
            ("Keys to Edges Ratio", "@KeysToEdgesRatio{(0.000)}"),
            ("Keys to Vertices Ratio", "@KeysToVerticesRatio{(0.000)}"),
            ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
            ("Clamp Edges?", "@ClampNumberOfEdges"),
        ]
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = list(bp.Spectral11) + list(bp.Category20[20])

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    if clamp_edges is None:
        clamp_edges = ('Y', 'N')

    if hash_funcs is None:
        hash_funcs = (
            df.HashFunction
                .value_counts()
                .sort_index()
                .index
                .values
        )

    num_resizes = (
        df.NumberOfTableResizeEvents
            .value_counts()
            .sort_index()
            .index
            .values
    )

    if min_num_resizes is None:
        min_num_resizes = num_resizes.min()

    if max_num_resizes is None:
        max_num_resizes = num_resizes.max()

    if sample_frac is None:
        sample_frac = 0.2

    targets = [
        (hf, ce) for (hf, ce) in product(hash_funcs, clamp_edges)
    ]

    for (hash_func, clamp) in targets:

        sample = source_df.sample(frac=sample_frac, replace=False)
        df = sample.query(
            f'NumberOfEdges >= {min_num_edges} and '
            f'NumberOfEdges <= {max_num_edges} and '
            f'NumberOfTableResizeEvents >= {min_num_resizes} and '
            f'NumberOfTableResizeEvents <= {max_num_resizes} and '
            f'HashFunction == "{hash_func}" and '
            f'ClampNumberOfEdges == "{clamp}"'
        ).copy()

        df['NumberOfTableResizeEventsStr'] = (
            df.NumberOfTableResizeEvents.values.astype(str)
        )

        num_resizes = (
            df.NumberOfTableResizeEvents
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_resizes_str = [ str(e) for e in num_resizes ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_resizes)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_resizes_str) }

        df['Color'] = [
            colors_map[e] for e in df.NumberOfTableResizeEventsStr.values
        ]

        df['KeysToVerticesRatio'] = np.around(df.KeysToVerticesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        sdf = (
            df.groupby(['KeysToVerticesRatio', 'SolutionsFoundRatio'])
              .size()
              .reset_index()
              .rename(columns={0: 'Size'})
        )

        sdf['LogSize'] = np.log(sdf['Size'].values * 2)

        size_map = {}

        for (i, row) in sdf.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            v = (row.Size, row.LogSize)

            size_map[k] = v

        df['Size'] = float(0)
        df['LogSize'] = float(0)

        for (i, row) in df.iterrows():
            k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
            (size, log_size) = size_map[k]
            df.at[i, 'Size'] = size * 3
            df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfTableResizeEvents'])

        tools = 'pan,wheel_zoom,box_select,lasso_select,reset,tap'
        if use_tooltips:
            tools += ',hover'

        p = figure(
            width=900,
            height=900,
            tools=tools,
            **figure_kwds,
        )

        p.title.text = f'{hash_func}, Clamp: {clamp}'
        p.xaxis.axis_label = 'Keys to Vertices Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'
        p.y_range = y_range
        p.x_range = Range1d(0, 0.6)
        if None:
            p.x_range = Range1d(
                df.KeysToVerticesRatio.min(),
                df.KeysToVerticesRatio.max()
            )

        renderers = []
        resize_legend_items = []

        target_df = df
        source = ColumnDataSource(target_df)
        #glyph = getattr(p, BOKEH_GLYPHS[resize])
        glyph = getattr(p, 'scatter')
        g = glyph(
            'KeysToVerticesRatio',
            'SolutionsFoundRatio',
            color='Color',
            size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_group='NumberOfTableResizeEvents',
            **scatter_kwds,
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        code = """
            const d = s.data;
            const selected_index = s.selected.indices[0];
            const selected_num_resizes =
                d['NumberOfTableResizeEvents'][selected_index];
            var num_resizes;
            var new_selection = [];

            for (var i = 0; i < d['index'].length; i++) {
                num_resizes = d['NumberOfTableResizeEvents'][i];
                if (num_resizes == selected_num_resizes) {
                    new_selection.push(i);
                }
            }

            s.selected.indices = new_selection;

            s.change.emit();
        """

        args = {
            's': source,
            'colors_map': colors_map,
        }

        callback = CustomJS(args=args, code=code)

        taptool = p.select(type=TapTool)
        taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = 2

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if 'sizing_mode' in figure_kwds:
        setattr(p, 'sizing_mode', figure_kwds['sizing_mode'])

    if show_plot:
        show(grid)

    return p

def grid9(df, lrdf, min_num_edges=None, max_num_edges=None,
          show_plot=True, figure_kwds=None, scatter_kwds=None,
          color_category=None, ncols=None, hash_funcs=None,
          clamp_edges=None, min_num_resizes=None, max_num_resizes=None,
          sample_frac=None, use_tooltips=False):
    """
    Like grid7() but with slopes.
    """

    import textwrap
    from itertools import product
    from tqdm import tqdm

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Slope,
        Legend,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        LegendItem,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.core.enums import (
        LegendLocation,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {}

    if use_tooltips and 'tooltips' not in figure_kwds:
        tooltips = [
            ("Index", "@index"),
            ("Keys", "@KeysName"),
            ("Number of Keys", "@NumberOfKeys"),
            ("Number of Edges", "@NumberOfEdges"),
            ("Number of Vertices", "@NumberOfVertices"),
            ("Number Of Resizes", "@NumberOfTableResizeEvents"),
            ("Keys to Edges Ratio", "@KeysToEdgesRatio{(0.000)}"),
            ("Keys to Vertices Ratio", "@KeysToVerticesRatio{(0.000)}"),
            ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
            ("Clamp Edges?", "@ClampNumberOfEdges"),
        ]
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = list(bp.Spectral11) + list(bp.Category20[20])

    if clamp_edges is None:
        clamp_edges = False

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    if hash_funcs is None:
        hash_funcs = (
            df.HashFunction
                .value_counts()
                .sort_index()
                .index
                .values
        )

    num_resizes = (
        df.NumberOfTableResizeEvents
            .value_counts()
            .sort_index()
            .index
            .values
    )

    if min_num_resizes is None:
        min_num_resizes = num_resizes.min()

    if max_num_resizes is None:
        max_num_resizes = num_resizes.max()

    if sample_frac is None:
        sample_frac = 0.1

    resize_range = list(range(min_num_resizes, max_num_resizes+1))
    targets = list(product(hash_funcs, resize_range))

    clamp = 'N' if not clamp_edges else 'Y'

    for target in tqdm(targets):

        (hash_func, num_resize) = target

        sample = source_df.sample(frac=sample_frac, replace=False)
        query = (
            f'HashFunction == "{hash_func}" and '
            f'NumberOfEdges >= {min_num_edges} and '
            f'NumberOfEdges <= {max_num_edges} and '
            f'NumberOfTableResizeEvents == {num_resize} and '
            f'ClampNumberOfEdges == "{clamp}"'
        )
        df = sample.query(query).copy()

        df['NumberOfVerticesStr'] = df.NumberOfVertices.values.astype(str)

        num_vertices = (
            df.NumberOfVertices
                .value_counts()
                .sort_index()
                .index
                .values
        )
        num_vertices_str = [ str(e) for e in num_vertices ]

        if isinstance(color_category, dict):
            cat = color_category[len(num_vertices)]
        else:
            cat = color_category

        colors_map = { e: cat[i] for (i, e) in enumerate(num_vertices_str) }

        df['Color'] = [ colors_map[e] for e in df.NumberOfVerticesStr.values ]

        df['KeysToVerticesRatio'] = np.around(df.KeysToVerticesRatio.values, 3)
        df['SolutionsFoundRatio'] = np.around(df.SolutionsFoundRatio.values, 3)

        if None:
            sdf = (
                df.groupby(['KeysToVerticesRatio', 'SolutionsFoundRatio'])
                  .size()
                  .reset_index()
                  .rename(columns={0: 'Size'})
            )

            sdf['LogSize'] = np.log(sdf['Size'].values * 2)

            size_map = {}

            for (i, row) in sdf.iterrows():
                k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
                v = (row.Size, row.LogSize)

                size_map[k] = v

            df['Size'] = float(0)
            df['LogSize'] = float(0)

            for (i, row) in df.iterrows():
                k = (row.KeysToVerticesRatio, row.SolutionsFoundRatio)
                (size, log_size) = size_map[k]
                df.at[i, 'Size'] = size * 3
                df.at[i, 'LogSize'] = log_size

        df.sort_values(by=['NumberOfVertices'])

        tools = 'pan,wheel_zoom,box_select,lasso_select,reset,tap'
        if use_tooltips:
            tools += ',hover'

        p = figure(
            width=500,
            height=500,
            tools=tools,
            **figure_kwds,
        )

        failed = df.FailedAttempts.values.sum()
        cyclic_fail = df.CyclicGraphFailures.values.sum()
        if False:
            pre_fail = df.PreMaskedVertexCollisionFailures.values.sum()
            post_fail = df.PostMaskedVertexCollisionFailures.values.sum()
            p.title.text = (
                f'{hash_func}/R:{num_resize}/C:{clamp}/F:{failed} '
                f'({pre_fail}/{post_fail}/{cyclic_fail})'
            )
        else:
            p.title.text = (
                f'{hash_func}/R:{num_resize}/C:{clamp}/F:{failed} '
                f'({cyclic_fail})'
            )
        p.xaxis.axis_label = 'Keys to Vertices Ratio'
        p.yaxis.axis_label = 'Probability of Finding Solution'
        p.y_range = y_range
        #p.x_range = Range1d(0, 0.6)
        if True:
            p.x_range = Range1d(
                df.KeysToVerticesRatio.min(),
                df.KeysToVerticesRatio.max()
            )

        target_df = df
        source = ColumnDataSource(target_df)
        #glyph = getattr(p, BOKEH_GLYPHS[resize])
        glyph = getattr(p, 'scatter')
        g = glyph(
            'KeysToVerticesRatio',
            'SolutionsFoundRatio',
            color='Color',
            #size='Size',
            fill_alpha=0.5,
            line_alpha=1.0,
            source=source,
            legend_group='NumberOfVertices',
            **scatter_kwds,
        )

        legend_items = []
        make_legend = False

        for (i, num_vertices) in enumerate(df.NumberOfVertices.values):
            lr_query = (
                f'HashFunction == "{hash_func}" and '
                f'ClampNumberOfEdges == "{clamp}" and '
                f'NumberOfVertices == {num_vertices} and '
                f'NumberOfTableResizeEvents == {num_resize}'
            )
            lr = lrdf.query(lr_query)
            slope = Slope(
                gradient=lr.Slope.values[0],
                y_intercept=lr.Intercept.values[0],
                line_color=colors_map[str(num_vertices)],
                line_width=2.0,
            )
            p.add_layout(slope)
            if make_legend:
                legend_items.append(
                    LegendItem(
                        label=lr.Label.values[0],
                        renderers=[g],
                        index=i,
                    )
                )

        if make_legend:
            legend = Legend(
                title='Linear Regression Details',
                location=LegendLocation.top_left,
                items=legend_items,
            )
            p.add_layout(legend)

        if 0:
            main_legend_items = []
            for (i, num_vert) in enumerate(num_vertices_str):
                main_legend_items.append(
                    LegendItem(
                        label=num_vert,
                        renderers=renderers,
                        index=i,
                    )
                )

            legend = Legend(
                title='Number of Vertices',
                location=LegendLocation.top_right,
                items=main_legend_items,
            )

        if None:
            legend = Legend(
                title='Clamp Number of Edges?',
                location=LegendLocation.top_left,
                items=legend_items,
            )

            p.add_layout(legend)
        elif 0:
            legend = Legend(
                title='Number Of Table Resizes',
                location=LegendLocation.top_left,
                items=resize_legend_items,
            )

            p.add_layout(legend)


        p.legend.title = 'Number of Vertices'
        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"

        if use_tooltips:

            code = textwrap.dedent("""
                const d = s.data;
                const selected_index = s.selected.indices[0];
                const selected_num_vertices =
                    d['NumberOfVertices'][selected_index];
                var num_vertices;
                var new_selection = [];

                for (var i = 0; i < d['index'].length; i++) {
                    num_vertices = d['NumberOfVertices'][i];
                    if (num_vertices == selected_num_vertices) {
                        new_selection.push(i);
                    }
                }

                s.selected.indices = new_selection;

                s.change.emit();
            """)

            args = {
                's': source,
                'colors_map': colors_map,
            }

            callback = CustomJS(args=args, code=code)

            taptool = p.select(type=TapTool)
            taptool.callback = callback

        figures.append(p)

    if not ncols:
        ncols = len(resize_range)

    #figures.insert(1, None)
    grid = gridplot(figures, ncols=ncols)

    if 'sizing_mode' in figure_kwds:
        setattr(p, 'sizing_mode', figure_kwds['sizing_mode'])

    if show_plot:
        show(grid)

    return p

def gridplot_hashfunc_seed_byte_count(
        source_df, hash_func, seed_num, which_bytes,
        num_resizes=0, clamp='N',
        width=None, height=None,
        min_num_vertices=None, max_num_vertices=None,
        show_plot=True, figure_kwds=None, sizing_mode=None,
        separate_num_vertices=None, persist_dfs=None):

    assert seed_num in (3, 6), seed_num
    assert num_resizes >= 0 and num_resizes <= 5, num_resizes

    from itertools import product

    import numpy as np

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        FactorRange,
        ColumnDataSource,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp

    if width is None:
        width = 500

    if height is None:
        height = 300

    if figure_kwds is None:
        figure_kwds = {}

    if separate_num_vertices is None:
        separate_num_vertices = False

    if min_num_vertices is None:
        min_num_vertices = max(source_df.NumberOfVertices.min(), 512)

    if max_num_vertices is None:
        max_num_vertices = source_df.NumberOfVertices.max()

    figures = []

    tools = 'pan,wheel_zoom,box_select,lasso_select,reset,tap,hover'

    tooltips = [
        ("Byte Value", "@index"),
        ("Count", "@Count"),
        ("Probability", "@Probability{(0.0000)}"),
    ]

    source_df = source_df.copy()
    source_df = update_df_with_seed_bytes(source_df)

    if 'ClampNumberOfEdges' not in source_df.columns:
        source_df['ClampNumberOfEdges'] = 'N'

    if 'NumberOfTableResizeEvents' not in source_df.columns:
        source_df['NumberOfTableResizeEvents'] = 0

    query = (
        f'HashFunction == "{hash_func}" and '
        f'NumberOfVertices >= {min_num_vertices} and '
        f'NumberOfVertices <= {max_num_vertices} and '
        f'NumberOfTableResizeEvents == {num_resizes} and '
        f'ClampNumberOfEdges == "{clamp}"'
    )
    dfq = source_df.query(query)

    assert not dfq.empty

    num_vertices = dfq.NumberOfVertices.value_counts().sort_index().index

    mask_byte_range = tuple(range(0, 0x1f+1))

    if separate_num_vertices:
        targets = product(num_vertices, which_bytes)
    else:
        targets = which_bytes

    for target in targets:

        if separate_num_vertices:
            (num_vertex, byte_num) = target
        else:
            byte_num = target

        key = f'Seed{seed_num}_Byte{byte_num}'

        if separate_num_vertices:
            df = dfq[dfq.NumberOfVertices == num_vertex]
        else:
            df = dfq

        df = (
            df.groupby(key)
                .size()
                .reindex(mask_byte_range, fill_value=0)
                .reset_index()
                .rename(columns={ 0: 'Count' })
        )

        total = float(df.Count.sum())
        df['Probability'] = df.Count / total
        df['Color'] = bp.viridis(len(df))
        df['ByteValue'] = df[key].apply(str)

        if separate_num_vertices:
            title = f'{hash_func}, {key}, NumberOfVertices: {num_vertex}'
            persist_key = f'Seed{seed_num}_Byte{byte_num}_Vertices{num_vertex}'
        else:
            title = f'{hash_func}, {key}'
            persist_key = key

        if persist_dfs is not None:
            persist_dfs[persist_key] = df

        p = figure(
            x_range=FactorRange(*tuple(str(i) for i in mask_byte_range)),
            y_range=(df.Count.min(), df.Count.max()),
            width=width,
            height=height,
            tools=tools,
            title=title,
            tooltips=tooltips,
            **figure_kwds
        )

        p.background_fill_color = "#eeeeee"
        p.grid.grid_line_color = "white"
        p.xaxis.axis_label = 'Byte Value'
        #p.yaxis.axis_label = 'Probability'
        p.yaxis.axis_label = 'Count'

        source = ColumnDataSource(df)

        b = p.vbar(
            x='ByteValue',
            #top='Probability',
            top='Count',
            fill_color='Color',
            width=1,
            line_color='white',
            fill_alpha=0.8,
            source=source,
        )

        figures.append(p)

    ncols = len(which_bytes)
    grid = gridplot(figures, ncols=ncols)

    if 'sizing_mode' in figure_kwds:
        setattr(p, 'sizing_mode', figure_kwds['sizing_mode'])

    if show_plot:
        show(grid)

    return grid

#===============================================================================
# Bar Plots
#===============================================================================

def bar1(df, lrdf, min_num_edges=None, max_num_edges=None,
         show_plot=True, figure_kwds=None, scatter_kwds=None,
         color_category=None, ncols=None, hash_funcs=None,
         clamp_edges=None, min_num_resizes=None, max_num_resizes=None):

    import textwrap
    from itertools import product
    from tqdm import tqdm

    import numpy as np
    import pandas as pd

    from bokeh.io import (
        show,
    )

    from bokeh.models import (
        Tabs,
        Slope,
        Legend,
        Select,
        TapTool,
        Range1d,
        ColorBar,
        CustomJS,
        LegendItem,
        RangeSlider,
        MultiSelect,
        ColumnDataSource,
        RadioButtonGroup,
    )

    from bokeh.core.enums import (
        LegendLocation,
    )

    from bokeh.plotting import (
        figure,
    )

    from bokeh.layouts import (
        gridplot,
    )

    import bokeh.palettes as bp
    import bokeh.transform as bt

    if figure_kwds is None:
        figure_kwds = {}

    if use_tooltips and 'tooltips' not in figure_kwds:
        tooltips = [
            ("Index", "@index"),
            ("Keys", "@KeysName"),
            ("Number of Keys", "@NumberOfKeys"),
            ("Number of Edges", "@NumberOfEdges"),
            ("Number of Vertices", "@NumberOfVertices"),
            ("Number Of Resizes", "@NumberOfTableResizeEvents"),
            ("Keys to Edges Ratio", "@KeysToEdgesRatio{(0.000)}"),
            ("Keys to Vertices Ratio", "@KeysToVerticesRatio{(0.000)}"),
            ("Solutions Found Ratio", "@SolutionsFoundRatio{(0.000)}"),
            ("Clamp Edges?", "@ClampNumberOfEdges"),
        ]
        figure_kwds['tooltips'] = tooltips

    if scatter_kwds is None:
        scatter_kwds = {}

    if min_num_edges is None:
        min_num_edges = 256

    if max_num_edges is None:
        max_num_edges = df.NumberOfEdges.max()

    if color_category is None:
        color_category = list(bp.Spectral11) + list(bp.Category20[20])

    if clamp_edges is None:
        clamp_edges = False

    figures = []

    source_df = df

    y_range = Range1d(0, 1.0)

    if hash_funcs is None:
        hash_funcs = (
            df.HashFunction
                .value_counts()
                .sort_index()
                .index
                .values
        )

    num_resizes = (
        df.NumberOfTableResizeEvents
            .value_counts()
            .sort_index()
            .index
            .values
    )

    if min_num_resizes is None:
        min_num_resizes = num_resizes.min()

    if max_num_resizes is None:
        max_num_resizes = num_resizes.max()

    resize_range = list(range(min_num_resizes, max_num_resizes+1))
    targets = list(product(hash_funcs, resize_range))

    clamp = 'N' if not clamp_edges else 'Y'

#===============================================================================
# Heat Maps
#===============================================================================
def cache_line_occupancy_heatmap(values, figsize=None,
                                 sort_values=False, cmap=None):
    """
    vals = get_vals_from_mcdf(get_mcdf_for_df_row(df[df.index == 15], mcdf))
    cache_line_occupancy_heatmap(vals)
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    (f, ax) = plt.subplots(figsize=figsize if figsize else (12, 12))

    if cmap is None:
        cmap = sns.color_palette('Spectral_r', n_colors=17)

    if sort_values:
        shape = values.shape
        values = np.flip(np.sort(values.reshape(shape[0] * shape[1])))
        values = values.reshape(shape[0], shape[1])

    ax = sns.heatmap(values, fmt='d', linewidths=0.5, ax=ax,
                     xticklabels=False,
                     yticklabels=False,
                     cbar_kws={
                         'ticks': range(0, 17),
                         'label': 'Number of Assigned per Cache Line',
                     },
                     cmap=cmap)
    return ax

#===============================================================================
# Helpers
#===============================================================================

def get_addresses():
    return

# vim:set ts=8 sw=4 sts=4 tw=80 et                                             :
