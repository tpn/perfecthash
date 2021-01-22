/*++

Copyright (c) 2021 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashEventsPrivate.h

Abstract:

    This is the private header file for the PerfectHashEvents component.  It is
    intended to be included by modules needing ETW facilities (instead of
    including the <PerfectHashEvents.h> header that ships in the ../../include
    directory).

--*/

#pragma once

//
// 4514: unreferenced inline function removed
//
// 4710: function not inlined
//
// 4820: padding added after member
//
// 5045: spectre mitigation warning
//
// 26451: arithmetic overflow warning
//

#pragma warning(push)
#pragma warning(disable: 4514 4710 4820 5045 26451)
#define RtlZeroMemory(Destination,Length) memset((Destination),0,(Length))
#include <PerfectHashEvents.h>
#undef RtlZeroMemory
#pragma warning(pop)

#define EVENT_WRITE_RTL_RANDOM_BYTES_START(Size) \
    EventWriteRtlGenerateRandomBytesStartEvent(  \
        NULL,                                    \
        Size                                     \
    )

#define EVENT_WRITE_RTL_RANDOM_BYTES_STOP(Size, Result) \
    EventWriteRtlGenerateRandomBytesStopEvent(          \
        NULL,                                           \
        Size,                                           \
        Result                                          \
    )

#define EVENT_WRITE_GRAPH(Name)   \
    EventWriteGraph##Name##Event( \
        &Graph->Activity,         \
        Graph->KeysFileName,      \
        Edge,                     \
        NumberOfKeys,             \
        Key,                      \
        Result,                   \
        Cycles,                   \
        Microseconds,             \
        Graph->Seed1,             \
        Graph->Seed2,             \
        Graph->Seed3,             \
        Graph->Seed4,             \
        Graph->Seed5,             \
        Graph->Seed6,             \
        Graph->Seed7,             \
        Graph->Seed8              \
    )

#define EVENT_WRITE_GRAPH_ADD_KEYS() EVENT_WRITE_GRAPH(AddKeys)
#define EVENT_WRITE_GRAPH_HASH_KEYS() EVENT_WRITE_GRAPH(HashKeys)

#define EVENT_WRITE_GRAPH_HASH_KEYS2() \
    EventWriteGraphHashKeysEvent(      \
        &Graph->Activity,              \
        Graph->KeysFileName,           \
        NumberOfKeys,                  \
        Cycles,                        \
        Microseconds                   \
    )

#define EVENT_WRITE_GRAPH_ADD_HASHED_KEYS() \
    EventWriteGraphAddHashedKeysEvent(      \
        &Graph->Activity,                   \
        Graph->KeysFileName,                \
        NumberOfKeys,                       \
        Cycles,                             \
        Microseconds                        \
    )

#define EVENT_WRITE_GRAPH_IS_ACYCLIC() \
    EventWriteGraphIsAcyclicEvent(     \
        &Graph->Activity,              \
        Graph->KeysFileName,           \
        Graph->Attempt,                \
        Graph->Impl,                   \
        Cycles,                        \
        Microseconds,                  \
        Graph->NumberOfKeys,           \
        Graph->NumberOfVertices,       \
        Graph->Flags.IsAcyclic         \
    )

#define EVENT_WRITE_GRAPH_ASSIGN_START() \
    EventWriteGraphAssignStartEvent(     \
        &Graph->Activity,                \
        Graph->KeysFileName,             \
        Graph->Attempt,                  \
        Graph->NumberOfKeys,             \
        Graph->NumberOfVertices          \
    )

#define EVENT_WRITE_GRAPH_ASSIGN_STOP() \
    EventWriteGraphAssignStopEvent(     \
        &Graph->Activity,               \
        Graph->KeysFileName,            \
        Graph->Attempt,                 \
        Graph->NumberOfKeys,            \
        Graph->NumberOfVertices,        \
        Graph->NumberOfEmptyVertices,   \
        Graph->MaximumTraversalDepth,   \
        Graph->TotalTraversals          \
    )

#define EVENT_WRITE_GRAPH_ASSIGN_RESULT() \
    EventWriteGraphAssignResultEvent(     \
        &Graph->Activity,                 \
        Graph->KeysFileName,              \
        Graph->Attempt,                   \
        Graph->Impl,                      \
        Cycles,                           \
        Microseconds,                     \
        Graph->NumberOfKeys,              \
        Graph->NumberOfVertices           \
    )

#define EVENT_WRITE_GRAPH_FOUND(Name)                             \
    EventWriteGraph##Name##Event(                                 \
        &Graph->Activity,                                         \
        Graph->KeysFileName,                                      \
        Attempt,                                                  \
        Graph->SolutionNumber,                                    \
        ElapsedMilliseconds,                                      \
        (ULONG)CoverageType,                                      \
        CoverageValue,                                            \
        CoverageValueAsDouble,                                    \
        (StopGraphSolving != FALSE),                              \
        (FoundBestGraph != FALSE),                                \
        (FoundEqualBestGraph != FALSE),                           \
        (IsCoverageValueDouble != FALSE),                         \
        EqualCount,                                               \
        Coverage->TotalNumberOfPages,                             \
        Coverage->TotalNumberOfLargePages,                        \
        Coverage->TotalNumberOfCacheLines,                        \
        Coverage->NumberOfUsedPages,                              \
        Coverage->NumberOfUsedLargePages,                         \
        Coverage->NumberOfUsedCacheLines,                         \
        Coverage->NumberOfEmptyPages,                             \
        Coverage->NumberOfEmptyLargePages,                        \
        Coverage->NumberOfEmptyCacheLines,                        \
        Coverage->FirstPageUsed,                                  \
        Coverage->FirstLargePageUsed,                             \
        Coverage->FirstCacheLineUsed,                             \
        Coverage->LastPageUsed,                                   \
        Coverage->LastLargePageUsed,                              \
        Coverage->LastCacheLineUsed,                              \
        Coverage->TotalNumberOfAssigned,                          \
        Coverage->NumberOfKeysWithVerticesMappingToSamePage,      \
        Coverage->NumberOfKeysWithVerticesMappingToSameLargePage, \
        Coverage->NumberOfKeysWithVerticesMappingToSameCacheLine, \
        Coverage->MaxGraphTraversalDepth,                         \
        Coverage->TotalGraphTraversals,                           \
        Graph->Seeds[0],                                          \
        Graph->Seeds[1],                                          \
        Graph->Seeds[2],                                          \
        Graph->Seeds[3],                                          \
        Graph->Seeds[4],                                          \
        Graph->Seeds[5],                                          \
        Graph->Seeds[6],                                          \
        Graph->Seeds[7],                                          \
        Coverage->NumberOfAssignedPerCacheLineCounts[0],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[1],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[2],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[3],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[4],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[5],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[6],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[7],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[8],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[9],          \
        Coverage->NumberOfAssignedPerCacheLineCounts[10],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[11],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[12],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[13],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[14],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[15],         \
        Coverage->NumberOfAssignedPerCacheLineCounts[16],         \
        Coverage->Slope,                                          \
        Coverage->Intercept,                                      \
        Coverage->CorrelationCoefficient,                         \
        Coverage->Score,                                          \
        Coverage->Rank                                            \
    )

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
