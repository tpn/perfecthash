/*++

Copyright (c) 2021 Trent Nelson <trent@trent.me>

Module Name:

    GraphCounters.h

Abstract:

    This is the header file for graph counters, shared by PerfectHashTable.h
    and Graph.h.  It defines helper macros for managing the elapsed cycles and
    microsecond counters associated with graph activities (and replicated in
    the table struct such that the information is available for writing to the
    CSV output if active).

--*/

#pragma once

//
// N.B. Normally we'd use an X-macro for this, but it feels like a bit of
//      overkill in this situatation, especially considering order-preserving
//      isn't important.  (Or maybe I'm just being lazy and didn't mind copying
//      and pasting the counter names below.)
//

#define DECL_GRAPH_COUNTER_STRUCT_FIELDS(Name) \
    LARGE_INTEGER Name##ElapsedCycles;         \
    LARGE_INTEGER Name##ElapsedMicroseconds

#define DECL_GRAPH_COUNTER_LOCAL_VARS() \
    LONGLONG Cycles;                    \
    LONGLONG Microseconds;              \
    LARGE_INTEGER Start;                \
    LARGE_INTEGER End

#define START_GRAPH_COUNTER() \
    QueryPerformanceCounter(&Start)

#define STOP_GRAPH_COUNTER(Name)                                            \
    QueryPerformanceCounter(&End);                                          \
    Graph->##Name##ElapsedCycles.QuadPart = Cycles = (                      \
        End.QuadPart - Start.QuadPart                                       \
    );                                                                      \
    Microseconds = (Cycles * 1000000) / Graph->Context->Frequency.QuadPart; \
    Graph->##Name##ElapsedMicroseconds.QuadPart = Microseconds

#define RESET_GRAPH_COUNTER(Name)                   \
    Graph->##Name##ElapsedCycles.QuadPart = 0;      \
    Graph->##Name##ElapsedMicroseconds.QuadPart = 0

#define COPY_GRAPH_COUNTER(Name)                  \
    Table->##Name##ElapsedCycles.QuadPart =       \
        Graph->##Name##ElapsedCycles.QuadPart;    \
    Table->##Name##ElapsedMicroseconds.QuadPart = \
        Graph->##Name##ElapsedCycles.QuadPart

#define DECL_GRAPH_COUNTERS_WITHIN_STRUCT()          \
    DECL_GRAPH_COUNTER_STRUCT_FIELDS(AddKeys);       \
    DECL_GRAPH_COUNTER_STRUCT_FIELDS(HashKeys);      \
    DECL_GRAPH_COUNTER_STRUCT_FIELDS(AddHashedKeys); \
    DECL_GRAPH_COUNTER_STRUCT_FIELDS(Assign)

#define RESET_GRAPH_COUNTERS()          \
    RESET_GRAPH_COUNTER(AddKeys);       \
    RESET_GRAPH_COUNTER(HashKeys);      \
    RESET_GRAPH_COUNTER(AddHashedKeys); \
    RESET_GRAPH_COUNTER(Assign)

#define COPY_GRAPH_COUNTERS_FROM_GRAPH_TO_TABLE() \
    COPY_GRAPH_COUNTER(AddKeys);                  \
    COPY_GRAPH_COUNTER(HashKeys);                 \
    COPY_GRAPH_COUNTER(AddHashedKeys);            \
    COPY_GRAPH_COUNTER(Assign)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
