/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTimestamp.h

Abstract:

    This is the private header file for timestamp related functionality for the
    perfect hash library.  It defines the TIMESTAMP structure and supporting
    macros for initializing, starting and stopping timestamps.

--*/

#define TIMESTAMP_TO_MICROSECONDS 1000000ULL
#define TIMESTAMP_TO_NANOSECONDS  1000000000ULL

typedef struct _TIMESTAMP {
    ULONGLONG Count;
    union {
        ULONG Aux;
        ULONG CpuId[4];
    };
    LARGE_INTEGER Start;
    LARGE_INTEGER End;
    ULARGE_INTEGER StartTsc;
    ULARGE_INTEGER EndTsc;
    ULARGE_INTEGER Tsc;
    ULARGE_INTEGER TotalTsc;
    ULARGE_INTEGER MinimumTsc;
    ULARGE_INTEGER MaximumTsc;
    ULARGE_INTEGER Cycles;
    ULARGE_INTEGER MinimumCycles;
    ULARGE_INTEGER MaximumCycles;
    ULARGE_INTEGER TotalCycles;
    ULARGE_INTEGER Nanoseconds;
    ULARGE_INTEGER TotalNanoseconds;
    ULARGE_INTEGER MinimumNanoseconds;
    ULARGE_INTEGER MaximumNanoseconds;
} TIMESTAMP;
typedef TIMESTAMP *PTIMESTAMP;

#define INIT_TIMESTAMP(Timestamp)                            \
    ZeroStructPointer((Timestamp));                          \
    (Timestamp)->TotalTsc.QuadPart = 0;                      \
    (Timestamp)->TotalCycles.QuadPart = 0;                   \
    (Timestamp)->TotalNanoseconds.QuadPart = 0;              \
    (Timestamp)->MinimumTsc.QuadPart = (ULONGLONG)-1;        \
    (Timestamp)->MinimumCycles.QuadPart = (ULONGLONG)-1;     \
    (Timestamp)->MinimumNanoseconds.QuadPart = (ULONGLONG)-1

#define RESET_TIMESTAMP(Timestamp)                            \
    (Timestamp)->Count = 0;                                   \
    (Timestamp)->TotalTsc.QuadPart = 0;                       \
    (Timestamp)->TotalCycles.QuadPart = 0;                    \
    (Timestamp)->TotalNanoseconds.QuadPart = 0;               \
    (Timestamp)->MinimumTsc.QuadPart = (ULONGLONG)-1;         \
    (Timestamp)->MaximumTsc.QuadPart = 0;                     \
    (Timestamp)->MinimumCycles.QuadPart = (ULONGLONG)-1;      \
    (Timestamp)->MaximumCycles.QuadPart = 0;                  \
    (Timestamp)->MinimumNanoseconds.QuadPart = (ULONGLONG)-1; \
    (Timestamp)->MaximumNanoseconds.QuadPart = 0

#define START_TIMESTAMP_CPUID(Timestamp)          \
    ++(Timestamp)->Count;                         \
    QueryPerformanceCounter(&(Timestamp)->Start); \
    __cpuid((PULONG)&(Timestamp)->CpuId, 0);      \
    (Timestamp)->StartTsc.QuadPart = __rdtsc()

#define START_TIMESTAMP_RDTSCP(Timestamp)                                      \
    ++(Timestamp)->Count;                                                      \
    QueryPerformanceCounter(&(Timestamp)->Start);                              \
    (Timestamp)->StartTsc.QuadPart = __rdtscp((unsigned int *)&Timestamp->Aux)

#define START_TIMESTAMP_RDTSC(Timestamp)          \
    ++(Timestamp)->Count;                         \
    QueryPerformanceCounter(&(Timestamp)->Start); \
    (Timestamp)->StartTsc.QuadPart = __rdtsc()

#define END_TIMESTAMP_COMMON(Timestamp)                      \
    (Timestamp)->Tsc.QuadPart = (                            \
        (Timestamp)->EndTsc.QuadPart -                       \
        (Timestamp)->StartTsc.QuadPart                       \
    );                                                       \
    (Timestamp)->Cycles.QuadPart = (                         \
        (Timestamp)->End.QuadPart -                          \
        (Timestamp)->Start.QuadPart                          \
    );                                                       \
    (Timestamp)->TotalTsc.QuadPart += (                      \
        (Timestamp)->Tsc.QuadPart                            \
    );                                                       \
    (Timestamp)->TotalCycles.QuadPart += (                   \
        (Timestamp)->Cycles.QuadPart                         \
    );                                                       \
    (Timestamp)->Nanoseconds.QuadPart = (                    \
        (Timestamp)->Cycles.QuadPart *                       \
        TIMESTAMP_TO_NANOSECONDS                             \
    );                                                       \
    (Timestamp)->Nanoseconds.QuadPart /= Frequency.QuadPart; \
    (Timestamp)->TotalNanoseconds.QuadPart += (              \
        (Timestamp)->Nanoseconds.QuadPart                    \
    );                                                       \
    if ((Timestamp)->MinimumNanoseconds.QuadPart >           \
        (Timestamp)->Nanoseconds.QuadPart) {                 \
            (Timestamp)->MinimumNanoseconds.QuadPart = (     \
                (Timestamp)->Nanoseconds.QuadPart            \
            );                                               \
    }                                                        \
    if ((Timestamp)->MaximumNanoseconds.QuadPart <           \
        (Timestamp)->Nanoseconds.QuadPart) {                 \
            (Timestamp)->MaximumNanoseconds.QuadPart = (     \
                (Timestamp)->Nanoseconds.QuadPart            \
            );                                               \
    }                                                        \
    if ((Timestamp)->MinimumTsc.QuadPart >                   \
        (Timestamp)->Tsc.QuadPart) {                         \
            (Timestamp)->MinimumTsc.QuadPart = (             \
                (Timestamp)->Tsc.QuadPart                    \
            );                                               \
    }                                                        \
    if ((Timestamp)->MaximumTsc.QuadPart <                   \
        (Timestamp)->Tsc.QuadPart) {                         \
            (Timestamp)->MaximumTsc.QuadPart = (             \
                (Timestamp)->Tsc.QuadPart                    \
            );                                               \
    }                                                        \
    if ((Timestamp)->MinimumCycles.QuadPart >                \
        (Timestamp)->Cycles.QuadPart) {                      \
            (Timestamp)->MinimumCycles.QuadPart = (          \
                (Timestamp)->Cycles.QuadPart                 \
            );                                               \
    }                                                        \
    if ((Timestamp)->MaximumCycles.QuadPart <                \
        (Timestamp)->Cycles.QuadPart) {                      \
            (Timestamp)->MaximumCycles.QuadPart = (          \
                (Timestamp)->Cycles.QuadPart                 \
            );                                               \
    }

#define END_TIMESTAMP_CPUID(Timestamp)          \
    __cpuid((PULONG)&(Timestamp)->CpuId, 0);    \
    (Timestamp)->EndTsc.QuadPart = __rdtsc();   \
    QueryPerformanceCounter(&(Timestamp)->End); \
    END_TIMESTAMP_COMMON(Id)

#define END_TIMESTAMP_RDTSCP(Timestamp)             \
    (Timestamp)->EndTsc.QuadPart = (                \
        __rdtscp((unsigned int *)&(Timestamp)->Aux) \
    );                                              \
    QueryPerformanceCounter(&(Timestamp)->End);     \
    END_TIMESTAMP_COMMON(Timestamp)

#define END_TIMESTAMP_RDTSC(Timestamp)          \
    (Timestamp)->EndTsc.QuadPart = __rdtsc();   \
    QueryPerformanceCounter(&(Timestamp)->End); \
    END_TIMESTAMP_COMMON(Timestamp)

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
