/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Graph.h

Abstract:

    This is the header file for the Graph module of the perfect hash library.
    It defines the GRAPH structure and supporting enums and structs, the
    GRAPH_VTBL structure and all associated function pointer typedefs, and
    the ASSIGNED_MEMORY_COVERAGE structures which are used to evaluate the
    cache line occupancy of the assigned data table after a solution has been
    found, if we're in the "find best graph" mode.

    The specifics of the hypergraph logic are contained within the impl files,
    e.g., GraphImpl1.[ch], GraphImpl2.[ch], etc.

    The GRAPH structure and supporting methods are the workhorse of the entire
    perfect hash library.  This module has evolved significantly during the
    course of development as various areas of research were pursued.  It is
    one of the venerable constantly-hacked-on kitchen-sink modules (like the
    context and Chm01 modules), versus the more stable modules like path and
    file, which are much cleaner, and have basically not changed since initial
    implementation.

    For example, when the module was first written, there was no notion of
    finding a best graph, or calculating assigned memory coverage, etc.  These
    concepts and supporting code have all been implemented well after the core
    CHM-based functionality was stable.

    Thus, its best to view the graph module (including supporting impl files)
    as primarily being a vehicle for exploratory research -- only a minimal
    set of features was designed from the start (or rather, based on the CHM
    approach), everything else has evolved organically over the course of the
    project.

--*/

#pragma once

#include "stdafx.h"

#if 0
#ifndef PH_CUDA
#include "stdafx.h"
#else
#include "Rtl.h"
#include "PerfectHashPrivate.h"
#endif
#endif

#include "GraphCounters.h"

//
// Define the primitive key, edge and vertex types and pointers to said types.
//

typedef ULONG KEY;
typedef ULONG EDGE;
typedef ULONG VERTEX;
typedef ULONG DEGREE;
typedef LONG ORDER;
typedef KEY *PKEY;
typedef EDGE *PEDGE;
typedef VERTEX *PVERTEX;
typedef DEGREE *PDEGREE;
typedef union _VERTEX_PAIR {
    struct {
        VERTEX Vertex1;
        VERTEX Vertex2;
    };
    LONGLONG AsLongLong;
    ULONGLONG AsULongLong;
    ULARGE_INTEGER AsULargeInteger;
} VERTEX_PAIR, *PVERTEX_PAIR;

//
// Our third graph implementation uses the following structures.  The 3 suffix
// on the EDGE3 and VERTEX3 type names solely represents the version 3 of the
// implementation (and not, for example, a 3-part hypergraph).
//

typedef union _EDGE3 {
    struct {
        VERTEX Vertex1;
        VERTEX Vertex2;
    };
    VERTEX_PAIR AsVertexPair;
    ULONGLONG AsULongLong;
} EDGE3, *PEDGE3;

typedef union _VERTEX3 {
    struct {

        //
        // The degree of connections for this vertex.
        //

        DEGREE Degree;

        //
        // All edges for this vertex; an incidence list constructed via XOR'ing all
        // edges together (aka "the XOR-trick").
        //

        EDGE Edges;
    };

    ULONGLONG_BYTES Combined;
} VERTEX3, *PVERTEX3;

//
// A core concept of the 2-part hypergraph algorithm for generating a perfect
// hash solution is the "assigned" array.  The size of this array is equal to
// the number of vertices in the graph.  The number of vertices in the graph is
// equal to the number of edges in the graph, rounded up to the next power of
// 2.  The number of edges in the graph is equal to the number of keys in the
// input set, rounded up to the next power of 2.  That is:
//
//  a) Number of keys in the input set.
//  b) Number of edges = number of keys, rounded up to a power of 2.
//  c) Number of vertices = number of edges, rounded up to the next power of 2.
//  d) Number of assigned elements = number of vertices.
//
// For example, KernelBase-2415.keys has 2415 keys, 4096 edges, and 8192
// vertices.  Thus, the assigned array will feature 8192 elements of the given
// underlying type (currently an unsigned 32-bit integer; ULONG).
//
// If, after a certain number of attempts, no graph solution has been found,
// the edges and vertices (and thus, assigned array) get doubled, and more
// attempts are made.  This is referred to as a table resize event.
//
// Poorly performing hash functions will often require numerous table resize
// events before finding a solution.  On the opposite side of the spectrum,
// good hash functions will often require no table resize events.  (We've yet
// to observe the Jenkins hash function resulting in a table resize event*;
// however, its latency is much greater than that of our best performing
// "multiply-shift"-derived routines.)
//
// [*]: This was written well before we found a swath of much faster routines
//      (i.e., MultiplyShiftR, RotateMultiplyXorRotate etc.) that also have
//      never been observed to require a table resize in order to solve; albeit
//      at much poorer solving rates for certain key sets.  Table resizes are
//      still a useful tool to immediately improve the solving rate for a given
//      hash function by many orders of magnitude, so the point still stands.
//
// Typically, the more complex the hash function, the better it performs, and
// vice versa.  Complexity in this case usually correlates directly to the
// number of CPU instructions required to calculate the hash, and thus, the
// latency of the routine.  Less instructions, lower latency, and vice versa.
// (Ignoring outlier instructions like idiv which can take upward of 90 cycles
//  to execute.)
//
// Hash routines with more instructions usually have long dependency chains,
// too, which further inhibits the performance.  That is, each calculation is
// dependent upon the results of the previous calculation, limiting the out-of-
// order and speculative execution capabilities of the CPU.
//
// Evaluating the latency of a given hash function is straight forward: you just
// measure the latency to perform the Index() routine for a given table (which
// results in two executions of the hash function (generating two hash codes),
// each with a different seed, and then two memory lookups into the assigned
// array, using the masked version of each hash code as the index).
//
// The performance of the memory lookups, however, can be highly variable, as
// they are at the mercy of the CPU cache.  The best scenario we can hope for is
// if the containing cache lines for both indices are in the L1 cache.  If we're
// benchmarking a single Index() call (with a constant key), we're guaranteed to
// get L1 cache hits after the first call.  So, the measured latency will tell
// us the best possible performance we can expect; micro-benchmarking at its
// finest.
//
// From a macro-benchmarking perspective, though, the size of the assigned array
// plays a large role.  In fact, it is not the size of the array per se, but the
// distribution of values throughout the array that will ultimately govern the
// likelihood of a cache hit or miss for a given key, assuming the keys being
// looked up are essentially random during the lifetime of the table (i.e. no
// key is any more probable of being looked up than another key).
//
// Are truly random key lookups a good measure of what happens in the real world
// though?  One could argue that a common use case would be hash tables where
// 90% of the lookups are performed against 10% of the keys, for example.  This
// may mean that under realistic workloads, the number of cache lines used by
// those 10% of keys is the critical factor; the remaining 90% of the keys are
// looked up so infrequently that the cost of a cache miss at all levels and
// subsequent memory fetch (so hundreds of cycles instead of 10-20 cycles) is
// ultimately irrelevant.  So, a table with 65536 elements may perform just as
// good, perhaps even better, than a table with 8192 elements, depending upon
// the actual real world workload*.
//
// [*]: We have verified this experimentally (some 3-4 years after writing the
//      above prose): on certain real world workloads, such as runtime function
//      tracing, where keys (relative RIPs of return addresses) have dramatic
//      cardinalities (i.e. top 10 values dominate 99.99999% of all lookups),
//      the size of the assigned table has virtually no impact.  (Because it
//      doesn't impact the cache residency of those frequent top-10 values,
//      which will always be present in the L1 cache.)
//
//      On the flip side, if a real world workload (such as index join on a
//      relatively well-distributed key set) regularly used a much larger key
//      set, then the table size would definitely impact performance, as a
//      larger memory footprint would inevitably lead to more cache misses,
//      and greater cache contention.
//
// In order to quantify the impact of memory coverage of the assigned array,
// we need to be able to measure it.  That is the role of the following types
// being defined.
//
// N.B. Memory coverage is an active work-in-progress.
//
// N.B. The above sentence was written 3-4 years ago.  We can now conclusively
//      say that the assigned memory coverage concept is an integral part of
//      our solution, and the key mechanism behind our notion of finding a
//      "best" graph.
//

typedef VERTEX ASSIGNED;
typedef ASSIGNED *PASSIGNED;

//
// ASSIGNED_SHIFT represents the left-shift amount needed to convert a single
// "ASSIGNED" unit into a corresponding number of bytes.  This is similar to
// the NT kernel macro PAGE_SHIFT, which is the left-shift amount used to take
// a number of pages and convert it into the number of bytes for those pages.
//

#define ASSIGNED_SHIFT 2
C_ASSERT((sizeof(ASSIGNED) >> 1) == ASSIGNED_SHIFT);

#ifndef PAGE_SHIFT
#define PAGE_SHIFT 12
#endif

#ifndef PAGE_SIZE
#define PAGE_SIZE (1 << PAGE_SHIFT) // 4096
#endif

#ifndef LARGE_PAGE_SHIFT
#define LARGE_PAGE_SHIFT 21
#endif

#ifndef LARGE_PAGE_SIZE
#define LARGE_PAGE_SIZE (1 << LARGE_PAGE_SHIFT) // 2097152, or 2MB.
#endif

#ifndef CACHE_LINE_SHIFT
#define CACHE_LINE_SHIFT 6
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE (1 << CACHE_LINE_SHIFT) // 64
#endif

#define NUM_ASSIGNED_PER_PAGE       (PAGE_SIZE       / sizeof(ASSIGNED))
#define NUM_ASSIGNED_PER_LARGE_PAGE (LARGE_PAGE_SIZE / sizeof(ASSIGNED))
#define NUM_ASSIGNED_PER_CACHE_LINE (CACHE_LINE_SIZE / sizeof(ASSIGNED))
#define NUM_CACHE_LINES_PER_PAGE    (PAGE_SIZE / CACHE_LINE_SIZE)

//
// For the human readers that don't like doing C preprocessor mental math,
// some C_ASSERTs to clarify the sizes above:
//

C_ASSERT(NUM_ASSIGNED_PER_PAGE       == 1024);      // Fits within USHORT.
C_ASSERT(NUM_ASSIGNED_PER_LARGE_PAGE == 524288);    // Fits within ULONG.
C_ASSERT(NUM_ASSIGNED_PER_CACHE_LINE == 16);        // Fits within BYTE.
C_ASSERT(NUM_CACHE_LINES_PER_PAGE    == 64);        // Fits within BYTE.

typedef ASSIGNED ASSIGNED_PAGE[NUM_ASSIGNED_PER_PAGE];
typedef ASSIGNED_PAGE *PASSIGNED_PAGE;

typedef ASSIGNED ASSIGNED_LARGE_PAGE[NUM_ASSIGNED_PER_LARGE_PAGE];
typedef ASSIGNED_LARGE_PAGE *PASSIGNED_LARGE_PAGE;

typedef ASSIGNED ASSIGNED_CACHE_LINE[NUM_ASSIGNED_PER_CACHE_LINE];
typedef ASSIGNED_CACHE_LINE *PASSIGNED_CACHE_LINE;

typedef USHORT ASSIGNED_PAGE_COUNT;
typedef ASSIGNED_PAGE_COUNT *PASSIGNED_PAGE_COUNT;

typedef ULONG ASSIGNED_LARGE_PAGE_COUNT;
typedef ASSIGNED_LARGE_PAGE_COUNT *PASSIGNED_LARGE_PAGE_COUNT;

typedef BYTE ASSIGNED_CACHE_LINE_COUNT;
typedef ASSIGNED_CACHE_LINE_COUNT *PASSIGNED_CACHE_LINE_COUNT;

typedef ASSIGNED_CACHE_LINE_COUNT
    PAGE_CACHE_LINE_COUNT[NUM_CACHE_LINES_PER_PAGE];

typedef struct _ASSIGNED_MEMORY_COVERAGE {

    ULONG TotalNumberOfPages;
    ULONG TotalNumberOfLargePages;
    ULONG TotalNumberOfCacheLines;

    union {
        ULONG NumberOfUsedPages;
        ULONG NumberOfPagesUsedByKeysSubset;
    };

    union {
        ULONG NumberOfUsedLargePages;
        ULONG NumberOfLargePagesUsedByKeysSubset;
    };

    union {
        ULONG NumberOfUsedCacheLines;
        ULONG NumberOfCacheLinesUsedByKeysSubset;
    };

    ULONG NumberOfEmptyPages;
    ULONG NumberOfEmptyLargePages;
    ULONG NumberOfEmptyCacheLines;

    ULONG FirstPageUsed;
    ULONG FirstLargePageUsed;
    ULONG FirstCacheLineUsed;

    ULONG LastPageUsed;
    ULONG LastLargePageUsed;
    ULONG LastCacheLineUsed;

    ULONG TotalNumberOfAssigned;

    _Writable_elements_(TotalNumberOfPages)
    PASSIGNED_PAGE_COUNT NumberOfAssignedPerPage;

    _Writable_elements_(TotalNumberOfLargePages)
    PASSIGNED_LARGE_PAGE_COUNT NumberOfAssignedPerLargePage;

    _Writable_elements_(TotalNumberOfCacheLines)
    PASSIGNED_CACHE_LINE_COUNT NumberOfAssignedPerCacheLine;

    //
    // Histogram of cache line counts.  The +1 accounts for the fact that we
    // want to count the number of times 0 occurs, as well, so we need 17 array
    // elements, not 16.
    //

#define TOTAL_NUM_ASSIGNED_PER_CACHE_LINE NUM_ASSIGNED_PER_CACHE_LINE + 1

    ULONG NumberOfAssignedPerCacheLineCounts[TOTAL_NUM_ASSIGNED_PER_CACHE_LINE];
    union {
        ULONG MaxAssignedPerCacheLineCount;
        ULONG MaxAssignedPerCacheLineCountForKeysSubset;
    };

    //
    // If we're calculating memory coverage for a subset of keys, the following
    // counts will reflect the situation where the two vertices for a given key
    // are co-located within the same page, large page and cache line.
    //

    ULONG NumberOfKeysWithVerticesMappingToSamePage;
    ULONG NumberOfKeysWithVerticesMappingToSameLargePage;
    ULONG NumberOfKeysWithVerticesMappingToSameCacheLine;

    ULONG MaxGraphTraversalDepth;
    ULONG TotalGraphTraversals;
    ULONG NumberOfEmptyVertices;
    ULONG NumberOfCollisionsDuringAssignment;

    //
    // Stores the best graph number if applicable.
    //

    ULONG BestGraphNumber;

    //
    // The solution number with respect to other graphs that have been solved.
    //

    ULONGLONG SolutionNumber;

    //
    // Stores Graph->Attempt at the time the memory coverage was captured.
    //

    LONGLONG Attempt;

    //
    // Linear regression performed against NumberOfAssignedPerCacheLineCounts.
    //

    DOUBLE Slope;
    DOUBLE Intercept;
    DOUBLE CorrelationCoefficient;
    DOUBLE PredictedNumberOfFilledCacheLines;

    //
    // Score and rank for the NumberOfAssignedPerCacheLineCounts array.
    //

    ULONGLONG Score;
    DOUBLE Rank;

} ASSIGNED_MEMORY_COVERAGE;
typedef ASSIGNED_MEMORY_COVERAGE *PASSIGNED_MEMORY_COVERAGE;
typedef const ASSIGNED_MEMORY_COVERAGE *PCASSIGNED_MEMORY_COVERAGE;

FORCEINLINE
VOID
CopyCoverage(
    _Out_writes_bytes_all_(sizeof(*Dest)) PASSIGNED_MEMORY_COVERAGE Dest,
    _In_reads_(sizeof(*Source)) PCASSIGNED_MEMORY_COVERAGE Source
    )
{
    //
    // Copy the structure, then clear the pointers.
    //

    CopyInline(Dest, Source, sizeof(*Dest));

    Dest->NumberOfAssignedPerPage = NULL;
    Dest->NumberOfAssignedPerLargePage = NULL;
    Dest->NumberOfAssignedPerCacheLine = NULL;
}

//
// The concept of assigned memory coverage, once introduced, instantly became
// foundational to our notion of finding a best graph.  The main workhorse is
// the ASSIGNED_MEMORY_COVERAGE structure, which is predicated upon the assigned
// table data being an array of ULONGs.
//
// Many years ago, when we first implemented support for writing the source
// code files as part of "compiled perfect hash table" generation, we added
// some logic that dynamically sized the assigned table data based on the max
// C type needed for a given number of edges.  E.g., tables with 65,354 vertices
// and less only needed to use a USHORT array to capture the assigned table
// data, not ULONG.  The resulting compiled perfect hash files enjoyed large
// speed benefits if they could be downsized from ULONG to USHORT, as the memory
// footprint effectively halved, which means fewer cache misses.  The effect
// is this improvement was most notable on the BenchmarkFull exes, as they
// would walk the entire set of keys as part of their benchmark work (versus
// BenchmarkIndex which just called Index() for one key over and over).
//
// However, during solving, we originally just stuck with the assigned table
// data being ULONG, and this trickled through to every part of the coverage
// struct (ASSIGNED_MEMORY_COVERAGE) and supporting methods used to calculate
// coverage (e.g., CalculateAssignedMemoryCoverage() and AVX2 variants).
//
// Issue 14 (https://github.com/tpn/perfecthash/issues/14) was created.  It
// notes that the assigned memory coverage stats aren't accurate if evaluating
// the compiled perfect hash table (when vertices <= 65,354), because they assumed
// assigned table data was ULONG, not USHORT.  i.e., a USHORT-based table data
// array can hold up to 32 assigned elements in a single 64-byte cache line, not
// 16 like the ULONG-based table.
//
// To solve the problem, though, not only do we need to alter the routines that
// calculate the memory coverage, we need to alter the actual data types used
// by graphs for vertices <= 65,354 during solving.
//
// If this were C++, we'd use a template to abstract this detail away.  However,
// we're C, so we just proliferate a bunch of new typedefs for the USHORT,
// 16-bit assigned table data types and supporting routines.  This isn't
// particularly elegant, and you can clearly tell it's been bolted on after the
// fact, *but*, the inelegant code duplication is very much worth it, not just
// so the assigned coverage stats are accurate for vertices <= 65,354, but
// solving time is greatly improved, often by 50% or more.
//
// This is due to the fact that using the ASSIGNED16-derived data types halves
// the memory requirements for each graph instance, which translates to massive
// solving speedups in the real world, particularly on CPUs with a lot of cores
// but weak-sauce L3 caches.
//
// (For example, my 12/24 core AMD Ryzen 9 3900X is a cheap consumer CPU that
//  has an unusually beefy L3 cache of 64MB.  This box will regularly trounce
//  a much more powerful (on paper) Intel Xeon W-2275 box I have that sports
//  14/28 cores but has a weak-sauce 19.2MB L3 cache.  When setting solving
//  concurrency to core count, which is how the whole library was designed to
//  run in order to max perform on underlying hardware, the Xeon box will
//  struggle to maintain 35,000-45,000 solving attempts per second (even with
//  AVX-512 hash function enabled) for HologramWorld-31016.keys+MultiplyShiftR,
//  as all graphs will be thrashing and competing for the L3 cache, whereas the
//  AMD box happily churns out 60,000+ attempts per second, with little to no
//  last-level (L3) cache thrashing.  So, the significant solving rate gained
//  by the introduction of this "ASSIGNED16" derivative is, in large part, an
//  artifact of the reduced memory footprint of each solver graph.)
//

typedef USHORT EDGE16;
typedef USHORT ASSIGNED16;
typedef ASSIGNED16 *PASSIGNED16;

typedef USHORT KEY16;
typedef SHORT ORDER16;
typedef USHORT EDGE16;
typedef USHORT VERTEX16;
typedef USHORT DEGREE16;

typedef EDGE16 *PEDGE16;
typedef VERTEX16 *PVERTEX16;
typedef DEGREE16 *PDEGREE16;

typedef union _VERTEX16_PAIR {
    struct {
        VERTEX16 Vertex1;
        VERTEX16 Vertex2;
    };
    ULONG AsULong;
    ULONG_INTEGER AsULongInteger;
} VERTEX16_PAIR, *PVERTEX16_PAIR;
C_ASSERT(sizeof(VERTEX16_PAIR) == sizeof(ULONG));

//
// Our third graph implementation uses the following structures.  The 3 suffix
// on the EDGE163 and VERTEX163 type names solely represents the version 3 of
// the implementation (and not, for example, a 3-part hypergraph).
//

typedef union _EDGE163 {
    struct {
        VERTEX16 Vertex1;
        VERTEX16 Vertex2;
    };
    VERTEX16_PAIR AsVertex16Pair;
    ULONG AsULong;
} EDGE163, *PEDGE163;

typedef union _VERTEX163 {

    struct {

        //
        // The degree of connections for this vertex.
        //

        DEGREE16 Degree;

        //
        // All edges for this vertex; an incidence list constructed via XOR'ing all
        // edges together (aka "the XOR-trick").
        //

        EDGE16 Edges;
    };

    ULONG_BYTES Combined;

} VERTEX163, *PVERTEX163;

//
// ASSIGNED16_SHIFT represents the left-shift amount needed to convert a single
// "ASSIGNED16" unit into a corresponding number of bytes.  This is similar to
// the NT kernel macro PAGE_SHIFT, which is the left-shift amount used to take
// a number of pages and convert it into the number of bytes for those pages.
//

#define ASSIGNED16_SHIFT 1
C_ASSERT((sizeof(ASSIGNED16) >> 1) == ASSIGNED16_SHIFT);

#define NUM_ASSIGNED16_PER_PAGE       (PAGE_SIZE       / sizeof(ASSIGNED16))
#define NUM_ASSIGNED16_PER_LARGE_PAGE (LARGE_PAGE_SIZE / sizeof(ASSIGNED16))
#define NUM_ASSIGNED16_PER_CACHE_LINE (CACHE_LINE_SIZE / sizeof(ASSIGNED16))

//
// For the human readers that don't like doing C preprocessor mental math,
// some C_ASSERTs to clarify the sizes above:
//

C_ASSERT(NUM_ASSIGNED16_PER_PAGE       == 2048);        // Fits within USHORT.
C_ASSERT(NUM_ASSIGNED16_PER_LARGE_PAGE == 1048576);     // Fits within ULONG.
C_ASSERT(NUM_ASSIGNED16_PER_CACHE_LINE == 32);          // Fits within BYTE.

typedef ASSIGNED16 ASSIGNED16_PAGE[NUM_ASSIGNED16_PER_PAGE];
typedef ASSIGNED16_PAGE *PASSIGNED16_PAGE;

typedef ASSIGNED16 ASSIGNED16_LARGE_PAGE[NUM_ASSIGNED16_PER_LARGE_PAGE];
typedef ASSIGNED16_LARGE_PAGE *PASSIGNED16_LARGE_PAGE;

typedef ASSIGNED16 ASSIGNED16_CACHE_LINE[NUM_ASSIGNED16_PER_CACHE_LINE];
typedef ASSIGNED16_CACHE_LINE *PASSIGNED16_CACHE_LINE;

typedef USHORT ASSIGNED16_PAGE_COUNT;
typedef ASSIGNED16_PAGE_COUNT *PASSIGNED16_PAGE_COUNT;

typedef ULONG ASSIGNED16_LARGE_PAGE_COUNT;
typedef ASSIGNED16_LARGE_PAGE_COUNT *PASSIGNED16_LARGE_PAGE_COUNT;

typedef BYTE ASSIGNED16_CACHE_LINE_COUNT;
typedef ASSIGNED16_CACHE_LINE_COUNT *PASSIGNED16_CACHE_LINE_COUNT;

typedef ASSIGNED16_CACHE_LINE_COUNT
    PAGE_CACHE_LINE_COUNT[NUM_CACHE_LINES_PER_PAGE];

//
// This is the USHORT, 16-bit counterpart to the ULONG, 32-bit based original
// ASSIGNED_MEMORY_COVERAGE.  Many of the scalar total/number fields have been
// kept as ULONG, mainly because I didn't want to invest the cognitive energy
// to determine whether each one could be downscaled to USHORT.  Additionally,
// this variant is larger than ASSIGNED_MEMORY_COVERAGE anyway, as the cache
// line occupancy histogram NumberOfAssignedPerCacheLineCounts is 33 in size,
// not 17 like the original version.
//

typedef struct _ASSIGNED16_MEMORY_COVERAGE {

    ULONG TotalNumberOfPages;
    ULONG TotalNumberOfLargePages;
    ULONG TotalNumberOfCacheLines;

    union {
        ULONG NumberOfUsedPages;
        ULONG NumberOfPagesUsedByKeysSubset;
    };

    union {
        ULONG NumberOfUsedLargePages;
        ULONG NumberOfLargePagesUsedByKeysSubset;
    };

    union {
        ULONG NumberOfUsedCacheLines;
        ULONG NumberOfCacheLinesUsedByKeysSubset;
    };

    ULONG NumberOfEmptyPages;
    ULONG NumberOfEmptyLargePages;
    ULONG NumberOfEmptyCacheLines;

    ULONG FirstPageUsed;
    ULONG FirstLargePageUsed;
    ULONG FirstCacheLineUsed;

    ULONG LastPageUsed;
    ULONG LastLargePageUsed;
    ULONG LastCacheLineUsed;

    ULONG TotalNumberOfAssigned;

    _Writable_elements_(TotalNumberOfPages)
    PASSIGNED16_PAGE_COUNT NumberOfAssignedPerPage;

    _Writable_elements_(TotalNumberOfLargePages)
    PASSIGNED16_LARGE_PAGE_COUNT NumberOfAssignedPerLargePage;

    _Writable_elements_(TotalNumberOfCacheLines)
    PASSIGNED16_CACHE_LINE_COUNT NumberOfAssignedPerCacheLine;

    //
    // Histogram of cache line counts.  The +1 accounts for the fact that we
    // want to count the number of times 0 occurs, as well, so we need 33 array
    // elements, not 32.
    //

#define TOTAL_NUM_ASSIGNED16_PER_CACHE_LINE NUM_ASSIGNED16_PER_CACHE_LINE + 1

    ULONG NumberOfAssignedPerCacheLineCounts[
                                           TOTAL_NUM_ASSIGNED16_PER_CACHE_LINE];
    union {
        ULONG MaxAssignedPerCacheLineCount;
        ULONG MaxAssignedPerCacheLineCountForKeysSubset;
    };

    //
    // If we're calculating memory coverage for a subset of keys, the following
    // counts will reflect the situation where the two vertices for a given key
    // are co-located within the same page, large page and cache line.
    //
    // N.B. Coverage for key subsets has not yet been implemented for the
    //      ASSIGNED16 logic yet.
    //

    ULONG NumberOfKeysWithVerticesMappingToSamePage;
    ULONG NumberOfKeysWithVerticesMappingToSameLargePage;
    ULONG NumberOfKeysWithVerticesMappingToSameCacheLine;

    ULONG MaxGraphTraversalDepth;
    ULONG TotalGraphTraversals;
    ULONG NumberOfEmptyVertices;
    ULONG NumberOfCollisionsDuringAssignment;

    //
    // Stores the best graph number if applicable.
    //

    ULONG BestGraphNumber;

    //
    // The solution number with respect to other graphs that have been solved.
    //

    ULONGLONG SolutionNumber;

    //
    // Stores Graph->Attempt at the time the memory coverage was captured.
    //

    LONGLONG Attempt;

    //
    // Linear regression performed against NumberOfAssigned16PerCacheLineCounts.
    //

    DOUBLE Slope;
    DOUBLE Intercept;
    DOUBLE CorrelationCoefficient;
    DOUBLE PredictedNumberOfFilledCacheLines;

    //
    // Score and rank for the NumberOfAssigned16PerCacheLineCounts array.
    //

    ULONGLONG Score;
    DOUBLE Rank;

} ASSIGNED16_MEMORY_COVERAGE;
typedef ASSIGNED16_MEMORY_COVERAGE *PASSIGNED16_MEMORY_COVERAGE;
typedef const ASSIGNED16_MEMORY_COVERAGE *PCASSIGNED16_MEMORY_COVERAGE;

//
// The 16-bit coverage struct always needs to be equal or greater than the size
// of the normal version, as it simplifies things like when the table does this:
//
//  Table->Coverage16 = Allocator->Vtbl->Calloc(Allocator,
//                                              1,
//                                              sizeof(*Table->Coverage16));
//
//

C_ASSERT(sizeof(ASSIGNED16_MEMORY_COVERAGE) >=
         sizeof(ASSIGNED_MEMORY_COVERAGE));

FORCEINLINE
VOID
CopyCoverage16(
    _Out_writes_bytes_all_(sizeof(*Dest)) PASSIGNED16_MEMORY_COVERAGE Dest,
    _In_reads_(sizeof(*Source)) PCASSIGNED16_MEMORY_COVERAGE Source
    )
{
    //
    // Copy the structure, then clear the pointers.
    //

    CopyInline(Dest, Source, sizeof(*Dest));

    Dest->NumberOfAssignedPerPage = NULL;
    Dest->NumberOfAssignedPerLargePage = NULL;
    Dest->NumberOfAssignedPerCacheLine = NULL;
}

//
// End of USHORT, 16-bit assigned derivatives.
//

//
// Define graph flags.
//

typedef union _GRAPH_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Indicates we've started deletion of edges from the graph.
        //

        ULONG Shrinking:1;

        //
        // When set, indicates the graph has been determined to be acyclic.
        // (This bit is set by IsGraphAcyclic() if the graph is acyclic; it
        //  is checked in GraphAssign().)
        //

        ULONG IsAcyclic:1;

        //
        // When set, indicates graph information has been set via the SetInfo()
        // call.
        //

        ULONG IsInfoSet:1;

        //
        // When set, indicates graph information has been loaded (at least once)
        // via a call to LoadInfo().
        //

        ULONG IsInfoLoaded:1;

        //
        // When set, turns the Graph->Vtbl->Verify() step into a no-op.
        //

        ULONG SkipVerification:1;

        //
        // When set, indicates this graph is the "spare graph" for the context.
        // When attempting to find the best graph solution, a worker thread may
        // register its graph as the current best solution, then use the spare
        // graph (or previous best graph) to continue solving attempts.
        //

        ULONG IsSpare:1;

        //
        // When set, indicates the graph should calculate assigned memory
        // coverage after a solution has been found.
        //

        ULONG WantsAssignedMemoryCoverage:1;

        //
        // When set, indicates the graph wants assigned memory coverage
        // information for a subset of keys (Graph->Context->KeysSubset).
        //

        ULONG WantsAssignedMemoryCoverageForKeysSubset:1;

        //
        // When set, enables additional redundant checks in IsGraphAcyclic()
        // with regards to counting deleted edges.
        //

        ULONG Paranoid:1;

        //
        // When set, indicates the VertexPairs array was successfully allocated
        // with large pages.  (Only applies when the --HashAllKeysFirst table
        // create flag is present.)
        //

        ULONG VertexPairsArrayUsesLargePages:1;

        //
        // When set, indicates the vertex pairs array wants write-combined
        // memory, when applicable.
        //

        ULONG WantsWriteCombiningForVertexPairsArray:1;

        //
        // When set, indicates the vertex pairs array is currently allocated
        // with write-combined page protection.
        //

        ULONG VertexPairsArrayIsWriteCombined:1;

        //
        // When set, indicates that the vertex pairs array wants to have the
        // write-combine page protection removed after all keys have been hashed
        // and prior to adding the vertices to the graph.
        //

        ULONG RemoveWriteCombineAfterSuccessfulHashKeys:1;

        //
        // When set, indicates that the graph used an optimized AVX2 version
        // of the hash function during graph solving.
        //

        ULONG UsedAvx2HashFunction:1;

        //
        // When set, indicates that the graph used an optimized AVX512 version
        // of the hash function during graph solving.
        //

        ULONG UsedAvx512HashFunction:1;

        //
        // When set, indicates the AVX2 memory coverage function was used.
        //

        ULONG UsedAvx2MemoryCoverageFunction:1;

        //
        // When set, indicates the 16-bit hash/assigned infrastructure is
        // active.
        //

        ULONG UsingAssigned16:1;

        //
        // When set, indicates that this is a CUDA graph.
        //

        ULONG IsCuGraph:1;

        //
        // When set, always try to respect the kernel runtime limit (supplied
        // via --CuDevicesKernelRuntimeTargetInMilliseconds), even if the device
        // indicates it has no kernel runtime limit (i.e. is in TCC mode).
        //

        ULONG AlwaysRespectCuKernelRuntimeLimit:1;

        //
        // When set, indicates we're in "find best graph" solving mode.  When
        // clear, indicates we're in "first graph wins" mode.
        //

        ULONG FindBestGraph:1;

        //
        // When set, indicates the current algorithm uses seed masks (which will
        // be populated in Graph->SeedMasks).
        //

        ULONG HasSeedMasks:1;

        //
        // When set, indicates the user has supplied seeds (which will be
        // populated in Graph->FirstSeed onward).
        //

        ULONG HasUserSeeds:1;

        //
        // Unused bits.
        //

        ULONG Unused:10;
    };
    LONG AsLong;
    ULONG AsULong;
} GRAPH_FLAGS;
typedef GRAPH_FLAGS *PGRAPH_FLAGS;
C_ASSERT(sizeof(GRAPH_FLAGS) == sizeof(ULONG));

#define IsGraphInfoSet(Graph) ((Graph)->Flags.IsInfoSet != FALSE)
#define IsGraphInfoLoaded(Graph) ((Graph)->Flags.IsInfoLoaded != FALSE)
#define IsSpareGraph(Graph) ((Graph)->Flags.IsSpare != FALSE)
#define IsCuGraph(Graph) ((Graph)->Flags.IsCuGraph != FALSE)
#define SkipGraphVerification(Graph) ((Graph)->Flags.SkipVerification != FALSE)
#define HasSeedMasks(Graph) ((Graph)->Flags.HasSeedMasks != FALSE)
#define HasUserSeeds(Graph) ((Graph)->Flags.HasUserSeeds != FALSE)
#define AlwaysRespectCuKernelRuntimeLimit(Graph) \
    ((Graph)->Flags.AlwaysRespectCuKernelRuntimeLimit != FALSE)
#define WantsAssignedMemoryCoverage(Graph) \
    ((Graph)->Flags.WantsAssignedMemoryCoverage)
#define WantsAssignedMemoryCoverageForKeysSubset(Graph) \
    ((Graph)->Flags.WantsAssignedMemoryCoverageForKeysSubset)
#define WantsCuRandomHostSeeds(Graph) \
    ((Graph)->Flags.WantsCuRandomHostSeeds != FALSE)
#define IsGraphParanoid(Graph) ((Graph)->Flags.Paranoid != FALSE)
#define IsUsingAssigned16(Graph) ((Graph)->Flags.UsingAssigned16 != FALSE)

#define SetSpareGraph(Graph) (Graph->Flags.IsSpareGraph = TRUE)
#define SetSpareCuGraph(Graph) (Graph->Flags.IsSpareCuGraph = TRUE)

DEFINE_UNUSED_STATE(GRAPH);


//
// Default version of the graph implementation used (i.e. GraphImp11.c vs
// GraphImpl2.c vs GraphImpl3.c).
//

#define DEFAULT_GRAPH_IMPL_VERSION 3

//
// Define the primary dimensions governing the graph size.
//

typedef struct _GRAPH_DIMENSIONS {

    //
    // Number of edges in the graph.  This corresponds to the number of keys
    // in our input set.  If modulus masking is active, the number of keys and
    // the number of edges will be identical.  Otherwise, the number of edges
    // will be the number of keys rounded up to a power of 2.
    //

    ULONG NumberOfEdges;

    //
    // Total number of edges in the graph.  This will be twice the size of the
    // NumberOfEdges value above, due to the quirky way the underlying r-graph
    // algorithm captures two hash values in the same list and offsets the
    // second set after the first, e.g.:
    //
    //      Edge2 = Edge1 + Graph->NumberOfEdges;
    //

    ULONG TotalNumberOfEdges;

    //
    // Number of vertices in the graph.  This will vary based on the masking
    // type.  It is doubled every time a graph resize event is encountered.
    //

    ULONG NumberOfVertices;

    //
    // The number of edges in the graph, rounded up to a power of 2, and then
    // shifted the appropriate amount to extract the exponent part for 2^n.
    //

    BYTE NumberOfEdgesPowerOf2Exponent;

    //
    // As above, but rounded up to the next power of 2 first.
    //

    BYTE NumberOfEdgesNextPowerOf2Exponent;

    //
    // The same exponent logic applied to the number of vertices as per the
    // NumberOfVertices field above.
    //

    BYTE NumberOfVerticesPowerOf2Exponent;

    //
    // And again for the next value.
    //

    BYTE NumberOfVerticesNextPowerOf2Exponent;

} GRAPH_DIMENSIONS;
C_ASSERT(sizeof(GRAPH_DIMENSIONS) == 16);
typedef GRAPH_DIMENSIONS *PGRAPH_DIMENSIONS;

//
// Define various memory offsets associated with a given graph structure.
// This allows parallel worker threads to reset their local GRAPH instance
// back to the initial state each time they want to try a new random seed.
//

typedef struct _GRAPH_INFO {

    //
    // Total number of graphs created.  This will match the maximum concurrency
    // level of the upstream context.
    //

    ULONG NumberOfGraphs;

    //
    // Number of RTL_BITMAP structures used by the graph.
    //

    USHORT NumberOfBitmaps;

    //
    // Pad out to a 4 byte boundary.
    //

    USHORT Padding;

    //
    // Size of the GRAPH structure.
    //

    ULONG SizeOfGraphStruct;

    //
    // System allocation granularity.  We align the memory map for the on-disk
    // structure using this value initially.
    //

    ULONG AllocationGranularity;

    //
    // If a masking type other than modulus is active, the AbsoluteEdge() needs
    // a way to mask edge values that exceed the number of edges in the table.
    // It does this via EdgeMask, which is initialized to the number of edges
    // (which will be power-of-2 sized for non-modulus masking), minus 1, such
    // that all lower bits will be set.
    //

    ULONG EdgeMask;

    //
    // Also capture the mask required to isolate vertices.
    //

    ULONG VertexMask;

    //
    // Number of pages, large pages and cache lines covered by the assigned
    // array.
    //

    ULONG AssignedArrayNumberOfPages;
    ULONG AssignedArrayNumberOfLargePages;
    ULONG AssignedArrayNumberOfCacheLines;

    //
    // Graph dimensions.  This information is duplicated in the graph due to
    // it being accessed frequently.
    //

    GRAPH_DIMENSIONS Dimensions;

    //
    // Pad out to an 8 byte boundary.
    //

    ULONG Padding2;

    //
    // Pointer to the owning context.
    //

    struct _PERFECT_HASH_CONTEXT *Context;

    //
    // Pointer to the previous Info, if applicable.
    //

    struct _GRAPH_INFO *PrevInfo;

    //
    // Array sizes.
    //

    ULONGLONG EdgesSizeInBytes;
    ULONGLONG NextSizeInBytes;
    ULONGLONG FirstSizeInBytes;
    ULONGLONG OrderSizeInBytes;
    ULONGLONG CountsSizeInBytes;
    ULONGLONG DeletedSizeInBytes;
    ULONGLONG Vertices3SizeInBytes;
    ULONGLONG VertexPairsSizeInBytes;
    ULONGLONG ValuesSizeInBytes;

    //
    // We use a union for the Assigned size in order to work with macros in the
    // CUDA GraphCuLoadInfo() routine.
    //

    union {
        ULONGLONG AssignedSizeInBytes;
        ULONGLONG AssignedHostSizeInBytes;
        ULONGLONG AssignedDeviceSizeInBytes;
    };

    //
    // Bitmap buffer sizes.
    //

    ULONGLONG DeletedEdgesBitmapBufferSizeInBytes;
    ULONGLONG VisitedVerticesBitmapBufferSizeInBytes;
    ULONGLONG AssignedBitmapBufferSizeInBytes;
    ULONGLONG IndexBitmapBufferSizeInBytes;

    //
    // Assigned memory coverage buffer sizes for counts.
    //

    ULONGLONG NumberOfAssignedPerCacheLineSizeInBytes;
    ULONGLONG NumberOfAssignedPerPageSizeInBytes;
    ULONGLONG NumberOfAssignedPerLargePageSizeInBytes;

    //
    // The allocation size of all the arrays, bitmap buffers, and memory
    // coverage arrays, rounded up to the nearest page size.
    //

    ULONGLONG AllocSize;

} GRAPH_INFO;
typedef GRAPH_INFO *PGRAPH_INFO;

//
// Declare graph component and define vtbl methods.
//

DECLARE_COMPONENT(Graph, GRAPH);

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(&Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_SET_INFO)(
    _In_ PGRAPH Graph,
    _In_ PGRAPH_INFO Info
    );
typedef GRAPH_SET_INFO *PGRAPH_SET_INFO;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_lock_not_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ENTER_SOLVING_LOOP)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_ENTER_SOLVING_LOOP *PGRAPH_ENTER_SOLVING_LOOP;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_SOLVE)(
    _In_ PGRAPH Graph,
    _Inout_ PGRAPH *NewGraphPointer
    );
typedef GRAPH_SOLVE *PGRAPH_SOLVE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_LOAD_INFO)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_LOAD_INFO *PGRAPH_LOAD_INFO;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_RESET)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_RESET *PGRAPH_RESET;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_LOAD_NEW_SEEDS)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_LOAD_NEW_SEEDS *PGRAPH_LOAD_NEW_SEEDS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_VERIFY)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_VERIFY *PGRAPH_VERIFY;

typedef
_Requires_exclusive_lock_held_(Graph->Lock)
VOID
(STDAPICALLTYPE GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE
      *PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE;

typedef
_Requires_exclusive_lock_held_(Graph->Lock)
VOID
(STDAPICALLTYPE GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET
      *PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_REGISTER_SOLVED)(
    _In_ PGRAPH Graph,
    _Inout_ PGRAPH *NewGraphPointer
    );
typedef GRAPH_REGISTER_SOLVED *PGRAPH_REGISTER_SOLVED;

typedef
_Must_inspect_result_
_Requires_exclusive_lock_held_(Graph->Lock)
BOOLEAN
(STDAPICALLTYPE GRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE)(
    _In_ struct _GRAPH *Graph
    );
typedef GRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE
      *PGRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ADD_KEYS)(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    );
typedef GRAPH_ADD_KEYS *PGRAPH_ADD_KEYS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_HASH_KEYS)(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    );
typedef GRAPH_HASH_KEYS *PGRAPH_HASH_KEYS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ADD_HASHED_KEYS)(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PVERTEX_PAIR VertexPairs
    );
typedef GRAPH_ADD_HASHED_KEYS *PGRAPH_ADD_HASHED_KEYS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ADD_HASHED_KEYS16)(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PVERTEX16_PAIR VertexPairs
    );
typedef GRAPH_ADD_HASHED_KEYS16 *PGRAPH_ADD_HASHED_KEYS16;


typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_IS_ACYCLIC)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_IS_ACYCLIC *PGRAPH_IS_ACYCLIC;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(STDAPICALLTYPE GRAPH_ASSIGN)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_ASSIGN *PGRAPH_ASSIGN;

//
// Some CUDA-specific structs/glue.
//

typedef struct _PH_CU_RANDOM_HOST_SEEDS {
    ULONG TotalNumberOfSeeds;
    ULONG UsedNumberOfSeeds;

    _Writable_elements_(TotalNumberOfSeeds)
    PULONG Seeds;
} PH_CU_RANDOM_HOST_SEEDS;
typedef PH_CU_RANDOM_HOST_SEEDS *PPH_CU_RANDOM_HOST_SEEDS;

typedef struct _GRAPH_SHARED {
    HRESULT HashKeysResult;
    ULONG Padding;
    PHRESULT HashKeysBlockResults;
} GRAPH_SHARED;
typedef GRAPH_SHARED *PGRAPH_SHARED;

typedef struct _GRAPH_SEEDS {

    //
    // Capture the seeds used for each hash function employed by the graph.
    //

    ULONG NumberOfSeeds;
    ULONG Padding;

    union {
        ULONG Seeds[MAX_NUMBER_OF_SEEDS];
        struct {
            union {
                struct {
                    union {
                        ULONG Seed1;
                        ULONG FirstSeed;
                        ULONG_BYTES Seed1Bytes;
                    };
                    union {
                        ULONG Seed2;
                        ULONG_BYTES Seed2Bytes;
                    };
                };
                ULARGE_INTEGER Seeds12;
            };
            union {
                struct {
                    union {
                        ULONG Seed3;
                        ULONG_BYTES Seed3Bytes;
                    };
                    union {
                        ULONG Seed4;
                        ULONG_BYTES Seed4Bytes;
                    };
                };
                ULARGE_INTEGER Seeds34;
            };
            union {
                struct {
                    union {
                        ULONG Seed5;
                        ULONG_BYTES Seed5Bytes;
                    };
                    union {
                        ULONG Seed6;
                        ULONG_BYTES Seed6Bytes;
                    };
                };
                ULARGE_INTEGER Seeds56;
            };
            union {
                struct {
                    union {
                        ULONG Seed7;
                        ULONG_BYTES Seed7Bytes;
                    };
                    union {
                        ULONG Seed8;
                        ULONG LastSeed;
                        ULONG_BYTES Seed8Bytes;
                    };
                };
                ULARGE_INTEGER Seeds78;
            };
        };
    };
} GRAPH_SEEDS;
typedef GRAPH_SEEDS *PGRAPH_SEEDS;

#if 0
//
// cuRAND-specific glue.
//

#ifndef __CUDA_ARCH__
#pragma pack(push, 1)
typedef struct _CU_RNG_STATE_PHILOX4_32_10 {
    DECLSPEC_ALIGN(16)
    ULONG Counter[4];

    DECLSPEC_ALIGN(16)
    ULONG Output[4];

    ULONG Key[4];
    ULONG State;
    ULONG BoxMullerFlag;
    ULONG BoxMullerFlagDouble;
    FLOAT BoxMullerExtra;
    DOUBLE BoxMullerExtraDouble;
} CU_RNG_STATE_PHILOX4_32_10;
#pragma pack(pop)
#else
typedef struct _CU_RNG_STATE_PHILOX4_32_10 {
    UINT4 Counter;
    UINT4 Output;
    UINT4 Key;
    UINT State;
    ULONG BoxMullerFlag;
    ULONG BoxMullerFlagDouble;
    FLOAT BoxMullerExtra;
    DOUBLE BoxMullerExtraDouble;
} CU_RNG_STATE_PHILOX4_32_10;
#endif
typedef CU_RNG_STATE_PHILOX4_32_10 *PCU_RNG_STATE_PHILOX4_32_10;

typedef struct _CU_RNG_STATE {
    PERFECT_HASH_CU_RNG_ID CuRngId;
    ULONG AllocSizeInBytes;
    union {
        CU_RNG_STATE_PHILOX4_32_10 AsPhilox43210;
    };
} CU_RNG_STATE;
typedef CU_RNG_STATE *PCU_RNG_STATE;
#endif

//
// Vtbl.
//

typedef struct _GRAPH_VTBL {
    DECLARE_COMPONENT_VTBL_HEADER(GRAPH);
    PGRAPH_SET_INFO SetInfo;
    PGRAPH_ENTER_SOLVING_LOOP EnterSolvingLoop;
    PGRAPH_VERIFY Verify;
    PGRAPH_LOAD_INFO LoadInfo;
    PGRAPH_RESET Reset;
    PGRAPH_LOAD_NEW_SEEDS LoadNewSeeds;
    PGRAPH_SOLVE Solve;
    PGRAPH_IS_ACYCLIC IsAcyclic;
    PGRAPH_ASSIGN Assign;
    PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE CalculateAssignedMemoryCoverage;
    PGRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET
        CalculateAssignedMemoryCoverageForKeysSubset;
    PGRAPH_REGISTER_SOLVED RegisterSolved;
    PGRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE ShouldWeContinueTryingToSolve;
    PGRAPH_ADD_KEYS AddKeys;
    PGRAPH_HASH_KEYS HashKeys;
} GRAPH_VTBL;
typedef GRAPH_VTBL *PGRAPH_VTBL;

//
// Define the graph structure.
//

typedef struct _Struct_size_bytes_(SizeOfStruct) _GRAPH {
    COMMON_COMPONENT_HEADER(GRAPH);

    //
    // Edge and vertex masks that can be used when non-modulus masking is in
    // place.  Both of these values are duplicated from the info structure as
    // they are accessed frequently.
    //
    //

    ULONG EdgeMask;
    ULONG VertexMask;

    //
    // Ditto for modulus-masking equivalents.
    //

    ULONG EdgeModulus;
    ULONG VertexModulus;

    //
    // Duplicate the mask type, as well, as this directs AbsoluteEdge()'s
    // decision to use the two masks above.
    //

    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;

    //
    // Duplicate the number of keys, as this is also frequently referenced.
    //

    ULONG NumberOfKeys;

    //
    // Pointer to the info structure describing various sizes.
    //

    PGRAPH_INFO Info;

    //
    // Graph attempt.  This ID is derived from an interlocked increment against
    // Context->Attempts, and represents the attempt number across all threads.
    //

    ULONGLONG Attempt;

    //
    // The solution number with respect to other graphs that have been solved.
    //

    ULONGLONG SolutionNumber;

    //
    // The local time the graph was solved, if applicable.
    //

    FILETIME64 SolvedTime;

    //
    // A localized attempt number that reflects the number of attempts made
    // by just this thread.
    //

    ULONG ThreadAttempt;

    //
    // Thread ID of the thread that owns us.  Each callback thread is provided
    // a single graph, and will attempt to solve the perfect hash table until
    // told otherwise.  Thus, there's a 1:1 relationship between graph instance
    // and owning thread.
    //

    ULONG ThreadId;

    //
    // Counter that is incremented each time we delete an edge during the
    // acyclic graph detection stage.
    //

    ULONG DeletedEdgeCount;

    //
    // Counter that is incremented each time we visit a vertex during the
    // assignment stage.
    //

    ULONG VisitedVerticesCount;

    //
    // Capture collisions during assignment step.
    //

    ULONG Collisions;

    //
    // Inline the GRAPH_DIMENSIONS structure.  This is available from the
    // GRAPH_INFO structure, however, it's accessed frequently, so we inline
    // it to avoid the extra level of indirection.
    //

    union {

        struct {
            ULONG NumberOfEdges;
            ULONG TotalNumberOfEdges;
            ULONG NumberOfVertices;
            BYTE NumberOfEdgesPowerOf2Exponent;
            BYTE NumberOfEdgesNextPowerOf2Exponent;
            BYTE NumberOfVerticesPowerOf2Exponent;
            BYTE NumberOfVerticesNextPowerOf2Exponent;
        };

        GRAPH_DIMENSIONS Dimensions;
    };

    //
    // The number of vertices when LoadInfo() was last called.
    //

    ULONG LastLoadedNumberOfVertices;

    //
    // Captures the 0-based index of this graph in the graphs array allocated
    // by the parent table creation thread.
    //

    ULONG Index;

    //
    // Current index into the Order array (used during assignment).
    //

    union {
        volatile LONG OrderIndex;
        volatile SHORT Order16Index;
    };

    volatile LONG OrderIndex1;
    volatile LONG OrderIndex2;
    volatile LONG OrderIndexBoth;
    volatile LONG OrderIndexNone;
    volatile LONG OrderIndexEither;

    ULONG NumberOfNonZeroVertices;

    //
    // Number of empty vertices encountered during the assignment step.
    //

    ULONG NumberOfEmptyVertices;

    //
    // Implementation version of this graph instance.
    //

    ULONG Impl;

    //
    // Duplicate the context pointer.  (This is also available from Info.)
    //

    struct _PERFECT_HASH_CONTEXT *Context;

    //
    // Pointer to an RNG instance.  Used to obtain new random data at the top of
    // each graph solving loop.
    //

    struct _RNG *Rng;

    //
    // As we include the file name of keys in ETW events, we keep a pointer
    // to it here to avoid having to look up six levels of indirection via:
    //      Graph->Context->Table->Keys->Path->FileName.Buffer
    //

    PCWSTR KeysFileName;

    //
    // And as we're poking into the innards of another class, keep a reference
    // to it so we can RELEASE() it during rundown.
    //

    struct _PERFECT_HASH_KEYS *Keys;

    //
    // GUID to use for activity tracking (i.e. the activity field of ETW
    // events).  Currently, this is just a randomly-created GUID.
    //

    GUID Activity;

    //
    // Edges array.
    //

    _Writable_elements_(TotalNumberOfEdges)
    union {
        PEDGE Edges;
        PEDGE16 Edges16;
    };

    //
    // Deletion order.
    //

    _Writable_elements_(NumberOfKeys)
    union {
        PLONG Order;
        PSHORT Order16;
    };

    _Writable_elements_(NumberOfVertices)
    union {
        PLONG OrderByThread;
        PSHORT Order16ByThread;
    };

    _Writable_elements_(NumberOfVertices)
    union {
        PLONG OrderByVertex;
        PSHORT Order16ByVertex;
    };

    _Writable_elements_(NumberOfVertices)
    union {
        PLONG OrderByEdge;
        PSHORT Order16ByEdge;
    };

    _Writable_elements_(NumberOfVertices)
    union {
        PLONG OrderByThreadOrder;
        PSHORT Order16ByThreadOrder;
    };

    volatile LONG OrderByVertexIndex;

    LONG CuScratch;

    union {
        PLONG SortedOrder16;
        PSHORT SortedOrder;
    };

    //
    // Array of the "next" edge array, as per the referenced papers.
    //

    _Writable_elements_(TotalNumberOfEdges)
    union {
        PEDGE Next;
        PEDGE16 Next16;
    };

    //
    // Array of vertices.
    //

    _Writable_elements_(NumberOfVertices)
    union {
        PVERTEX First;
        PVERTEX16 First16;
    };

    //
    // Array of assigned vertices.
    //

    _Writable_elements_(NumberOfVertices)
    union {
        PASSIGNED Assigned;
        PASSIGNED16 Assigned16;
        PASSIGNED AssignedHost;
        PASSIGNED16 Assigned16Host;
    };

    //
    // Array of VERTEX3 elements for the graph impl 3.
    //

    _Writable_elements_(NumberOfVertices)
    union {
        PVERTEX3 Vertices3;
        PVERTEX163 Vertices163;
        PVERTEX3 Vertices3Host;
        PVERTEX163 Vertices163Host;
    };

    PVERTEX3 SortedVertices3;
    PULONG SortedVertices3Indices;

    PEDGE3 OrderedVertices;
    PULONG Indices;

    //
    // Graph implementations 1 & 2: this is an optional array of vertex pairs,
    // indexed by the edge for the key (i.e. the 0-based offset of the key in
    // the keys array).  For implementation 3, this will always contain the
    // array of vertex pairs, indexed by edge for the key.
    //

    _Writable_elements_(NumberOfKeys)
    union {

        //
        // For ASSIGNED_MEMORY_COVERAGE.
        //

        union {
            PVERTEX_PAIR VertexPairs;
            PEDGE3 Edges3;
            PVERTEX_PAIR VertexPairsHost;
            PEDGE3 Edges3Host;
        };

        //
        // For ASSIGNED16_MEMORY_COVERAGE && GraphImpl==3.
        //

        union {
            PVERTEX16_PAIR Vertex16Pairs;
            PEDGE163 Edges163;
            PVERTEX16_PAIR Vertex16PairsHost;
            PEDGE163 Edges163Host;
        };
    };

    //
    // Array of values indexed by the offsets in the Assigned array.  This
    // essentially allows us to simulate a loaded table that supports the
    // Insert(), Index() and Lookup() routines as part of graph validation.
    //

    PULONG Values;

    //
    // Bitmap used to capture deleted edges as part of the acyclic detection
    // stage.  The SizeOfBitMap will reflect TotalNumberOfEdges.
    //

    RTL_BITMAP DeletedEdgesBitmap;

    //
    // Bitmap used to capture vertices visited as part of the assignment stage.
    // The SizeOfBitMap will reflect NumberOfVertices.
    //

    RTL_BITMAP VisitedVerticesBitmap;

    //
    // Bitmap used to test the correctness of the Assigned array.
    //

    RTL_BITMAP AssignedBitmap;

    //
    // Bitmap used to track indices during the assignment step.
    //

    RTL_BITMAP IndexBitmap;

    //
    // Memory coverage information for the assigned array.
    //

    union {
        ASSIGNED_MEMORY_COVERAGE AssignedMemoryCoverage;
        ASSIGNED16_MEMORY_COVERAGE Assigned16MemoryCoverage;
    };

    //
    // Counters to track elapsed cycles and microseconds of graph activities.
    // Each name (typically) maps 1:1 with a corresponding Graph*() function,
    // Elapsed microseconds of the GraphAddKeys() routine.
    //

    DECL_GRAPH_COUNTERS_WITHIN_STRUCT();

    //
    // If this is a GPU solver graph, this points to the solve context.
    //

    struct _PH_CU_SOLVE_CONTEXT *CuSolveContext;

    //
    // Capture the device-side host and device graph and spare graph instances.
    //

    struct _GRAPH *CuHostGraph;
    struct _GRAPH *CuHostSpareGraph;

    struct _GRAPH *CuDeviceGraph;
    struct _GRAPH *CuDeviceSpareGraph;

    //
    // Host and device pointers to keys array.
    //

    _Readable_elements_(NumberOfKeys)
    PKEY HostKeys;
    CU_DEVICE_POINTER DeviceKeys;

    //
    // Pointer to device memory view of GRAPH_INFO.
    //

    PGRAPH_INFO CuGraphInfo;

    //
    // Kernel launch parameters.
    //

    ULONG CuBlocksPerGrid;
    ULONG CuThreadsPerBlock;
    ULONG CuKernelRuntimeTargetInMilliseconds;
    ULONG CuJitMaxNumberOfRegisters;
    ULONG CuRandomNumberBatchSize;

    //
    // Used by CUDA kernels to communicate the result back to the host.
    //

    HRESULT CuKernelResult;

    //
    // Intermediate results communicated back between CUDA parent/child grids.
    //

    HRESULT CuHashKeysResult;
    HRESULT CuIsAcyclicResult;

    //
    // Index of this graph relative to all graphs created for the targeted
    // device.
    //

    LONG CuDeviceIndex;

    //
    // Capture the hash function ID so that CUDA kernels can resolve the correct
    // hash function.
    //

    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;

    //
    // Clock related fields.
    //

    ULONGLONG CuStartClock;
    ULONGLONG CuEndClock;
    ULONGLONG CuCycles;
    ULONGLONG CuElapsedMilliseconds;

    //
    // Various counters.
    //

    ULONG CuNoVertexCollisionFailures;
    ULONG CuVertexCollisionFailures;
    ULONG CuCyclicGraphFailures;
    ULONG CuFailedAttempts;
    ULONG CuFinishedCount;

    //
    // CUDA RNG details.
    //

    PERFECT_HASH_CU_RNG_ID CuRngId;
    ULONGLONG CuRngSeed;
    ULONGLONG CuRngSubsequence;
    ULONGLONG CuRngOffset;
    PVOID CuRngState;

    //
    // Pointer to device attributes in device memory.
    //

    CU_DEVICE_POINTER CuDeviceAttributes;

    //
    // Device addresses.
    //

    //
    // Array of assigned vertices.
    //

    _Writable_elements_(NumberOfVertices)
    union {
        PASSIGNED AssignedDevice;
        PASSIGNED16 Assigned16Device;
    };

    //
    // Array of VERTEX3 elements for the graph impl 3.
    //

    _Writable_elements_(NumberOfVertices)
    union {
        PVERTEX3 Vertices3Device;
        PVERTEX163 Vertices163Device;
    };

    //
    // Graph implementations 1 & 2: this is an optional array of vertex pairs,
    // indexed by the edge for the key (i.e. the 0-based offset of the key in
    // the keys array).  For implementation 3, this will always contain the
    // array of vertex pairs, indexed by edge for the key.
    //

    _Writable_elements_(NumberOfKeys)
    union {

        //
        // For ASSIGNED_MEMORY_COVERAGE.
        //

        union {
            PVERTEX_PAIR VertexPairsDevice;
            PEDGE3 Edges3Device;
        };

        //
        // For ASSIGNED16_MEMORY_COVERAGE && GraphImpl==3.
        //

        union {
            PVERTEX16_PAIR Vertex16PairsDevice;
            PEDGE163 Edges163Device;
        };
    };

    //
    // Optional array of vertex pairs, indexed by number of keys.
    //

    _Writable_elements_(NumberOfKeys)
    PVERTEX_PAIR SortedVertexPairsDevice;

    _Writable_elements_(NumberOfKeys)
    PULONG VertexPairsIndexDevice;

    //
    // CUDA vertex arrays.
    //

    _Writable_elements_(NumberOfKeys)
    PVERTEX Vertices1Device;

    _Writable_elements_(NumberOfKeys)
    PVERTEX Vertices2Device;

    _Writable_elements_(NumberOfKeys)
    PULONG Vertices1IndexDevice;

    _Writable_elements_(NumberOfKeys)
    PULONG Vertices2IndexDevice;

    //
    // CUDA arrays for capturing deleted edges and visited vertices.
    //

    _Writable_elements_(NumberOfVertices)
    volatile ULONG *DeletedDevice;

    _Writable_elements_(NumberOfKeys)
    volatile ULONG *VisitedDevice;

    //
    // CUDA array for capturing count of vertices.
    //

    _Writable_elements_(NumberOfVertices)
    volatile ULONG *CountsDevice;

    //
    // CUDA vertex locks.
    //

    _Writable_elements_(NumberOfVertices)
    PVOID VertexLocks;

    PVOID _Edges;
    PVOID _Degrees;
    LONG _SavedVertices3;
    LONG _SavedVertexPairs;
    PFILE_WORK_ITEM SaveVertices3FileWorkItem;
    PFILE_WORK_ITEM SaveVertexPairsFileWorkItem;
    struct _GRAPH *CpuGraph;

    //
    // Seed masks for the current hash function.
    //

    SEED_MASKS SeedMasks;

    //
    // Opaque context for kernels.
    //

    struct _CU_KERNEL_CONTEXT *CuKernelContext;

    //
    // The graph interface.
    //

    GRAPH_VTBL Interface;

    //
    // The current recursive traversal depth during assignment.
    //

    ULONG TraversalDepth;

    //
    // Maximum recursive traversal depth observed during assignment.
    //

    ULONG MaximumTraversalDepth;

    //
    // Total number of graph traversals performed during assignment.
    //

    ULONG TotalTraversals;

    volatile ULONG TotalNumberOfAssigned;

    union {

        GRAPH_SEEDS GraphSeeds;

        //
        // The GRAPH_SEEDS structure was introduced many years after the GRAPH
        // struct itself was written.  To avoid breaking compatibility with all
        // the existing code that uses the GRAPH struct, we inline the entire
        // GRAPH_SEEDS structure here.  This allows new GPU code to reliably
        // copy the entire GRAPH_SEEDS struct into constant memory, whilst not
        // breaking all of the existing code that features Graph->Seed1, etc.
        //

        struct {

            ULONG NumberOfSeeds;
            ULONG Padding3;

            union {
                ULONG Seeds[MAX_NUMBER_OF_SEEDS];
                struct {
                    union {
                        struct {
                            union {
                                ULONG Seed1;
                                ULONG FirstSeed;
                                ULONG_BYTES Seed1Bytes;
                            };
                            union {
                                ULONG Seed2;
                                ULONG_BYTES Seed2Bytes;
                            };
                        };
                        ULARGE_INTEGER Seeds12;
                    };
                    union {
                        struct {
                            union {
                                ULONG Seed3;
                                ULONG_BYTES Seed3Bytes;
                            };
                            union {
                                ULONG Seed4;
                                ULONG_BYTES Seed4Bytes;
                            };
                        };
                        ULARGE_INTEGER Seeds34;
                    };
                    union {
                        struct {
                            union {
                                ULONG Seed5;
                                ULONG_BYTES Seed5Bytes;
                            };
                            union {
                                ULONG Seed6;
                                ULONG_BYTES Seed6Bytes;
                            };
                        };
                        ULARGE_INTEGER Seeds56;
                    };
                    union {
                        struct {
                            union {
                                ULONG Seed7;
                                ULONG_BYTES Seed7Bytes;
                            };
                            union {
                                ULONG Seed8;
                                ULONG LastSeed;
                                ULONG_BYTES Seed8Bytes;
                            };
                        };
                        ULARGE_INTEGER Seeds78;
                    };
                };
            };
        };
    };

} GRAPH;
typedef GRAPH *PGRAPH;

//
// Locking macro helpers.
//

#define TryAcquireGraphLockExclusive(Graph) \
    TryAcquireSRWLockExclusive(&Graph->Lock)

#define AcquireGraphLockExclusive(Graph) \
    AcquireSRWLockExclusive(&Graph->Lock)

#define ReleaseGraphLockExclusive(Graph) \
    ReleaseSRWLockExclusive(&Graph->Lock)

#define TryAcquireGraphLockShared(Graph) \
    TryAcquireSRWLockShared(&Graph->Lock)

#define AcquireGraphLockShared(Graph) \
    AcquireSRWLockShared(&Graph->Lock)

#define ReleaseGraphLockShared(Graph) \
    ReleaseSRWLockShared(&Graph->Lock)

//
// Graph bit helpers.
//

#define TestGraphBit(Name, BitNumber) \
    (TestBit64(Graph->Name.Buffer, (LONGLONG)BitNumber))

#define SetGraphBit(Name, BitNumber) \
    SetBit64(Graph->Name.Buffer, (LONGLONG)BitNumber)

//
// Private non-vtbl methods.
//

typedef
HRESULT
(NTAPI GRAPH_INITIALIZE)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_INITIALIZE *PGRAPH_INITIALIZE;

typedef
VOID
(NTAPI GRAPH_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PGRAPH Graph
    );
typedef GRAPH_RUNDOWN *PGRAPH_RUNDOWN;

typedef
_Success_(return >= 0)
HRESULT
(NTAPI GRAPH_APPLY_USER_SEEDS)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_APPLY_USER_SEEDS *PGRAPH_APPLY_USER_SEEDS;

typedef
_Success_(return >= 0)
HRESULT
(NTAPI GRAPH_APPLY_SEED_MASKS)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_APPLY_SEED_MASKS *PGRAPH_APPLY_SEED_MASKS;

typedef
_Success_(return >= 0)
HRESULT
(NTAPI GRAPH_APPLY_WEIGHTED_SEED_MASKS)(
    _In_ PGRAPH Graph,
    _In_opt_ PCSEED_MASK_COUNTS SeedMaskCounts
    );
typedef GRAPH_APPLY_WEIGHTED_SEED_MASKS *PGRAPH_APPLY_WEIGHTED_SEED_MASKS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
(NTAPI GRAPH_POST_HASH_KEYS)(
    _In_ HRESULT HashResult,
    _In_ PGRAPH Graph
    );

typedef
_Requires_exclusive_lock_held_(Graph->Lock)
VOID
(NTAPI GRAPH_CALCULATE_MEMORY_COVERAGE_CACHE_LINE_COUNTS)(
    _In_ PGRAPH Graph
    );
typedef GRAPH_CALCULATE_MEMORY_COVERAGE_CACHE_LINE_COUNTS
      *PGRAPH_CALCULATE_MEMORY_COVERAGE_CACHE_LINE_COUNTS;

#ifndef __INTELLISENSE__

//
// Private non-vtbl methods.
//

extern GRAPH_INITIALIZE GraphInitialize;
extern GRAPH_RUNDOWN GraphRundown;
extern GRAPH_APPLY_USER_SEEDS GraphApplyUserSeeds;
extern GRAPH_APPLY_SEED_MASKS GraphApplySeedMasks;
extern GRAPH_APPLY_WEIGHTED_SEED_MASKS GraphApplyWeightedSeedMasks;
extern GRAPH_POST_HASH_KEYS GraphPostHashKeys;

//
// Private vtbl methods.
//
// N.B. These need to come after the GRAPH structure definition in order for
//      the SAL concurrency annotations to work (i.e. _Requires_lock_not_held_).
//

extern GRAPH_SET_INFO GraphSetInfo;
extern GRAPH_ENTER_SOLVING_LOOP GraphEnterSolvingLoop;
extern GRAPH_VERIFY GraphVerify;
extern GRAPH_LOAD_INFO GraphLoadInfo;
extern GRAPH_RESET GraphReset;
extern GRAPH_LOAD_NEW_SEEDS GraphLoadNewSeeds;
extern GRAPH_SOLVE GraphSolve;
extern GRAPH_IS_ACYCLIC GraphIsAcyclic;
extern GRAPH_IS_ACYCLIC GraphIsAcyclic16;
extern GRAPH_ASSIGN GraphAssign;
extern GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE
    GraphCalculateAssignedMemoryCoverage;
extern GRAPH_CALCULATE_ASSIGNED_MEMORY_COVERAGE_FOR_KEYS_SUBSET
    GraphCalculateAssignedMemoryCoverageForKeysSubset;
extern GRAPH_REGISTER_SOLVED GraphRegisterSolved;
#ifdef _M_X64
extern GRAPH_REGISTER_SOLVED GraphRegisterSolvedTsx;
#endif
extern GRAPH_SHOULD_WE_CONTINUE_TRYING_TO_SOLVE
    GraphShouldWeContinueTryingToSolve;
extern GRAPH_ADD_KEYS GraphAddKeys;
extern GRAPH_HASH_KEYS GraphHashKeys;
extern GRAPH_ADD_HASHED_KEYS GraphAddHashedKeys;
#endif

//
// Define a helper macro for hashing keys during graph creation.  Assumes a
// variable named Graph is in scope.  We can't use the Table->Vtbl->Hash()
// vtbl function during graph creation because it obtains the seed data
// from the table header -- which is the appropriate place to find it once
// we're dealing with a previously-created table that has been loaded, but
// won't have a value during the graph solving step because the seed data
// is located in the graph itself.
//

#define SEEDED_HASH(Key, Result)                             \
    if (FAILED(Table->Vtbl->SeededHash(Table,                \
                                       Key,                  \
                                       Graph->NumberOfSeeds, \
                                       &Graph->FirstSeed,    \
                                       Result))) {           \
        goto Error;                                          \
    }

//
// Define an on-disk representation of the graph's information.  This is stored
// in the NTFS stream extending from the backing file named :Info.  It is
// responsible for storing information about the on-disk mapping such that it
// can be reloaded from disk and used as a perfect hash table.  The structure
// must always embed the TABLE_INFO_ON_DISK_HEADER structure such that the
// generic loader routine can access on-disk versions saved by different algos
// in order to extract the algorithm ID and determine a suitable loader func to
// use.
//

typedef struct _Struct_size_bytes_(Header.SizeOfStruct) _GRAPH_INFO_ON_DISK {

    //
    // Include the required header.
    //

    TABLE_INFO_ON_DISK TableInfoOnDisk;

    //
    // Additional information we capture is mostly just for informational
    // and debugging purposes.
    //

    //
    // Inline the GRAPH_DIMENSIONS structure.
    //

    union {

        struct {
            ULONG NumberOfEdges;
            ULONG TotalNumberOfEdges;
            ULONG NumberOfVertices;
            BYTE NumberOfEdgesPowerOf2Exponent;
            BYTE NumberOfEdgesNextPowerOf2Exponent;
            BYTE NumberOfVerticesPowerOf2Exponent;
            BYTE NumberOfVerticesNextPowerOf2Exponent;
        };

        GRAPH_DIMENSIONS Dimensions;
    };

} GRAPH_INFO_ON_DISK;
C_ASSERT(sizeof(GRAPH_INFO_ON_DISK) <= PAGE_SIZE);
typedef GRAPH_INFO_ON_DISK *PGRAPH_INFO_ON_DISK;

//
// Define a helper macro for checking whether or not graph solving should stop.
//

#define MAYBE_STOP_GRAPH_SOLVING(Graph)                       \
    if (GraphShouldWeContinueTryingToSolve(Graph) == FALSE) { \
        return PH_S_GRAPH_SOLVING_STOPPED;                    \
    }

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
