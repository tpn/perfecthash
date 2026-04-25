/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    GraphImpl4.cpp

Abstract:

    Experimental CPU graph implementation that mirrors GraphImpl3's solver but
    uses a templated C++ core for storage-width and effective-key-width
    specialization.  The C ABI remains unchanged; only these entrypoints are
    exported to the C side.

--*/

#include "stdafx.h"
#include "GraphImpl4.h"
#include "PerfectHashEventsPrivate.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace {

template <typename T, typename NativeT>
union hash_pair_t {
    struct {
        T LowPart;
        T HighPart;
    };
    NativeT AsPair;
};

using hash_pair8_t = hash_pair_t<uint8_t, uint16_t>;
using hash_pair16_t = hash_pair_t<uint16_t, uint32_t>;
using hash_pair32_t = hash_pair_t<uint32_t, uint64_t>;

template <typename ValueType, typename ShiftType>
FORCEINLINE
ValueType
rotate_right(
    ValueType value,
    ShiftType shift
    ) noexcept
{
    constexpr ShiftType bits = static_cast<ShiftType>(sizeof(ValueType) * 8);

    if (shift == 0) {
        return value;
    }

    shift %= bits;
    return static_cast<ValueType>(
        (value >> shift) | (value << (bits - shift))
    );
}

template <typename ResultType, typename KeyType, typename VertexType>
FORCEINLINE
ResultType
hash_key_for_id(
    PERFECT_HASH_HASH_FUNCTION_ID id,
    KeyType key,
    PULONG seeds,
    VertexType mask
    ) noexcept
{
    ResultType result = {};
    VertexType vertex1 = 0;
    VertexType vertex2 = 0;
    VertexType downsized_key = static_cast<VertexType>(key);
    ULONG_BYTES seed3 = {};
    ULONG seed4 = 0;
    ULONG seed5 = 0;

    switch (id) {
        case PerfectHashHashMultiplyShiftRFunctionId:
            seed3.AsULong = seeds[2];
            vertex1 = downsized_key * seeds[0];
            vertex1 = vertex1 >> seed3.Byte1;
            vertex2 = downsized_key * seeds[1];
            vertex2 = vertex2 >> seed3.Byte2;
            result.LowPart = static_cast<decltype(result.LowPart)>(vertex1 & mask);
            result.HighPart = static_cast<decltype(result.HighPart)>(vertex2 & mask);
            return result;

        case PerfectHashHashMultiplyShiftRXFunctionId:
            seed3.AsULong = seeds[2];
            vertex1 = downsized_key * seeds[0];
            vertex1 = vertex1 >> seed3.Byte1;
            vertex2 = downsized_key * seeds[1];
            vertex2 = vertex2 >> seed3.Byte1;
            result.LowPart = static_cast<decltype(result.LowPart)>(vertex1);
            result.HighPart = static_cast<decltype(result.HighPart)>(vertex2);
            return result;

        case PerfectHashHashMulshrolate1RXFunctionId:
            seed3.AsULong = seeds[2];
            vertex1 = downsized_key * seeds[0];
            vertex1 = rotate_right(vertex1, seed3.Byte2);
            vertex1 = vertex1 >> seed3.Byte1;
            vertex2 = downsized_key * seeds[1];
            vertex2 = vertex2 >> seed3.Byte1;
            result.LowPart = static_cast<decltype(result.LowPart)>(vertex1);
            result.HighPart = static_cast<decltype(result.HighPart)>(vertex2);
            return result;

        case PerfectHashHashMulshrolate2RXFunctionId:
            seed3.AsULong = seeds[2];
            vertex1 = downsized_key * seeds[0];
            vertex1 = rotate_right(vertex1, seed3.Byte2);
            vertex1 = vertex1 >> seed3.Byte1;
            vertex2 = downsized_key * seeds[1];
            vertex2 = rotate_right(vertex2, seed3.Byte3);
            vertex2 = vertex2 >> seed3.Byte1;
            result.LowPart = static_cast<decltype(result.LowPart)>(vertex1);
            result.HighPart = static_cast<decltype(result.HighPart)>(vertex2);
            return result;

        case PerfectHashHashMulshrolate3RXFunctionId:
            seed3.AsULong = seeds[2];
            seed4 = seeds[3];
            vertex1 = downsized_key * seeds[0];
            vertex1 = rotate_right(vertex1, seed3.Byte2);
            vertex1 = vertex1 * seed4;
            vertex1 = vertex1 >> seed3.Byte1;
            vertex2 = downsized_key * seeds[1];
            vertex2 = rotate_right(vertex2, seed3.Byte3);
            vertex2 = vertex2 >> seed3.Byte1;
            result.LowPart = static_cast<decltype(result.LowPart)>(vertex1);
            result.HighPart = static_cast<decltype(result.HighPart)>(vertex2);
            return result;

        case PerfectHashHashMulshrolate4RXFunctionId:
            seed3.AsULong = seeds[2];
            seed4 = seeds[3];
            seed5 = seeds[4];
            vertex1 = downsized_key * seeds[0];
            vertex1 = rotate_right(vertex1, seed3.Byte2);
            vertex1 = vertex1 * seed4;
            vertex1 = vertex1 >> seed3.Byte1;
            vertex2 = downsized_key * seeds[1];
            vertex2 = rotate_right(vertex2, seed3.Byte3);
            vertex2 = vertex2 * seed5;
            vertex2 = vertex2 >> seed3.Byte1;
            result.LowPart = static_cast<decltype(result.LowPart)>(vertex1);
            result.HighPart = static_cast<decltype(result.HighPart)>(vertex2);
            return result;

        default:
            return result;
    }
}

template <typename T>
constexpr T
empty_value() noexcept
{
    return std::numeric_limits<T>::max();
}

template <typename T>
constexpr bool
is_empty(T value) noexcept
{
    return value == empty_value<T>();
}

template <typename StorageType>
struct storage_policy;

template <>
struct storage_policy<uint8_t> {
    using storage_type = uint8_t;
    using edge_type = uint8_t;
    using degree_type = uint8_t;
    using vertex_type = uint8_t;
    using order_type = int8_t;
    using assigned_type = uint8_t;
    using pair_type = VERTEX8_PAIR;
    using edge3_type = EDGE83;
    using vertex3_type = VERTEX83;
    using result_pair_type = hash_pair8_t;
    using hash_vertex_type = uint32_t;

    static assigned_type *Assigned(PGRAPH graph) noexcept
    {
        return reinterpret_cast<assigned_type *>(graph->Assigned8);
    }

    static assigned_type *Assigned(PPERFECT_HASH_TABLE table) noexcept
    {
        return reinterpret_cast<assigned_type *>(table->Assigned8);
    }

    static vertex3_type *Vertices(PGRAPH graph) noexcept
    {
        return reinterpret_cast<vertex3_type *>(graph->Vertices83);
    }

    static pair_type *Pairs(PGRAPH graph) noexcept
    {
        return reinterpret_cast<pair_type *>(graph->Vertex8Pairs);
    }

    static edge3_type *Edges(PGRAPH graph) noexcept
    {
        return reinterpret_cast<edge3_type *>(graph->Edges83);
    }

    static order_type *Order(PGRAPH graph) noexcept
    {
        return reinterpret_cast<order_type *>(graph->Order8);
    }

    static volatile order_type &OrderIndex(PGRAPH graph) noexcept
    {
        return reinterpret_cast<volatile order_type &>(graph->Order8Index);
    }
};

template <>
struct storage_policy<uint16_t> {
    using storage_type = uint16_t;
    using edge_type = uint16_t;
    using degree_type = uint16_t;
    using vertex_type = uint16_t;
    using order_type = int16_t;
    using assigned_type = uint16_t;
    using pair_type = VERTEX16_PAIR;
    using edge3_type = EDGE163;
    using vertex3_type = VERTEX163;
    using result_pair_type = hash_pair16_t;
    using hash_vertex_type = uint32_t;

    static assigned_type *Assigned(PGRAPH graph) noexcept
    {
        return reinterpret_cast<assigned_type *>(graph->Assigned16);
    }

    static assigned_type *Assigned(PPERFECT_HASH_TABLE table) noexcept
    {
        return reinterpret_cast<assigned_type *>(table->Assigned16);
    }

    static vertex3_type *Vertices(PGRAPH graph) noexcept
    {
        return reinterpret_cast<vertex3_type *>(graph->Vertices163);
    }

    static pair_type *Pairs(PGRAPH graph) noexcept
    {
        return reinterpret_cast<pair_type *>(graph->Vertex16Pairs);
    }

    static edge3_type *Edges(PGRAPH graph) noexcept
    {
        return reinterpret_cast<edge3_type *>(graph->Edges163);
    }

    static order_type *Order(PGRAPH graph) noexcept
    {
        return reinterpret_cast<order_type *>(graph->Order16);
    }

    static volatile order_type &OrderIndex(PGRAPH graph) noexcept
    {
        return reinterpret_cast<volatile order_type &>(graph->Order16Index);
    }
};

template <>
struct storage_policy<uint32_t> {
    using storage_type = uint32_t;
    using edge_type = uint32_t;
    using degree_type = uint32_t;
    using vertex_type = uint32_t;
    using order_type = int32_t;
    using assigned_type = uint32_t;
    using pair_type = VERTEX_PAIR;
    using edge3_type = EDGE3;
    using vertex3_type = VERTEX3;
    using result_pair_type = hash_pair32_t;
    using hash_vertex_type = uint32_t;

    static assigned_type *Assigned(PGRAPH graph) noexcept
    {
        return reinterpret_cast<assigned_type *>(graph->Assigned);
    }

    static assigned_type *Assigned(PPERFECT_HASH_TABLE table) noexcept
    {
        return reinterpret_cast<assigned_type *>(table->Assigned);
    }

    static vertex3_type *Vertices(PGRAPH graph) noexcept
    {
        return reinterpret_cast<vertex3_type *>(graph->Vertices3);
    }

    static pair_type *Pairs(PGRAPH graph) noexcept
    {
        return reinterpret_cast<pair_type *>(graph->VertexPairs);
    }

    static edge3_type *Edges(PGRAPH graph) noexcept
    {
        return reinterpret_cast<edge3_type *>(graph->Edges3);
    }

    static order_type *Order(PGRAPH graph) noexcept
    {
        return reinterpret_cast<order_type *>(graph->Order);
    }

    static volatile order_type &OrderIndex(PGRAPH graph) noexcept
    {
        return reinterpret_cast<volatile order_type &>(graph->OrderIndex);
    }
};

template <typename StoragePolicy>
inline void
set_pair(
    typename StoragePolicy::pair_type &pair,
    typename StoragePolicy::vertex_type low,
    typename StoragePolicy::vertex_type high
    ) noexcept
{
    pair.Vertex1 = low;
    pair.Vertex2 = high;
}

template <typename KeyType>
KeyType
compact_key(
    PPERFECT_HASH_TABLE table,
    ULONG key
    ) noexcept
{
    if (table->GraphImpl4EffectiveKeySizeInBytes >= sizeof(ULONG) ||
        table->GraphImpl4KeyDownsizeBitmap == 0) {
        return static_cast<KeyType>(key);
    }

    uint64_t value = static_cast<uint64_t>(key);
    uint64_t result;

    if (table->GraphImpl4KeyDownsizeContiguous != FALSE) {
        result = (
            value >> table->GraphImpl4KeyDownsizeTrailingZeros
        ) & table->GraphImpl4KeyDownsizeShiftedMask;
    } else {
        result = ExtractBits64(value, table->GraphImpl4KeyDownsizeBitmap);
    }

    return static_cast<KeyType>(result);
}

template <typename StoragePolicy>
void
add_edge_impl(
    PGRAPH graph,
    typename StoragePolicy::edge_type edge,
    typename StoragePolicy::vertex_type vertex1_index,
    typename StoragePolicy::vertex_type vertex2_index
    ) noexcept
{
    auto *vertex1 = &StoragePolicy::Vertices(graph)[vertex1_index];
    auto *vertex2 = &StoragePolicy::Vertices(graph)[vertex2_index];

    vertex1->Edges ^= edge;
    ++vertex1->Degree;

    vertex2->Edges ^= edge;
    ++vertex2->Degree;
}

template <typename StoragePolicy>
void
remove_vertex_impl(
    PGRAPH graph,
    typename StoragePolicy::vertex_type vertex_index
    ) noexcept
{
    using vertex_type = typename StoragePolicy::vertex_type;
    using edge_type = typename StoragePolicy::edge_type;
    using order_type = typename StoragePolicy::order_type;

    auto *vertex = &StoragePolicy::Vertices(graph)[vertex_index];
    if (vertex->Degree != 1) {
        return;
    }

    edge_type edge = vertex->Edges;
    auto *edge3 = &StoragePolicy::Edges(graph)[edge];

    if (is_empty<vertex_type>(edge3->Vertex1) ||
        is_empty<vertex_type>(edge3->Vertex2)) {
        return;
    }

    auto *vertex1 = &StoragePolicy::Vertices(graph)[edge3->Vertex1];
    if (vertex1->Degree >= 1) {
        vertex1->Edges ^= edge;
        --vertex1->Degree;
    }

    auto *vertex2 = &StoragePolicy::Vertices(graph)[edge3->Vertex2];
    if (vertex2->Degree >= 1) {
        vertex2->Edges ^= edge;
        --vertex2->Degree;
    }

    graph->DeletedEdgeCount++;

    volatile order_type &order_index = StoragePolicy::OrderIndex(graph);
    order_type next = static_cast<order_type>(order_index - 1);
    order_index = next;
    StoragePolicy::Order(graph)[next] = static_cast<order_type>(edge);
}

template <typename StoragePolicy, typename KeyType>
HRESULT
hash_keys_impl(
    PGRAPH graph,
    ULONG number_of_keys,
    PKEY keys
    )
{
    PGRAPH Graph = graph;
    using result_pair_type = typename StoragePolicy::result_pair_type;
    using vertex_type = typename StoragePolicy::vertex_type;
    using hash_vertex_type = typename StoragePolicy::hash_vertex_type;

    auto *table = graph->Context->Table;
    auto hash_mask = static_cast<hash_vertex_type>(table->HashMask);
    auto *seeds = &graph->FirstSeed;
    auto *pairs = StoragePolicy::Pairs(graph);

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    START_GRAPH_COUNTER();

    for (ULONG index = 0; index < number_of_keys; index++) {
        KeyType key = compact_key<KeyType>(table, keys[index]);
        result_pair_type hash = hash_key_for_id<result_pair_type,
                                                KeyType,
                                                hash_vertex_type>(
            table->HashFunctionId,
            key,
            seeds,
            hash_mask
        );

        if (hash.LowPart == hash.HighPart) {
            STOP_GRAPH_COUNTER(HashKeys);
            EVENT_WRITE_GRAPH(HashKeys);
            return PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
        }

        set_pair<StoragePolicy>(pairs[index],
                                static_cast<vertex_type>(hash.LowPart),
                                static_cast<vertex_type>(hash.HighPart));
    }

    STOP_GRAPH_COUNTER(HashKeys);
    EVENT_WRITE_GRAPH(HashKeys);

    return S_OK;
}

template <typename StoragePolicy, typename KeyType>
HRESULT
add_keys_impl(
    PGRAPH graph,
    ULONG number_of_keys,
    PKEY keys
    )
{
    PGRAPH Graph = graph;
    using result_pair_type = typename StoragePolicy::result_pair_type;
    using vertex_type = typename StoragePolicy::vertex_type;
    using edge_type = typename StoragePolicy::edge_type;
    using hash_vertex_type = typename StoragePolicy::hash_vertex_type;

    auto *table = graph->Context->Table;
    auto hash_mask = static_cast<hash_vertex_type>(table->HashMask);
    auto *seeds = &graph->FirstSeed;
    auto *pairs = StoragePolicy::Pairs(graph);

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    START_GRAPH_COUNTER();

    for (ULONG index = 0; index < number_of_keys; index++) {
        KeyType key = compact_key<KeyType>(table, keys[index]);
        result_pair_type hash = hash_key_for_id<result_pair_type,
                                                KeyType,
                                                hash_vertex_type>(
            table->HashFunctionId,
            key,
            seeds,
            hash_mask
        );

        if (hash.LowPart == hash.HighPart) {
            STOP_GRAPH_COUNTER(AddKeys);
            EVENT_WRITE_GRAPH(AddKeys);
            return PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
        }

        set_pair<StoragePolicy>(pairs[index],
                                static_cast<vertex_type>(hash.LowPart),
                                static_cast<vertex_type>(hash.HighPart));

        add_edge_impl<StoragePolicy>(graph,
                                     static_cast<edge_type>(index),
                                     static_cast<vertex_type>(hash.LowPart),
                                     static_cast<vertex_type>(hash.HighPart));
    }

    STOP_GRAPH_COUNTER(AddKeys);
    EVENT_WRITE_GRAPH(AddKeys);

    return S_OK;
}

template <typename StoragePolicy>
HRESULT
add_hashed_keys_impl(
    PGRAPH graph,
    ULONG number_of_keys
    )
{
    PGRAPH Graph = graph;
    using edge_type = typename StoragePolicy::edge_type;

    auto *pairs = StoragePolicy::Pairs(graph);

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    START_GRAPH_COUNTER();

    for (ULONG index = 0; index < number_of_keys; index++) {
        auto pair = pairs[index];
        add_edge_impl<StoragePolicy>(graph,
                                     static_cast<edge_type>(index),
                                     pair.Vertex1,
                                     pair.Vertex2);
    }

    STOP_GRAPH_COUNTER(AddHashedKeys);
    EVENT_WRITE_GRAPH_ADD_HASHED_KEYS();

    return S_OK;
}

template <typename StoragePolicy>
HRESULT
is_acyclic_impl(
    PGRAPH graph
    )
{
    PGRAPH Graph = graph;
    using vertex_type = typename StoragePolicy::vertex_type;
    using order_type = typename StoragePolicy::order_type;
    using edge_type = typename StoragePolicy::edge_type;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    graph->Flags.Shrinking = TRUE;
    StoragePolicy::OrderIndex(graph) = static_cast<order_type>(graph->NumberOfKeys);

    START_GRAPH_COUNTER();
    for (ULONG vertex = 0; vertex < graph->NumberOfVertices; vertex++) {
        remove_vertex_impl<StoragePolicy>(graph, static_cast<vertex_type>(vertex));
    }
    STOP_GRAPH_COUNTER(IsAcyclicPhase1);

    START_GRAPH_COUNTER();
    for (order_type index = static_cast<order_type>(graph->NumberOfKeys);
         StoragePolicy::OrderIndex(graph) > 0 &&
         index > StoragePolicy::OrderIndex(graph);
         NOTHING) {

        edge_type edge_index = static_cast<edge_type>(StoragePolicy::Order(graph)[--index]);
        auto *edge3 = &StoragePolicy::Edges(graph)[edge_index];
        remove_vertex_impl<StoragePolicy>(graph, edge3->Vertex1);
        remove_vertex_impl<StoragePolicy>(graph, edge3->Vertex2);
    }
    STOP_GRAPH_COUNTER(IsAcyclicPhase2);

    if (graph->DeletedEdgeCount != graph->NumberOfKeys) {
        EVENT_WRITE_GRAPH_IS_ACYCLIC();
        return PH_E_GRAPH_CYCLIC_FAILURE;
    }

    graph->Flags.IsAcyclic = TRUE;
    EVENT_WRITE_GRAPH_IS_ACYCLIC();
    return S_OK;
}

template <typename StoragePolicy>
HRESULT
assign_impl(
    PGRAPH graph
    )
{
    PGRAPH Graph = graph;
    using vertex_type = typename StoragePolicy::vertex_type;
    using assigned_type = typename StoragePolicy::assigned_type;
    using order_type = typename StoragePolicy::order_type;
    using edge_type = typename StoragePolicy::edge_type;

    auto *table = graph->Context->Table;
    auto *assigned = StoragePolicy::Assigned(graph);
    auto *order = StoragePolicy::Order(graph);
    auto *edges = StoragePolicy::Edges(graph);
    const assigned_type number_of_edges =
        static_cast<assigned_type>(graph->NumberOfEdges);

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    UNREFERENCED_PARAMETER(table);

    ASSERT(graph->Flags.IsAcyclic);

    EVENT_WRITE_GRAPH_ASSIGN_START();
    START_GRAPH_COUNTER();

    for (ULONG index = 0; index < graph->NumberOfKeys; index++) {
        order_type order_value = order[index];
        auto *edge = &edges[static_cast<edge_type>(order_value)];

        vertex_type vertex1;
        vertex_type vertex2;

        if (!TestGraphBit(VisitedVerticesBitmap, edge->Vertex1)) {
            vertex1 = edge->Vertex1;
            vertex2 = edge->Vertex2;
        } else {
            vertex1 = edge->Vertex2;
            vertex2 = edge->Vertex1;
        }

        assigned_type value = static_cast<assigned_type>(
            static_cast<assigned_type>(order_value) - assigned[vertex2]
        );
        if (value >= number_of_edges) {
            value = static_cast<assigned_type>(value + number_of_edges);
        }

        assigned[vertex1] = value;
        SetGraphBit(VisitedVerticesBitmap, vertex1);
        SetGraphBit(VisitedVerticesBitmap, vertex2);
    }

    STOP_GRAPH_COUNTER(Assign);
    EVENT_WRITE_GRAPH_ASSIGN_STOP();
    EVENT_WRITE_GRAPH_ASSIGN_RESULT();

    return S_OK;
}

template <typename StoragePolicy, typename KeyType>
HRESULT
index_impl_with_effective_key(
    PERFECT_HASH_HASH_FUNCTION_ID hash_function_id,
    PULONG seeds,
    typename StoragePolicy::assigned_type *assigned,
    ULONG hash_mask,
    ULONG index_mask,
    ULONG key,
    PULONG index
    )
{
    using result_pair_type = typename StoragePolicy::result_pair_type;
    using assigned_type = typename StoragePolicy::assigned_type;
    using hash_vertex_type = typename StoragePolicy::hash_vertex_type;

    KeyType effective_key = static_cast<KeyType>(key);
    result_pair_type hash = hash_key_for_id<result_pair_type,
                                            KeyType,
                                            hash_vertex_type>(
        hash_function_id,
        effective_key,
        seeds,
        static_cast<hash_vertex_type>(hash_mask)
    );

    if (hash.LowPart == hash.HighPart) {
        *index = 0;
        return E_FAIL;
    }

    assigned_type vertex1 = assigned[hash.LowPart];
    assigned_type vertex2 = assigned[hash.HighPart];
    *index = static_cast<ULONG>(
        (static_cast<uint32_t>(vertex1) + static_cast<uint32_t>(vertex2)) &
        index_mask
    );
    return S_OK;
}

template <typename StoragePolicy, typename KeyType>
HRESULT
index_impl_with_assigned(
    PERFECT_HASH_HASH_FUNCTION_ID hash_function_id,
    PULONG seeds,
    typename StoragePolicy::assigned_type *assigned,
    ULONG hash_mask,
    ULONG index_mask,
    PPERFECT_HASH_TABLE table,
    ULONG key,
    PULONG index
    )
{
    return index_impl_with_effective_key<StoragePolicy, KeyType>(
        hash_function_id,
        seeds,
        assigned,
        hash_mask,
        index_mask,
        compact_key<KeyType>(table, key),
        index
    );
}

template <typename StoragePolicy, typename KeyType>
HRESULT
index_impl(
    PPERFECT_HASH_TABLE table,
    ULONG key,
    PULONG index
    )
{
    return index_impl_with_assigned<StoragePolicy, KeyType>(
        table->HashFunctionId,
        &table->TableInfoOnDisk->FirstSeed,
        StoragePolicy::Assigned(table),
        table->HashMask,
        table->IndexMask,
        table,
        key,
        index
    );
}

template <typename StorageType, typename KeyType>
HRESULT
index_effective_key_dispatch_impl(PPERFECT_HASH_TABLE table,
                                  ULONG key,
                                  PULONG index)
{
    //
    // The key has already passed through the composed 64-bit-to-effective-key
    // bitmap used by loaded-table Index64.  Do not apply inner compact-key
    // extraction again here.
    //
    return index_impl_with_effective_key<storage_policy<StorageType>, KeyType>(
        table->HashFunctionId,
        &table->TableInfoOnDisk->FirstSeed,
        storage_policy<StorageType>::Assigned(table),
        table->HashMask,
        table->IndexMask,
        key,
        index
    );
}

template <typename StoragePolicy, typename KeyType>
HRESULT
verify_impl(
    PGRAPH graph
    )
{
    PGRAPH Graph = graph;
    PRTL rtl;
    PALLOCATOR allocator;
    PPERFECT_HASH_TABLE table;
    PULONG values = nullptr;
    ULONG number_of_assignments;
    ULONG collisions = 0;

    ASSERT(graph != nullptr);

    if (SkipGraphVerification(graph)) {
        return PH_S_GRAPH_VERIFICATION_SKIPPED;
    }

    table = graph->Context->Table;
    rtl = graph->Context->Rtl;
    allocator = graph->Allocator;

    number_of_assignments = rtl->RtlNumberOfSetBits(&graph->AssignedBitmap);
    ASSERT(number_of_assignments == 0);

    values = graph->Values;
    if (!values) {
        values = graph->Values = static_cast<PULONG>(
            allocator->Vtbl->Calloc(
                allocator,
                graph->Info->ValuesSizeInBytes,
                sizeof(*graph->Values)
            )
        );
    }

    if (!values) {
        return E_OUTOFMEMORY;
    }

    auto *keys = static_cast<PKEY>(table->Keys->KeyArrayBaseAddress);
    auto *assigned = StoragePolicy::Assigned(graph);
    auto *seeds = &graph->FirstSeed;
    for (ULONG edge = 0; edge < graph->NumberOfKeys; edge++) {
        ULONG key = keys[edge];
        ULONG value_index = 0;
        HRESULT result = index_impl_with_assigned<StoragePolicy, KeyType>(
            table->HashFunctionId,
            seeds,
            assigned,
            table->HashMask,
            table->IndexMask,
            table,
            key,
            &value_index
        );
        if (FAILED(result)) {
            allocator->Vtbl->FreePointer(allocator, PPV(&graph->Values));
            return result;
        }

        if (TestGraphBit(AssignedBitmap, value_index)) {
            collisions++;
        }

        SetGraphBit(AssignedBitmap, value_index);
        values[value_index] = key;
    }

    number_of_assignments = rtl->RtlNumberOfSetBits(&graph->AssignedBitmap);

    allocator->Vtbl->FreePointer(allocator, PPV(&graph->Values));

    if (collisions) {
        return PH_E_COLLISIONS_ENCOUNTERED_DURING_GRAPH_VERIFICATION;
    }

    if (number_of_assignments != graph->NumberOfKeys) {
        return PH_E_NUM_ASSIGNMENTS_NOT_EQUAL_TO_NUM_KEYS_DURING_GRAPH_VERIFICATION;
    }

    return S_OK;
}

template <typename StorageType, typename KeyType>
HRESULT
hash_keys_dispatch_impl(PGRAPH graph, ULONG number_of_keys, PKEY keys)
{
    return hash_keys_impl<storage_policy<StorageType>, KeyType>(
        graph, number_of_keys, keys
    );
}

template <typename StorageType, typename KeyType>
HRESULT
add_keys_dispatch_impl(PGRAPH graph, ULONG number_of_keys, PKEY keys)
{
    return add_keys_impl<storage_policy<StorageType>, KeyType>(
        graph, number_of_keys, keys
    );
}

template <typename StorageType, typename KeyType>
HRESULT
verify_dispatch_impl(PGRAPH graph)
{
    return verify_impl<storage_policy<StorageType>, KeyType>(graph);
}

template <typename StorageType, typename KeyType>
HRESULT
index_dispatch_impl(PPERFECT_HASH_TABLE table, ULONG key, PULONG index)
{
    return index_impl<storage_policy<StorageType>, KeyType>(table, key, index);
}

template <typename KeyType, typename Fn>
HRESULT
dispatch_storage_width(
    PGRAPH graph,
    Fn &&fn
    )
{
    auto *table = graph->Context->Table;
    switch (table->TableInfoOnDisk->AssignedElementSizeInBytes) {
        case 1:
            return fn.template operator()<uint8_t, KeyType>();
        case 2:
            return fn.template operator()<uint16_t, KeyType>();
        case 4:
            return fn.template operator()<uint32_t, KeyType>();
        default:
            return PH_E_NOT_IMPLEMENTED;
    }
}

template <typename Fn>
HRESULT
dispatch_key_width(
    PGRAPH graph,
    Fn &&fn
    )
{
    auto *table = graph->Context->Table;
    switch (table->GraphImpl4EffectiveKeySizeInBytes) {
        case 1:
            return dispatch_storage_width<uint8_t>(graph, fn);
        case 2:
            return dispatch_storage_width<uint16_t>(graph, fn);
        case 4:
            return dispatch_storage_width<uint32_t>(graph, fn);
        default:
            return PH_E_NOT_IMPLEMENTED;
    }
}

template <typename Fn>
HRESULT
dispatch_table_width(
    PPERFECT_HASH_TABLE table,
    Fn &&fn
    )
{
    switch (table->GraphImpl4EffectiveKeySizeInBytes) {
        case 1:
            switch (table->TableInfoOnDisk->AssignedElementSizeInBytes) {
                case 1: return fn.template operator()<uint8_t, uint8_t>();
                case 2: return fn.template operator()<uint16_t, uint8_t>();
                case 4: return fn.template operator()<uint32_t, uint8_t>();
                default: return PH_E_NOT_IMPLEMENTED;
            }
        case 2:
            switch (table->TableInfoOnDisk->AssignedElementSizeInBytes) {
                case 1: return fn.template operator()<uint8_t, uint16_t>();
                case 2: return fn.template operator()<uint16_t, uint16_t>();
                case 4: return fn.template operator()<uint32_t, uint16_t>();
                default: return PH_E_NOT_IMPLEMENTED;
            }
        case 4:
            switch (table->TableInfoOnDisk->AssignedElementSizeInBytes) {
                case 1: return fn.template operator()<uint8_t, uint32_t>();
                case 2: return fn.template operator()<uint16_t, uint32_t>();
                case 4: return fn.template operator()<uint32_t, uint32_t>();
                default: return PH_E_NOT_IMPLEMENTED;
            }
        default:
            return PH_E_NOT_IMPLEMENTED;
    }
}

} // namespace

extern "C"
HRESULT
NTAPI
GraphHashKeys4(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
{
    auto dispatch = [=]<typename StorageType, typename KeyType>() -> HRESULT {
        return hash_keys_dispatch_impl<StorageType, KeyType>(
            Graph, NumberOfKeys, Keys
        );
    };

    return dispatch_key_width(Graph, dispatch);
}

extern "C"
HRESULT
NTAPI
GraphAddKeys4(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
{
    auto dispatch = [=]<typename StorageType, typename KeyType>() -> HRESULT {
        return add_keys_dispatch_impl<StorageType, KeyType>(
            Graph, NumberOfKeys, Keys
        );
    };

    return dispatch_key_width(Graph, dispatch);
}

extern "C"
HRESULT
NTAPI
GraphHashKeysThenAdd4(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
{
    HRESULT result = GraphHashKeys4(Graph, NumberOfKeys, Keys);
    if (FAILED(result)) {
        return result;
    }

    switch (Graph->Context->Table->TableInfoOnDisk->AssignedElementSizeInBytes) {
        case 1:
            return add_hashed_keys_impl<storage_policy<uint8_t>>(Graph, NumberOfKeys);
        case 2:
            return add_hashed_keys_impl<storage_policy<uint16_t>>(Graph, NumberOfKeys);
        case 4:
            return add_hashed_keys_impl<storage_policy<uint32_t>>(Graph, NumberOfKeys);
        default:
            return PH_E_NOT_IMPLEMENTED;
    }
}

extern "C"
HRESULT
NTAPI
GraphIsAcyclic4(
    PGRAPH Graph
    )
{
    switch (Graph->Context->Table->TableInfoOnDisk->AssignedElementSizeInBytes) {
        case 1:
            return is_acyclic_impl<storage_policy<uint8_t>>(Graph);
        case 2:
            return is_acyclic_impl<storage_policy<uint16_t>>(Graph);
        case 4:
            return is_acyclic_impl<storage_policy<uint32_t>>(Graph);
        default:
            return PH_E_NOT_IMPLEMENTED;
    }
}

extern "C"
HRESULT
NTAPI
GraphAssign4(
    PGRAPH Graph
    )
{
    switch (Graph->Context->Table->TableInfoOnDisk->AssignedElementSizeInBytes) {
        case 1:
            return assign_impl<storage_policy<uint8_t>>(Graph);
        case 2:
            return assign_impl<storage_policy<uint16_t>>(Graph);
        case 4:
            return assign_impl<storage_policy<uint32_t>>(Graph);
        default:
            return PH_E_NOT_IMPLEMENTED;
    }
}

extern "C"
HRESULT
NTAPI
GraphVerify4(
    PGRAPH Graph
    )
{
    auto dispatch = [=]<typename StorageType, typename KeyType>() -> HRESULT {
        return verify_dispatch_impl<StorageType, KeyType>(Graph);
    };

    return dispatch_key_width(Graph, dispatch);
}

extern "C"
HRESULT
NTAPI
PerfectHashTableIndexImpl4Chm01(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
{
    //
    // The table Index() entry point receives keys in the table's 32-bit
    // domain.  Loaded downsized-64 callers must apply the outer downsize
    // bitmap first; this implementation then applies any GraphImpl4 inner
    // compact-key metadata before hashing.
    //
    auto dispatch = [=]<typename StorageType, typename KeyType>() -> HRESULT {
        return index_dispatch_impl<StorageType, KeyType>(Table, Key, Index);
    };

    return dispatch_table_width(Table, dispatch);
}

extern "C"
HRESULT
NTAPI
PerfectHashTableIndexImpl4EffectiveKeyChm01(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
{
    auto dispatch = [=]<typename StorageType, typename KeyType>() -> HRESULT {
        return index_effective_key_dispatch_impl<StorageType, KeyType>(
            Table,
            Key,
            Index
        );
    };

    return dispatch_table_width(Table, dispatch);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
