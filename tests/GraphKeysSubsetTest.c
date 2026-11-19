#include "stdafx.h"

// Each key joins vertex zero to a different cache line, forming an acyclic
// star. The center exceeds both the cache-line BYTE and page USHORT limits.
static HRESULT STDAPICALLTYPE
HashStar(PPERFECT_HASH_TABLE Table, ULONG Key, ULONG Count, PULONG Seeds,
         PULONGLONG Hash)
{
    (void)Table;
    (void)Count;
    (void)Seeds;
    *Hash = (ULONGLONG)(Key * 16) << 32;
    return S_OK;
}

static HRESULT STDAPICALLTYPE
MaskVertex(PPERFECT_HASH_TABLE Table, ULONG Vertex, PULONG Masked)
{
    (void)Table;
    *Masked = Vertex;
    return S_OK;
}

int main(void)
{
    static PERFECT_HASH_CONTEXT Context;
    static PERFECT_HASH_TABLE Table;
    static PERFECT_HASH_TABLE_VTBL Vtbl;
    static GRAPH Graph;
    static KEYS_SUBSET Subset;
    static ULONG Keys[65536], Pages[1025], LargePages[3], Lines[65537];
    PASSIGNED_MEMORY_COVERAGE Coverage = &Graph.AssignedMemoryCoverage;
    ULONG Index;

    for (Index = 0; Index < ARRAYSIZE(Keys); Index++) {
        Keys[Index] = Index + 1;
    }
    Subset.Values = Keys;
    Subset.NumberOfValues = ARRAYSIZE(Keys);
    Vtbl.SeededHash = HashStar;
    Vtbl.MaskHash = MaskVertex;
    Table.Vtbl = &Vtbl;
    Context.Table = &Table;
    Context.KeysSubset = &Subset;
    Graph.Context = &Context;
    Coverage->TotalNumberOfPages = ARRAYSIZE(Pages);
    Coverage->TotalNumberOfLargePages = ARRAYSIZE(LargePages);
    Coverage->TotalNumberOfCacheLines = ARRAYSIZE(Lines);
    Coverage->SubsetHitsPerPage = Pages;
    Coverage->NumberOfAssignedPerLargePage = LargePages;
    Coverage->SubsetHitsPerCacheLine = Lines;

    GraphCalculateAssignedMemoryCoverageForKeysSubset(&Graph);
    if (Coverage->MaxAssignedPerCacheLineCount != ARRAYSIZE(Keys) ||
        Coverage->NumberOfUsedCacheLines != ARRAYSIZE(Lines) ||
        Coverage->NumberOfAssignedPerCacheLineCounts[1] != ARRAYSIZE(Keys) ||
        Coverage->NumberOfUsedPages != ARRAYSIZE(Pages) ||
        Coverage->NumberOfUsedLargePages != ARRAYSIZE(LargePages) ||
        Lines[0] != ARRAYSIZE(Keys) || Pages[0] != ARRAYSIZE(Keys) + 63) {
        return 1;
    }
    for (Index = 0; Index < ARRAYSIZE(Coverage->NumberOfAssignedPerCacheLineCounts); Index++) {
        if (Index != 1 && Coverage->NumberOfAssignedPerCacheLineCounts[Index] != 0) {
            return 1;
        }
    }
    return 0;
}
