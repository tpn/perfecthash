/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    GraphVertexCollision.c

Abstract:

    This module implements vertex collision functionality.  Routines are
    provided for creating and destroying a graph's vertex collision database,
    and the vertex collision callback.

--*/

#include "stdafx.h"
#include "PerfectHashEventsPrivate.h"

//
// Helper macros for pointer comparisons.
//

#define GreaterThanOrEqual(Left, Right) \
    (((ULONG_PTR)Left) >= ((ULONG_PTR)Right))

#define LessThanOrEqual(Left, Right) \
    (((ULONG_PTR)Left) <= ((ULONG_PTR)Right))

_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
GraphCreateVertexCollisionDb (
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    This routine creates a vertex collision database for the provided graph.

    N.B. If the vertex collision database has already been created, this routine
         is a no-op.

Arguments:

    Graph - Supplies a pointer to the graph instance for which the vertex
        collision database will be created.

Return Value:

    S_OK - Database created successfully.

    S_FALSE - Database already exists.

    E_OUTOFMEMORY - Out of memory.

--*/
{
    PALLOCATOR Allocator;
    PVERTEX_COLLISION_DB Db;

    if (Graph->VertexCollisionDb != NULL) {
        return S_FALSE;
    }

    Allocator = Graph->Allocator;
    Db = (PVERTEX_COLLISION_DB)(
        Allocator->Vtbl->Calloc(Allocator, 1, sizeof(*Db))
    );

    if (!Db) {
        return E_OUTOFMEMORY;
    }

    Graph->VertexCollisionDb = Db;

    Db->SizeOfStruct = sizeof(*Db);
    Db->TotalNumberOfElements = VERTEX_COLLISION_ARRAY_SIZE;
    Db->TotalNumberOfLruCacheEntries = VERTEX_COLLISION_LRU_CACHE_SIZE;

    //
    // We're done, update the graph flags and return success.
    //

    Graph->Flags.UsingVertexCollisionCallback = TRUE;

    return S_OK;
}

_Requires_exclusive_lock_held_(Graph->Lock)
VOID
GraphDestroyVertexCollisionDb (
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    This routine destroys a vertex collision database for the provided graph.

    N.B. If the vertex collision database has not been created, this routine
         is a no-op.

Arguments:

    Graph - Supplies a pointer to the graph instance for which the vertex
        collision database will be destroyed.

Return Value:

    None.

--*/
{
    PALLOCATOR Allocator;
    PVERTEX_COLLISION_DB Db;

    //
    // Initialize aliases.
    //

    Db = Graph->VertexCollisionDb;

    //
    // If there's no database, we're done.
    //

    if (!Db) {
        return;
    }

    //
    // Perform some invariant checks, then proceed with deallocation.
    //

    ASSERT(Db->SizeOfStruct == sizeof(*Db));
    ASSERT(Db->NumberOfElements <= Db->TotalNumberOfElements);
    ASSERT(Db->NumberOfLruCacheEntries <=
           Db->TotalNumberOfLruCacheEntries);

    Allocator = Graph->Allocator;
    Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Graph->VertexCollisionDb);

    return;
}

static
VOID
SortCollisionEntry(
    _In_ PVERTEX_COLLISION_DB Db,
    _In_ PVERTEX_COLLISION Start,
    _In_ PVERTEX_COLLISION Last,
    _In_ PVERTEX_COLLISION End,
    _In_ PVERTEX_COLLISION Entry
    )
{
    ULONG Count;
    ULONG_PTR EndPtr;
    ULONG_PTR LastPtr;
    ULONG_PTR StartPtr;
    ULONG_PTR EntryPtr;
    VERTEX_COLLISION Temp;
    PVERTEX_COLLISION Prev;
    PVERTEX_COLLISION Next;

    UNREFERENCED_PARAMETER(Db);

    //
    // Initialize aliases.
    //

    EndPtr = (ULONG_PTR)End;
    LastPtr = (ULONG_PTR)Last;
    StartPtr = (ULONG_PTR)Start;
    EntryPtr = (ULONG_PTR)Entry;

    //
    // Invariant checks.
    //

#ifdef _DEBUG
    ASSERT(EntryPtr >= StartPtr);
    ASSERT(EntryPtr <= LastPtr);
    ASSERT(StartPtr <= LastPtr);
    ASSERT(LastPtr <= EndPtr);
#endif

    //
    // If this is the first element, we're done.
    //

    if (Entry == Start) {
        return;
    }

    //
    // If this isn't the last element, verify our count is equal to or greater
    // than the next element's count (as we expect the list to already be sorted
    // in descending order according to count).
    //

    if (EntryPtr < LastPtr) {
        Next = Entry + 1;
        ASSERT(Entry->Count >= Next->Count);
    }

    Count = Entry->Count;

    for (Prev = Entry - 1; GreaterThanOrEqual(Prev, Start); Prev--) {

        if (Count > Prev->Count) {
            Temp.Key = Prev->Key;
            Temp.Count = Prev->Count;

            Prev->Key = Entry->Key;
            Prev->Count = Entry->Count;

            Entry->Key = Temp.Key;
            Entry->Count = Temp.Count;

            Entry = Prev;
        } else {
            break;
        }
    }
}

static
VOID
DeleteSortedCollision (
    _In_ PRTL Rtl,
    _In_ PVERTEX_COLLISION_DB Db,
    _In_ PVERTEX_COLLISION Start,
    _In_ PVERTEX_COLLISION Last,
    _In_ PVERTEX_COLLISION End,
    _In_ PVERTEX_COLLISION Entry,
    _Inout_ PSHORT NumberOfElements
    )
{
    ULONG_PTR EndPtr;
    ULONG_PTR LastPtr;
    ULONG_PTR StartPtr;
    ULONG_PTR EntryPtr;
    ULONG_PTR BytesToMove;

    UNREFERENCED_PARAMETER(Db);

    //
    // Initialize aliases.
    //

    EndPtr = (ULONG_PTR)End;
    LastPtr = (ULONG_PTR)Last;
    StartPtr = (ULONG_PTR)Start;
    EntryPtr = (ULONG_PTR)Entry;

    //
    // Invariant checks.
    //

#ifdef _DEBUG
    ASSERT(EntryPtr >= StartPtr);
    ASSERT(EntryPtr <= LastPtr);
    ASSERT(StartPtr <= LastPtr);
    ASSERT(LastPtr <= EndPtr);
    ASSERT(*NumberOfElements > 0);
#endif

    //
    // Decrement the count up-front.
    //

    *NumberOfElements -= 1;

    //
    // If this is the last element, we don't need to memmove.
    //

    if (Entry == Last) {
        Entry->Count = 0;
        Entry->Key = 0;
        return;
    }

    //
    // Move everything up from our deletion point, then clear the final
    // element.
    //

    BytesToMove = LastPtr - EntryPtr;

    MoveMemory(Entry, Entry+1, BytesToMove);

    ASSERT(Last != Start);

    Last->Count = 0;
    Last->Key = 0;
}

static
VOID
InsertSortedCollision (
    _In_ PRTL Rtl,
    _In_ PVERTEX_COLLISION_DB Db,
    _In_ PVERTEX_COLLISION Start,
    _In_opt_ PVERTEX_COLLISION Last,
    _In_ PVERTEX_COLLISION End,
    _In_ PVERTEX_COLLISION Entry,
    _Inout_ PSHORT NumberOfElements,
    _In_ BOOLEAN ReplaceLast
    )
{
    ULONG Count;
    BOOLEAN Full;
    ULONG_PTR EndPtr;
    ULONG_PTR LastPtr;
    ULONG_PTR StartPtr;
    ULONG_PTR EntryPtr;
    ULONG_PTR ThisPtr;
    ULONG_PTR BytesToMove;
    ULONG_PTR ElementCount;
    PVERTEX_COLLISION This;

    UNREFERENCED_PARAMETER(Db);

    //
    // Initialize aliases.
    //

    EndPtr = (ULONG_PTR)End;
    LastPtr = (ULONG_PTR)Last;
    StartPtr = (ULONG_PTR)Start;
    EntryPtr = (ULONG_PTR)Entry;

    Full = (Last == End);

    //
    // Invariant checks.
    //

#ifdef _DEBUG
    if (LastPtr != 0) {
        ASSERT(StartPtr <= LastPtr);
    }
    ASSERT(StartPtr <= EndPtr);
    ASSERT(*NumberOfElements >= 0);
    if (Full) {
        ASSERT(*NumberOfElements == VERTEX_COLLISION_LRU_CACHE_SIZE);
    }
#endif

    //
    // If the list is empty, the insertion point is the first element.
    //

    if (Last == NULL) {
        ASSERT(*NumberOfElements == 0);
        This = Start;
        *NumberOfElements += 1;
        This->Count = Entry->Count;
        This->Key = Entry->Key;
        return;
    }

#ifdef _DEBUG
    Count = 0;
    for (This = Start; LessThanOrEqual(This, Last); This++) {
        Count++;
        ASSERT(This->Key != Entry->Key);
        ASSERT(This->Key != ((This+1)->Key));
    }
#endif

    //
    // Walk the list and find the first entry with a count equal to or less
    // than our entry's count.
    //

    Count = 0;

    for (This = Start; LessThanOrEqual(This, Last); This++) {

        Count++;

        ASSERT(This->Count > 0);
        ASSERT(This->Key > 0);

        if (This->Count <= Entry->Count) {

            //
            // We found an appropriate insertion point.
            //

            if (This == End) {

                //
                // We're at the end of the list; overwrite this entry, but don't
                // update the count.
                //

                This->Count = Entry->Count;
                This->Key = Entry->Key;
                ASSERT(*NumberOfElements == VERTEX_COLLISION_LRU_CACHE_SIZE);
                return;

            } else {

                //
                // We're not at the end of the list.  Move everything down if
                // applicable, overwrite our new entry at this insertion point,
                // and update the count.
                //

                ThisPtr = (ULONG_PTR)This;

                BytesToMove = LastPtr - ThisPtr;
                if (!Full) {
                    BytesToMove += sizeof(*This);
                }

                ElementCount = BytesToMove / sizeof(*This);

                if (!Full) {
                    ASSERT(ElementCount == ((ULONG_PTR)(*NumberOfElements)));
                }

                ASSERT(ElementCount < VERTEX_COLLISION_LRU_CACHE_SIZE);
                ASSERT(BytesToMove > 0);

                MoveMemory(This + 1, This, BytesToMove);

                This->Count = Entry->Count;
                This->Key = Entry->Key;

                if (!Full) {
                    *NumberOfElements += 1;
                }

                return;
            }
        }
    }

    //
    // If we get here, we didn't find an appropriate insertion point in the
    // existing list.
    //

    if (Last == End) {

        //
        // The list is full.  Replace the last element if requested, otherwise,
        // we don't need to take any action.  (ReplaceLast would only be used
        // for LRU cache entries.)  Count is not updated.
        //

        if (ReplaceLast) {
            Last->Count = Entry->Count;
            Last->Key = Entry->Key;
        }

        return;
    }

    //
    // Otherwise, add this entry to the end of the list and update the count.
    //

    This = Last + 1;
    This->Count = Entry->Count;
    This->Key = Entry->Key;
    ASSERT(*NumberOfElements < VERTEX_COLLISION_LRU_CACHE_SIZE);
    *NumberOfElements += 1;

    return;
}

_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
GraphVertexCollisionCallback (
    _In_ PGRAPH Graph,
    _In_ KEY Key
    )
/*++

Routine Description:

    This routine is called when a graph encounters a vertex collision.

Arguments:

    Graph - Supplies a pointer to the graph instance.

    Key - Supplies the key that resulted in the collision.

Return Value:

    S_OK on success, otherwise, an appropriate error code.

--*/
{
    PRTL Rtl;
    SHORT Index;
    HRESULT Result;
    ULONG MinimumCount;
    BOOLEAN ReplaceLast;
    PVERTEX_COLLISION_DB Db;
    PVERTEX_COLLISION Lru;
    PVERTEX_COLLISION StartLru;
    PVERTEX_COLLISION EndLru;
    PVERTEX_COLLISION LastLru;
    VERTEX_COLLISION NewLru;
    VERTEX_COLLISION TempLru;
    PVERTEX_COLLISION Collision;
    PVERTEX_COLLISION StartCollision;
    PVERTEX_COLLISION EndCollision;
    PVERTEX_COLLISION LastCollision;

    //
    // Initialize aliases.
    //

    Db = Graph->VertexCollisionDb;
    Rtl = Graph->Rtl;

    //
    // Start refers to the first entry in the list.  Last refers to the last
    // valid entry in the list.  End refers to the final element in the list.
    //
    // Invariants:
    //
    //      If list empty:  Start == Last
    //      If list full:   Last == End
    //      At all times:   Start <= Last <= End
    //

    StartLru = &Db->LruCache[0];
    EndLru = &Db->LruCache[Db->TotalNumberOfLruCacheEntries-1];

    if (!IsLruCacheEmpty(Db)) {
        LastLru = &Db->LruCache[Db->NumberOfLruCacheEntries-1];
    } else {
        LastLru = NULL;
    }

    StartCollision = &Db->Collisions[0];
    EndCollision = &Db->Collisions[Db->TotalNumberOfElements-1];

    if (!IsCollisionsEmpty(Db)) {
        LastCollision = &Db->Collisions[Db->NumberOfElements-1];
    } else {
        LastCollision = NULL;
    }

    //
    // Invariant checks.
    //

#ifdef _DEBUG
    ASSERT((ULONG_PTR)LastLru <= (ULONG_PTR)EndLru);
    ASSERT((ULONG_PTR)LastCollision <= (ULONG_PTR)EndCollision);
    if (IsLruCacheFull(Db)) {
        ASSERT(LastLru == EndLru);
    }
#endif


    MinimumCount = MINIMUM_VERTEX_COLLISION_COUNT;

    //
    // Perform a linear scan through the existing collisions to see if we have
    // already seen this key.
    //

    Collision = NULL;
    for (Index = 0; Index < Db->NumberOfElements; Index++) {
        if (Key == Db->Collisions[Index].Key) {
            Collision = &Db->Collisions[Index];
            break;
        }
    }

    if (Collision) {

        //
        // We found an existing collision; update the count and sort.
        //

        Collision->Count++;

        SortCollisionEntry(Db,
                           StartCollision,
                           LastCollision,
                           EndCollision,
                           Collision);

        goto End;

    }

    //
    // No existing collision was found.  Attempt to find it in the LRU
    // cache.
    //

    Lru = NULL;
    for (Index = 0; Index < Db->NumberOfLruCacheEntries; Index++) {
        ASSERT(Db->LruCache[Index].Count > 0);
        ASSERT(Db->LruCache[Index].Key > 0);
        if (Key != Db->LruCache[Index].Key) {
            continue;
        }
        Lru = &Db->LruCache[Index];
        break;
    }

    if (Lru) {

        //
        // We've found the key in the LRU cache list, increment the count
        // and check to see if it meets our minimum count threshold.
        //

        if (++Lru->Count >= MinimumCount) {

            //
            // The LRU entry is suitable for migration to the official collision
            // table.  How we handle this is dependent up on whether or not the
            // collision table is already full.  If it is, then this LRU entry's
            // count needs to equal or exceed the last collision entry in order
            // for migration.
            //

            if (IsCollisionsFull(Db)) {

                _Analysis_assume_(LastCollision != NULL);
                if (LastCollision->Count <= Lru->Count) {

                    //
                    // The last collision has a count equal to or less than
                    // our LRU count, and the collision database is full, which
                    // means our LRU entry will get promoted to the collision
                    // list and the collision entry will get demoted back to
                    // the LRU cache, assuming an appropriate insertion point
                    // can be found for it.
                    //

                    NewLru.Count = LastCollision->Count;
                    NewLru.Key = LastCollision->Key;

                    //
                    // Delete the LRU entry first.
                    //

                    TempLru.Count = Lru->Count;
                    TempLru.Key = Lru->Key;

#ifdef _DEBUG
                    BOOLEAN Found = FALSE;
                    BOOLEAN Found2 = FALSE;

                    for (Index = 0; Index < Db->NumberOfLruCacheEntries; Index++) {
                        Entry = &Db->LruCache[Index];
                        if (Index < (Db->NumberOfLruCacheEntries-1)) {
                            ASSERT(Entry->Key != ((Entry+1)->Key));
                        }
                        if (Entry->Key == Key) {
                            ASSERT(!Found);
                            Found = TRUE;
                        }
                        if (Entry->Key == NewLru.Key) {
                            ASSERT(!Found2);
                            Found2 = TRUE;
                        }
                    }

                    ASSERT(Found);
                    //ASSERT(!Found2);

                    //TempLru.Count = Lru->Count;
                    //TempLru.Key = Lru->Key;
#endif

                    DeleteSortedCollision(Rtl,
                                          Db,
                                          StartLru,
                                          LastLru,
                                          EndLru,
                                          Lru,
                                          &Db->NumberOfLruCacheEntries);

                    ASSERT(Db->NumberOfLruCacheEntries <= VERTEX_COLLISION_LRU_CACHE_SIZE);

#ifdef _DEBUG
                    Found = FALSE;
                    for (Index = 0; Index < Db->NumberOfLruCacheEntries-1; Index++) {
                        Entry = &Db->LruCache[Index];
                        ASSERT(Entry->Key != ((Entry+1)->Key));
                        if (Entry->Key == Key) {
                            ASSERT(!Found);
                            Found = TRUE;
                        }
                        ASSERT(Entry->Key != TempLru.Key);
                    }
                    ASSERT(!Found);
#endif


                    Db->NumberOfLruDeletions++;
                    LastLru -= 1;

                    //
                    // Then attempt to insert the collision back into the LRU
                    // cache.
                    //

                    ReplaceLast = FALSE;

#ifdef _DEBUG
                    for (Index = 0; Index < Db->NumberOfLruCacheEntries-1; Index++) {
                        Entry = &Db->LruCache[Index];
                        ASSERT(Entry->Key != ((Entry+1)->Key));
                        ASSERT(Entry->Key != NewLru.Key);
                    }
#endif

                    InsertSortedCollision(Rtl,
                                          Db,
                                          StartLru,
                                          LastLru,
                                          EndLru,
                                          &NewLru,
                                          &Db->NumberOfLruCacheEntries,
                                          ReplaceLast);
                    ASSERT(Db->NumberOfLruCacheEntries <= VERTEX_COLLISION_LRU_CACHE_SIZE);

#ifdef _DEBUG
                    for (Index = 0; Index < Db->NumberOfLruCacheEntries-1; Index++) {
                        Entry = &Db->LruCache[Index];
                        ASSERT(Entry->Key != ((Entry+1)->Key));
                    }
#endif

                    //
                    // Now overwrite the last collision with the LRU entry.
                    //

                    Collision = LastCollision;
                    Collision->Count = TempLru.Count;
                    Collision->Key = TempLru.Key;

                    //
                    // Dispatch a sort to ensure the collision list is always sorted
                    // after mutation.
                    //

                    SortCollisionEntry(Db,
                                       StartCollision,
                                       LastCollision,
                                       EndCollision,
                                       Collision);

                    goto End;

                } else {

                    //
                    // The collision database is full, but this LRU entry's
                    // count is less than the count of the last collision entry,
                    // so, it doesn't get promoted.
                    //

                    goto End;
                }

            } else {

                //
                // The collision list isn't full, so we can add this LRU entry
                // to the end of the collision list, then delete it from the
                // LRU cache.
                //

                //
                // Add a new entry to the collision list.
                //

                if (LastCollision == NULL) {
                    LastCollision = StartCollision;
                }
                LastCollision += 1;
                Collision = LastCollision;
                Collision->Count = Lru->Count;
                Collision->Key = Lru->Key;
                Db->NumberOfElements++;

                //
                // Dispatch a sort to ensure the collision list is always sorted
                // after mutation.
                //

                SortCollisionEntry(Db,
                                   StartCollision,
                                   LastCollision,
                                   EndCollision,
                                   Collision);

                //
                // Then delete the LRU entry.
                //

                DeleteSortedCollision(Rtl,
                                      Db,
                                      StartLru,
                                      LastLru,
                                      EndLru,
                                      Lru,
                                      &Db->NumberOfLruCacheEntries);
                ASSERT(Db->NumberOfLruCacheEntries <= VERTEX_COLLISION_LRU_CACHE_SIZE);
                Db->NumberOfLruDeletions++;

            }

        } else {

            //
            // The LRU entry's count isn't high enough to warrant migration
            // to the collision list.  As we adjusted the count, the entry
            // may need to be moved up higher in the LRU cache list, though,
            // as long as it wasn't the first entry.
            //

            if (Index > 0) {
                SortCollisionEntry(Db, StartLru, LastLru, EndLru, Lru);
            }
        }

    } else {

        //
        // No existing LRU entry found for this key, so add it to the end of
        // the list, explicitly replacing the last element if applicable (to
        // ensure a new LRU cache element has a chance to get onto a full
        // list).
        //

        NewLru.Count = 1;
        NewLru.Key = Key;
        if (IsLruCacheFull(Db)) {
            ReplaceLast = TRUE;
        } else {
            ReplaceLast = FALSE;
        }

#ifdef _DEBUG
        for (Index = 0; Index < Db->NumberOfLruCacheEntries-1; Index++) {
            Entry = &Db->LruCache[Index];
            ASSERT(Entry->Key != ((Entry+1)->Key))
        }
#endif

        InsertSortedCollision(Rtl,
                              Db,
                              StartLru,
                              LastLru,
                              EndLru,
                              &NewLru,
                              &Db->NumberOfLruCacheEntries,
                              ReplaceLast);
        ASSERT(Db->NumberOfLruCacheEntries <= VERTEX_COLLISION_LRU_CACHE_SIZE);

#ifdef _DEBUG
        for (Index = 0; Index < Db->NumberOfLruCacheEntries-1; Index++) {
            Entry = &Db->LruCache[Index];
            ASSERT(Entry->Key != ((Entry+1)->Key))
        }
#endif

    }

    //
    // Intentional follow-on to End.
    //

End:

    Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
