
#ifndef CPH_INDEX_ONLY

DECLARE_LOOKUP_ROUTINE()
{
    CPHINDEX Index;

    Index = INDEX_ROUTINE(Key);
    return TABLE_VALUES[Index];
}

DECLARE_INSERT_ROUTINE()
{
    CPHINDEX Index;
    CPHVALUE Previous;

    Index = INDEX_ROUTINE(Key);
    Previous = TABLE_VALUES[Index];
    TABLE_VALUES[Index] = Value;
    return Previous;
}

DECLARE_DELETE_ROUTINE()
{
    CPHINDEX Index;
    CPHVALUE Previous;

    Index = INDEX_ROUTINE(Key);
    Previous = TABLE_VALUES[Index];
    TABLE_VALUES[Index] = 0;
    return Previous;
}

DECLARE_INTERLOCKED_INCREMENT_ROUTINE()
{
    CPHINDEX Index;
    CPHVALUE New;

    Index = INDEX_ROUTINE(Key);
#ifdef _WIN32
    New = _InterlockedIncrement((volatile LONG *)&TABLE_VALUES[Index]);
#else
    New = __sync_add_and_fetch(&TABLE_VALUES[Index], 1);
#endif
    return New;
}

#ifdef CPH_HAS_KEYS
DECLARE_INDEX_BSEARCH_ROUTINE()
{
    CPHINDEX Low;
    CPHINDEX High;
    CPHINDEX Middle;
    CPHDKEY Value;

    Low = 0;
    High = NUMBER_OF_KEYS - 1;
    Middle = (Low + High) / 2;

    while (Low <= High) {

        Value = KEYS[Middle];

        if (Value == Key) {
            break;
        } else if (Value < Key) {
            Low = Middle + 1;
        } else {
            High = Middle - 1;
        }

        Middle = (Low + High) / 2;
    }

    return Middle;
}
#endif

#endif
