
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

#endif
