
//
// Disable the anonymous union/struct warning.
//

#pragma warning(push)
#pragma warning(disable: 4201 4094)

typedef union _LARGE_INTEGER {
    struct {
        ULONG LowPart;
        LONG HighPart;
    };
    LONGLONG QuadPart;
} LARGE_INTEGER;
typedef LARGE_INTEGER *PLARGE_INTEGER;

typedef union _ULARGE_INTEGER {
    struct {
        ULONG LowPart;
        ULONG HighPart;
    };
    ULONGLONG QuadPart;
} ULARGE_INTEGER;
typedef ULARGE_INTEGER *PULARGE_INTEGER;

#pragma warning(pop)

extern
BOOLEAN
QueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    );

extern volatile ULONG CtrlCPressed;

extern void ExitProcess(ULONG);

#define FOR_EACH_KEY \
    for (Index = 0, Source = KEYS; Index < NUMBER_OF_KEYS; Index++)

