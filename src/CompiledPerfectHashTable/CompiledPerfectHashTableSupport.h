
//
// Disable the anonymous union/struct warning.
//

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable: 4201 4094)
#endif

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

#ifdef _WIN32
#pragma warning(pop)
#endif

extern
void
CphQueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    );

extern volatile ULONG CtrlCPressed;

#define FOR_EACH_KEY \
    for (Index = 0, Source = KEYS; Index < NUMBER_OF_KEYS; Index++)

#ifdef _WIN32

extern
BOOLEAN
QueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    );

#define CPH_MAIN() \
VOID               \
__stdcall          \
mainCRTStartup(    \
    void           \
    )

extern void ExitProcess(ULONG);
#define CPH_EXIT(Code) ExitProcess(Code)

#elif defined(__linux__) || defined(__APPLE__)

#include <stdio.h>

#define CPH_MAIN() \
int                \
main(              \
    int argc,      \
    char **argv    \
    )

#define CPH_EXIT(Code)    \
    printf("%d\n", Code); \
    return Code

#endif

