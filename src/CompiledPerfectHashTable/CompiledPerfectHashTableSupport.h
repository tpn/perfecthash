

extern
void
CphQueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    );

// ARM64 PMU counters are per-core. Use a dedicated pinned worker so the
// caller's complete scheduling eligibility (including processor groups) stays
// unchanged. Thread setup is outside the timed region.
typedef ULONG (CPHCALLTYPE *CPH_CYCLE_BENCHMARK_ROUTINE)(ULONG Seconds);
#if defined(_WIN32) && defined(_M_ARM64)
#ifdef __cplusplus
extern "C" {
#endif
extern ULONG __stdcall GetCurrentProcessorNumber(void);
extern uintptr_t __stdcall SetThreadAffinityMask(PVOID Thread, uintptr_t Mask);
extern PVOID __stdcall GetProcessHeap(void);
extern PVOID __stdcall HeapAlloc(PVOID Heap, ULONG Flags, uintptr_t Size);
extern BOOL __stdcall HeapFree(PVOID Heap, ULONG Flags, PVOID Memory);
extern PVOID __stdcall CreateThread(PVOID Attributes, uintptr_t StackSize,
    ULONG (__stdcall *Start)(PVOID), PVOID Parameter, ULONG Flags, PULONG Id);
extern ULONG __stdcall WaitForSingleObject(PVOID Handle, ULONG Milliseconds);
extern BOOL __stdcall GetExitCodeThread(PVOID Thread, PULONG Code);
extern BOOL __stdcall CloseHandle(PVOID Handle);
#ifdef __cplusplus
}
#endif
typedef struct _CPH_CYCLE_BENCHMARK_CONTEXT {
    CPH_CYCLE_BENCHMARK_ROUTINE Routine;
    ULONG Seconds;
} CPH_CYCLE_BENCHMARK_CONTEXT;
static ULONG __stdcall CphCycleBenchmarkWorker(PVOID Parameter)
{
    CPH_CYCLE_BENCHMARK_CONTEXT Context =
        *(CPH_CYCLE_BENCHMARK_CONTEXT *)Parameter;
    ULONG Processor = GetCurrentProcessorNumber();
    HeapFree(GetProcessHeap(), 0, Parameter);
    if (!SetThreadAffinityMask((PVOID)(intptr_t)-2,
                               (uintptr_t)1 << Processor)) {
        return (ULONG)-1;
    }
    return Context.Routine(Context.Seconds);
}
static inline ULONG CphRunCycleBenchmark(CPH_CYCLE_BENCHMARK_ROUTINE Routine,
                                         ULONG Seconds)
{
    PVOID Thread;
    ULONG Cycles = (ULONG)-1;
    CPH_CYCLE_BENCHMARK_CONTEXT *Context =
        (CPH_CYCLE_BENCHMARK_CONTEXT *)HeapAlloc(GetProcessHeap(), 0,
                                               sizeof(*Context));
    if (!Context) {
        return Cycles;
    }
    Context->Routine = Routine;
    Context->Seconds = Seconds;
    Thread = CreateThread(0, 0, CphCycleBenchmarkWorker, Context, 0, 0);
    if (!Thread) {
        HeapFree(GetProcessHeap(), 0, Context);
        return Cycles;
    }
    // The worker owns Context, including when this wait fails.
    if (WaitForSingleObject(Thread, (ULONG)-1) == 0) {
        if (!GetExitCodeThread(Thread, &Cycles)) {
            Cycles = (ULONG)-1;
        }
    }
    CloseHandle(Thread);
    return Cycles;
}
#else
static inline ULONG CphRunCycleBenchmark(CPH_CYCLE_BENCHMARK_ROUTINE Routine,
                                         ULONG Seconds)
{
    return Routine(Seconds);
}
#endif

extern volatile ULONG CtrlCPressed;

#define FOR_EACH_KEY \
    for (Index = 0, Source = KEYS; Index < NUMBER_OF_KEYS; Index++)

#ifdef _WIN32

extern
BOOLEAN
QueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    );

#ifndef PH_UNITY

#define CPH_MAIN() \
VOID               \
__stdcall          \
mainCRTStartup(    \
    void           \
    )

extern void ExitProcess(ULONG);
#define CPH_EXIT(Code) ExitProcess(Code)

#else

#define CPH_MAIN() \
int                \
main(              \
    int argc,      \
    char **argv    \
    )

#define CPH_EXIT(Code)    \
    return Code

#endif

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

