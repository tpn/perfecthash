
volatile ULONG CtrlCPressed = 0;

void
CphQueryPerformanceCounter(
    _Out_ PLARGE_INTEGER Count
    )
{
    Count->QuadPart = __rdtsc();
}

