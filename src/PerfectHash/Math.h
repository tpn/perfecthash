/*++

Copyright (c) 2021 Trent Nelson <trent@trent.me>

Module Name:

    Math.h

Abstract:

    This is the header file for math functionality of the perfect hash library.

    In the beginning, the most complex math we did was multiplication, and life
    was good.  Then we started wanting to calculate probabilities and perform
    linear regressions, and thus, we needed routines like pow() and sqrt().

    Unfortunately, because we don't link against a standard C runtime, we don't
    have these common math routines readily available.  So, we included some
    open source implementations of said functions (see _pow.c), and introduced
    this file as the header to contain the necessary function declarations,
    plus house the function decls of math routines specific to the perfect hash
    library.

--*/

#pragma once

#include "stdafx.h"

//
// Function decls and helper short-name for standard C runtime functions
// provided by _pow.c.
//

DOUBLE __ieee754_pow(DOUBLE x, DOUBLE y);
DOUBLE __ieee754_sqrt(DOUBLE x);

#define pow __ieee754_pow
#define sqrt __ieee754_sqrt

//
// _dtoa() and _freedtoa(), provided by _dtoa.c.
//

extern PALLOCATOR _dtoa_Allocator;

char *
_dtoa(double dd, int mode, int ndigits,
      int *decpt, int *sign, char **rve);

void
_freedtoa(char *s);

//
// Helper inlines.
//

FORCEINLINE
DOUBLE
sqr(DOUBLE x)
{
    return x * x;
}

//
// Function decls specific to the perfect hash library.
//

HRESULT
CalculatePredictedAttempts(
    _In_ DOUBLE SolutionsFoundRatio,
    _Out_ PULONG PredictedAttempts
    );

VOID
LinearRegressionNumberOfAssignedPerCacheLineCounts(
    _In_reads_(TOTAL_NUM_ASSIGNED_PER_CACHE_LINE) PULONG YCounts,
    _Out_ PDOUBLE SlopePointer,
    _Out_ PDOUBLE InterceptPointer,
    _Out_ PDOUBLE CorrelationCoefficientPointer,
    _Out_ PDOUBLE PredictedNumberOfFilledCacheLinesPointer
    );

VOID
ScoreNumberOfAssignedPerCacheLineCounts(
    _In_reads_(TOTAL_NUM_ASSIGNED_PER_CACHE_LINE) PULONG YCounts,
    _In_ ULONG TotalNumberOfAssigned,
    _Out_ PULONGLONG Score,
    _Out_ PDOUBLE Rank
    );

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
