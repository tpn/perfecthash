/*++

Copyright (c) 2021 Trent Nelson <trent@trent.me>

Module Name:

    Math.c

Abstract:

    This module contains implementations of math-specific routines for the
    perfect hash library.  Routines are provided to calculate the predicted
    number of attempts required to solve a graph (based on a solutions-found
    ratio supplied), and perform linear regression.

--*/

#include "stdafx.h"

HRESULT
CalculatePredictedAttempts(
    _In_ DOUBLE SolutionsFoundRatio,
    _Out_ PULONG PredictedAttempts
    )
/*++

Routine Description:

    Given a solutions-found ratio, calculate the predicted number of attempts
    required to solve the graph.  As the solutions-found ratio is a probability
    following the geometric distribution, we can predict the number of attempts
    required to solve a graph by the formula:

        p * (q ** (r - 1))

    Where p is the probability of success, q is (1 - p), and r is the attempt.
    A cumulative probability is calcualted by multiplying this result by the
    attempt number, and returning the attempt once the delta between this value
    and the prior loop's value is greater than 0.0.

Arguments:

    SolutionsFoundRatio - Supplies the solutions found ratio.  Must be a value
        less than 1.0 and greater than 0.0.

    PredictedAttempts - Supplies a pointer to a variable that receives the
        predicted number of attempts if the routine was successful.

Return Value:

    S_OK - Success.

    E_POINTER - PredictedAttempts was NULL.

    PH_E_INVALID_SOLUTIONS_FOUND_RATIO - Solutions found ratio was invalid
        (i.e. greater than 1.0 or less than 0.0).

--*/
{
    ULONG Attempt;
    DOUBLE Delta;
    DOUBLE Success;
    DOUBLE Failure;
    DOUBLE Cumulative;
    DOUBLE Probability;
    DOUBLE LastCumulative;

    if (!ARGUMENT_PRESENT(PredictedAttempts)) {
        return E_POINTER;
    }

    if (SolutionsFoundRatio == 1.0) {
        Attempt = 1;
        goto End;
    }

    if (SolutionsFoundRatio > 1.0 || SolutionsFoundRatio <= 0.0) {
        return PH_E_INVALID_SOLUTIONS_FOUND_RATIO;
    }

    LastCumulative = 0.0;
    Success = SolutionsFoundRatio;
    Failure = 1 - SolutionsFoundRatio;

    for (Attempt = 1; ; Attempt++) {
        Probability = Success * pow(Failure, (DOUBLE)(Attempt - 1));
        Cumulative = (DOUBLE)Attempt * Probability;
        Delta = LastCumulative - Cumulative;
        if (Delta > 0.0) {
            break;
        }
        LastCumulative = Cumulative;
    }

    //
    // Update the caller's pointer and return success.
    //

End:

    *PredictedAttempts = Attempt;
    return S_OK;
}

VOID
LinearRegressionNumberOfAssignedPerCacheLineCounts(
    _In_reads_(TOTAL_NUM_ASSIGNED_PER_CACHE_LINE) PULONG YCounts,
    _Out_ PDOUBLE SlopePointer,
    _Out_ PDOUBLE InterceptPointer,
    _Out_ PDOUBLE CorrelationCoefficientPointer,
    _Out_ PDOUBLE PredictedNumberOfFilledCacheLinesPointer
    )
/*++

Routine Description:

    Given an array of 17 cache line counts, perform a linear regression and
    return the slope, intercept, correlation coefficient, and predicted number
    of filled cache lines (i.e. y for `y = mx + b` where x == 16).

Arguments:

    YCounts - Supplies the array of cache line counts.

    SlopePointer - Receives the slope.

    InterceptPointer - Receives the intercept.

    CorrelationCoefficientPointer - Receives the correlation coefficient.

Return Value:

    None.

--*/
{
    BYTE Index;
    DOUBLE X;
    DOUBLE Y;
    DOUBLE Y2;
    DOUBLE SumY;
    DOUBLE SumY2;
    DOUBLE SumXY;
    DOUBLE Slope;
    DOUBLE Intercept;
    DOUBLE Predicted;
    DOUBLE CorrelationCoefficient;
    CONST DOUBLE N = TOTAL_NUM_ASSIGNED_PER_CACHE_LINE;
    CONST DOUBLE SumX = 136.0;
    CONST DOUBLE SumX2 = 1496.0;
    CONST DOUBLE SumXSquared = 18496.0;
    CONST DOUBLE Denominator = ((N * SumX2) - SumXSquared);
    CONST BYTE Total = TOTAL_NUM_ASSIGNED_PER_CACHE_LINE;
    CONST BYTE XCounts[TOTAL_NUM_ASSIGNED_PER_CACHE_LINE] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };

    SumY = 0.0;
    SumXY = 0.0;
    SumY2 = 0.0;

    for (Index = 1; Index < Total; Index++) {
        X = (DOUBLE)XCounts[Index];

        Y = (DOUBLE)YCounts[Index];
        SumY += Y;

        Y2 = sqr(Y);
        SumY2 += Y2;

        SumXY += X * Y;
    }

    Slope = ((N * SumXY) - (SumX * SumY)) / Denominator;
    Intercept = ((SumY * SumX2) - (SumX * SumXY)) / Denominator;
    CorrelationCoefficient = (
        (SumXY - ((SumX * SumY) / N)) / sqrt(
            (SumX2 - sqr(SumX) / N) *
            (SumY2 - sqr(SumY) / N)
        )
    );

    Predicted = (Slope * 16.0) + Intercept;

    *SlopePointer = Slope;
    *InterceptPointer = Intercept;
    *CorrelationCoefficientPointer = CorrelationCoefficient;
    *PredictedNumberOfFilledCacheLinesPointer = Predicted;

    return;
}

VOID
ScoreNumberOfAssignedPerCacheLineCounts(
    _In_reads_(TOTAL_NUM_ASSIGNED_PER_CACHE_LINE) PULONG YCounts,
    _In_ ULONG TotalNumberOfAssigned,
    _Out_ PULONGLONG Score,
    _Out_ PDOUBLE Rank
    )
/*++

Routine Description:

    Given an array of 17 cache line counts, construct a score that is obtained
    by multiplying each array element by its relevant position squared in the
    array, then summing the results.  E.g. the number of assigned in the 3
    bucket is multiplied by 9, 5 is multiplied by 25, etc.

    The rank is (or at least should be) a decimal value between (0,1] that
    attempts to capture the score relative to the maximum possible score.

Arguments:

    YCounts - Supplies the array of cache line counts.

    Score - Supplies a pointer to a variable that receives the score.

    Rank - Supplies a pointer to a variable that receives the rank.

Return Value:

    None.

--*/
{
    BYTE Index;
    ULONGLONG X;
    ULONGLONG Y;
    ULONGLONG Sum;
    DOUBLE MaxScore;
    CONST BYTE Total = TOTAL_NUM_ASSIGNED_PER_CACHE_LINE;
    CONST SHORT XMultipliers[TOTAL_NUM_ASSIGNED_PER_CACHE_LINE] = {
        0, 1, 4, 9, 16, 25, 36, 49, 64, 81,
        100, 121, 144, 169, 196, 225, 256
    };

    Sum = 0;

    //
    // We start at 1 to ignore the count of cache lines with no assigned
    // elements.
    //

    for (Index = 1; Index < Total; Index++) {
        X = (ULONGLONG)XMultipliers[Index];
        Y = (ULONGLONG)YCounts[Index];

        Sum += (Y * X);
    }

    MaxScore = ((DOUBLE)TotalNumberOfAssigned / 16.0) * (DOUBLE)256;

    *Score = Sum;
    *Rank = Sum / MaxScore;

    return;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
