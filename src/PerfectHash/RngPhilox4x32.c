/*++

Copyright (c) 2021 Trent Nelson <trent@trent.me>

Module Name:

    RngPhilox4x32.c

Abstract:

    This module implements the Philox 4x32 random number generator.  It is
    based off curand_philox4x32_x.h from CUDA (which is based off the 2011
    paper by D.E. Shaw).  The only changes from the cuRAND version is the
    conversion to our coding style (with some C-related tweaks to return types
    and function parameter types).

    N.B. I've included both the NVIDIA and D.E. Shaw copyrights as that's what
         they do in curand_philox4x32_x.h (despite not looking anything like
         D.E. Shaw's original code).

--*/

/* Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * The source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * The Licensed Deliverables contained herein are PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and are being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
/*
   Copyright 2010-2011, D. E. Shaw Research.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions, and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions, and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 * Neither the name of D. E. Shaw Research nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "stdafx.h"

#define PHILOX_W32_0   (0x9E3779B9)
#define PHILOX_W32_1   (0xBB67AE85)
#define PHILOX_M4x32_0 (0xD2511F53)
#define PHILOX_M4x32_1 (0xCD9E8D57)

VOID
PhiloxStateIncrement(
    PRNG_STATE_PHILOX43210 State,
    ULONGLONG Offset
    )
{
    ULONG Low = (ULONG)(Offset);
    ULONG High = (ULONG)(Offset >> 32);

    State->Counter.X += Low;

    if (State->Counter.X < Low) {
        High++;
    }

    State->Counter.Y += High;

    if (High <= State->Counter.Y) {
       return;
    }

    if (++State->Counter.Z) {
       return;
    }

    ++State->Counter.W;
}

VOID
PhiloxStateIncrementHigh(
    PRNG_STATE_PHILOX43210 State,
    ULONGLONG Number
    )
{
    ULONG Low = (ULONG)(Number);
    ULONG High = (ULONG)(Number >> 32);

    State->Counter.Z += Low;

    if (State->Counter.Z < Low) {
        High++;
    }

    State->Counter.W += High;
}

VOID
PhiloxStateIncrementNoOffset(
    PRNG_STATE_PHILOX43210 State
    )
{
    if (++State->Counter.X) {
        return;
    }

    if (++State->Counter.Y) {
        return;
    }

    if (++State->Counter.Z) {
        return;
    }

    ++State->Counter.W;
}

ULONG
MulHighLow32(
    ULONG A,
    ULONG B,
    PULONG Out
    )
{
    ULONGLONG Product = ((ULONGLONG)A) * ((ULONGLONG)B);
    *Out = (ULONG)(Product >> 32ULL);
    return (ULONG)Product;
}

XMMWORD
Philox4x32(
    XMMWORD CounterXmm,
    UINT2 Key
    )
{
    ULONG High0;
    ULONG High1;
    ULONG Low0;
    ULONG Low1;
    UINT4 Output;
    UINT4 Counter;

    Counter.AsXmmWord = CounterXmm;
    Low0 = MulHighLow32(PHILOX_M4x32_0, Counter.X, &High0);
    Low1 = MulHighLow32(PHILOX_M4x32_1, Counter.Z, &High1);

    Output.X = High1 ^ Counter.Y ^ Key.X;
    Output.Y = Low1;
    Output.Z = High0 ^ Counter.W ^ Key.Y;
    Output.W = Low0;

    return Output.AsXmmWord;
}

XMMWORD
Philox4x32_10(
    XMMWORD Counter,
    UINT2 Key
    )
{
    UINT4 Output;

#define DO_ROUND()                      \
    Counter = Philox4x32(Counter, Key); \
    Key.X += PHILOX_W32_0;              \
    Key.Y += PHILOX_W32_1

    DO_ROUND(); //  1
    DO_ROUND(); //  2
    DO_ROUND(); //  3
    DO_ROUND(); //  4
    DO_ROUND(); //  5
    DO_ROUND(); //  6
    DO_ROUND(); //  7
    DO_ROUND(); //  8
    DO_ROUND(); //  9

    //
    // Tenth round.
    //

    Output.AsXmmWord = Philox4x32(Counter, Key);
    return Output.AsXmmWord;
}


//
// Helper macro for calling the generator correctly.
//

#define DO_PHILOX_4x32_10(State)             \
    State->Output.AsXmmWord = Philox4x32_10( \
        State->Counter.AsXmmWord,            \
        State->Key                           \
    )

LONG
GetRandomLong(PRNG_STATE_PHILOX43210 State)
{
    ULONG Value = 0;

    State->TotalCount += 1;
    State->TotalBytes += sizeof(ULONG);

    switch (State->CurrentCount++) {
        case 0:
            Value = State->Output.X;
            break;
        case 1:
            Value = State->Output.Y;
            break;
        case 2:
            Value = State->Output.Z;
            break;
        case 3:
            Value = State->Output.W;
            break;
        default:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;
    }
    if (State->CurrentCount == 4) {
        PhiloxStateIncrementNoOffset(State);
        DO_PHILOX_4x32_10(State);
        State->CurrentCount = 0;
    }
    return (LONG)Value;
}

VOID
Skipahead(
    PRNG_STATE_PHILOX43210 State,
    ULONGLONG Offset
    )
{
    State->CurrentCount += (Offset & 3);
    Offset /= 4;

    if (State->CurrentCount > 3) {
        Offset += 1;
        State->CurrentCount -= 4;
    }
    PhiloxStateIncrement(State, Offset);
    DO_PHILOX_4x32_10(State);
}

VOID
SkipaheadSubsequence(
    PRNG_STATE_PHILOX43210 State,
    ULONGLONG Subsequence
    )
{
    PhiloxStateIncrementHigh(State, Subsequence);
    DO_PHILOX_4x32_10(State);
}

VOID
RngPhilox43210Init(
    PRNG_STATE_PHILOX43210 State
    )
{
    PRNG Rng = STATE_TO_RNG(State);

    State->Counter.X = 0;
    State->Counter.Y = 0;
    State->Counter.Z = 0;
    State->Counter.W = 0;
    State->Output.X = 0;
    State->Output.Y = 0;
    State->Output.Z = 0;
    State->Output.W = 0;
    State->Key.X = (ULONG)Rng->Seed;
    State->Key.Y = (ULONG)(Rng->Seed >> 32);
    State->CurrentCount = 0;
    State->TotalCount = 0;
    State->TotalBytes = 0ULL;
    SkipaheadSubsequence(State, Rng->Subsequence);
    Skipahead(State, Rng->Offset);
}

HRESULT
RngPhilox43210GenerateRandomBytes(
    PRNG_STATE_PHILOX43210 State,
    SIZE_T SizeOfBufferInBytes,
    PBYTE Buffer
    )
{
    ULONG Index;
    ULONG NumberOfLongs;
    LONG Random;
    PLONG Long;

    NumberOfLongs = (ULONG)(SizeOfBufferInBytes >> 2);
    Long = (PLONG)Buffer;

    for (Index = 0; Index < NumberOfLongs; Index++) {
        Random = GetRandomLong(State);
        ASSERT(Random != 0);
        *Long++ = Random;
    }

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
