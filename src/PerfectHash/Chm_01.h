/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm_01.h

Abstract:

    This is the header file for the Chm_01.c module, which is our first pass
    at the CHM perfect hash table algorithm.  It defines types related to the
    implementation of the CHM algorithm.

--*/

#include "stdafx.h"

//
// Declare the main work and file work callback functions.
//

extern PERFECT_HASH_MAIN_WORK_CALLBACK ProcessGraphCallbackChm01;
extern PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;

extern FILE_WORK_CALLBACK_IMPL PrepareTableCallbackChm01;
extern FILE_WORK_CALLBACK_IMPL SaveTableCallbackChm01;
extern FILE_WORK_CALLBACK_IMPL PrepareHeaderCallbackChm01;
extern FILE_WORK_CALLBACK_IMPL PrepareHeaderFileChm01;
extern FILE_WORK_CALLBACK_IMPL SaveHeaderCallbackChm01;
extern FILE_WORK_CALLBACK_IMPL SaveHeaderFileChm01;

extern SHOULD_WE_CONTINUE_TRYING_TO_SOLVE_GRAPH
    ShouldWeContinueTryingToSolveGraphChm01;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
