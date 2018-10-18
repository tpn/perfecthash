/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableCompile.c

Abstract:

    This module implements functionality for compiling perfect hash tables into
    a more optimal format.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_COMPILE PerfectHashTableCompile;

_Use_decl_annotations_
HRESULT
PerfectHashTableCompile(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlagsPointer,
    PERFECT_HASH_CPU_ARCH_ID CpuArchId
    )
/*++

Routine Description:

    Compiles a loaded perfect hash table into an optimized format.

Arguments:

    Table - Supplies a pointer to the PERFECT_HASH_TABLE interface for which
        the compilation is to be performed.

    TableCompileFlags - Optionally supplies a pointer to a table compile flags
        structure that can be used to customize the compilation behavior.

    CpuArchId - Supplies the CPU architecture for which the perfect hash table
        compilation is to target.  If this differs from the current CPU arch,
        cross-compilation must be supported by the underlying algorith, hash
        function and masking type.

Return Value:

    S_OK - Table compiled successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table was NULL.

    E_UNEXPECTED - General error.

    E_OUTOFMEMORY - Out of memory.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags provided.

    PH_E_TABLE_LOCKED - The table is locked.

    PH_E_TABLE_NOT_CREATED - The table has not been created.

    PH_E_INVALID_CPU_ARCH_ID - Invalid CPU architecture ID.

    PH_E_TABLE_COMPILATION_FAILED - Table compilation failed.

--*/
{
    PRTL Rtl;
    PACL Acl = NULL;
    BOOL Success;
    BOOL InheritHandles;
    ULONG ExitCode;
    ULONG WaitResult;
    ULONG CreationFlags;
    HRESULT Result = S_OK;
    NTSTATUS Status;
    PWSTR Environment;
    PWSTR ApplicationName;
    PWSTR CurrentDirectory;
    EXPLICIT_ACCESS_W ExplicitAccess;
    PSECURITY_ATTRIBUTES ThreadAttributes;
    PSECURITY_ATTRIBUTES ProcessAttributes;
    SECURITY_ATTRIBUTES SecurityAttributes;
    SECURITY_DESCRIPTOR SecurityDescriptor;
    STARTUPINFOW StartupInfo;
    PROCESS_INFORMATION ProcessInfo;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;
    WCHAR CommandBuffer[COMPILE_COMMANDLINE_BUFFER_SIZE_IN_CHARS];
    PCUNICODE_STRING Source;
    UNICODE_STRING Command;
    const UNICODE_STRING Null = RTL_CONSTANT_STRING(L"\0");
    const UNICODE_STRING Prefix = RTL_CONSTANT_STRING(
        L"msbuild /nologo /noconlog /m /t:Rebuild "
        L"/p:Configuration=Release;Platform=x64 "
    );

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!IsValidPerfectHashCpuArchId(CpuArchId)) {
        return PH_E_INVALID_CPU_ARCH_ID;
    }

    VALIDATE_FLAGS(TableCompile, TABLE_COMPILE);

    if (!TryAcquirePerfectHashTableLockExclusive(Table)) {
        return PH_E_TABLE_LOCKED;
    }

    if (!Table->Flags.Created) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_NOT_CREATED;
    }

    //
    // Argument validation complete.
    //

    Rtl = Table->Rtl;
    ZeroStruct(ProcessInfo);
    ZeroStruct(StartupInfo);

    //
    // Initialize the command buffer and corresponding UNICODE_STRING struct.
    //

    ZeroArray(CommandBuffer);

    Command.Length = 0;
    Command.MaximumLength = sizeof(CommandBuffer);
    Command.Buffer = (PWSTR)CommandBuffer;

    //
    // Construct the command line for compiling the table.
    //

    Source = &Prefix;
    Status = Rtl->RtlAppendUnicodeStringToString(&Command, Source);
    if (FAILED(Status)) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(PerfectHashTableCompile_AppendPrefix, Result);
        goto Error;
    }

    Source = &Table->VSSolutionFile->Path->FileName;
    Status = Rtl->RtlAppendUnicodeStringToString(&Command, Source);
    if (FAILED(Status)) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(PerfectHashTableCompile_AppendSolutionFile, Result);
        goto Error;
    }

    Source = &Null;
    Status = Rtl->RtlAppendUnicodeStringToString(&Command, Source);
    if (FAILED(Status)) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(PerfectHashTableCompile_AppendNull, Result);
        goto Error;
    }

    //
    // Create an exclusive DACL for the process and thread.
    //

    Result = CreateExclusiveDaclForCurrentUser(Rtl,
                                               &SecurityAttributes,
                                               &SecurityDescriptor,
                                               &ExplicitAccess,
                                               &Acl);

    if (FAILED(Result)) {
        PH_ERROR(CreateExclusiveDaclForCurrentUser, Result);
        goto Error;
    }

    ProcessAttributes = ThreadAttributes = &SecurityAttributes;

    //
    // Create the process.
    //

    ApplicationName = NULL;
    InheritHandles = FALSE;
    CreationFlags = DETACHED_PROCESS;
    Environment = NULL;
    CurrentDirectory = Table->OutputDirectory->Path->FullPath.Buffer;
    StartupInfo.cb = sizeof(StartupInfo);

    Success = CreateProcessW(ApplicationName,
                             Command.Buffer,
                             ProcessAttributes,
                             ThreadAttributes,
                             InheritHandles,
                             CreationFlags,
                             Environment,
                             CurrentDirectory,
                             &StartupInfo,
                             &ProcessInfo);

    if (!Success) {
        SYS_ERROR(CreateProcessW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Process was created successfully.  Wait for it to finish.
    //

    WaitResult = WaitForSingleObject(ProcessInfo.hProcess, INFINITE);
    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Get the exit code.
    //

    ExitCode = (ULONG)-1;
    Success = GetExitCodeProcess(ProcessInfo.hProcess, &ExitCode);
    if (!Success) {
        SYS_ERROR(GetExitCodeProcess);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (ExitCode != 0) {
        Result = PH_E_TABLE_COMPILATION_FAILED;
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Free the Acl structure (allocated by CreateExclusiveDaclForCurrentUser)
    // if applicable.
    //

    if (Acl) {
        LocalFree(Acl);
        Acl = NULL;
    }

    //
    // Close the process and thread handles if applicable.
    //

    if (ProcessInfo.hProcess) {
        if (!CloseHandle(ProcessInfo.hProcess)) {
            SYS_ERROR(CloseHandle);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
    }

    if (ProcessInfo.hThread) {
        if (!CloseHandle(ProcessInfo.hThread)) {
            SYS_ERROR(CloseHandle);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
    }

    //
    // Release the table lock and return.
    //

    ReleasePerfectHashTableLockExclusive(Table);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
