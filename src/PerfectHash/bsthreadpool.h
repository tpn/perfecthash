#pragma once

#ifndef EXTERN_C
#define EXTERN_C extern "C"
#endif

#ifdef __cplusplus
EXTERN_C {
#endif

typedef struct _THREADPOOL THREADPOOL;
typedef struct _THREADPOOL *PTHREADPOOL;

typedef void (*PTHREADPOOL_CALLBACK)(void*);

PTHREADPOOL
ThreadpoolInit (
    int NumberOfThreads
    );

void
ThreadpoolAddWork(
    PTHREADPOOL Threadpool,
    PTHREADPOOL_CALLBACK Function,
    void* Argument
    );

void
ThreadpoolWait(
    PTHREADPOOL Threadpool
    );

void
ThreadpoolDestroy(
    PTHREADPOOL Threadpool
    );

#ifdef __cplusplus
}
#endif

