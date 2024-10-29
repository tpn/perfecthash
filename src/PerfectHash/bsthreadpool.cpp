#include "bsthreadpool.h"
#include "BS_thread_pool_light.hpp"

EXTERN_C
PTHREADPOOL
ThreadpoolInit (
    int NumberOfThreads
    )
{
    return (PTHREADPOOL) new BS::thread_pool_light(NumberOfThreads);
}

EXTERN_C
void
ThreadpoolAddWork(
    PTHREADPOOL Threadpool,
    PTHREADPOOL_CALLBACK Function,
    void* Argument
    )
{
    ((BS::thread_pool_light*)Threadpool)->push_task(Function, Argument);
}

EXTERN_C
void
ThreadpoolWait(
    PTHREADPOOL Threadpool
    )
{
    ((BS::thread_pool_light*)Threadpool)->wait_for_tasks();
}

EXTERN_C
void
ThreadpoolDestroy(
    PTHREADPOOL Threadpool
    )
{
    delete (BS::thread_pool_light*)Threadpool;
}

