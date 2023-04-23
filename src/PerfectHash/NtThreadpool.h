/*++

Module Name:

    NtThreadpool.h

Abstract:

    Private header file for NT threadpool types.

--*/

#include "stdafx.h"

struct _TPP_BARRIER;
enum _TPP_CALLBACK_RUN_TYPE;
struct _TPP_CALLER;
struct _TPP_CLEANUP_GROUP_MEMBER;
struct _TPP_CLEANUP_GROUP_MEMBER_VFUNCS;
union _TPP_FLAGS_COUNT;
struct _TPP_ITE;
struct _TPP_ITE_WAITER;
struct _TPP_NUMA_NODE;
struct _TPP_PH;
struct _TPP_PH_LINKS;
union _TPP_POOL_QUEUE_STATE;
struct _TPP_QUEUE;
struct _TPP_REFCOUNT;
struct _TPP_TIMER_QUEUE;
struct _TPP_TIMER_SUBQUEUE;
union _TPP_WORK_STATE;
struct _TP_ALPC;
struct _TP_CALLBACK_ENVIRON_V3;
struct _TP_CALLBACK_INSTANCE;
enum _TP_CALLBACK_PRIORITY;
struct _TP_CLEANUP_GROUP;
struct _TP_DIRECT;
struct _TP_IO;
struct _TP_JOB;
struct _TP_POOL;
struct _TP_TASK;
struct _TP_TASK_CALLBACKS;
struct _TP_TIMER;
struct _TP_WAIT;
struct _TP_WORK;

typedef union _TPP_FLAGS_COUNT
{
  union
  {
    /* 0x0000 */ uint64_t Count : 60; /* bit position: 0 */
    /* 0x0000 */ uint64_t Flags : 4; /* bit position: 60 */
    /* 0x0000 */ int64_t Data;
  }; /* size: 0x0008 */
} TPP_FLAGS_COUNT, *PTPP_FLAGS_COUNT; /* size: 0x0008 */

typedef struct _TPP_ITE
{
  /* 0x0000 */ struct _TPP_ITE_WAITER* First;
} TPP_ITE, *PTPP_ITE; /* size: 0x0008 */

typedef struct _TPP_ITE_WAITER
{
  /* 0x0000 */ struct _TPP_ITE_WAITER* Next;
  /* 0x0008 */ void* ThreadId;
} TPP_ITE_WAITER, *PTPP_ITE_WAITER; /* size: 0x0010 */

typedef struct _TPP_CALLER
{
  /* 0x0000 */ void* ReturnAddress;
} TPP_CALLER, *PTPP_CALLER; /* size: 0x0008 */


typedef struct _TPP_QUEUE
{
  /* 0x0000 */ struct _LIST_ENTRY Queue;
  /* 0x0010 */ struct _RTL_SRWLOCK Lock;
} TPP_QUEUE, *PTPP_QUEUE; /* size: 0x0018 */

typedef struct _TPP_REFCOUNT
{
  /* 0x0000 */ volatile int32_t Refcount;
} TPP_REFCOUNT, *PTPP_REFCOUNT; /* size: 0x0004 */

typedef union _TPP_POOL_QUEUE_STATE
{
  union
  {
    /* 0x0000 */ int64_t Exchange;
    struct
    {
      /* 0x0000 */ int32_t RunningThreadGoal : 16; /* bit position: 0 */
      /* 0x0000 */ uint32_t PendingReleaseCount : 16; /* bit position: 16 */
      /* 0x0004 */ uint32_t QueueLength;
    }; /* size: 0x0008 */
  }; /* size: 0x0008 */
} TPP_POOL_QUEUE_STATE, *PTPP_POOL_QUEUE_STATE; /* size: 0x0008 */

typedef struct _TP_TASK_CALLBACKS
{
  /* 0x0000 */ void* ExecuteCallback /* function */;
  /* 0x0008 */ void* Unposted /* function */;
} TP_TASK_CALLBACKS, *PTP_TASK_CALLBACKS; /* size: 0x0010 */

typedef struct _TPP_BARRIER
{
  /* 0x0000 */ volatile union _TPP_FLAGS_COUNT Ptr;
  /* 0x0008 */ struct _RTL_SRWLOCK WaitLock;
  /* 0x0010 */ struct _TPP_ITE WaitList;
} TPP_BARRIER, *PTPP_BARRIER; /* size: 0x0018 */

typedef struct _TP_CLEANUP_GROUP
{
  /* 0x0000 */ struct _TPP_REFCOUNT Refcount;
  /* 0x0004 */ int32_t Released;
  /* 0x0008 */ struct _RTL_SRWLOCK MemberLock;
  /* 0x0010 */ struct _LIST_ENTRY MemberList;
  /* 0x0020 */ struct _TPP_BARRIER Barrier;
  /* 0x0038 */ struct _RTL_SRWLOCK CleanupLock;
  /* 0x0040 */ struct _LIST_ENTRY CleanupList;
} TP_CLEANUP_GROUP, *PTP_CLEANUP_GROUP; /* size: 0x0050 */

typedef struct _ALPC_WORK_ON_BEHALF_TICKET
{
  /* 0x0000 */ uint32_t ThreadId;
  /* 0x0004 */ uint32_t ThreadCreationTimeLow;
} ALPC_WORK_ON_BEHALF_TICKET, *PALPC_WORK_ON_BEHALF_TICKET; /* size: 0x0008 */

typedef struct _TPP_CLEANUP_GROUP_MEMBER
{
  /* 0x0000 */ struct _TPP_REFCOUNT Refcount;
  /* 0x0004 */ long Padding_233;
  /* 0x0008 */ const struct _TPP_CLEANUP_GROUP_MEMBER_VFUNCS* VFuncs;
  /* 0x0010 */ struct _TP_CLEANUP_GROUP* CleanupGroup;
  /* 0x0018 */ void* CleanupGroupCancelCallback /* function */;
  /* 0x0020 */ void* FinalizationCallback /* function */;
  /* 0x0028 */ struct _LIST_ENTRY CleanupGroupMemberLinks;
  /* 0x0038 */ struct _TPP_BARRIER CallbackBarrier;
  union
  {
    /* 0x0050 */ void* Callback;
    /* 0x0050 */ void* WorkCallback /* function */;
    /* 0x0050 */ void* SimpleCallback /* function */;
    /* 0x0050 */ void* TimerCallback /* function */;
    /* 0x0050 */ void* WaitCallback /* function */;
    /* 0x0050 */ void* IoCallback /* function */;
    /* 0x0050 */ void* AlpcCallback /* function */;
    /* 0x0050 */ void* AlpcCallbackEx /* function */;
    /* 0x0050 */ void* JobCallback /* function */;
  }; /* size: 0x0008 */
  /* 0x0058 */ void* Context;
  /* 0x0060 */ struct _ACTIVATION_CONTEXT* ActivationContext;
  /* 0x0068 */ void* SubProcessTag;
  /* 0x0070 */ struct _GUID ActivityId;
  /* 0x0080 */ struct _ALPC_WORK_ON_BEHALF_TICKET WorkOnBehalfTicket;
  /* 0x0088 */ void* RaceDll;
  /* 0x0090 */ struct _TP_POOL* Pool;
  /* 0x0098 */ struct _LIST_ENTRY PoolObjectLinks;
  union
  {
    /* 0x00a8 */ volatile int32_t Flags;
    /* 0x00a8 */ uint32_t LongFunction : 1; /* bit position: 0 */
    /* 0x00a8 */ uint32_t Persistent : 1; /* bit position: 1 */
    /* 0x00a8 */ uint32_t UnusedPublic : 14; /* bit position: 2 */
    /* 0x00a8 */ uint32_t Released : 1; /* bit position: 16 */
    /* 0x00a8 */ uint32_t CleanupGroupReleased : 1; /* bit position: 17 */
    /* 0x00a8 */ uint32_t InCleanupGroupCleanupList : 1; /* bit position: 18 */
    /* 0x00a8 */ uint32_t UnusedPrivate : 13; /* bit position: 19 */
  }; /* size: 0x0004 */
  /* 0x00ac */ long Padding_234;
  /* 0x00b0 */ struct _TPP_CALLER AllocCaller;
  /* 0x00b8 */ struct _TPP_CALLER ReleaseCaller;
  /* 0x00c0 */ enum _TP_CALLBACK_PRIORITY CallbackPriority;
  /* 0x00c4 */ int32_t __PADDING__[1];
} TPP_CLEANUP_GROUP_MEMBER, *PTPP_CLEANUP_GROUP_MEMBER; /* size: 0x00c8 */

typedef struct _TPP_CLEANUP_GROUP_MEMBER_VFUNCS
{
  /* 0x0000 */ void* Free /* function */;
  /* 0x0008 */ void* CallbackEpilog /* function */;
  /* 0x0010 */ void* StopCallbackGeneration /* function */;
  /* 0x0018 */ void* CancelPendingCallbacks /* function */;
} TPP_CLEANUP_GROUP_MEMBER_VFUNCS, *PTPP_CLEANUP_GROUP_MEMBER_VFUNCS; /* size: 0x0020 */

typedef struct _TP_TASK
{
  /* 0x0000 */ const struct _TP_TASK_CALLBACKS* Callbacks;
  /* 0x0008 */ uint32_t NumaNode;
  /* 0x000c */ uint8_t IdealProcessor;
  /* 0x000d */ char Padding_242[3];
  /* 0x0010 */ struct _LIST_ENTRY ListEntry;
} TP_TASK, *PTP_TASK; /* size: 0x0020 */

typedef struct _TP_DIRECT
{
  /* 0x0000 */ struct _TP_TASK Task;
  /* 0x0020 */ uint64_t Lock;
  /* 0x0028 */ struct _LIST_ENTRY IoCompletionInformationList;
  /* 0x0038 */ void* Callback /* function */;
  /* 0x0040 */ uint32_t NumaNode;
  /* 0x0044 */ uint8_t IdealProcessor;
  /* 0x0045 */ char __PADDING__[3];
} TP_DIRECT, *PTP_DIRECT; /* size: 0x0048 */

typedef struct _TPP_PH
{
  /* 0x0000 */ struct _TPP_PH_LINKS* Root;
} TPP_PH, *PTPP_PH; /* size: 0x0008 */

typedef struct _TPP_PH_LINKS
{
  /* 0x0000 */ struct _LIST_ENTRY Siblings;
  /* 0x0010 */ struct _LIST_ENTRY Children;
  /* 0x0020 */ int64_t Key;
} TPP_PH_LINKS, *PTPP_PH_LINKS; /* size: 0x0028 */

typedef struct _TPP_TIMER_SUBQUEUE
{
  /* 0x0000 */ int64_t Expiration;
  /* 0x0008 */ struct _TPP_PH WindowStart;
  /* 0x0010 */ struct _TPP_PH WindowEnd;
  /* 0x0018 */ void* Timer;
  /* 0x0020 */ void* TimerPkt;
  /* 0x0028 */ struct _TP_DIRECT Direct;
  /* 0x0070 */ uint32_t ExpirationWindow;
  /* 0x0074 */ int32_t __PADDING__[1];
} TPP_TIMER_SUBQUEUE, *PTPP_TIMER_SUBQUEUE; /* size: 0x0078 */

typedef struct _TPP_TIMER_QUEUE
{
  /* 0x0000 */ struct _RTL_SRWLOCK Lock;
  /* 0x0008 */ struct _TPP_TIMER_SUBQUEUE AbsoluteQueue;
  /* 0x0080 */ struct _TPP_TIMER_SUBQUEUE RelativeQueue;
  /* 0x00f8 */ int32_t AllocatedTimerCount;
  /* 0x00fc */ int32_t __PADDING__[1];
} TPP_TIMER_QUEUE, *PTPP_TIMER_QUEUE; /* size: 0x0100 */

typedef union _TPP_WORK_STATE
{
  union
  {
    /* 0x0000 */ int32_t Exchange;
    /* 0x0000 */ uint32_t Insertable : 1; /* bit position: 0 */
    /* 0x0000 */ uint32_t PendingCallbackCount : 31; /* bit position: 1 */
  }; /* size: 0x0004 */
} TPP_WORK_STATE, *PTPP_WORK_STATE; /* size: 0x0004 */

typedef struct _TP_POOL
{
  /* 0x0000 */ struct _TPP_REFCOUNT Refcount;
  /* 0x0004 */ long Padding_62;
  /* 0x0008 */ volatile union _TPP_POOL_QUEUE_STATE QueueState;
  /* 0x0010 */ struct _TPP_QUEUE* TaskQueue[3];
  /* 0x0028 */ struct _TPP_NUMA_NODE* NumaNode;
  /* 0x0030 */ struct _GROUP_AFFINITY* ProximityInfo;
  /* 0x0038 */ void* WorkerFactory;
  /* 0x0040 */ void* CompletionPort;
  /* 0x0048 */ struct _RTL_SRWLOCK Lock;
  /* 0x0050 */ struct _LIST_ENTRY PoolObjectList;
  /* 0x0060 */ struct _LIST_ENTRY WorkerList;
  /* 0x0070 */ struct _TPP_TIMER_QUEUE TimerQueue;
  /* 0x0170 */ struct _RTL_SRWLOCK ShutdownLock;
  /* 0x0178 */ uint8_t ShutdownInitiated;
  /* 0x0179 */ uint8_t Released;
  /* 0x017a */ uint16_t PoolFlags;
  /* 0x017c */ long Padding_63;
  /* 0x0180 */ struct _LIST_ENTRY PoolLinks;
  /* 0x0190 */ struct _TPP_CALLER AllocCaller;
  /* 0x0198 */ struct _TPP_CALLER ReleaseCaller;
  /* 0x01a0 */ volatile int32_t AvailableWorkerCount;
  /* 0x01a4 */ volatile int32_t LongRunningWorkerCount;
  /* 0x01a8 */ uint32_t LastProcCount;
  /* 0x01ac */ volatile int32_t NodeStatus;
  /* 0x01b0 */ volatile int32_t BindingCount;
  /* 0x01b4 */ uint32_t CallbackChecksDisabled : 1; /* bit position: 0 */
  /* 0x01b4 */ uint32_t TrimTarget : 11; /* bit position: 1 */
  /* 0x01b4 */ uint32_t TrimmedThrdCount : 11; /* bit position: 12 */
  /* 0x01b8 */ uint32_t SelectedCpuSetCount;
  /* 0x01bc */ long Padding_64;
  /* 0x01c0 */ struct _RTL_CONDITION_VARIABLE TrimComplete;
  /* 0x01c8 */ struct _LIST_ENTRY TrimmedWorkerList;

    DWORD ThreadMinimum;
    DWORD ThreadMaximum;

} TP_POOL, *PTP_POOL; /* size: 0x01d8 */

typedef struct _TP_IO
{
  /* 0x0000 */ struct _TPP_CLEANUP_GROUP_MEMBER CleanupGroupMember;
  /* 0x00c8 */ struct _TP_DIRECT Direct;
  /* 0x0110 */ void* File;
  /* 0x0118 */ volatile int32_t PendingIrpCount;
  /* 0x011c */ int32_t __PADDING__[1];
} TP_IO, *PTP_IO; /* size: 0x0120 */

typedef struct _TP_JOB
{
  /* 0x0000 */ struct _TP_DIRECT Direct;
  /* 0x0048 */ struct _TPP_CLEANUP_GROUP_MEMBER CleanupGroupMember;
  /* 0x0110 */ void* JobHandle;
  union
  {
    /* 0x0118 */ volatile int64_t CompletionState;
    /* 0x0118 */ int64_t Rundown : 1; /* bit position: 0 */
    /* 0x0118 */ int64_t CompletionCount : 63; /* bit position: 1 */
  }; /* size: 0x0008 */
  /* 0x0120 */ struct _RTL_SRWLOCK RundownLock;
} TP_JOB, *PTP_JOB; /* size: 0x0128 */

// --

typedef enum _TPP_CALLBACK_RUN_TYPE
{
  TppCallbackRunTypeNormal = 0,
  TppCallbackRunTypeLong = 1,
  TppCallbackRunTypeIndependent = 2,
  TppCallbackRunTypeShort = 3,
  TppCallbackRunTypeShortUsed = 4,
} TPP_CALLBACK_RUN_TYPE, *PTPP_CALLBACK_RUN_TYPE;

typedef struct _TPP_NUMA_NODE
{
  /* 0x0000 */ int32_t WorkerCount;
} TPP_NUMA_NODE, *PTPP_NUMA_NODE; /* size: 0x0004 */


typedef struct _TP_ALPC
{
  /* 0x0000 */ struct _TP_DIRECT Direct;
  /* 0x0048 */ struct _TPP_CLEANUP_GROUP_MEMBER CleanupGroupMember;
  /* 0x0110 */ void* AlpcPort;
  /* 0x0118 */ int32_t DeferredSendCount;
  /* 0x011c */ int32_t LastConcurrencyCount;
  union
  {
    /* 0x0120 */ uint32_t Flags;
    /* 0x0120 */ uint32_t ExTypeCallback : 1; /* bit position: 0 */
    /* 0x0120 */ uint32_t CompletionListRegistered : 1; /* bit position: 1 */
    /* 0x0120 */ uint32_t Reserved : 30; /* bit position: 2 */
  }; /* size: 0x0004 */
  /* 0x0124 */ int32_t __PADDING__[1];
} TP_ALPC, *PTP_ALPC; /* size: 0x0128 */

typedef struct _RTL_ACTIVATION_CONTEXT_STACK_FRAME
{
  /* 0x0000 */ struct _RTL_ACTIVATION_CONTEXT_STACK_FRAME* Previous;
  /* 0x0008 */ struct _ACTIVATION_CONTEXT* ActivationContext;
  /* 0x0010 */ uint32_t Flags;
  /* 0x0014 */ int32_t __PADDING__[1];
} RTL_ACTIVATION_CONTEXT_STACK_FRAME, *PRTL_ACTIVATION_CONTEXT_STACK_FRAME; /* size: 0x0018 */

typedef struct _RTL_CALLER_ALLOCATED_ACTIVATION_CONTEXT_STACK_FRAME_EXTENDED
{
  /* 0x0000 */ uint64_t Size;
  /* 0x0008 */ uint32_t Format;
  /* 0x000c */ long Padding_45;
  /* 0x0010 */ struct _RTL_ACTIVATION_CONTEXT_STACK_FRAME Frame;
  /* 0x0028 */ void* Extra1;
  /* 0x0030 */ void* Extra2;
  /* 0x0038 */ void* Extra3;
  /* 0x0040 */ void* Extra4;
} RTL_CALLER_ALLOCATED_ACTIVATION_CONTEXT_STACK_FRAME_EXTENDED, *PRTL_CALLER_ALLOCATED_ACTIVATION_CONTEXT_STACK_FRAME_EXTENDED; /* size: 0x0048 */

typedef struct _WORKER_FACTORY_DEFERRED_WORK
{
  /* 0x0000 */ struct _PORT_MESSAGE* AlpcSendMessage;
  /* 0x0008 */ void* AlpcSendMessagePort;
  /* 0x0010 */ uint32_t AlpcSendMessageFlags;
  /* 0x0014 */ uint32_t Flags;
} WORKER_FACTORY_DEFERRED_WORK, *PWORKER_FACTORY_DEFERRED_WORK; /* size: 0x0018 */

typedef struct _TP_CALLBACK_INSTANCE
{
  /* 0x0000 */ struct _RTL_CALLER_ALLOCATED_ACTIVATION_CONTEXT_STACK_FRAME_EXTENDED ActivationFrame;
  /* 0x0048 */ enum _TPP_CALLBACK_RUN_TYPE CallbackRunType;
  /* 0x004c */ uint8_t ActivationFrameUsed : 1; /* bit position: 0 */
  /* 0x004c */ uint8_t Disassociated : 1; /* bit position: 1 */
  /* 0x004c */ uint8_t UnrecoverableErrorDetected : 1; /* bit position: 2 */
  /* 0x004d */ char Padding_236[3];
  /* 0x0050 */ void* SubProcessTag;
  union
  {
    /* 0x0058 */ void* Callback;
    /* 0x0058 */ void* WorkCallback /* function */;
    /* 0x0058 */ void* SimpleCallback /* function */;
    /* 0x0058 */ void* TimerCallback /* function */;
    /* 0x0058 */ void* WaitCallback /* function */;
    /* 0x0058 */ void* IoCallback /* function */;
    /* 0x0058 */ void* AlpcCallback /* function */;
    /* 0x0058 */ void* AlpcCallbackEx /* function */;
    /* 0x0058 */ void* JobCallback /* function */;
    /* 0x0058 */ void* FinalizationCallback /* function */;
    /* 0x0058 */ void* DirectCallback /* function */;
    /* 0x0058 */ void* TaskCallback /* function */;
  }; /* size: 0x0008 */
  /* 0x0060 */ void* Context;
  union
  {
    /* 0x0068 */ uint32_t SkipPostThreadFlags;
    /* 0x0068 */ uint32_t SPActFrame : 1; /* bit position: 0 */
    /* 0x0068 */ uint32_t SPSubProcTag : 1; /* bit position: 1 */
    /* 0x0068 */ uint32_t SPImpersonation : 1; /* bit position: 2 */
    /* 0x0068 */ uint32_t SPThreadInfo : 1; /* bit position: 3 */
    /* 0x0068 */ uint32_t SPTransaction : 1; /* bit position: 4 */
    /* 0x0068 */ uint32_t SPLdrLock : 1; /* bit position: 5 */
    /* 0x0068 */ uint32_t SPLanguages : 1; /* bit position: 6 */
    /* 0x0068 */ uint32_t SPThreadPriBack : 1; /* bit position: 7 */
  }; /* size: 0x0004 */
  union
  {
    /* 0x006c */ uint32_t VerifyThreadFlags;
    /* 0x006c */ uint32_t VThreadInfo : 1; /* bit position: 0 */
  }; /* size: 0x0004 */
  /* 0x0070 */ int32_t PreCallThrdPriority;
  /* 0x0074 */ long Padding_237;
  /* 0x0078 */ uint64_t PreCallThrdAffinity;
  /* 0x0080 */ struct _TP_POOL* Pool;
  /* 0x0088 */ struct _TP_ALPC* AlpcWorkItem;
  /* 0x0090 */ uint32_t CallbackEpilogFlags;
  /* 0x0094 */ uint32_t Event;
  /* 0x0098 */ uint32_t Mutex;
  /* 0x009c */ uint32_t Semaphore;
  /* 0x00a0 */ uint32_t SemaphoreReleaseCount;
  /* 0x00a4 */ long Padding_238;
  /* 0x00a8 */ void* RaceDll;
  /* 0x00b0 */ struct _TP_CLEANUP_GROUP* CleanupGroup;
  /* 0x00b8 */ struct _TPP_CLEANUP_GROUP_MEMBER* CleanupGroupMember;
  /* 0x00c0 */ struct _RTL_CRITICAL_SECTION* CriticalSection;
  /* 0x00c8 */ void* DllHandle;
  /* 0x00d0 */ struct _WORKER_FACTORY_DEFERRED_WORK DeferredWork;
  /* 0x00e8 */ struct _GUID PreviousActivityId;
  /* 0x00f8 */ uint64_t WorkOnBehalf;
} TP_CALLBACK_INSTANCE, *PTP_CALLBACK_INSTANCE; /* size: 0x0100 */

typedef struct _TP_WORK
{
  /* 0x0000 */ struct _TPP_CLEANUP_GROUP_MEMBER CleanupGroupMember;
  /* 0x00c8 */ struct _TP_TASK Task;
  /* 0x00e8 */ volatile union _TPP_WORK_STATE WorkState;
  /* 0x00ec */ int32_t __PADDING__[1];
} TP_WORK, *PTP_WORK; /* size: 0x00f0 */

typedef struct _TP_TIMER
{
  /* 0x0000 */ struct _TP_WORK Work;
  /* 0x00f0 */ struct _RTL_SRWLOCK Lock;
  union
  {
    /* 0x00f8 */ struct _TPP_PH_LINKS WindowEndLinks;
    /* 0x00f8 */ struct _LIST_ENTRY ExpirationLinks;
  }; /* size: 0x0028 */
  /* 0x0120 */ struct _TPP_PH_LINKS WindowStartLinks;
  /* 0x0148 */ int64_t DueTime;
  /* 0x0150 */ struct _TPP_ITE CancelIte;
  /* 0x0158 */ uint32_t Window;
  /* 0x015c */ uint32_t Period;
  /* 0x0160 */ uint8_t Inserted;
  /* 0x0161 */ uint8_t WaitTimer;
  union
  {
    /* 0x0162 */ uint8_t TimerStatus;
    /* 0x0162 */ uint8_t InQueue : 1; /* bit position: 0 */
    /* 0x0162 */ uint8_t Absolute : 1; /* bit position: 1 */
    /* 0x0162 */ uint8_t Cancelled : 1; /* bit position: 2 */
  }; /* size: 0x0001 */
  /* 0x0163 */ uint8_t BlockInsert;
  /* 0x0164 */ int32_t __PADDING__[1];
} TP_TIMER, *PTP_TIMER; /* size: 0x0168 */

typedef struct _TP_WAIT
{
  /* 0x0000 */ struct _TP_TIMER Timer;
  /* 0x0168 */ void* Handle;
  /* 0x0170 */ void* WaitPkt;
  /* 0x0178 */ void* NextWaitHandle;
  /* 0x0180 */ union _LARGE_INTEGER NextWaitTimeout;
  /* 0x0188 */ struct _TP_DIRECT Direct;
  union
  {
    union
    {
      /* 0x01d0 */ uint8_t AllFlags;
      /* 0x01d0 */ uint8_t NextWaitActive : 1; /* bit position: 0 */
      /* 0x01d0 */ uint8_t NextTimeoutActive : 1; /* bit position: 1 */
      /* 0x01d0 */ uint8_t CallbackCounted : 1; /* bit position: 2 */
      /* 0x01d0 */ uint8_t Spare : 5; /* bit position: 3 */
    }; /* size: 0x0001 */
  } /* size: 0x0001 */ WaitFlags;
  /* 0x01d1 */ char __PADDING__[7];
} TP_WAIT, *PTP_WAIT; /* size: 0x01d8 */

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
