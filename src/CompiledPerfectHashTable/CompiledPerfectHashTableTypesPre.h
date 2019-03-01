
#ifndef BASETYPES

//
// Define basic NT types.
//

typedef int BOOL;
typedef char BOOLEAN;
typedef unsigned char BYTE;
typedef BYTE *PBYTE;
typedef short SHORT;
typedef short *PSHORT;
typedef unsigned short USHORT;
typedef unsigned short *PUSHORT;
typedef long long LONGLONG;
typedef long long *PLONGLONG;
typedef unsigned long long ULONGLONG;
typedef unsigned long long *PULONGLONG;
typedef void *PVOID;

#define VOID void

#ifdef _WIN32
typedef long LONG;
typedef long *PLONG;
typedef unsigned long ULONG;
typedef unsigned long *PULONG;
#elif __linux__
typedef int LONG;
typedef int *PLONG;
typedef unsigned int ULONG;
typedef unsigned int *PULONG;
#endif

#define TRUE 1
#define FALSE 0


#define BASETYPES

#endif
