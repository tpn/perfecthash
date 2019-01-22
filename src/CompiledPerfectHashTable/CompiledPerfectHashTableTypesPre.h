
//
// Define start/end markers for IACA.
//

#define IACA_VC_START() __writegsbyte(111, 111)
#define IACA_VC_END()   __writegsbyte(222, 222)

//
// Define basic NT types and macros used by this header file.
//

#define CPHCALLTYPE __stdcall
#define FORCEINLINE __forceinline

typedef char BOOLEAN;
typedef unsigned char BYTE;
typedef BYTE *PBYTE;
typedef short SHORT;
typedef unsigned short USHORT;
typedef long LONG;
typedef long long LONGLONG;
typedef unsigned long ULONG;
typedef unsigned long *PULONG;
typedef unsigned long long ULONGLONG;
typedef void *PVOID;

