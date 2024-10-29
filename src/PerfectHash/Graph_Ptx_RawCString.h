#ifdef _DEBUG
#include "GraphDebug_Ptx_RawCString.h"
#define GraphPtxRawCStr GraphDebugPtxRawCStr
#define GraphPtxRawCString GraphDebugPtxRawCString
#else
#include "GraphRelease_Ptx_RawCString.h"
#define GraphPtxRawCStr GraphReleasePtxRawCStr
#define GraphPtxRawCString GraphReleasePtxRawCString
#endif
