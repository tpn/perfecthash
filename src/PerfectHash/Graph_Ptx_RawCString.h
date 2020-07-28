#ifdef _DEBUG
#include "GraphDebug_Ptx_RawCString.h"
#define GraphPtxRawCStr GraphDebugPtxRawCStr
#else
#include "GraphRelease_Ptx_RawCString.h"
#define GraphPtxRawCStr GraphReleasePtxRawCStr
#endif
