//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR GraphReleasePtxRawCStr[] =
    "//\n"
    "// Generated by NVIDIA NVVM Compiler\n"
    "//\n"
    "// Compiler Build ID: CL-27506705\n"
    "// Cuda compilation tools, release 10.2, V10.2.89\n"
    "// Based on LLVM 3.4svn\n"
    "//\n"
    "\n"
    ".version 6.5\n"
    ".target sm_30\n"
    ".address_size 64\n"
    "\n"
    "	// .globl	PerfectHashCudaSeededHashAllMultiplyShiftR2\n"
    "\n"
    ".visible .entry PerfectHashCudaSeededHashAllMultiplyShiftR2(\n"
    "	.param .u64 PerfectHashCudaSeededHashAllMultiplyShiftR2_param_0,\n"
    "	.param .u32 PerfectHashCudaSeededHashAllMultiplyShiftR2_param_1,\n"
    "	.param .u64 PerfectHashCudaSeededHashAllMultiplyShiftR2_param_2,\n"
    "	.param .u64 PerfectHashCudaSeededHashAllMultiplyShiftR2_param_3,\n"
    "	.param .u32 PerfectHashCudaSeededHashAllMultiplyShiftR2_param_4\n"
    ")\n"
    "{\n"
    "	.reg .pred 	%p<3>;\n"
    "	.reg .b32 	%r<21>;\n"
    "	.reg .b64 	%rd<11>;\n"
    "\n"
    "\n"
    "	ld.param.u64 	%rd3, [PerfectHashCudaSeededHashAllMultiplyShiftR2_param_0];\n"
    "	ld.param.u32 	%r10, [PerfectHashCudaSeededHashAllMultiplyShiftR2_param_1];\n"
    "	ld.param.u64 	%rd4, [PerfectHashCudaSeededHashAllMultiplyShiftR2_param_2];\n"
    "	ld.param.u64 	%rd5, [PerfectHashCudaSeededHashAllMultiplyShiftR2_param_3];\n"
    "	.loc 1 129 10\n"
    "	mov.u32 	%r1, %ntid.x;\n"
    "	mov.u32 	%r11, %ctaid.x;\n"
    "	mov.u32 	%r12, %tid.x;\n"
    "	mad.lo.s32 	%r20, %r1, %r11, %r12;\n"
    "	.loc 1 129 5\n"
    "	setp.ge.u32	%p1, %r20, %r10;\n"
    "	@%p1 bra 	BB0_3;\n"
    "\n"
    "	.loc 1 125 5\n"
    "	cvta.to.global.u64 	%rd6, %rd5;\n"
    "	ld.global.u32 	%r3, [%rd6];\n"
    "	.loc 1 126 5\n"
    "	ld.global.u32 	%r4, [%rd6+4];\n"
    "	.loc 1 127 5\n"
    "	ld.global.u32 	%r13, [%rd6+8];\n"
    "	.loc 1 132 9\n"
    "	and.b32  	%r5, %r13, 255;\n"
    "	.loc 1 133 9\n"
    "	bfe.u32 	%r6, %r13, 8, 8;\n"
    "	.loc 1 129 79\n"
    "	mov.u32 	%r14, %nctaid.x;\n"
    "	mul.lo.s32 	%r7, %r14, %r1;\n"
    "	.loc 1 125 5\n"
    "	cvta.to.global.u64 	%rd1, %rd3;\n"
    "	cvta.to.global.u64 	%rd2, %rd4;\n"
    "\n"
    "BB0_2:\n"
    "	.loc 1 130 9\n"
    "	mul.wide.u32 	%rd7, %r20, 4;\n"
    "	add.s64 	%rd8, %rd1, %rd7;\n"
    "	ld.global.u32 	%r15, [%rd8];\n"
    "	.loc 1 132 9\n"
    "	mul.lo.s32 	%r16, %r15, %r3;\n"
    "	.loc 1 133 9\n"
    "	mul.lo.s32 	%r17, %r15, %r4;\n"
    "	.loc 1 135 9\n"
    "	mul.wide.u32 	%rd9, %r20, 8;\n"
    "	add.s64 	%rd10, %rd2, %rd9;\n"
    "	.loc 1 133 9\n"
    "	shr.u32 	%r18, %r17, %r6;\n"
    "	.loc 1 132 9\n"
    "	shr.u32 	%r19, %r16, %r5;\n"
    "	.loc 1 136 9\n"
    "	st.global.v2.u32 	[%rd10], {%r19, %r18};\n"
    "	.loc 1 129 79\n"
    "	add.s32 	%r20, %r7, %r20;\n"
    "	.loc 1 129 5\n"
    "	setp.lt.u32	%p2, %r20, %r10;\n"
    "	@%p2 bra 	BB0_2;\n"
    "\n"
    "BB0_3:\n"
    "	.loc 1 138 1\n"
    "	ret;\n"
    "}\n"
    "\n"
    "	// .globl	PerfectHashCudaEnterSolvingLoop\n"
    ".visible .entry PerfectHashCudaEnterSolvingLoop(\n"
    "	.param .u64 PerfectHashCudaEnterSolvingLoop_param_0\n"
    ")\n"
    "{\n"
    "	.reg .pred 	%p<2>;\n"
    "	.reg .b64 	%rd<5>;\n"
    "\n"
    "\n"
    "	.loc 2 193 22\n"
    "	// inline asm\n"
    "	mov.u64 	%rd2, %clock64;\n"
    "	// inline asm\n"
    "\n"
    "BB1_1:\n"
    "	.loc 2 196 18\n"
    "	// inline asm\n"
    "	mov.u64 	%rd3, %clock64;\n"
    "	// inline asm\n"
    "	sub.s64 	%rd4, %rd3, %rd2;\n"
    "	.loc 1 146 5\n"
    "	setp.lt.s64	%p1, %rd4, 1000;\n"
    "	@%p1 bra 	BB1_1;\n"
    "\n"
    "	.loc 1 147 1\n"
    "	ret;\n"
    "}\n"
    "\n"
    "	.file	1 \"c:\\\\src\\\\perfecthash\\\\src\\\\PerfectHash\\\\../PerfectHashCuda/Graph.cu\", 1595881408, 3073\n"
    "	.file	2 \"c:\\\\src\\\\perfecthash\\\\src\\\\PerfectHashCuda\\\\../PerfectHash/Cu.cuh\", 1595880544, 4451\n"
    "\n"
;

const STRING GraphReleasePtxRawCString = {
    sizeof(GraphReleasePtxRawCStr) - sizeof(CHAR),
    sizeof(GraphReleasePtxRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&GraphReleasePtxRawCStr,
};

#ifndef RawCString
#define RawCString (&GraphReleasePtxRawCString)
#endif
