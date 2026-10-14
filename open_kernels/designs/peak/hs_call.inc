asm(R"hs(
	.text
	.globl	hs_call
	.p2align	4
	.type	hs_call,@function
hs_call:
	movx r24, #0
	movx r25, #0
	mova r3, #780
	movxm r17, #2048
	movxm r7, #1152
	mov r4, p0
	mov r6, p1
	mov r16, p2
	add r5, r4, r7
	add r20, r16, r17
	add r18, r20, r17
	add r19, r18, r17
	mov p3, r20
	mov p4, r16
	mov p5, r20
	vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vldb.pop.576 ex0, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vldb.pop.576 ex2, [p1, lf1, r25]
	vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vlda.conv.fp32.bf16 cml1, [p2], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vlda.conv.fp32.bf16 cmh1, [p2], #64;		vldb.pop.576 ex6, [p1, lf1, r25]
	nop
	vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm0, ex0, ex1, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24]
	vmac.f dm3, dm1, ex0, ex2, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vlda.conv.fp32.bf16 cml0, [p3], #64
	vlda.conv.fp32.bf16 cmh0, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm1, dm2, ex5, ex3, r3;		vldb.pop.576 ex0, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm3, ex5, ex6, r3;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm3, dm4, ex4, ex1, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm0, ex4, ex2, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex9, ex8, r3;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm2, ex9, ex10, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm2, dm3, ex7, ex3, r3
	vmac.f dm3, dm4, ex7, ex6, r3;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm4, dm0, ex0, ex5, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex0, ex1, r3;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm2, ex11, ex8, r3
	vmac.f dm2, dm3, ex11, ex10, r3;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm3, dm4, ex4, ex9, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm0, ex4, ex3, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm0, dm1, ex2, ex5, r3
	vmac.f dm1, dm2, ex2, ex1, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex7, ex0, r3;		vldb.pop.576 ex2, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm4, ex7, ex8, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm4, dm0, ex6, ex9, r3;		mov p0, r4;		movx r24, #0
	vmac.f dm0, dm1, ex6, ex3, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex11, ex4, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm3, ex11, ex5, r3;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm3, dm4, ex10, ex0, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm4, dm0, ex10, ex8, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm1, dm1, ex2, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex3, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm3, ex1, ex4, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm0, dm4, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm2, dm0, ex6, ex11, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm3, dm2, ex6, ex0, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm4, ex9, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex9, ex3, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex8, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm0, dm3, ex2, ex4, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm1, dm1, ex10, ex11, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex10, ex0, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex6, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm0, ex6, ex7, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm0, dm1, ex5, ex8, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex5, ex4, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm4, dm2, ex9, ex2, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm2, dm3, ex9, ex11, r3;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm3, dm0, ex3, ex1, r3
	vmac.f dm0, dm1, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex10, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm4, dm2, ex10, ex8, r3;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm2, dm3, ex0, ex2, r3
	vmac.f dm3, dm0, ex0, ex11, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm1, ex5, ex9, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm1, dm4, ex5, ex1, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm2, ex4, ex6, r3;		mov p0, r4;		movx r24, #0
	vmac.f dm2, dm3, ex4, ex8, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm3, dm0, ex7, ex10, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm1, ex7, ex2, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm4, ex3, ex9, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm2, ex3, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex11, ex5, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex4, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm4, dm1, ex0, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm2, dm4, ex0, ex2, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm2, ex8, ex7, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex8, [p0, lf0, r24]
	vmac.f dm1, dm0, ex8, ex9, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm4, ex6, ex5, r3;		vst.conv.bf16.fp32 cml3, [p4], #64
	vmac.f dm0, dm2, ex6, ex4, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex1, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm1, ex11, ex10, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm3, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm4, dm4, ex3, ex9, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex8, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm2, ex8, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64
	vmac.f dm3, dm4, ex2, ex10, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm0, ex6, ex11, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm0, dm1, ex6, ex7, r3;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm1, dm2, ex4, ex0, r3
	vmac.f dm2, dm3, ex4, ex5, r3;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm3, dm4, ex9, ex8, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm4, dm0, ex9, ex1, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm1, ex3, ex11, r3
	vmac.f dm1, dm2, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm2, dm3, ex10, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm3, dm4, ex10, ex0, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm0, ex2, ex8, r3;		mov p0, r4;		movx r24, #0
	vmac.f dm0, dm1, ex2, ex1, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex5, ex9, r3;		vldb.pop.576 ex2, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm3, ex5, ex11, r3;		vldb.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm4, ex4, ex6, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm0, ex4, ex0, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm1, dm1, ex7, ex10, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex7, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm4, dm3, ex3, ex9, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm0, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm2, dm0, ex2, ex5, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex6, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm4, ex8, ex10, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex8, ex1, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex0, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm3, ex7, ex9, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex5, r3;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm4, dm4, ex4, ex6, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex3, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm3, dm0, ex2, ex10, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex11, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex11, ex9, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex9, [p0, lf0, r24]
	vmac.f dm4, dm2, ex8, ex7, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm3, ex8, ex5, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm3, dm0, ex1, ex3, r3
	vmac.f dm0, dm1, ex1, ex10, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex6, ex2, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm4, dm2, ex6, ex0, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm3, ex4, ex7, r3
	vmac.f dm3, dm0, ex4, ex5, r3;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm0, dm1, ex11, ex8, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm1, dm4, ex11, ex3, r3;		vldb.pop.576 ex2, [p0, lf0, r24];		vlda.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm2, ex9, ex2, r3;		mov p0, r4;		movx r24, #0
	vmac.f dm2, dm3, ex9, ex0, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm3, dm0, ex10, ex6, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm1, ex10, ex7, r3;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm4, ex1, ex8, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm2, ex1, ex3, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex5, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm1, dm0, ex5, ex0, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm4, dm1, ex4, ex6, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm2, dm4, ex4, ex7, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		vldb.pop.576 ex7, [p0, lf0, r24]
	vmac.f dm0, dm2, ex9, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24]
	vmac.f dm1, dm0, ex9, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm4, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64
	vmac.f dm0, dm2, ex2, ex0, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm0, ex5, ex1, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm1, ex5, ex6, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm3, dm3, ex3, ex10, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm4, dm4, ex3, ex8, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex9, ex4, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm2, ex9, ex11, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm2, dm3, ex7, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64
	vmac.f dm3, dm4, ex7, ex6, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm4, dm0, ex2, ex5, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm0, dm1, ex2, ex10, r3;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm1, dm2, ex0, ex4, r3
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex8, ex9, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm4, dm0, ex8, ex1, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm0, dm1, ex3, ex5, r3
	vmac.f dm1, dm2, ex3, ex10, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm2, dm3, ex7, ex2, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm4, ex7, ex4, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm0, ex6, ex9, r3;		mov p0, r4;		movx r24, #0
	vmac.f dm0, dm1, ex6, ex1, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex11, ex8, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm3, ex11, ex5, r3;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm3, dm4, ex0, ex2, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm0, ex0, ex4, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm1, dm1, ex10, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm3, dm2, ex10, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm4, dm3, ex3, ex8, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm0, dm4, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm2, dm0, ex6, ex11, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm3, dm2, ex6, ex2, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm4, ex9, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex9, ex1, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm2, ex10, ex0, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm3, ex10, ex8, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm1, dm1, ex4, ex11, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm4, ex4, ex2, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex6, ex3, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm0, ex6, ex7, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex5, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex5, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm4, dm2, ex9, ex10, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm2, dm3, ex9, ex11, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm3, dm0, ex1, ex3, r3
	vmac.f dm0, dm1, ex1, ex7, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex4, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm2, ex4, ex0, r3;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex10, r3
	vmac.f dm3, dm0, ex2, ex11, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex8, ex9, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm4, ex8, ex3, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm2, ex5, ex6, r3;		mov p0, r4;		movx r24, #0
	vmac.f dm2, dm3, ex5, ex0, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm3, dm0, ex7, ex4, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm1, ex7, ex10, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm4, ex1, ex9, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm2, ex1, ex3, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex11, ex8, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex0, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm1, ex2, ex4, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm2, dm4, ex2, ex10, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm0, dm2, ex5, ex7, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm1, dm0, ex5, ex9, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm4, ex6, ex8, r3;		vst.conv.bf16.fp32 cml3, [p4], #64
	vmac.f dm0, dm2, ex6, ex0, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex1, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm1, ex11, ex4, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm3, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm4, dm4, ex3, ex9, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex5, ex2, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm1, dm2, ex5, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm2, dm3, ex10, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64
	vmac.f dm3, dm4, ex10, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm4, dm0, ex6, ex11, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm0, dm1, ex6, ex7, r3;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm1, dm2, ex0, ex2, r3
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex9, ex5, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm4, dm0, ex9, ex1, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm1, ex3, ex11, r3
	vmac.f dm1, dm2, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm2, dm3, ex10, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm3, dm4, ex10, ex2, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm0, ex4, ex5, r3;		mov p0, r4;		movx r24, #0
	vmac.f dm0, dm1, ex4, ex1, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex8, ex9, r3;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm3, ex8, ex11, r3;		vldb.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm4, ex0, ex6, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm0, ex0, ex2, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm1, ex7, ex10, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex7, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm4, dm3, ex3, ex9, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm0, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm2, dm0, ex4, ex8, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm3, dm2, ex4, ex6, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm4, ex5, ex10, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex5, ex1, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex0, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm3, ex7, ex9, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm1, ex2, ex8, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm4, ex2, ex6, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex4, ex3, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm3, dm0, ex4, ex10, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex11, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex11, ex9, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex9, [p0, lf0, r24]
	vmac.f dm4, dm2, ex5, ex7, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm3, ex5, ex8, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm3, dm0, ex1, ex3, r3
	vmac.f dm0, dm1, ex1, ex10, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex6, ex4, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm4, dm2, ex6, ex0, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex7, r3
	vmac.f dm3, dm0, ex2, ex8, r3;		vldb.pop.576 ex2, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm0, dm1, ex11, ex5, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm1, dm4, ex11, ex3, r3;		vldb.pop.576 ex0, [p0, lf0, r24];		mov p1, r6;		movx r25, #0
	vmac.f dm4, dm2, ex9, ex4, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm2, dm3, ex9, ex0, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm3, dm0, ex10, ex6, r3;		mov p2, r18;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm1, ex10, ex7, r3;		mov p3, r19;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm1, dm4, ex1, ex5, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm2, ex1, ex3, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex11, ex8, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex4, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm4, dm1, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm2, dm4, ex2, ex7, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		vldb.pop.576 ex7, [p0, lf0, r24]
	vmac.f dm0, dm2, ex9, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24]
	vmac.f dm1, dm0, ex9, ex5, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm4, ex0, ex8, r3;		vst.conv.bf16.fp32 cml3, [p4], #64
	vmac.f dm0, dm2, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex1, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm1, ex11, ex6, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm3, dm3, ex3, ex10, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm4, dm4, ex3, ex5, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex9, ex2, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm2, ex9, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm2, dm3, ex7, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64
	vmac.f dm3, dm4, ex7, ex6, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm4, dm0, ex4, ex11, r3;		mov p4, r18;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm0, dm1, ex4, ex10, r3;		mov p5, r19;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm1, dm2, ex0, ex2, r3
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex5, ex9, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm4, dm0, ex5, ex1, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm1, ex3, ex11, r3
	vmac.f dm1, dm2, ex3, ex10, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm2, dm3, ex7, ex4, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm4, ex7, ex2, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm0, ex6, ex9, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm0, dm1, ex6, ex1, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex8, ex5, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm3, ex8, ex11, r3;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm3, dm4, ex0, ex4, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm0, ex0, ex2, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm1, ex10, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm3, dm2, ex10, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm4, dm3, ex3, ex5, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm0, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm2, dm0, ex6, ex8, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm3, dm2, ex6, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm4, ex9, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex9, ex1, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm2, ex10, ex0, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm3, ex10, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm1, ex2, ex8, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm4, ex2, ex4, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex6, ex3, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm0, ex6, ex7, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex11, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex11, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm4, dm2, ex9, ex10, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm2, dm3, ex9, ex8, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm3, dm0, ex1, ex3, r3
	vmac.f dm0, dm1, ex1, ex7, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex4, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm2, ex4, ex0, r3;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex10, r3
	vmac.f dm3, dm0, ex2, ex8, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex11, ex9, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm1, dm4, ex11, ex3, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm2, ex5, ex6, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm2, dm3, ex5, ex0, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm3, dm0, ex7, ex4, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm1, ex7, ex10, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm4, ex1, ex9, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm2, ex1, ex3, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex8, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex8, [p0, lf0, r24]
	vmac.f dm1, dm0, ex8, ex0, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm1, ex2, ex4, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm2, dm4, ex2, ex10, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm0, dm2, ex5, ex7, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm1, dm0, ex5, ex9, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm4, ex6, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64
	vmac.f dm0, dm2, ex6, ex0, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm0, ex8, ex1, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm1, ex8, ex4, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm3, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm4, dm4, ex3, ex9, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex5, ex2, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm1, dm2, ex5, ex11, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm2, dm3, ex10, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64
	vmac.f dm3, dm4, ex10, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm4, dm0, ex6, ex8, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm0, dm1, ex6, ex7, r3;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm1, dm2, ex0, ex2, r3
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex9, ex5, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm4, dm0, ex9, ex1, r3;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm0, dm1, ex3, ex8, r3
	vmac.f dm1, dm2, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm2, dm3, ex10, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm3, dm4, ex10, ex2, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm0, ex4, ex5, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm0, dm1, ex4, ex1, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex11, ex9, r3;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm3, ex11, ex8, r3;		vldb.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm4, ex0, ex6, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm0, ex0, ex2, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm1, ex7, ex10, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex7, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm4, dm3, ex3, ex9, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm0, dm4, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex8, [p0, lf0, r24]
	vmac.f dm2, dm0, ex4, ex11, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm3, dm2, ex4, ex6, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm4, ex5, ex10, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex5, ex1, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex0, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm3, ex7, ex9, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm1, dm1, ex2, ex11, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm4, ex2, ex6, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex4, ex3, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm3, dm0, ex4, ex10, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex8, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex8, ex9, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex8, [p0, lf0, r24]
	vmac.f dm4, dm2, ex5, ex7, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm3, ex5, ex11, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm3, dm0, ex1, ex3, r3
	vmac.f dm0, dm1, ex1, ex10, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex6, ex4, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm4, dm2, ex6, ex0, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex7, r3
	vmac.f dm3, dm0, ex2, ex11, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex9, ex5, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm4, ex9, ex3, r3;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm2, ex8, ex4, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm2, dm3, ex8, ex0, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm3, dm0, ex10, ex6, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm1, ex10, ex7, r3;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm1, dm4, ex1, ex5, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm2, ex1, ex3, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex11, ex9, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex0, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm4, dm1, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm2, dm4, ex2, ex7, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		vldb.pop.576 ex7, [p0, lf0, r24]
	vmac.f dm0, dm2, ex8, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex8, [p0, lf0, r24]
	vmac.f dm1, dm0, ex8, ex5, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm2, dm4, ex4, ex9, r3;		vst.conv.bf16.fp32 cml3, [p4], #64
	vmac.f dm0, dm2, ex4, ex0, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm0, ex11, ex1, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm1, ex11, ex6, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm3, dm3, ex3, ex10, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm4, dm4, ex3, ex5, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex8, ex2, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm2, ex8, ex9, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm2, dm3, ex7, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64
	vmac.f dm3, dm4, ex7, ex6, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm4, dm0, ex4, ex11, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm0, dm1, ex4, ex10, r3;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm1, dm2, ex0, ex2, r3
	vmac.f dm2, dm3, ex0, ex9, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex5, ex8, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm4, dm0, ex5, ex1, r3;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm0, dm1, ex3, ex11, r3
	vmac.f dm1, dm2, ex3, ex10, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm2, dm3, ex7, ex4, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm4, ex7, ex2, r3;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm0, ex6, ex8, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm0, dm1, ex6, ex1, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex9, ex5, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm2, dm3, ex9, ex11, r3;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm3, dm4, ex0, ex4, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm0, ex0, ex2, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm1, ex10, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm3, dm2, ex10, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm4, dm3, ex3, ex5, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm0, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex11, [p0, lf0, r24]
	vmac.f dm2, dm0, ex6, ex9, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex6, [p0, lf0, r24]
	vmac.f dm3, dm2, ex6, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm4, ex8, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex8, ex1, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm2, ex10, ex0, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm3, ex10, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm1, ex2, ex9, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm4, ex2, ex4, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex6, ex3, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm0, ex6, ex7, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex11, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex11, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm4, dm2, ex8, ex10, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm2, dm3, ex8, ex9, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm3, dm0, ex1, ex3, r3
	vmac.f dm0, dm1, ex1, ex7, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex4, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm2, ex4, ex0, r3;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex10, r3
	vmac.f dm3, dm0, ex2, ex9, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex11, ex8, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm1, dm4, ex11, ex3, r3;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm2, ex5, ex6, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm2, dm3, ex5, ex0, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm3, dm0, ex7, ex4, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm1, ex7, ex10, r3;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm1, dm4, ex1, ex8, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm2, ex1, ex3, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex9, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex9, [p0, lf0, r24]
	vmac.f dm1, dm0, ex9, ex0, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm1, ex2, ex4, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm2, dm4, ex2, ex10, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		vldb.pop.576 ex10, [p0, lf0, r24]
	vmac.f dm0, dm2, ex5, ex7, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex5, [p0, lf0, r24]
	vmac.f dm1, dm0, ex5, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm4, ex6, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64
	vmac.f dm0, dm2, ex6, ex0, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm0, ex9, ex1, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm2, dm1, ex9, ex4, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm3, dm3, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm4, dm4, ex3, ex8, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm0, dm1, ex5, ex2, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.pop.576 ex8, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm1, dm2, ex5, ex11, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm2, dm3, ex10, ex1, r3;		vst.conv.bf16.fp32 cml0, [p5], #64
	vmac.f dm3, dm4, ex10, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm4, dm0, ex6, ex9, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm0, dm1, ex6, ex7, r3;		vldb.pop.576 ex2, [p1, lf1, r25]
	vmac.f dm1, dm2, ex0, ex2, r3
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex8, ex5, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm4, dm0, ex8, ex1, r3;		vldb.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm0, dm1, ex3, ex9, r3
	vmac.f dm1, dm2, ex3, ex7, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm2, dm3, ex10, ex6, r3;		vldb.pop.576 ex7, [p0, lf0, r24];		vlda.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm3, dm4, ex10, ex2, r3;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex1, [p1, lf1, r25]
	vmac.f dm4, dm0, ex4, ex5, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm0, dm1, ex4, ex1, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm1, dm2, ex11, ex8, r3;		vldb.pop.576 ex4, [p0, lf0, r24];		vlda.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm2, dm3, ex11, ex9, r3;		vldb.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm3, dm4, ex0, ex6, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm4, dm0, ex0, ex2, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm1, ex7, ex10, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.pop.576 ex7, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex1, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.pop.576 ex8, [p1, lf1, r25]
	vmac.f dm4, dm3, ex3, ex8, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm0, dm4, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		vldb.pop.576 ex9, [p0, lf0, r24]
	vmac.f dm2, dm0, ex4, ex11, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64;		vldb.pop.576 ex4, [p0, lf0, r24]
	vmac.f dm3, dm2, ex4, ex6, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.pop.576 ex10, [p1, lf1, r25]
	vmac.f dm0, dm4, ex5, ex10, r3;		vst.conv.bf16.fp32 cml1, [p4], #64
	vmac.f dm2, dm0, ex5, ex1, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm2, ex7, ex0, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.pop.576 ex5, [p0, lf0, r24];		vlda.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm0, dm3, ex7, ex8, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex11, [p1, lf1, r25]
	vmac.f dm1, dm1, ex2, ex11, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm4, dm4, ex2, ex6, r3;		vldb.fill.512 [p0, lf0, r24];		vlda.fill.512 [p1, lf1, r25]
	vmac.f dm2, dm3, ex4, ex3, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.pop.576 ex6, [p0, lf0, r24];		vlda.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm3, dm0, ex4, ex10, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex0, [p1, lf1, r25]
	vmac.f dm0, dm1, ex9, ex0, r3;		vst.conv.bf16.fp32 cml2, [p5], #64
	vmac.f dm1, dm4, ex9, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex8, [p0, lf0, r24]
	vmac.f dm4, dm2, ex5, ex7, r3;		vldb.pop.576 ex9, [p0, lf0, r24];		vlda.pop.576 ex5, [p1, lf1, r25]
	vmac.f dm2, dm3, ex5, ex11, r3;		vldb.pop.576 ex3, [p1, lf1, r25]
	vmac.f dm3, dm0, ex1, ex3, r3
	vmac.f dm0, dm1, ex1, ex10, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex6, ex4, r3;		vldb.pop.576 ex10, [p0, lf0, r24];		vlda.pop.576 ex6, [p1, lf1, r25]
	vmac.f dm4, dm2, ex6, ex0, r3;		vldb.pop.576 ex7, [p1, lf1, r25]
	vmac.f dm2, dm3, ex2, ex7, r3
	vmac.f dm3, dm0, ex2, ex11, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex9, ex5, r3;		vldb.pop.576 ex11, [p0, lf0, r24];		vlda.pop.576 ex9, [p1, lf1, r25]
	vmac.f dm1, dm4, ex9, ex3, r3;		vldb.pop.576 ex4, [p1, lf1, r25]
	vmac.f dm4, dm2, ex8, ex4, r3
	vmac.f dm2, dm3, ex8, ex0, r3
	vmac.f dm3, dm0, ex10, ex6, r3
	vmac.f dm0, dm1, ex10, ex7, r3
	vmac.f dm1, dm4, ex1, ex5, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex3, r3
	vmac.f dm2, dm3, ex11, ex9, r3
	vmac.f dm3, dm0, ex11, ex4, r3
	vmac.f dm0, dm1, ex2, ex6, r3
	vmac.f dm1, dm4, ex2, ex7, r3
	nop
	nop
	vmac.f dm4, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cml2, [p4], #64
	vmac.f dm2, dm1, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh2, [p4], #64
	vst.conv.bf16.fp32 cml3, [p4], #64
	vst.conv.bf16.fp32 cmh3, [p4], #64
	nop
	nop
	vst.conv.bf16.fp32 cml4, [p5], #64
	vst.conv.bf16.fp32 cmh4, [p5], #64
	vst.conv.bf16.fp32 cml2, [p5], #64
	vst.conv.bf16.fp32 cmh2, [p5], #64
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lhs_end:
	.size	hs_call, .Lhs_end-hs_call
)hs");
