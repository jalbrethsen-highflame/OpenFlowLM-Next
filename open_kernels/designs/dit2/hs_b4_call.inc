asm(R"hs(
	.text
	.globl	hs_b4_call
	.p2align	4
	.type	hs_b4_call,@function
hs_b4_call:
	paddxm [sp], #64
	st p6, [sp, #-64]
	mov p6, p3
	mov r31, p3
	mov crunpacksize, #0
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
	lda r21, [p6], #4
	nop
	lda r22, [p6], #4
	nop
	lda r23, [p6], #4
	nop
	lda r26, [p6], #4;		vldb.fill.512 [p0, lf0, r24]
	mov el6, r21;		vlda.pop.576 ex0, [p0, lf0, r24]
	vldb.unpack x6, unpacksign1, [p1], #32;		mov el7, r21;		vlda.pop.576 ex1, [p0, lf0, r24]
	vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex2, [p0, lf0, r24]
	vlda.conv.fp32.bf16 cml1, [p2], #64;		mov el8, r23
	vlda.conv.fp32.bf16 cmh1, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26;		vlda.pop.576 ex3, [p0, lf0, r24]
	mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex0, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm1, ex0, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vlda.conv.fp32.bf16 cml0, [p3], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vlda.conv.fp32.bf16 cmh0, [p3], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex2, ex7, r3;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r4;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		lda r21, [p6], #4
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex5, r3;		lda r22, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		lda r23, [p6], #4;		mov p0, r4;		movx r24, #0
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex4, r3;		lda r26, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex10, r3;		mov el6, r21;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex4, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		mov el7, r21;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vmac.f dm4, dm2, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex1, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		mov el8, r23
	vmac.f dm1, dm0, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vmac.f dm4, dm1, ex2, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26
	vmac.f dm2, dm4, ex2, ex4, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm2, ex3, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm1, dm0, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm4, ex0, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm0, dm2, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm0, ex2, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm1, ex2, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm3, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r4;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		lda r21, [p6], #4
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex5, r3;		lda r22, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		lda r23, [p6], #4;		mov p0, r4;		movx r24, #0
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex4, r3;		lda r26, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex10, r3;		mov el6, r21;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex4, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		mov el7, r21;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vmac.f dm4, dm2, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex1, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		mov el8, r23
	vmac.f dm1, dm0, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vmac.f dm4, dm1, ex2, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26
	vmac.f dm2, dm4, ex2, ex4, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm2, ex3, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm1, dm0, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm4, ex0, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm0, dm2, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm0, ex2, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm1, ex2, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm3, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r4;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		lda r21, [p6], #4
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex5, r3;		lda r22, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		lda r23, [p6], #4;		mov p0, r4;		movx r24, #0
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex4, r3;		lda r26, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex10, r3;		mov el6, r21;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex4, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		mov el7, r21;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vmac.f dm4, dm2, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex1, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		mov el8, r23
	vmac.f dm1, dm0, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vmac.f dm4, dm1, ex2, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26
	vmac.f dm2, dm4, ex2, ex4, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm2, ex3, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm1, dm0, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm4, ex0, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm0, dm2, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm0, ex2, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm1, ex2, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm3, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r4;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		mov p6, r31
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		lda r21, [p6], #4
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex5, r3;		lda r22, [p6], #4;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		mov p0, r5;		movx r24, #0
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		lda r23, [p6], #4;		mov el6, r21
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		mov el7, r21;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex4, r3;		mov p1, r6;		lda r26, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex10, r3;		mov p2, r18;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex4, r3;		mov p3, r19;		vldb.unpack x6, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vmac.f dm4, dm2, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex1, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		mov el8, r23
	vmac.f dm1, dm0, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vmac.f dm4, dm1, ex2, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26
	vmac.f dm2, dm4, ex2, ex4, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm2, ex3, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm1, dm0, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm4, ex0, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm0, dm2, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm0, ex2, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm1, ex2, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm3, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		mov p4, r18;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		mov p5, r19;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r5;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		lda r21, [p6], #4
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex5, r3;		lda r22, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		lda r23, [p6], #4;		mov p0, r5;		movx r24, #0
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex4, r3;		lda r26, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex10, r3;		mov el6, r21;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex4, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		mov el7, r21;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vmac.f dm4, dm2, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex1, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		mov el8, r23
	vmac.f dm1, dm0, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vmac.f dm4, dm1, ex2, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26
	vmac.f dm2, dm4, ex2, ex4, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm2, ex3, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm1, dm0, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm4, ex0, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm0, dm2, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm0, ex2, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm1, ex2, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm3, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r5;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		lda r21, [p6], #4
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex5, r3;		lda r22, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		lda r23, [p6], #4;		mov p0, r5;		movx r24, #0
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex4, r3;		lda r26, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex10, r3;		mov el6, r21;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex4, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		mov el7, r21;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vmac.f dm4, dm2, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex1, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		mov el8, r23
	vmac.f dm1, dm0, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vmac.f dm4, dm1, ex2, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26
	vmac.f dm2, dm4, ex2, ex4, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm2, ex3, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm1, dm0, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm4, ex0, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm0, dm2, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm0, ex2, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm1, ex2, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm3, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r5;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		lda r21, [p6], #4
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex5, r3;		lda r22, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32;		lda r23, [p6], #4;		mov p0, r5;		movx r24, #0
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex4, r3;		lda r26, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex10, r3;		mov el6, r21;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex4, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		mov el7, r21;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex11, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		vldb.unpack x8, unpacksign1, [p1], #32;		mov eh6, r22
	vmac.f dm4, dm2, ex3, ex5, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		mov eh7, r22;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm3, ex1, ex11, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		mov el8, r23
	vmac.f dm1, dm0, ex1, ex5, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		vldb.unpack x7, unpacksign1, [p1], #32;		mov el9, r23
	vmac.f dm4, dm1, ex2, ex10, r3;		vlda.conv.fp32.bf16 cml3, [p3], #64;		vldb.unpack x9, unpacksign1, [p1], #32;		mov eh8, r26
	vmac.f dm2, dm4, ex2, ex4, r3;		vlda.conv.fp32.bf16 cmh3, [p3], #64;		mov eh9, r26;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm0, dm2, ex3, ex6, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm1, dm0, ex3, ex8, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm4, ex0, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm0, dm2, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm0, ex2, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm1, ex2, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm3, dm3, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex8, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex0, ex6, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex8, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.unpack x6, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex3, ex7, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vldb.unpack x8, unpacksign1, [p1], #32
	vmac.f dm3, dm4, ex3, ex9, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm0, ex1, ex7, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32
	vmac.f dm1, dm2, ex2, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex2, ex8, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex6, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm0, ex3, ex8, r3;		vldb.unpack x6, unpacksign1, [p1], #32;		lda r27, [p6], #4
	vmac.f dm0, dm1, ex0, ex7, r3;		vldb.unpack x8, unpacksign1, [p1], #32;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex9, r3;		lda r28, [p6], #4;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex2, ex7, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm3, dm4, ex2, ex9, r3;		vldb.unpack x7, unpacksign1, [p1], #32;		lda r29, [p6], #4;		mov p0, r5;		movx r24, #0
	vmac.f dm4, dm0, ex1, ex6, r3;		vldb.unpack x9, unpacksign1, [p1], #32;		vlda.fill.512 [p0, lf0, r24]
	vmac.f dm0, dm1, ex1, ex8, r3;		lda r30, [p6], #4;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm2, ex0, ex6, r3;		mov el10, r27;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex8, r3;		vldb.unpack x10, unpacksign1, [p1], #32;		mov el11, r27;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm4, ex3, ex7, r3;		vlda.conv.fp32.bf16 cml0, [p2], #64;		vldb.unpack x4, unpacksign1, [p1], #32;		mov eh10, r28
	vmac.f dm4, dm0, ex3, ex9, r3;		vlda.conv.fp32.bf16 cmh0, [p2], #64;		mov eh11, r28;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm1, ex1, ex7, r3;		vlda.conv.fp32.bf16 cml2, [p2], #64;		mov el4, r29
	vmac.f dm3, dm2, ex1, ex9, r3;		vlda.conv.fp32.bf16 cmh2, [p2], #64;		vldb.unpack x11, unpacksign1, [p1], #32;		mov el5, r29
	vmac.f dm4, dm3, ex2, ex6, r3;		vlda.conv.fp32.bf16 cml1, [p3], #64;		vldb.unpack x5, unpacksign1, [p1], #32;		mov eh4, r30
	vmac.f dm0, dm4, ex2, ex8, r3;		vlda.conv.fp32.bf16 cmh1, [p3], #64;		mov eh5, r30;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm2, dm0, ex3, ex10, r3;		vlda.conv.fp32.bf16 cml4, [p3], #64
	vmac.f dm3, dm2, ex3, ex4, r3;		vlda.conv.fp32.bf16 cmh4, [p3], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm4, ex0, ex7, r3;		vst.conv.bf16.fp32 cml1, [p4], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm2, dm0, ex0, ex9, r3;		vst.conv.bf16.fp32 cmh1, [p4], #64;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm3, dm2, ex2, ex11, r3;		vst.conv.bf16.fp32 cml3, [p4], #64;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm0, dm3, ex2, ex5, r3;		vst.conv.bf16.fp32 cmh3, [p4], #64;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm1, dm1, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm4, ex1, ex4, r3;		vldb.fill.512 [p0, lf0, r24]
	vmac.f dm2, dm3, ex0, ex10, r3;		vst.conv.bf16.fp32 cml0, [p5], #64;		vlda.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm3, dm0, ex0, ex4, r3;		vst.conv.bf16.fp32 cmh0, [p5], #64;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex3, ex11, r3;		vst.conv.bf16.fp32 cml2, [p5], #64;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm1, dm4, ex3, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p5], #64;		vldb.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm4, dm2, ex1, ex11, r3;		vlda.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm2, dm3, ex1, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex2, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm0, dm1, ex2, ex4, r3;		vldb.pop.576 ex3, [p0, lf0, r24]
	vmac.f dm1, dm4, ex3, ex10, r3;		vlda.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex4, r3;		vldb.unpack x10, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex0, ex11, r3;		vldb.unpack x4, unpacksign1, [p1], #32
	vmac.f dm3, dm0, ex0, ex5, r3;		vldb.pop.576 ex2, [p0, lf0, r24]
	vmac.f dm0, dm1, ex2, ex11, r3;		vlda.pop.576 ex1, [p0, lf0, r24]
	vmac.f dm1, dm4, ex2, ex5, r3;		vldb.unpack x11, unpacksign1, [p1], #32
	vmac.f dm4, dm2, ex1, ex10, r3;		vldb.unpack x5, unpacksign1, [p1], #32
	vmac.f dm2, dm3, ex1, ex4, r3
	vmac.f dm3, dm0, ex0, ex10, r3
	vmac.f dm0, dm1, ex0, ex4, r3
	vmac.f dm1, dm4, ex3, ex11, r3;		vldb.pop.576 ex0, [p0, lf0, r24]
	vmac.f dm4, dm2, ex3, ex5, r3
	vmac.f dm2, dm3, ex1, ex11, r3
	vmac.f dm3, dm0, ex1, ex5, r3
	vmac.f dm0, dm1, ex2, ex10, r3
	vmac.f dm1, dm4, ex2, ex4, r3
	nop
	nop
	vmac.f dm4, dm0, ex0, ex11, r3;		vst.conv.bf16.fp32 cml2, [p4], #64
	vmac.f dm2, dm1, ex0, ex5, r3;		vst.conv.bf16.fp32 cmh2, [p4], #64
	vst.conv.bf16.fp32 cml3, [p4], #64
	vst.conv.bf16.fp32 cmh3, [p4], #64
	nop
	nop
	vst.conv.bf16.fp32 cml4, [p5], #64
	vst.conv.bf16.fp32 cmh4, [p5], #64
	vst.conv.bf16.fp32 cml2, [p5], #64
	vst.conv.bf16.fp32 cmh2, [p5], #64
	lda p6, [sp, #-64]
	nop
	nop
	ret lr
	nop
	nop
	nop
	paddxm [sp], #-64
	nop
.Lhs_b4_call_end:
	.size	hs_b4_call, .Lhs_b4_call_end-hs_b4_call
)hs");
