asm(R"hs(
	.text
	.globl	b4q_none
	.p2align	4
	.type	b4q_none,@function
b4q_none:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_none_end:
	.size	b4q_none, .Lb4q_none_end-b4q_none
	.text
	.globl	b4q_p1u1
	.p2align	4
	.type	b4q_p1u1,@function
b4q_p1u1:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	vldb.pop.576 ex1, [p0, lf0, r24]
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p1u1_end:
	.size	b4q_p1u1, .Lb4q_p1u1_end-b4q_p1u1
	.text
	.globl	b4q_p1u2
	.p2align	4
	.type	b4q_p1u2,@function
b4q_p1u2:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p1u2_end:
	.size	b4q_p1u2, .Lb4q_p1u2_end-b4q_p1u2
	.text
	.globl	b4q_p1u3
	.p2align	4
	.type	b4q_p1u3,@function
b4q_p1u3:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p1u3_end:
	.size	b4q_p1u3, .Lb4q_p1u3_end-b4q_p1u3
	.text
	.globl	b4q_p2u1
	.p2align	4
	.type	b4q_p2u1,@function
b4q_p2u1:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p2u1_end:
	.size	b4q_p2u1, .Lb4q_p2u1_end-b4q_p2u1
	.text
	.globl	b4q_p2u2
	.p2align	4
	.type	b4q_p2u2,@function
b4q_p2u2:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p2u2_end:
	.size	b4q_p2u2, .Lb4q_p2u2_end-b4q_p2u2
	.text
	.globl	b4q_p2u3
	.p2align	4
	.type	b4q_p2u3,@function
b4q_p2u3:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p2u3_end:
	.size	b4q_p2u3, .Lb4q_p2u3_end-b4q_p2u3
	.text
	.globl	b4q_p3u1
	.p2align	4
	.type	b4q_p3u1,@function
b4q_p3u1:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p3u1_end:
	.size	b4q_p3u1, .Lb4q_p3u1_end-b4q_p3u1
	.text
	.globl	b4q_p3u2
	.p2align	4
	.type	b4q_p3u2,@function
b4q_p3u2:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p3u2_end:
	.size	b4q_p3u2, .Lb4q_p3u2_end-b4q_p3u2
	.text
	.globl	b4q_p3u3
	.p2align	4
	.type	b4q_p3u3,@function
b4q_p3u3:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p3u3_end:
	.size	b4q_p3u3, .Lb4q_p3u3_end-b4q_p3u3
	.text
	.globl	b4q_p4u1
	.p2align	4
	.type	b4q_p4u1,@function
b4q_p4u1:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	nop
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p4u1_end:
	.size	b4q_p4u1, .Lb4q_p4u1_end-b4q_p4u1
	.text
	.globl	b4q_p4u2
	.p2align	4
	.type	b4q_p4u2,@function
b4q_p4u2:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	nop
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p4u2_end:
	.size	b4q_p4u2, .Lb4q_p4u2_end-b4q_p4u2
	.text
	.globl	b4q_p4u3
	.p2align	4
	.type	b4q_p4u3,@function
b4q_p4u3:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p2, #0]
	lda r17, [p2, #4]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x6, unpacksign1, [p1, #0]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	mov el6, r16
	mov eh6, r16
	nop
	nop
	nop
	nop
	vlda.conv.fp32.bf16 cml0, [p3, #0]
	vlda.conv.fp32.bf16 cmh0, [p3, #64]
	vlda.conv.fp32.bf16 cml1, [p3, #128]
	vlda.conv.fp32.bf16 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	movx r24, #0
	nop
	vldb.fill.512 [p0, lf0, r24]
	vldb.pop.576 ex0, [p0, lf0, r24]
	nop
	nop
	nop
	vldb.pop.576 ex1, [p0, lf0, r24]
	nop
	nop
	vldb.unpack x8, unpacksign1, [p1, #32]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
	vmac.f dm1, dm1, ex1, ex6, r3
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vst.conv.bf16.fp32 cml0, [p3, #0]
	vst.conv.bf16.fp32 cmh0, [p3, #64]
	vst.conv.bf16.fp32 cml1, [p3, #128]
	vst.conv.bf16.fp32 cmh1, [p3, #192]
	nop
	nop
	nop
	nop
	ret lr
	nop
	nop
	nop
	nop
	nop
.Lb4q_p4u3_end:
	.size	b4q_p4u3, .Lb4q_p4u3_end-b4q_p4u3
)hs");
extern "C" void b4q_none(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p1u1(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p1u2(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p1u3(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p2u1(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p2u2(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p2u3(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p3u1(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p3u2(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p3u3(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p4u1(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p4u2(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4q_p4u3(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);
static void (*const b4q_cases[])(const bfp16ebs8 *, const uint8_t *, const int32_t *, bfloat16 *) = {b4q_none, b4q_p1u1, b4q_p1u2, b4q_p1u3, b4q_p2u1, b4q_p2u2, b4q_p2u3, b4q_p3u1, b4q_p3u2, b4q_p3u3, b4q_p4u1, b4q_p4u2, b4q_p4u3};
static const unsigned b4q_n = 13;
static const char *const b4q_names[] = {"none", "p1u1", "p1u2", "p1u3", "p2u1", "p2u2", "p2u3", "p3u1", "p3u2", "p3u3", "p4u1", "p4u2", "p4u3"};
