asm(R"hs(
	.text
	.globl	b4p_unp4
	.p2align	4
	.type	b4p_unp4,@function
b4p_unp4:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp4_end:
	.size	b4p_unp4, .Lb4p_unp4_end-b4p_unp4
	.text
	.globl	b4p_unp5
	.p2align	4
	.type	b4p_unp5,@function
b4p_unp5:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp5_end:
	.size	b4p_unp5, .Lb4p_unp5_end-b4p_unp5
	.text
	.globl	b4p_unp6
	.p2align	4
	.type	b4p_unp6,@function
b4p_unp6:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp6_end:
	.size	b4p_unp6, .Lb4p_unp6_end-b4p_unp6
	.text
	.globl	b4p_unp7
	.p2align	4
	.type	b4p_unp7,@function
b4p_unp7:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp7_end:
	.size	b4p_unp7, .Lb4p_unp7_end-b4p_unp7
	.text
	.globl	b4p_unp8
	.p2align	4
	.type	b4p_unp8,@function
b4p_unp8:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp8_end:
	.size	b4p_unp8, .Lb4p_unp8_end-b4p_unp8
	.text
	.globl	b4p_unp9
	.p2align	4
	.type	b4p_unp9,@function
b4p_unp9:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp9_end:
	.size	b4p_unp9, .Lb4p_unp9_end-b4p_unp9
	.text
	.globl	b4p_unp10
	.p2align	4
	.type	b4p_unp10,@function
b4p_unp10:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp10_end:
	.size	b4p_unp10, .Lb4p_unp10_end-b4p_unp10
	.text
	.globl	b4p_unp11
	.p2align	4
	.type	b4p_unp11,@function
b4p_unp11:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp11_end:
	.size	b4p_unp11, .Lb4p_unp11_end-b4p_unp11
	.text
	.globl	b4p_unp12
	.p2align	4
	.type	b4p_unp12,@function
b4p_unp12:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_unp12_end:
	.size	b4p_unp12, .Lb4p_unp12_end-b4p_unp12
	.text
	.globl	b4p_eclob
	.p2align	4
	.type	b4p_eclob,@function
b4p_eclob:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #32]
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
	mov el6, r18
	mov eh6, r19
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_eclob_end:
	.size	b4p_eclob, .Lb4p_eclob_end-b4p_eclob
	.text
	.globl	b4p_mov0
	.p2align	4
	.type	b4p_mov0,@function
b4p_mov0:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	mov el6, r18
	mov eh6, r19
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
	mov el6, r20
	mov eh6, r21;		vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_mov0_end:
	.size	b4p_mov0, .Lb4p_mov0_end-b4p_mov0
	.text
	.globl	b4p_mov1
	.p2align	4
	.type	b4p_mov1,@function
b4p_mov1:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	mov el6, r18
	mov eh6, r19
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
	mov el6, r20
	mov eh6, r21
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_mov1_end:
	.size	b4p_mov1, .Lb4p_mov1_end-b4p_mov1
	.text
	.globl	b4p_mov2
	.p2align	4
	.type	b4p_mov2,@function
b4p_mov2:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	mov el6, r18
	mov eh6, r19
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
	mov el6, r20
	mov eh6, r21
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_mov2_end:
	.size	b4p_mov2, .Lb4p_mov2_end-b4p_mov2
	.text
	.globl	b4p_mov3
	.p2align	4
	.type	b4p_mov3,@function
b4p_mov3:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	mov el6, r18
	mov eh6, r19
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
	mov el6, r20
	mov eh6, r21
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_mov3_end:
	.size	b4p_mov3, .Lb4p_mov3_end-b4p_mov3
	.text
	.globl	b4p_mov4
	.p2align	4
	.type	b4p_mov4,@function
b4p_mov4:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	mov el6, r18
	mov eh6, r19
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
	mov el6, r20
	mov eh6, r21
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_mov4_end:
	.size	b4p_mov4, .Lb4p_mov4_end-b4p_mov4
	.text
	.globl	b4p_mov5
	.p2align	4
	.type	b4p_mov5,@function
b4p_mov5:
	mova r3, #780
	mov crunpacksize, #0
	lda r16, [p1, #0]
	lda r17, [p1, #4]
	lda r18, [p1, #8]
	lda r19, [p1, #12]
	lda r20, [p1, #16]
	lda r21, [p1, #20]
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	vldb.unpack x0, unpacksign1, [p0, #0]
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
	mov el0, r16
	mov eh0, r17
	vlda.conv.fp32.bf16 cml0, [p2, #0]
	vlda.conv.fp32.bf16 cmh0, [p2, #64]
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
	vldb.unpack x6, unpacksign1, [p0, #64]
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
	mov el6, r18
	mov eh6, r19
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
	mov el6, r20
	mov eh6, r21
	nop
	nop
	nop
	nop
	vmac.f dm0, dm0, ex0, ex6, r3
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
	vst.conv.bf16.fp32 cml0, [p2, #0]
	vst.conv.bf16.fp32 cmh0, [p2, #64]
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
.Lb4p_mov5_end:
	.size	b4p_mov5, .Lb4p_mov5_end-b4p_mov5
)hs");
extern "C" void b4p_unp4(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp5(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp6(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp7(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp8(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp9(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp10(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp11(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_unp12(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_eclob(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_mov0(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_mov1(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_mov2(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_mov3(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_mov4(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
extern "C" void b4p_mov5(const uint8_t *nib, const int32_t *exp, bfloat16 *c);
static void (*const b4p_cases[])(const uint8_t *, const int32_t *, bfloat16 *) = {b4p_unp4, b4p_unp5, b4p_unp6, b4p_unp7, b4p_unp8, b4p_unp9, b4p_unp10, b4p_unp11, b4p_unp12, b4p_eclob, b4p_mov0, b4p_mov1, b4p_mov2, b4p_mov3, b4p_mov4, b4p_mov5};
static const unsigned b4p_n = 16;
