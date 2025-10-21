	.file	"MM2.c"
	.text
.Ltext0:
	.globl	drand48
	.type	drand48, @function
drand48:
.LFB27:
	.file 1 "MM2.c"
	.loc 1 48 0
	.cfi_startproc
	.loc 1 50 0
	movl	SEED(%rip), %edx
	movl	%edx, %eax
	andl	$1, %eax
	imull	$1588635695, %eax, %eax
	shrl	%edx
	imull	$1117695901, %edx, %edx
	subl	%edx, %eax
	movl	%eax, SEED(%rip)
	.loc 1 51 0
	movl	%eax, %eax
	cvtsi2sdq	%rax, %xmm0
	divsd	.LC0(%rip), %xmm0
	.loc 1 52 0
	ret
	.cfi_endproc
.LFE27:
	.size	drand48, .-drand48
	.globl	srand48
	.type	srand48, @function
srand48:
.LFB28:
	.loc 1 55 0
	.cfi_startproc
.LVL0:
	.loc 1 55 0
	testl	%edi, %edi
	je	.L4
	.loc 1 55 0 is_stmt 0 discriminator 1
	movl	%edi, SEED(%rip)
.L4:
	rep ret
	.cfi_endproc
.LFE28:
	.size	srand48, .-srand48
	.globl	mm2
	.type	mm2, @function
mm2:
.LFB29:
	.loc 1 70 0 is_stmt 1
	.cfi_startproc
.LVL1:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdx, %rbx
.LVL2:
	.loc 1 76 0
	movl	$0, %edx
.LVL3:
	jmp	.L7
.LVL4:
.L10:
	.loc 1 76 0 is_stmt 0 discriminator 2
	movsd	(%r10,%r8), %xmm0
	mulsd	(%r9,%rax), %xmm0
	addsd	(%rcx,%rax), %xmm0
	movsd	%xmm0, (%rcx,%rax)
.LVL5:
	addq	$8, %rax
	.loc 1 75 0 is_stmt 1 discriminator 2
	cmpq	$16640, %rax
	jne	.L10
.LVL6:
	addq	$8, %r8
	addq	$16640, %r11
	.loc 1 74 0
	cmpq	$16640, %r8
	je	.L9
.L12:
.LVL7:
	.loc 1 76 0 discriminator 1
	movq	%r11, %r9
	movl	$0, %eax
	jmp	.L10
.LVL8:
.L9:
	addq	$16640, %rdx
	.loc 1 73 0
	cmpq	$34611200, %rdx
	je	.L6
.L7:
.LVL9:
	leaq	(%rbx,%rdx), %rcx
	leaq	(%rdi,%rdx), %r10
	movq	%rsi, %r11
	.loc 1 76 0 discriminator 1
	movl	$0, %r8d
	jmp	.L12
.LVL10:
.L6:
	.loc 1 80 0
	popq	%rbx
	.cfi_def_cfa_offset 8
.LVL11:
	ret
	.cfi_endproc
.LFE29:
	.size	mm2, .-mm2
	.globl	fillmatrix
	.type	fillmatrix, @function
fillmatrix:
.LFB30:
	.loc 1 84 0
	.cfi_startproc
.LVL12:
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdi, %r13
.LVL13:
	.loc 1 90 0
	movl	$2080, %r12d
	jmp	.L15
.LVL14:
.L18:
	.loc 1 90 0 is_stmt 0 discriminator 2
	movl	$0, %eax
	call	drand48
.LVL15:
	movsd	%xmm0, 0(%rbp,%rbx)
.LVL16:
	addq	$8, %rbx
	.loc 1 89 0 is_stmt 1 discriminator 2
	cmpq	$16640, %rbx
	jne	.L18
.LVL17:
	addq	$16640, %r13
	.loc 1 88 0
	subl	$1, %r12d
.LVL18:
	je	.L14
.LVL19:
.L15:
	.loc 1 90 0 discriminator 1
	movq	%r13, %rbp
	movl	$0, %ebx
	jmp	.L18
.LVL20:
.L14:
	.loc 1 93 0
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
.LVL21:
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE30:
	.size	fillmatrix, .-fillmatrix
	.globl	zeromatrix
	.type	zeromatrix, @function
zeromatrix:
.LFB31:
	.loc 1 97 0
	.cfi_startproc
.LVL22:
	.loc 1 102 0
	movl	$2080, %ecx
	xorpd	%xmm0, %xmm0
	jmp	.L21
.LVL23:
.L24:
	.loc 1 102 0 is_stmt 0 discriminator 2
	movsd	%xmm0, (%rdx,%rax)
.LVL24:
	addq	$8, %rax
	.loc 1 101 0 is_stmt 1 discriminator 2
	cmpq	$16640, %rax
	jne	.L24
.LVL25:
	addq	$16640, %rdi
	.loc 1 100 0
	subl	$1, %ecx
.LVL26:
	je	.L20
.LVL27:
.L21:
	.loc 1 102 0 discriminator 1
	movq	%rdi, %rdx
	movl	$0, %eax
	jmp	.L24
.LVL28:
.L20:
	rep ret
	.cfi_endproc
.LFE31:
	.size	zeromatrix, .-zeromatrix
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"%1.1f "
	.text
	.globl	printmatrix
	.type	printmatrix, @function
printmatrix:
.LFB32:
	.loc 1 109 0
	.cfi_startproc
.LVL29:
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	movq	%rdi, %r13
.LVL30:
	.loc 1 114 0
	movl	$2080, %r12d
	jmp	.L26
.LVL31:
.L29:
.LBB10:
.LBB11:
	.file 2 "/usr/include/x86_64-linux-gnu/bits/stdio2.h"
	.loc 2 104 0 discriminator 2
	movsd	0(%rbp,%rbx), %xmm0
	movl	$.LC2, %esi
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk
.LVL32:
	addq	$8, %rbx
.LBE11:
.LBE10:
	.loc 1 113 0 discriminator 2
	cmpq	$16640, %rbx
	jne	.L29
.LVL33:
.LBB12:
.LBB13:
	.loc 2 104 0
	movl	$10, %edi
	call	putchar
.LVL34:
	addq	$16640, %r13
.LBE13:
.LBE12:
	.loc 1 112 0
	subl	$1, %r12d
.LVL35:
	je	.L25
.LVL36:
.L26:
	.loc 1 114 0 discriminator 1
	movq	%r13, %rbp
	movl	$0, %ebx
	jmp	.L29
.LVL37:
.L25:
	.loc 1 118 0
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
.LVL38:
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE32:
	.size	printmatrix, .-printmatrix
	.section	.rodata.str1.1
.LC3:
	.string	"Initialisation: %ld\n"
.LC6:
	.string	"mm2: %f s, %f MFLOPS\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB33:
	.loc 1 121 0
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	.loc 1 126 0
	call	clock
.LVL39:
	movq	%rax, %rbx
.LVL40:
.LBB20:
.LBB21:
	.loc 1 55 0
	movl	$8897, SEED(%rip)
.LBE21:
.LBE20:
	.loc 1 129 0
	movl	$A, %edi
	movl	$0, %eax
.LVL41:
	call	fillmatrix
.LVL42:
	.loc 1 130 0
	movl	$B, %edi
	movl	$0, %eax
	call	fillmatrix
.LVL43:
	.loc 1 132 0
	call	clock
.LVL44:
	subq	%rbx, %rax
	movq	%rax, %rdx
.LBB22:
.LBB23:
	.loc 2 104 0
	movl	$.LC3, %esi
	movl	$1, %edi
	movl	$0, %eax
	call	__printf_chk
.LVL45:
.LBE23:
.LBE22:
	.loc 1 133 0
	call	clock
.LVL46:
	.loc 1 143 0
	movl	$C, %edi
	movl	$0, %eax
	call	zeromatrix
.LVL47:
	.loc 1 144 0
	call	clock
.LVL48:
	movq	%rax, %rbx
.LVL49:
	.loc 1 145 0
	movl	$C, %edx
	movl	$B, %esi
	movl	$A, %edi
	movl	$0, %eax
.LVL50:
	call	mm2
.LVL51:
	.loc 1 146 0
	call	clock
.LVL52:
	subq	%rbx, %rax
	cvtsi2sdq	%rax, %xmm0
	movsd	.LC4(%rip), %xmm2
	divsd	%xmm2, %xmm0
.LVL53:
	.loc 1 147 0
	movsd	.LC5(%rip), %xmm1
	divsd	%xmm0, %xmm1
.LVL54:
.LBB24:
.LBB25:
	.loc 2 104 0
	divsd	%xmm2, %xmm1
.LVL55:
	movl	$.LC6, %esi
	movl	$1, %edi
	movl	$2, %eax
	call	__printf_chk
.LVL56:
.LBE25:
.LBE24:
	.loc 1 155 0
	movl	$0, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
.LVL57:
	ret
	.cfi_endproc
.LFE33:
	.size	main, .-main
	.data
	.align 4
	.type	SEED, @object
	.size	SEED, 4
SEED:
	.long	93186752
	.comm	C,34611200,32
	.comm	B,34611200,32
	.comm	A,34611200,32
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	4284481536
	.long	1106247679
	.align 8
.LC4:
	.long	0
	.long	1093567616
	.align 8
.LC5:
	.long	0
	.long	1108394756
	.text
.Letext0:
	.file 3 "/usr/lib/gcc/x86_64-linux-gnu/4.8/include/stddef.h"
	.file 4 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file 5 "/usr/include/libio.h"
	.file 6 "/usr/include/time.h"
	.file 7 "/usr/include/stdio.h"
	.file 8 "<built-in>"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x7f6
	.value	0x4
	.long	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.long	.LASF67
	.byte	0x1
	.long	.LASF68
	.long	.LASF69
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.long	.Ldebug_line0
	.uleb128 0x2
	.byte	0x8
	.byte	0x4
	.long	.LASF0
	.uleb128 0x3
	.long	.LASF8
	.byte	0x3
	.byte	0xd4
	.long	0x3f
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF1
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.long	.LASF2
	.uleb128 0x2
	.byte	0x2
	.byte	0x7
	.long	.LASF3
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.long	.LASF4
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF5
	.uleb128 0x2
	.byte	0x2
	.byte	0x5
	.long	.LASF6
	.uleb128 0x4
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.long	.LASF7
	.uleb128 0x3
	.long	.LASF9
	.byte	0x4
	.byte	0x83
	.long	0x70
	.uleb128 0x3
	.long	.LASF10
	.byte	0x4
	.byte	0x84
	.long	0x70
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.long	.LASF11
	.uleb128 0x3
	.long	.LASF12
	.byte	0x4
	.byte	0x87
	.long	0x70
	.uleb128 0x5
	.byte	0x8
	.uleb128 0x6
	.byte	0x8
	.long	0xa7
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.long	.LASF13
	.uleb128 0x7
	.long	.LASF43
	.byte	0xd8
	.byte	0x5
	.byte	0xf5
	.long	0x22e
	.uleb128 0x8
	.long	.LASF14
	.byte	0x5
	.byte	0xf6
	.long	0x69
	.byte	0
	.uleb128 0x8
	.long	.LASF15
	.byte	0x5
	.byte	0xfb
	.long	0xa1
	.byte	0x8
	.uleb128 0x8
	.long	.LASF16
	.byte	0x5
	.byte	0xfc
	.long	0xa1
	.byte	0x10
	.uleb128 0x8
	.long	.LASF17
	.byte	0x5
	.byte	0xfd
	.long	0xa1
	.byte	0x18
	.uleb128 0x8
	.long	.LASF18
	.byte	0x5
	.byte	0xfe
	.long	0xa1
	.byte	0x20
	.uleb128 0x8
	.long	.LASF19
	.byte	0x5
	.byte	0xff
	.long	0xa1
	.byte	0x28
	.uleb128 0x9
	.long	.LASF20
	.byte	0x5
	.value	0x100
	.long	0xa1
	.byte	0x30
	.uleb128 0x9
	.long	.LASF21
	.byte	0x5
	.value	0x101
	.long	0xa1
	.byte	0x38
	.uleb128 0x9
	.long	.LASF22
	.byte	0x5
	.value	0x102
	.long	0xa1
	.byte	0x40
	.uleb128 0x9
	.long	.LASF23
	.byte	0x5
	.value	0x104
	.long	0xa1
	.byte	0x48
	.uleb128 0x9
	.long	.LASF24
	.byte	0x5
	.value	0x105
	.long	0xa1
	.byte	0x50
	.uleb128 0x9
	.long	.LASF25
	.byte	0x5
	.value	0x106
	.long	0xa1
	.byte	0x58
	.uleb128 0x9
	.long	.LASF26
	.byte	0x5
	.value	0x108
	.long	0x266
	.byte	0x60
	.uleb128 0x9
	.long	.LASF27
	.byte	0x5
	.value	0x10a
	.long	0x26c
	.byte	0x68
	.uleb128 0x9
	.long	.LASF28
	.byte	0x5
	.value	0x10c
	.long	0x69
	.byte	0x70
	.uleb128 0x9
	.long	.LASF29
	.byte	0x5
	.value	0x110
	.long	0x69
	.byte	0x74
	.uleb128 0x9
	.long	.LASF30
	.byte	0x5
	.value	0x112
	.long	0x77
	.byte	0x78
	.uleb128 0x9
	.long	.LASF31
	.byte	0x5
	.value	0x116
	.long	0x4d
	.byte	0x80
	.uleb128 0x9
	.long	.LASF32
	.byte	0x5
	.value	0x117
	.long	0x5b
	.byte	0x82
	.uleb128 0x9
	.long	.LASF33
	.byte	0x5
	.value	0x118
	.long	0x272
	.byte	0x83
	.uleb128 0x9
	.long	.LASF34
	.byte	0x5
	.value	0x11c
	.long	0x282
	.byte	0x88
	.uleb128 0x9
	.long	.LASF35
	.byte	0x5
	.value	0x125
	.long	0x82
	.byte	0x90
	.uleb128 0x9
	.long	.LASF36
	.byte	0x5
	.value	0x12e
	.long	0x9f
	.byte	0x98
	.uleb128 0x9
	.long	.LASF37
	.byte	0x5
	.value	0x12f
	.long	0x9f
	.byte	0xa0
	.uleb128 0x9
	.long	.LASF38
	.byte	0x5
	.value	0x130
	.long	0x9f
	.byte	0xa8
	.uleb128 0x9
	.long	.LASF39
	.byte	0x5
	.value	0x131
	.long	0x9f
	.byte	0xb0
	.uleb128 0x9
	.long	.LASF40
	.byte	0x5
	.value	0x132
	.long	0x34
	.byte	0xb8
	.uleb128 0x9
	.long	.LASF41
	.byte	0x5
	.value	0x134
	.long	0x69
	.byte	0xc0
	.uleb128 0x9
	.long	.LASF42
	.byte	0x5
	.value	0x136
	.long	0x288
	.byte	0xc4
	.byte	0
	.uleb128 0xa
	.long	.LASF70
	.byte	0x5
	.byte	0x9a
	.uleb128 0x7
	.long	.LASF44
	.byte	0x18
	.byte	0x5
	.byte	0xa0
	.long	0x266
	.uleb128 0x8
	.long	.LASF45
	.byte	0x5
	.byte	0xa1
	.long	0x266
	.byte	0
	.uleb128 0x8
	.long	.LASF46
	.byte	0x5
	.byte	0xa2
	.long	0x26c
	.byte	0x8
	.uleb128 0x8
	.long	.LASF47
	.byte	0x5
	.byte	0xa6
	.long	0x69
	.byte	0x10
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x235
	.uleb128 0x6
	.byte	0x8
	.long	0xae
	.uleb128 0xb
	.long	0xa7
	.long	0x282
	.uleb128 0xc
	.long	0x8d
	.byte	0
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x22e
	.uleb128 0xb
	.long	0xa7
	.long	0x298
	.uleb128 0xc
	.long	0x8d
	.byte	0x13
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x29e
	.uleb128 0xd
	.long	0xa7
	.uleb128 0x2
	.byte	0x4
	.byte	0x4
	.long	.LASF48
	.uleb128 0x3
	.long	.LASF49
	.byte	0x6
	.byte	0x3b
	.long	0x94
	.uleb128 0xe
	.long	.LASF52
	.byte	0x2
	.byte	0x66
	.long	0x69
	.byte	0x3
	.long	0x2d2
	.uleb128 0xf
	.long	.LASF50
	.byte	0x2
	.byte	0x66
	.long	0x298
	.uleb128 0x10
	.byte	0
	.uleb128 0x11
	.long	.LASF55
	.byte	0x1
	.byte	0x2f
	.long	0x2d
	.quad	.LFB27
	.quad	.LFE27-.LFB27
	.uleb128 0x1
	.byte	0x9c
	.long	0x322
	.uleb128 0x12
	.string	"a"
	.byte	0x1
	.byte	0x31
	.long	0x322
	.long	0x5eb0a82f
	.uleb128 0x13
	.string	"m"
	.byte	0x1
	.byte	0x31
	.long	0x327
	.sleb128 -5
	.uleb128 0x14
	.string	"q"
	.byte	0x1
	.byte	0x31
	.long	0x32c
	.byte	0x2
	.uleb128 0x12
	.string	"r"
	.byte	0x1
	.byte	0x31
	.long	0x331
	.long	0x429eaf9d
	.byte	0
	.uleb128 0xd
	.long	0x54
	.uleb128 0xd
	.long	0x54
	.uleb128 0xd
	.long	0x54
	.uleb128 0xd
	.long	0x54
	.uleb128 0x15
	.long	.LASF54
	.byte	0x1
	.byte	0x37
	.byte	0x1
	.long	0x34e
	.uleb128 0xf
	.long	.LASF51
	.byte	0x1
	.byte	0x37
	.long	0x54
	.byte	0
	.uleb128 0x16
	.long	0x336
	.quad	.LFB28
	.quad	.LFE28-.LFB28
	.uleb128 0x1
	.byte	0x9c
	.long	0x371
	.uleb128 0x17
	.long	0x342
	.uleb128 0x1
	.byte	0x55
	.byte	0
	.uleb128 0x18
	.string	"mm2"
	.byte	0x1
	.byte	0x44
	.quad	.LFB29
	.quad	.LFE29-.LFB29
	.uleb128 0x1
	.byte	0x9c
	.long	0x3d9
	.uleb128 0x19
	.string	"A"
	.byte	0x1
	.byte	0x45
	.long	0x3ea
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x19
	.string	"B"
	.byte	0x1
	.byte	0x45
	.long	0x3ea
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x1a
	.string	"C"
	.byte	0x1
	.byte	0x45
	.long	0x3ea
	.long	.LLST0
	.uleb128 0x1b
	.string	"i"
	.byte	0x1
	.byte	0x47
	.long	0x69
	.long	.LLST1
	.uleb128 0x1b
	.string	"j"
	.byte	0x1
	.byte	0x47
	.long	0x69
	.long	.LLST2
	.uleb128 0x1b
	.string	"k"
	.byte	0x1
	.byte	0x47
	.long	0x69
	.long	.LLST3
	.byte	0
	.uleb128 0xb
	.long	0x2d
	.long	0x3ea
	.uleb128 0x1c
	.long	0x8d
	.value	0x81f
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.long	0x3d9
	.uleb128 0x1d
	.long	.LASF53
	.byte	0x1
	.byte	0x52
	.quad	.LFB30
	.quad	.LFE30-.LFB30
	.uleb128 0x1
	.byte	0x9c
	.long	0x453
	.uleb128 0x1a
	.string	"A"
	.byte	0x1
	.byte	0x53
	.long	0x3ea
	.long	.LLST4
	.uleb128 0x1b
	.string	"i"
	.byte	0x1
	.byte	0x55
	.long	0x69
	.long	.LLST5
	.uleb128 0x1b
	.string	"j"
	.byte	0x1
	.byte	0x55
	.long	0x69
	.long	.LLST6
	.uleb128 0x1e
	.long	.LASF55
	.byte	0x1
	.byte	0x56
	.long	0x2d
	.long	0x445
	.uleb128 0x10
	.byte	0
	.uleb128 0x1f
	.quad	.LVL15
	.long	0x2d2
	.byte	0
	.uleb128 0x1d
	.long	.LASF56
	.byte	0x1
	.byte	0x5f
	.quad	.LFB31
	.quad	.LFE31-.LFB31
	.uleb128 0x1
	.byte	0x9c
	.long	0x498
	.uleb128 0x1a
	.string	"A"
	.byte	0x1
	.byte	0x60
	.long	0x3ea
	.long	.LLST7
	.uleb128 0x1b
	.string	"i"
	.byte	0x1
	.byte	0x62
	.long	0x69
	.long	.LLST8
	.uleb128 0x1b
	.string	"j"
	.byte	0x1
	.byte	0x62
	.long	0x69
	.long	.LLST9
	.byte	0
	.uleb128 0x1d
	.long	.LASF57
	.byte	0x1
	.byte	0x6b
	.quad	.LFB32
	.quad	.LFE32-.LFB32
	.uleb128 0x1
	.byte	0x9c
	.long	0x552
	.uleb128 0x1a
	.string	"A"
	.byte	0x1
	.byte	0x6c
	.long	0x3ea
	.long	.LLST10
	.uleb128 0x1b
	.string	"i"
	.byte	0x1
	.byte	0x6e
	.long	0x69
	.long	.LLST11
	.uleb128 0x1b
	.string	"j"
	.byte	0x1
	.byte	0x6e
	.long	0x69
	.long	.LLST12
	.uleb128 0x20
	.long	0x2b5
	.quad	.LBB10
	.quad	.LBE10-.LBB10
	.byte	0x1
	.byte	0x72
	.long	0x521
	.uleb128 0x21
	.long	0x2c5
	.long	.LLST13
	.uleb128 0x22
	.quad	.LVL32
	.long	0x7ba
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x31
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	.LC2
	.byte	0
	.byte	0
	.uleb128 0x24
	.long	0x2b5
	.quad	.LBB12
	.quad	.LBE12-.LBB12
	.byte	0x1
	.byte	0x74
	.uleb128 0x25
	.long	0x2c5
	.uleb128 0x22
	.quad	.LVL34
	.long	0x7d5
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x3a
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x11
	.long	.LASF58
	.byte	0x1
	.byte	0x78
	.long	0x69
	.quad	.LFB33
	.quad	.LFE33-.LFB33
	.uleb128 0x1
	.byte	0x9c
	.long	0x73e
	.uleb128 0x26
	.long	.LASF59
	.byte	0x1
	.byte	0x7a
	.long	0x2d
	.long	.LLST15
	.uleb128 0x26
	.long	.LASF60
	.byte	0x1
	.byte	0x7a
	.long	0x2d
	.long	.LLST16
	.uleb128 0x26
	.long	.LASF61
	.byte	0x1
	.byte	0x7b
	.long	0x70
	.long	.LLST17
	.uleb128 0x27
	.long	.LASF66
	.byte	0x1
	.byte	0x7c
	.long	0x2aa
	.uleb128 0x20
	.long	0x336
	.quad	.LBB20
	.quad	.LBE20-.LBB20
	.byte	0x1
	.byte	0x80
	.long	0x5ce
	.uleb128 0x28
	.long	0x342
	.value	0x22c1
	.byte	0
	.uleb128 0x20
	.long	0x2b5
	.quad	.LBB22
	.quad	.LBE22-.LBB22
	.byte	0x1
	.byte	0x84
	.long	0x61a
	.uleb128 0x17
	.long	0x2c5
	.uleb128 0xa
	.byte	0x3
	.quad	.LC3
	.byte	0x9f
	.uleb128 0x22
	.quad	.LVL45
	.long	0x7ba
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x31
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	.LC3
	.byte	0
	.byte	0
	.uleb128 0x20
	.long	0x2b5
	.quad	.LBB24
	.quad	.LBE24-.LBB24
	.byte	0x1
	.byte	0x94
	.long	0x666
	.uleb128 0x17
	.long	0x2c5
	.uleb128 0xa
	.byte	0x3
	.quad	.LC6
	.byte	0x9f
	.uleb128 0x22
	.quad	.LVL56
	.long	0x7ba
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x31
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	.LC6
	.byte	0
	.byte	0
	.uleb128 0x1f
	.quad	.LVL39
	.long	0x7ee
	.uleb128 0x29
	.quad	.LVL42
	.long	0x3f0
	.long	0x692
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	A
	.byte	0
	.uleb128 0x29
	.quad	.LVL43
	.long	0x3f0
	.long	0x6b1
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	B
	.byte	0
	.uleb128 0x1f
	.quad	.LVL44
	.long	0x7ee
	.uleb128 0x1f
	.quad	.LVL46
	.long	0x7ee
	.uleb128 0x29
	.quad	.LVL47
	.long	0x453
	.long	0x6ea
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	C
	.byte	0
	.uleb128 0x1f
	.quad	.LVL48
	.long	0x7ee
	.uleb128 0x29
	.quad	.LVL51
	.long	0x371
	.long	0x730
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	A
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	B
	.uleb128 0x23
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x9
	.byte	0x3
	.quad	C
	.byte	0
	.uleb128 0x1f
	.quad	.LVL52
	.long	0x7ee
	.byte	0
	.uleb128 0x2a
	.long	.LASF62
	.byte	0x1
	.byte	0x2c
	.long	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	SEED
	.uleb128 0x2b
	.long	.LASF63
	.byte	0x7
	.byte	0xa8
	.long	0x26c
	.uleb128 0x2b
	.long	.LASF64
	.byte	0x7
	.byte	0xa9
	.long	0x26c
	.uleb128 0xb
	.long	0x2d
	.long	0x781
	.uleb128 0x1c
	.long	0x8d
	.value	0x81f
	.uleb128 0x1c
	.long	0x8d
	.value	0x81f
	.byte	0
	.uleb128 0x2c
	.string	"A"
	.byte	0x1
	.byte	0x1e
	.long	0x769
	.uleb128 0x9
	.byte	0x3
	.quad	A
	.uleb128 0x2c
	.string	"B"
	.byte	0x1
	.byte	0x20
	.long	0x769
	.uleb128 0x9
	.byte	0x3
	.quad	B
	.uleb128 0x2c
	.string	"C"
	.byte	0x1
	.byte	0x22
	.long	0x769
	.uleb128 0x9
	.byte	0x3
	.quad	C
	.uleb128 0x2d
	.long	.LASF71
	.byte	0x2
	.byte	0x57
	.long	0x69
	.long	0x7d5
	.uleb128 0x2e
	.long	0x69
	.uleb128 0x2e
	.long	0x298
	.uleb128 0x10
	.byte	0
	.uleb128 0x2f
	.long	.LASF65
	.byte	0x8
	.byte	0
	.long	.LASF72
	.long	0x69
	.long	0x7ee
	.uleb128 0x2e
	.long	0x69
	.byte	0
	.uleb128 0x27
	.long	.LASF66
	.byte	0x6
	.byte	0xbd
	.long	0x70
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x1b
	.uleb128 0xe
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xa
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xc
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xd
	.uleb128 0x26
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xe
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xf
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x10
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x11
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x12
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1c
	.uleb128 0x6
	.byte	0
	.byte	0
	.uleb128 0x13
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1c
	.uleb128 0xd
	.byte	0
	.byte	0
	.uleb128 0x14
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1c
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x15
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x16
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x17
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x18
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x19
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x1a
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x1b
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x1c
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0x5
	.byte	0
	.byte	0
	.uleb128 0x1d
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1e
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1f
	.uleb128 0x4109
	.byte	0
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x20
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x21
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x22
	.uleb128 0x4109
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x23
	.uleb128 0x410a
	.byte	0
	.uleb128 0x2
	.uleb128 0x18
	.uleb128 0x2111
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x24
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0xb
	.uleb128 0x59
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x25
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x26
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x27
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x28
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x1c
	.uleb128 0x5
	.byte	0
	.byte	0
	.uleb128 0x29
	.uleb128 0x4109
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2a
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x2b
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x2c
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x2d
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2e
	.uleb128 0x5
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2f
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
.LLST0:
	.quad	.LVL1-.Ltext0
	.quad	.LVL3-.Ltext0
	.value	0x1
	.byte	0x51
	.quad	.LVL3-.Ltext0
	.quad	.LVL11-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	.LVL11-.Ltext0
	.quad	.LFE29-.Ltext0
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x51
	.byte	0x9f
	.quad	0
	.quad	0
.LLST1:
	.quad	.LVL2-.Ltext0
	.quad	.LVL4-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST2:
	.quad	.LVL7-.Ltext0
	.quad	.LVL8-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST3:
	.quad	.LVL9-.Ltext0
	.quad	.LVL10-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST4:
	.quad	.LVL12-.Ltext0
	.quad	.LVL14-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL14-.Ltext0
	.quad	.LFE30-.Ltext0
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.quad	0
	.quad	0
.LLST5:
	.quad	.LVL13-.Ltext0
	.quad	.LVL14-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	.LVL14-.Ltext0
	.quad	.LVL17-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x820
	.byte	0x7c
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	.LVL17-.Ltext0
	.quad	.LVL18-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x821
	.byte	0x7c
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	.LVL18-.Ltext0
	.quad	.LVL21-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x820
	.byte	0x7c
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	0
	.quad	0
.LLST6:
	.quad	.LVL19-.Ltext0
	.quad	.LVL20-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST7:
	.quad	.LVL22-.Ltext0
	.quad	.LVL23-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL23-.Ltext0
	.quad	.LFE31-.Ltext0
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.quad	0
	.quad	0
.LLST8:
	.quad	.LVL22-.Ltext0
	.quad	.LVL23-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	.LVL23-.Ltext0
	.quad	.LVL25-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x820
	.byte	0x72
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	.LVL25-.Ltext0
	.quad	.LVL26-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x821
	.byte	0x72
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	.LVL26-.Ltext0
	.quad	.LFE31-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x820
	.byte	0x72
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	0
	.quad	0
.LLST9:
	.quad	.LVL27-.Ltext0
	.quad	.LVL28-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST10:
	.quad	.LVL29-.Ltext0
	.quad	.LVL31-.Ltext0
	.value	0x1
	.byte	0x55
	.quad	.LVL31-.Ltext0
	.quad	.LFE32-.Ltext0
	.value	0x4
	.byte	0xf3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.quad	0
	.quad	0
.LLST11:
	.quad	.LVL30-.Ltext0
	.quad	.LVL31-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	.LVL31-.Ltext0
	.quad	.LVL34-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x820
	.byte	0x7c
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	.LVL34-.Ltext0
	.quad	.LVL35-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x821
	.byte	0x7c
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	.LVL35-.Ltext0
	.quad	.LVL38-.Ltext0
	.value	0x7
	.byte	0xa
	.value	0x820
	.byte	0x7c
	.sleb128 0
	.byte	0x1c
	.byte	0x9f
	.quad	0
	.quad	0
.LLST12:
	.quad	.LVL36-.Ltext0
	.quad	.LVL37-.Ltext0
	.value	0x2
	.byte	0x30
	.byte	0x9f
	.quad	0
	.quad	0
.LLST13:
	.quad	.LVL31-.Ltext0
	.quad	.LVL36-.Ltext0
	.value	0xa
	.byte	0x3
	.quad	.LC2
	.byte	0x9f
	.quad	.LVL37-.Ltext0
	.quad	.LFE32-.Ltext0
	.value	0xa
	.byte	0x3
	.quad	.LC2
	.byte	0x9f
	.quad	0
	.quad	0
.LLST15:
	.quad	.LVL53-.Ltext0
	.quad	.LVL56-1-.Ltext0
	.value	0x1
	.byte	0x61
	.quad	0
	.quad	0
.LLST16:
	.quad	.LVL53-.Ltext0
	.quad	.LVL54-.Ltext0
	.value	0x10
	.byte	0xf4
	.uleb128 0x2d
	.byte	0x8
	.long	0
	.long	0x4210c304
	.byte	0xf5
	.uleb128 0x11
	.uleb128 0x2d
	.byte	0x1b
	.byte	0x9f
	.quad	.LVL54-.Ltext0
	.quad	.LVL55-.Ltext0
	.value	0x1
	.byte	0x62
	.quad	.LVL55-.Ltext0
	.quad	.LVL56-1-.Ltext0
	.value	0x10
	.byte	0xf4
	.uleb128 0x2d
	.byte	0x8
	.long	0
	.long	0x4210c304
	.byte	0xf5
	.uleb128 0x11
	.uleb128 0x2d
	.byte	0x1b
	.byte	0x9f
	.quad	0
	.quad	0
.LLST17:
	.quad	.LVL40-.Ltext0
	.quad	.LVL41-.Ltext0
	.value	0x1
	.byte	0x50
	.quad	.LVL41-.Ltext0
	.quad	.LVL46-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	.LVL49-.Ltext0
	.quad	.LVL50-.Ltext0
	.value	0x1
	.byte	0x50
	.quad	.LVL50-.Ltext0
	.quad	.LVL57-.Ltext0
	.value	0x1
	.byte	0x53
	.quad	0
	.quad	0
	.section	.debug_aranges,"",@progbits
	.long	0x2c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.quad	0
	.quad	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF52:
	.string	"printf"
.LASF9:
	.string	"__off_t"
.LASF15:
	.string	"_IO_read_ptr"
.LASF27:
	.string	"_chain"
.LASF8:
	.string	"size_t"
.LASF33:
	.string	"_shortbuf"
.LASF69:
	.string	"/homes/phjk/ToyPrograms/ACA16/MM"
.LASF62:
	.string	"SEED"
.LASF51:
	.string	"init"
.LASF21:
	.string	"_IO_buf_base"
.LASF57:
	.string	"printmatrix"
.LASF59:
	.string	"time"
.LASF5:
	.string	"signed char"
.LASF28:
	.string	"_fileno"
.LASF16:
	.string	"_IO_read_end"
.LASF7:
	.string	"long int"
.LASF14:
	.string	"_flags"
.LASF22:
	.string	"_IO_buf_end"
.LASF31:
	.string	"_cur_column"
.LASF53:
	.string	"fillmatrix"
.LASF72:
	.string	"putchar"
.LASF0:
	.string	"double"
.LASF71:
	.string	"__printf_chk"
.LASF30:
	.string	"_old_offset"
.LASF35:
	.string	"_offset"
.LASF49:
	.string	"clock_t"
.LASF44:
	.string	"_IO_marker"
.LASF63:
	.string	"stdin"
.LASF4:
	.string	"unsigned int"
.LASF1:
	.string	"long unsigned int"
.LASF19:
	.string	"_IO_write_ptr"
.LASF65:
	.string	"__builtin_putchar"
.LASF46:
	.string	"_sbuf"
.LASF3:
	.string	"short unsigned int"
.LASF23:
	.string	"_IO_save_base"
.LASF67:
	.string	"GNU C 4.8.4 -mtune=generic -march=x86-64 -g -O1 -fstack-protector"
.LASF12:
	.string	"__clock_t"
.LASF34:
	.string	"_lock"
.LASF29:
	.string	"_flags2"
.LASF41:
	.string	"_mode"
.LASF64:
	.string	"stdout"
.LASF11:
	.string	"sizetype"
.LASF20:
	.string	"_IO_write_end"
.LASF70:
	.string	"_IO_lock_t"
.LASF43:
	.string	"_IO_FILE"
.LASF68:
	.string	"MM2.c"
.LASF48:
	.string	"float"
.LASF47:
	.string	"_pos"
.LASF55:
	.string	"drand48"
.LASF26:
	.string	"_markers"
.LASF54:
	.string	"srand48"
.LASF2:
	.string	"unsigned char"
.LASF6:
	.string	"short int"
.LASF61:
	.string	"lasttime"
.LASF32:
	.string	"_vtable_offset"
.LASF13:
	.string	"char"
.LASF60:
	.string	"mflops"
.LASF45:
	.string	"_next"
.LASF10:
	.string	"__off64_t"
.LASF17:
	.string	"_IO_read_base"
.LASF25:
	.string	"_IO_save_end"
.LASF50:
	.string	"__fmt"
.LASF36:
	.string	"__pad1"
.LASF37:
	.string	"__pad2"
.LASF38:
	.string	"__pad3"
.LASF39:
	.string	"__pad4"
.LASF40:
	.string	"__pad5"
.LASF42:
	.string	"_unused2"
.LASF24:
	.string	"_IO_backup_base"
.LASF56:
	.string	"zeromatrix"
.LASF66:
	.string	"clock"
.LASF58:
	.string	"main"
.LASF18:
	.string	"_IO_write_base"
	.ident	"GCC: (Ubuntu 4.8.4-2ubuntu1~14.04) 4.8.4"
	.section	.note.GNU-stack,"",@progbits
