	.text
	.file	"temp_weaker.cc"
	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	movb	$0, -16(%rsp)
	.p2align	4, 0x90
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
                                        #       Child Loop BB0_4 Depth 3
	movb	$1, %al
	xchgb	%al, -16(%rsp)
	testb	$1, %al
	je	.LBB0_5
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_4 Depth 3
	movq	$0, -8(%rsp)
	movq	-8(%rsp), %rax
	cmpq	$99, %rax
	ja	.LBB0_3
	.p2align	4, 0x90
.LBB0_4:                                #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_2 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	addq	$1, -8(%rsp)
	movq	-8(%rsp), %rax
	cmpq	$100, %rax
	jb	.LBB0_4
.LBB0_3:                                #   in Loop: Header=BB0_2 Depth=2
	movb	-16(%rsp), %al
	testb	$1, %al
	je	.LBB0_1
	jmp	.LBB0_2
.LBB0_5:
	movb	$0, -16(%rsp)
	xorl	%eax, %eax
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 10.0.0-4ubuntu1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
