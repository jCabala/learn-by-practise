	.file	"9_data_race_and_compiler_optimization.cc"
	.text
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text._ZNSt6thread24_M_thread_deps_never_runEv,"axG",@progbits,_ZNSt6thread24_M_thread_deps_never_runEv,comdat
	.p2align 4
	.weak	_ZNSt6thread24_M_thread_deps_never_runEv
	.type	_ZNSt6thread24_M_thread_deps_never_runEv, @function
_ZNSt6thread24_M_thread_deps_never_runEv:
.LFB2424:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE2424:
	.size	_ZNSt6thread24_M_thread_deps_never_runEv, .-_ZNSt6thread24_M_thread_deps_never_runEv
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv
	.type	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv, @function
_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv:
.LFB3125:
	.cfi_startproc
	endbr64
	movq	8(%rdi), %rdx
	movq	16(%rdi), %rax
	movq	%rdx, %rdi
	jmp	*%rax
	.cfi_endproc
.LFE3125:
	.size	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv, .-_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED5Ev,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev
	.type	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev, @function
_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev:
.LFB3122:
	.cfi_startproc
	endbr64
	leaq	16+_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE(%rip), %rax
	movq	%rax, (%rdi)
	jmp	_ZNSt6thread6_StateD2Ev@PLT
	.cfi_endproc
.LFE3122:
	.size	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev, .-_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED1Ev
	.set	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED1Ev,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED5Ev,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev
	.type	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev, @function
_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev:
.LFB3124:
	.cfi_startproc
	endbr64
	leaq	16+_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE(%rip), %rax
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movq	%rdi, %rbx
	movq	%rax, (%rdi)
	call	_ZNSt6thread6_StateD2Ev@PLT
	movq	%rbx, %rdi
	movl	$24, %esi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm@PLT
	.cfi_endproc
.LFE3124:
	.size	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev, .-_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"Thread 2 got woken up\n"
	.text
	.p2align 4
	.globl	_Z3BarRi
	.type	_Z3BarRi, @function
_Z3BarRi:
.LFB2457:
	.cfi_startproc
	endbr64
	movl	$22, %edx
	leaq	.LC0(%rip), %rsi
	leaq	_ZSt4cout(%rip), %rdi
	jmp	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	.cfi_endproc
.LFE2457:
	.size	_Z3BarRi, .-_Z3BarRi
	.section	.rodata.str1.1
.LC1:
	.string	"Waiting\n"
.LC3:
	.string	"Waking up thread 2\n"
	.text
	.p2align 4
	.globl	_Z3FooRi
	.type	_Z3FooRi, @function
_Z3FooRi:
.LFB2456:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	leaq	_ZSt4cout(%rip), %r12
	movl	$8, %edx
	leaq	.LC1(%rip), %rsi
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rdi, %rbp
	movq	%r12, %rdi
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$32, %rsp
	.cfi_def_cfa_offset 64
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	movq	%rsp, %rbx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movdqa	.LC2(%rip), %xmm0
	movaps	%xmm0, (%rsp)
.L10:
	movq	%rbx, %rsi
	movq	%rbx, %rdi
	call	nanosleep@PLT
	cmpl	$-1, %eax
	je	.L14
.L9:
	movl	$19, %edx
	leaq	.LC3(%rip), %rsi
	movq	%r12, %rdi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@PLT
	movl	$1, 0(%rbp)
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L15
	addq	$32, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L14:
	.cfi_restore_state
	call	__errno_location@PLT
	cmpl	$4, (%rax)
	jne	.L9
	jmp	.L10
.L15:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE2456:
	.size	_Z3FooRi, .-_Z3FooRi
	.section	.text._ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,"axG",@progbits,_ZNSt6threadC5IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,comdat
	.align 2
	.p2align 4
	.weak	_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.type	_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_, @function
_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_:
.LFB2763:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2763
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movq	%rdx, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rsi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rdi, %rbx
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
	movq	%fs:40, %rax
	movq	%rax, 8(%rsp)
	xorl	%eax, %eax
	movq	$0, (%rdi)
	movl	$24, %edi
.LEHB0:
	call	_Znwm@PLT
.LEHE0:
	movq	(%r12), %rdx
	movq	%rsp, %rsi
	movq	%rbx, %rdi
	leaq	16+_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE(%rip), %rcx
	movq	%rbp, 16(%rax)
	movq	%rdx, 8(%rax)
	leaq	_ZNSt6thread24_M_thread_deps_never_runEv(%rip), %rdx
	movq	%rcx, (%rax)
	movq	%rax, (%rsp)
.LEHB1:
	call	_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE@PLT
.LEHE1:
	movq	(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L16
	movq	(%rdi), %rax
	call	*8(%rax)
.L16:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L30
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L18:
	.cfi_restore_state
	movq	(%rsp), %rdi
	testq	%rdi, %rdi
	je	.L19
	movq	(%rdi), %rax
	call	*8(%rax)
.L19:
	movq	8(%rsp), %rax
	subq	%fs:40, %rax
	je	.L20
.L30:
	call	__stack_chk_fail@PLT
.L22:
	endbr64
	movq	%rax, %rbx
	jmp	.L18
.L20:
	movq	%rbx, %rdi
.LEHB2:
	call	_Unwind_Resume@PLT
.LEHE2:
	.cfi_endproc
.LFE2763:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table._ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,"aG",@progbits,_ZNSt6threadC5IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,comdat
.LLSDA2763:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2763-.LLSDACSB2763
.LLSDACSB2763:
	.uleb128 .LEHB0-.LFB2763
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB2763
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L22-.LFB2763
	.uleb128 0
	.uleb128 .LEHB2-.LFB2763
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
.LLSDACSE2763:
	.section	.text._ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,"axG",@progbits,_ZNSt6threadC5IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,comdat
	.size	_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_, .-_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.weak	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.set	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.section	.text.unlikely,"ax",@progbits
.LCOLDB4:
	.section	.text.startup,"ax",@progbits
.LHOTB4:
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB2458:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2458
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	leaq	_Z3FooRi(%rip), %rsi
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	movq	%fs:40, %rax
	movq	%rax, 40(%rsp)
	xorl	%eax, %eax
	leaq	32(%rsp), %r12
	leaq	16(%rsp), %rbp
	movl	$0, 12(%rsp)
	leaq	12(%rsp), %rbx
	movq	%r12, %rdx
	movq	%rbp, %rdi
	movq	%rbx, 32(%rsp)
.LEHB3:
	call	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
.LEHE3:
	movq	%rbx, 32(%rsp)
	movq	%r12, %rdx
	leaq	24(%rsp), %rbx
	leaq	_Z3BarRi(%rip), %rsi
	movq	%rbx, %rdi
.LEHB4:
	call	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
.LEHE4:
	movq	%rbp, %rdi
.LEHB5:
	call	_ZNSt6thread4joinEv@PLT
	movq	%rbx, %rdi
	call	_ZNSt6thread4joinEv@PLT
.LEHE5:
	cmpq	$0, 24(%rsp)
	jne	.L33
	cmpq	$0, 16(%rsp)
	jne	.L33
	movq	40(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L42
	addq	$48, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L33:
	.cfi_restore_state
	call	_ZSt9terminatev@PLT
.L42:
	call	__stack_chk_fail@PLT
.L39:
	endbr64
	movq	%rax, %rdi
	jmp	.L34
.L38:
	endbr64
	movq	%rax, %rdi
	jmp	.L35
	.section	.gcc_except_table,"a",@progbits
.LLSDA2458:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2458-.LLSDACSB2458
.LLSDACSB2458:
	.uleb128 .LEHB3-.LFB2458
	.uleb128 .LEHE3-.LEHB3
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB4-.LFB2458
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L38-.LFB2458
	.uleb128 0
	.uleb128 .LEHB5-.LFB2458
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L39-.LFB2458
	.uleb128 0
.LLSDACSE2458:
	.section	.text.startup
	.cfi_endproc
	.section	.text.unlikely
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDAC2458
	.type	main.cold, @function
main.cold:
.LFSB2458:
.L34:
	.cfi_def_cfa_offset 80
	.cfi_offset 3, -32
	.cfi_offset 6, -24
	.cfi_offset 12, -16
	cmpq	$0, 24(%rsp)
	jne	.L33
.L35:
	cmpq	$0, 16(%rsp)
	jne	.L33
	movq	40(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L43
.LEHB6:
	call	_Unwind_Resume@PLT
.LEHE6:
.L43:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE2458:
	.section	.gcc_except_table
.LLSDAC2458:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSEC2458-.LLSDACSBC2458
.LLSDACSBC2458:
	.uleb128 .LEHB6-.LCOLDB4
	.uleb128 .LEHE6-.LEHB6
	.uleb128 0
	.uleb128 0
.LLSDACSEC2458:
	.section	.text.unlikely
	.section	.text.startup
	.size	main, .-main
	.section	.text.unlikely
	.size	main.cold, .-main.cold
.LCOLDE4:
	.section	.text.startup
.LHOTE4:
	.weak	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.section	.rodata._ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,"aG",@progbits,_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,comdat
	.align 32
	.type	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, @object
	.size	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, 84
_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE:
	.string	"NSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE"
	.weak	_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.section	.data.rel.ro._ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,"awG",@progbits,_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,comdat
	.align 8
	.type	_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, @object
	.size	_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, 24
_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.quad	_ZTINSt6thread6_StateE
	.weak	_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.section	.data.rel.ro.local._ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,"awG",@progbits,_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,comdat
	.align 8
	.type	_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, @object
	.size	_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, 40
_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE:
	.quad	0
	.quad	_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.quad	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED1Ev
	.quad	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev
	.quad	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC2:
	.quad	1
	.quad	0
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"GCC: (Ubuntu 13.2.0-23ubuntu4) 13.2.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
