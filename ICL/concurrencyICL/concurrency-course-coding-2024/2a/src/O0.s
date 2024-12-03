	.file	"9_data_race_and_compiler_optimization.cc"
	.text
	.section	.text._ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_,"axG",@progbits,_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_,comdat
	.align 2
	.weak	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_
	.type	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_, @function
_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_:
.LFB279:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE279:
	.size	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_, .-_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_
	.section	.text._ZNSt6chrono15duration_valuesIlE4zeroEv,"axG",@progbits,_ZNSt6chrono15duration_valuesIlE4zeroEv,comdat
	.weak	_ZNSt6chrono15duration_valuesIlE4zeroEv
	.type	_ZNSt6chrono15duration_valuesIlE4zeroEv, @function
_ZNSt6chrono15duration_valuesIlE4zeroEv:
.LFB280:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$0, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE280:
	.size	_ZNSt6chrono15duration_valuesIlE4zeroEv, .-_ZNSt6chrono15duration_valuesIlE4zeroEv
	.section	.text._ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv,"axG",@progbits,_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv,comdat
	.align 2
	.weak	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	.type	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv, @function
_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv:
.LFB281:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE281:
	.size	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv, .-_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	.section	.text._ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_,"axG",@progbits,_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_,comdat
	.align 2
	.weak	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_
	.type	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_, @function
_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_:
.LFB300:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE300:
	.size	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_, .-_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_
#APP
	.globl _ZSt21ios_base_library_initv
#NO_APP
	.section	.text._ZNSt6thread2idC2Ev,"axG",@progbits,_ZNSt6thread2idC5Ev,comdat
	.align 2
	.weak	_ZNSt6thread2idC2Ev
	.type	_ZNSt6thread2idC2Ev, @function
_ZNSt6thread2idC2Ev:
.LFB2350:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	$0, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2350:
	.size	_ZNSt6thread2idC2Ev, .-_ZNSt6thread2idC2Ev
	.weak	_ZNSt6thread2idC1Ev
	.set	_ZNSt6thread2idC1Ev,_ZNSt6thread2idC2Ev
	.section	.text._ZNSt6thread24_M_thread_deps_never_runEv,"axG",@progbits,_ZNSt6thread24_M_thread_deps_never_runEv,comdat
	.weak	_ZNSt6thread24_M_thread_deps_never_runEv
	.type	_ZNSt6thread24_M_thread_deps_never_runEv, @function
_ZNSt6thread24_M_thread_deps_never_runEv:
.LFB2355:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2355:
	.size	_ZNSt6thread24_M_thread_deps_never_runEv, .-_ZNSt6thread24_M_thread_deps_never_runEv
	.section	.text._ZNSt6threadD2Ev,"axG",@progbits,_ZNSt6threadD5Ev,comdat
	.align 2
	.weak	_ZNSt6threadD2Ev
	.type	_ZNSt6threadD2Ev, @function
_ZNSt6threadD2Ev:
.LFB2358:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6thread8joinableEv
	testb	%al, %al
	je	.L11
	call	_ZSt9terminatev@PLT
.L11:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2358:
	.size	_ZNSt6threadD2Ev, .-_ZNSt6threadD2Ev
	.weak	_ZNSt6threadD1Ev
	.set	_ZNSt6threadD1Ev,_ZNSt6threadD2Ev
	.section	.text._ZNKSt6thread8joinableEv,"axG",@progbits,_ZNKSt6thread8joinableEv,comdat
	.align 2
	.weak	_ZNKSt6thread8joinableEv
	.type	_ZNKSt6thread8joinableEv, @function
_ZNKSt6thread8joinableEv:
.LFB2366:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	leaq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread2idC1Ev
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSteqNSt6thread2idES0_
	xorl	$1, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L14
	call	__stack_chk_fail@PLT
.L14:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2366:
	.size	_ZNKSt6thread8joinableEv, .-_ZNKSt6thread8joinableEv
	.section	.text._ZSteqNSt6thread2idES0_,"axG",@progbits,_ZSteqNSt6thread2idES0_,comdat
	.weak	_ZSteqNSt6thread2idES0_
	.type	_ZSteqNSt6thread2idES0_, @function
_ZSteqNSt6thread2idES0_:
.LFB2375:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdx
	movq	-16(%rbp), %rax
	cmpq	%rax, %rdx
	sete	%al
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2375:
	.size	_ZSteqNSt6thread2idES0_, .-_ZSteqNSt6thread2idES0_
	.section	.rodata
.LC0:
	.string	"Waiting\n"
.LC1:
	.string	"Waking up thread 2\n"
	.text
	.globl	_Z3FooRi
	.type	_Z3FooRi, @function
_Z3FooRi:
.LFB2387:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	movl	$1, -20(%rbp)
	leaq	-20(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IivEERKT_
	leaq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	movq	-40(%rbp), %rax
	movl	$1, (%rax)
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L18
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2387:
	.size	_Z3FooRi, .-_Z3FooRi
	.section	.rodata
.LC2:
	.string	"Thread 2 got woken up\n"
	.text
	.globl	_Z3BarRi
	.type	_Z3BarRi, @function
_Z3BarRi:
.LFB2388:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	nop
.L20:
	movq	-8(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L20
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	leaq	_ZSt4cout(%rip), %rax
	movq	%rax, %rdi
	call	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2388:
	.size	_Z3BarRi, .-_Z3BarRi
	.globl	main
	.type	main, @function
main:
.LFB2389:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2389
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movl	$0, -52(%rbp)
	leaq	-52(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt3refIiESt17reference_wrapperIT_ERS1_
	movq	%rax, -32(%rbp)
	leaq	-32(%rbp), %rdx
	leaq	-48(%rbp), %rax
	leaq	_Z3FooRi(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB0:
	call	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
.LEHE0:
	leaq	-52(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt3refIiESt17reference_wrapperIT_ERS1_
	movq	%rax, -32(%rbp)
	leaq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rax
	leaq	_Z3BarRi(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB1:
	call	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
.LEHE1:
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
.LEHB2:
	call	_ZNSt6thread4joinEv@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread4joinEv@PLT
.LEHE2:
	movl	$0, %ebx
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6threadD1Ev
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6threadD1Ev
	movl	%ebx, %eax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L29
.L28:
	endbr64
	movq	%rax, %rbx
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6threadD1Ev
	jmp	.L24
.L27:
	endbr64
	movq	%rax, %rbx
.L24:
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6threadD1Ev
	movq	%rbx, %rax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L25
	call	__stack_chk_fail@PLT
.L25:
	movq	%rax, %rdi
.LEHB3:
	call	_Unwind_Resume@PLT
.LEHE3:
.L29:
	call	__stack_chk_fail@PLT
.L26:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2389:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA2389:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2389-.LLSDACSB2389
.LLSDACSB2389:
	.uleb128 .LEHB0-.LFB2389
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB2389
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L27-.LFB2389
	.uleb128 0
	.uleb128 .LEHB2-.LFB2389
	.uleb128 .LEHE2-.LEHB2
	.uleb128 .L28-.LFB2389
	.uleb128 0
	.uleb128 .LEHB3-.LFB2389
	.uleb128 .LEHE3-.LEHB3
	.uleb128 0
	.uleb128 0
.LLSDACSE2389:
	.text
	.size	main, .-main
	.section	.text._ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv,"axG",@progbits,_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv,comdat
	.align 2
	.weak	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	.type	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv, @function
_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv:
.LFB2393:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2393:
	.size	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv, .-_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	.section	.text._ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC2IivEERKT_,"axG",@progbits,_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC5IivEERKT_,comdat
	.align 2
	.weak	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC2IivEERKT_
	.type	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC2IivEERKT_, @function
_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC2IivEERKT_:
.LFB2676:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2676:
	.size	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC2IivEERKT_, .-_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC2IivEERKT_
	.weak	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IivEERKT_
	.set	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IivEERKT_,_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC2IivEERKT_
	.section	.text._ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE,"axG",@progbits,_ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE,comdat
	.weak	_ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE
	.type	_ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE, @function
_ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE:
.LFB2678:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	call	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv
	movq	%rax, -32(%rbp)
	leaq	-32(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE
	testb	%al, %al
	jne	.L40
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE
	movq	%rax, -48(%rbp)
	leaq	-48(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_
	movq	%rax, -32(%rbp)
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE
	movq	%rax, -40(%rbp)
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	movq	%rax, -32(%rbp)
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1000000000EEE5countEv
	movq	%rax, -24(%rbp)
	nop
.L38:
	leaq	-32(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	nanosleep@PLT
	cmpl	$-1, %eax
	jne	.L36
	call	__errno_location@PLT
	movl	(%rax), %eax
	cmpl	$4, %eax
	jne	.L36
	movl	$1, %eax
	jmp	.L37
.L36:
	movl	$0, %eax
.L37:
	testb	%al, %al
	jne	.L38
	jmp	.L33
.L40:
	nop
.L33:
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L39
	call	__stack_chk_fail@PLT
.L39:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2678:
	.size	_ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE, .-_ZNSt11this_thread9sleep_forIlSt5ratioILl1ELl1EEEEvRKNSt6chrono8durationIT_T0_EE
	.section	.text._ZSt3refIiESt17reference_wrapperIT_ERS1_,"axG",@progbits,_ZSt3refIiESt17reference_wrapperIT_ERS1_,comdat
	.weak	_ZSt3refIiESt17reference_wrapperIT_ERS1_
	.type	_ZSt3refIiESt17reference_wrapperIT_ERS1_, @function
_ZSt3refIiESt17reference_wrapperIT_ERS1_:
.LFB2679:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-24(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt17reference_wrapperIiEC1IRivPiEEOT_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L43
	call	__stack_chk_fail@PLT
.L43:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2679:
	.size	_ZSt3refIiESt17reference_wrapperIT_ERS1_, .-_ZSt3refIiESt17reference_wrapperIT_ERS1_
	.section	.text._ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,"axG",@progbits,_ZNSt6threadC5IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,comdat
	.align 2
	.weak	_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.type	_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_, @function
_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_:
.LFB2693:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2693
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread2idC1Ev
	movl	$24, %edi
.LEHB4:
	call	_Znwm@PLT
.LEHE4:
	movq	%rax, %rbx
	movl	$1, %r13d
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %r12
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	movq	%r12, %rdx
	movq	%rax, %rsi
	movq	%rbx, %rdi
.LEHB5:
	call	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC1IJRS4_S7_EEEDpOT_
.LEHE5:
	movl	$0, %r13d
	leaq	-48(%rbp), %rax
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC1IS3_vEEPS1_
	leaq	-48(%rbp), %rcx
	movq	-56(%rbp), %rax
	leaq	_ZNSt6thread24_M_thread_deps_never_runEv(%rip), %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB6:
	call	_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE@PLT
.LEHE6:
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED1Ev
	nop
	movq	-40(%rbp), %rax
	subq	%fs:40, %rax
	je	.L49
	jmp	.L52
.L51:
	endbr64
	movq	%rax, %r12
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED1Ev
	jmp	.L46
.L50:
	endbr64
	movq	%rax, %r12
.L46:
	testb	%r13b, %r13b
	je	.L47
	movl	$24, %esi
	movq	%rbx, %rdi
	call	_ZdlPvm@PLT
.L47:
	movq	%r12, %rax
	movq	-40(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L48
	call	__stack_chk_fail@PLT
.L48:
	movq	%rax, %rdi
.LEHB7:
	call	_Unwind_Resume@PLT
.LEHE7:
.L52:
	call	__stack_chk_fail@PLT
.L49:
	addq	$56, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2693:
	.section	.gcc_except_table
.LLSDA2693:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2693-.LLSDACSB2693
.LLSDACSB2693:
	.uleb128 .LEHB4-.LFB2693
	.uleb128 .LEHE4-.LEHB4
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB5-.LFB2693
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L50-.LFB2693
	.uleb128 0
	.uleb128 .LEHB6-.LFB2693
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L51-.LFB2693
	.uleb128 0
	.uleb128 .LEHB7-.LFB2693
	.uleb128 .LEHE7-.LEHB7
	.uleb128 0
	.uleb128 0
.LLSDACSE2693:
	.section	.text._ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,"axG",@progbits,_ZNSt6threadC5IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,comdat
	.size	_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_, .-_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.weak	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.set	_ZNSt6threadC1IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_,_ZNSt6threadC2IRFvRiEJSt17reference_wrapperIiEEvEEOT_DpOT0_
	.section	.text._ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE,"axG",@progbits,_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE,comdat
	.weak	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE
	.type	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE, @function
_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE:
.LFB2697:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2697:
	.size	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE, .-_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1000000000EEEElS2_ILl1ELl1EEEENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES8_E4typeERKNS1_IT0_T1_EE
	.section	.text._ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv,"axG",@progbits,_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv,comdat
	.weak	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv
	.type	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv, @function
_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv:
.LFB2807:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	call	_ZNSt6chrono15duration_valuesIlE4zeroEv
	movq	%rax, -24(%rbp)
	leaq	-24(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L57
	call	__stack_chk_fail@PLT
.L57:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2807:
	.size	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv, .-_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEE4zeroEv
	.section	.text._ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE,"axG",@progbits,_ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE,comdat
	.weak	_ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE
	.type	_ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE, @function
_ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE:
.LFB2808:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rdx
	movq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE
	xorl	$1, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2808:
	.size	_ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE, .-_ZNSt6chronoleIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE
	.section	.text._ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE,"axG",@progbits,_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE,comdat
	.weak	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE
	.type	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE, @function
_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE:
.LFB2809:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2809:
	.size	_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE, .-_ZNSt6chrono13duration_castINS_8durationIlSt5ratioILl1ELl1EEEElS3_EENSt9enable_ifIXsrNS_13__is_durationIT_EE5valueES7_E4typeERKNS1_IT0_T1_EE
	.section	.text._ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_,"axG",@progbits,_ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_,comdat
	.weak	_ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_
	.type	_ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_, @function
_ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_:
.LFB2810:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$72, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	-72(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -56(%rbp)
	leaq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	movq	%rax, %rbx
	movq	-80(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -48(%rbp)
	leaq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	subq	%rax, %rbx
	movq	%rbx, %rdx
	movq	%rdx, -40(%rbp)
	leaq	-40(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6chrono8durationIlSt5ratioILl1ELl1EEEC1IlvEERKT_
	movq	-32(%rbp), %rax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L64
	call	__stack_chk_fail@PLT
.L64:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2810:
	.size	_ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_, .-_ZNSt6chronomiIlSt5ratioILl1ELl1EElS2_EENSt11common_typeIJNS_8durationIT_T0_EENS4_IT1_T2_EEEE4typeERKS7_RKSA_
	.section	.text._ZNSt17reference_wrapperIiEC2IRivPiEEOT_,"axG",@progbits,_ZNSt17reference_wrapperIiEC5IRivPiEEOT_,comdat
	.align 2
	.weak	_ZNSt17reference_wrapperIiEC2IRivPiEEOT_
	.type	_ZNSt17reference_wrapperIiEC2IRivPiEEOT_, @function
_ZNSt17reference_wrapperIiEC2IRivPiEEOT_:
.LFB2812:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE
	movq	%rax, %rdi
	call	_ZNSt17reference_wrapperIiE6_S_funERi
	movq	-8(%rbp), %rdx
	movq	%rax, (%rdx)
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2812:
	.size	_ZNSt17reference_wrapperIiEC2IRivPiEEOT_, .-_ZNSt17reference_wrapperIiEC2IRivPiEEOT_
	.weak	_ZNSt17reference_wrapperIiEC1IRivPiEEOT_
	.set	_ZNSt17reference_wrapperIiEC1IRivPiEEOT_,_ZNSt17reference_wrapperIiEC2IRivPiEEOT_
	.section	.text._ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE,"axG",@progbits,_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE,comdat
	.weak	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	.type	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE, @function
_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE:
.LFB2814:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2814:
	.size	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE, .-_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	.section	.text._ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE,"axG",@progbits,_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE,comdat
	.weak	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	.type	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE, @function
_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE:
.LFB2815:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2815:
	.size	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE, .-_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	.section	.text._ZNSt6thread6_StateC2Ev,"axG",@progbits,_ZNSt6thread6_StateC5Ev,comdat
	.align 2
	.weak	_ZNSt6thread6_StateC2Ev
	.type	_ZNSt6thread6_StateC2Ev, @function
_ZNSt6thread6_StateC2Ev:
.LFB2818:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	leaq	16+_ZTVNSt6thread6_StateE(%rip), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2818:
	.size	_ZNSt6thread6_StateC2Ev, .-_ZNSt6thread6_StateC2Ev
	.weak	_ZNSt6thread6_StateC1Ev
	.set	_ZNSt6thread6_StateC1Ev,_ZNSt6thread6_StateC2Ev
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC5IJRS4_S7_EEEDpOT_,comdat
	.align 2
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_
	.type	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_, @function
_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_:
.LFB2820:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2820
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread6_StateC2Ev
	leaq	16+_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-24(%rbp), %rax
	leaq	8(%rax), %rbx
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %r12
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	movq	%r12, %rdx
	movq	%rax, %rsi
	movq	%rbx, %rdi
.LEHB8:
	call	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC1IJRS3_S6_EEEDpOT_
.LEHE8:
	jmp	.L74
.L73:
	endbr64
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread6_StateD2Ev@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB9:
	call	_Unwind_Resume@PLT
.LEHE9:
.L74:
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2820:
	.section	.gcc_except_table
.LLSDA2820:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2820-.LLSDACSB2820
.LLSDACSB2820:
	.uleb128 .LEHB8-.LFB2820
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L73-.LFB2820
	.uleb128 0
	.uleb128 .LEHB9-.LFB2820
	.uleb128 .LEHE9-.LEHB9
	.uleb128 0
	.uleb128 0
.LLSDACSE2820:
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC5IJRS4_S7_EEEDpOT_,comdat
	.size	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_, .-_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC1IJRS4_S7_EEEDpOT_
	.set	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC1IJRS4_S7_EEEDpOT_,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEC2IJRS4_S7_EEEDpOT_
	.section	.text._ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI2St15__uniq_ptr_implIS1_S3_EEPS1_,"axG",@progbits,_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI5St15__uniq_ptr_implIS1_S3_EEPS1_,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI2St15__uniq_ptr_implIS1_S3_EEPS1_
	.type	_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI2St15__uniq_ptr_implIS1_S3_EEPS1_, @function
_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI2St15__uniq_ptr_implIS1_S3_EEPS1_:
.LFB2824:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2824:
	.size	_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI2St15__uniq_ptr_implIS1_S3_EEPS1_, .-_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI2St15__uniq_ptr_implIS1_S3_EEPS1_
	.weak	_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI1St15__uniq_ptr_implIS1_S3_EEPS1_
	.set	_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI1St15__uniq_ptr_implIS1_S3_EEPS1_,_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI2St15__uniq_ptr_implIS1_S3_EEPS1_
	.section	.text._ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC2IS3_vEEPS1_,"axG",@progbits,_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC5IS3_vEEPS1_,comdat
	.align 2
	.weak	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC2IS3_vEEPS1_
	.type	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC2IS3_vEEPS1_, @function
_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC2IS3_vEEPS1_:
.LFB2826:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_dataINSt6thread6_StateESt14default_deleteIS1_ELb1ELb1EECI1St15__uniq_ptr_implIS1_S3_EEPS1_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2826:
	.size	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC2IS3_vEEPS1_, .-_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC2IS3_vEEPS1_
	.weak	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC1IS3_vEEPS1_
	.set	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC1IS3_vEEPS1_,_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EEC2IS3_vEEPS1_
	.section	.text._ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED2Ev,"axG",@progbits,_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED5Ev,comdat
	.align 2
	.weak	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED2Ev
	.type	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED2Ev, @function
_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED2Ev:
.LFB2829:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	(%rax), %rax
	testq	%rax, %rax
	je	.L78
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_
	movq	(%rax), %rax
	movq	%rax, %rsi
	movq	%rbx, %rdi
	call	_ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_
.L78:
	movq	-24(%rbp), %rax
	movq	$0, (%rax)
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2829:
	.size	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED2Ev, .-_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED2Ev
	.weak	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED1Ev
	.set	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED1Ev,_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EED2Ev
	.section	.text._ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE,"axG",@progbits,_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE,comdat
	.weak	_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE
	.type	_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE, @function
_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE:
.LFB2831:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	imulq	$1000000000, %rax, %rax
	movq	%rax, -24(%rbp)
	leaq	-24(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEC1IlvEERKT_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L81
	call	__stack_chk_fail@PLT
.L81:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2831:
	.size	_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE, .-_ZNSt6chrono20__duration_cast_implINS_8durationIlSt5ratioILl1ELl1000000000EEEES2_ILl1000000000ELl1EElLb0ELb1EE6__castIlS2_ILl1ELl1EEEES4_RKNS1_IT_T0_EE
	.section	.text._ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE,"axG",@progbits,_ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE,comdat
	.weak	_ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE
	.type	_ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE, @function
_ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE:
.LFB2900:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -40(%rbp)
	leaq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	movq	%rax, %rbx
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -32(%rbp)
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6chrono8durationIlSt5ratioILl1ELl1EEE5countEv
	cmpq	%rax, %rbx
	setl	%al
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L84
	call	__stack_chk_fail@PLT
.L84:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2900:
	.size	_ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE, .-_ZNSt6chronoltIlSt5ratioILl1ELl1EElS2_EEbRKNS_8durationIT_T0_EERKNS3_IT1_T2_EE
	.section	.text._ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE,"axG",@progbits,_ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE,comdat
	.weak	_ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE
	.type	_ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE, @function
_ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE:
.LFB2901:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2901:
	.size	_ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE, .-_ZSt7forwardIRiEOT_RNSt16remove_referenceIS1_E4typeE
	.section	.text._ZNSt17reference_wrapperIiE6_S_funERi,"axG",@progbits,_ZNSt17reference_wrapperIiE6_S_funERi,comdat
	.weak	_ZNSt17reference_wrapperIiE6_S_funERi
	.type	_ZNSt17reference_wrapperIiE6_S_funERi, @function
_ZNSt17reference_wrapperIiE6_S_funERi:
.LFB2902:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt11__addressofIiEPT_RS0_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2902:
	.size	_ZNSt17reference_wrapperIiE6_S_funERi, .-_ZNSt17reference_wrapperIiE6_S_funERi
	.section	.text._ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC2IJRS3_S6_EEEDpOT_,"axG",@progbits,_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC5IJRS3_S6_EEEDpOT_,comdat
	.align 2
	.weak	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC2IJRS3_S6_EEEDpOT_
	.type	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC2IJRS3_S6_EEEDpOT_, @function
_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC2IJRS3_S6_EEEDpOT_:
.LFB2914:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-24(%rbp), %rbx
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %r12
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	movq	%r12, %rdx
	movq	%rax, %rsi
	movq	%rbx, %rdi
	call	_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC1IRS1_S4_Lb1EEEOT_OT0_
	nop
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2914:
	.size	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC2IJRS3_S6_EEEDpOT_, .-_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC2IJRS3_S6_EEEDpOT_
	.weak	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC1IJRS3_S6_EEEDpOT_
	.set	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC1IJRS3_S6_EEEDpOT_,_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEC2IJRS3_S6_EEEDpOT_
	.section	.text._ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_,"axG",@progbits,_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC5EPS1_,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_
	.type	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_, @function
_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_:
.LFB2918:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC1ILb1ELb1EEEv
	movq	-32(%rbp), %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv
	movq	%rbx, (%rax)
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2918:
	.size	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_, .-_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_
	.weak	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC1EPS1_
	.set	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC1EPS1_,_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EEC2EPS1_
	.section	.text._ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv,"axG",@progbits,_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv
	.type	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv, @function
_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv:
.LFB2920:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2920:
	.size	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv, .-_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE6_M_ptrEv
	.section	.text._ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv,"axG",@progbits,_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv,comdat
	.align 2
	.weak	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv
	.type	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv, @function
_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv:
.LFB2921:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2921:
	.size	_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv, .-_ZNSt10unique_ptrINSt6thread6_StateESt14default_deleteIS1_EE11get_deleterEv
	.section	.text._ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_,"axG",@progbits,_ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_,comdat
	.weak	_ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_
	.type	_ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_, @function
_ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_:
.LFB2922:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2922:
	.size	_ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_, .-_ZSt4moveIRPNSt6thread6_StateEEONSt16remove_referenceIT_E4typeEOS5_
	.section	.text._ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_,"axG",@progbits,_ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_,comdat
	.align 2
	.weak	_ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_
	.type	_ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_, @function
_ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_:
.LFB2923:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	testq	%rax, %rax
	je	.L99
	movq	(%rax), %rdx
	addq	$8, %rdx
	movq	(%rdx), %rdx
	movq	%rax, %rdi
	call	*%rdx
.L99:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2923:
	.size	_ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_, .-_ZNKSt14default_deleteINSt6thread6_StateEEclEPS1_
	.section	.text._ZSt11__addressofIiEPT_RS0_,"axG",@progbits,_ZSt11__addressofIiEPT_RS0_,comdat
	.weak	_ZSt11__addressofIiEPT_RS0_
	.type	_ZSt11__addressofIiEPT_RS0_, @function
_ZSt11__addressofIiEPT_RS0_:
.LFB2979:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2979:
	.size	_ZSt11__addressofIiEPT_RS0_, .-_ZSt11__addressofIiEPT_RS0_
	.section	.text._ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_,"axG",@progbits,_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC5IRS1_S4_Lb1EEEOT_OT0_,comdat
	.align 2
	.weak	_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_
	.type	_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_, @function
_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_:
.LFB2981:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2981
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-24(%rbp), %rbx
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %r12
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	movq	%r12, %rdx
	movq	%rax, %rsi
	movq	%rbx, %rdi
	call	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_
	nop
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2981:
	.section	.gcc_except_table
.LLSDA2981:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2981-.LLSDACSB2981
.LLSDACSB2981:
.LLSDACSE2981:
	.section	.text._ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_,"axG",@progbits,_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC5IRS1_S4_Lb1EEEOT_OT0_,comdat
	.size	_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_, .-_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_
	.weak	_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC1IRS1_S4_Lb1EEEOT_OT0_
	.set	_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC1IRS1_S4_Lb1EEEOT_OT0_,_ZNSt5tupleIJPFvRiESt17reference_wrapperIiEEEC2IRS1_S4_Lb1EEEOT_OT0_
	.section	.text._ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv,"axG",@progbits,_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC5ILb1ELb1EEEv,comdat
	.align 2
	.weak	_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv
	.type	_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv, @function
_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv:
.LFB2984:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA2984
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2984:
	.section	.gcc_except_table
.LLSDA2984:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE2984-.LLSDACSB2984
.LLSDACSB2984:
.LLSDACSE2984:
	.section	.text._ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv,"axG",@progbits,_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC5ILb1ELb1EEEv,comdat
	.size	_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv, .-_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv
	.weak	_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC1ILb1ELb1EEEv
	.set	_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC1ILb1ELb1EEEv,_ZNSt5tupleIJPNSt6thread6_StateESt14default_deleteIS1_EEEC2ILb1ELb1EEEv
	.section	.text._ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_,"axG",@progbits,_ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_,comdat
	.weak	_ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_
	.type	_ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_, @function
_ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_:
.LFB2986:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2986:
	.size	_ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_, .-_ZSt3getILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_
	.section	.text._ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv,"axG",@progbits,_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv,comdat
	.align 2
	.weak	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv
	.type	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv, @function
_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv:
.LFB2987:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2987:
	.size	_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv, .-_ZNSt15__uniq_ptr_implINSt6thread6_StateESt14default_deleteIS1_EE10_M_deleterEv
	.section	.text._ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC5IRS1_JS4_EvEEOT_DpOT0_,comdat
	.align 2
	.weak	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_
	.type	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_, @function
_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_:
.LFB3009:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	-40(%rbp), %rbx
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %rsi
	movq	%rbx, %rdi
	call	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_
	movq	-40(%rbp), %rax
	leaq	8(%rax), %rbx
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIRFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	movq	%rax, -32(%rbp)
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	movq	%rbx, %rdi
	call	_ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L109
	call	__stack_chk_fail@PLT
.L109:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3009:
	.size	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_, .-_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_
	.weak	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC1IRS1_JS4_EvEEOT_DpOT0_
	.set	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC1IRS1_JS4_EvEEOT_DpOT0_,_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEEC2IRS1_JS4_EvEEOT_DpOT0_
	.section	.text._ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev,"axG",@progbits,_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC5Ev,comdat
	.align 2
	.weak	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev
	.type	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev, @function
_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev:
.LFB3012:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3012:
	.size	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev, .-_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev
	.weak	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC1Ev
	.set	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC1Ev,_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEEC2Ev
	.section	.text._ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE
	.type	_ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE, @function
_ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE:
.LFB3014:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3014:
	.size	_ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE, .-_ZSt12__get_helperILm0EPNSt6thread6_StateEJSt14default_deleteIS1_EEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE
	.section	.text._ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_,"axG",@progbits,_ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_,comdat
	.weak	_ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_
	.type	_ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_, @function
_ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_:
.LFB3015:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3015:
	.size	_ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_, .-_ZSt3getILm1EJPNSt6thread6_StateESt14default_deleteIS1_EEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS9_
	.section	.text._ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_,"axG",@progbits,_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC5IS1_EEOT_,comdat
	.align 2
	.weak	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_
	.type	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_, @function
_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_:
.LFB3025:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-24(%rbp), %rbx
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %rsi
	movq	%rbx, %rdi
	call	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3025:
	.size	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_, .-_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_
	.weak	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC1IS1_EEOT_
	.set	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC1IS1_EEOT_,_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEEC2IS1_EEOT_
	.section	.text._ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_,"axG",@progbits,_ZNSt10_Head_baseILm0EPFvRiELb0EEC5ERKS2_,comdat
	.align 2
	.weak	_ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_
	.type	_ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_, @function
_ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_:
.LFB3028:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3028:
	.size	_ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_, .-_ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_
	.weak	_ZNSt10_Head_baseILm0EPFvRiELb0EEC1ERKS2_
	.set	_ZNSt10_Head_baseILm0EPFvRiELb0EEC1ERKS2_,_ZNSt10_Head_baseILm0EPFvRiELb0EEC2ERKS2_
	.section	.text._ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev,"axG",@progbits,_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC5Ev,comdat
	.align 2
	.weak	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev
	.type	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev, @function
_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev:
.LFB3031:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3031:
	.size	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev, .-_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev
	.weak	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC1Ev
	.set	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC1Ev,_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEEC2Ev
	.section	.text._ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev,"axG",@progbits,_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC5Ev,comdat
	.align 2
	.weak	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev
	.type	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev, @function
_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev:
.LFB3034:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	$0, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3034:
	.size	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev, .-_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev
	.weak	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC1Ev
	.set	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC1Ev,_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EEC2Ev
	.section	.text._ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_,comdat
	.weak	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_
	.type	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_, @function
_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_:
.LFB3036:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3036:
	.size	_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_, .-_ZNSt11_Tuple_implILm0EJPNSt6thread6_StateESt14default_deleteIS1_EEE7_M_headERS5_
	.section	.text._ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE
	.type	_ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE, @function
_ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE:
.LFB3037:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3037:
	.size	_ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE, .-_ZSt12__get_helperILm1ESt14default_deleteINSt6thread6_StateEEJEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE
	.section	.text._ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_,"axG",@progbits,_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC5IS1_EEOT_,comdat
	.align 2
	.weak	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_
	.type	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_, @function
_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_:
.LFB3044:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	-8(%rbp), %rdx
	movq	(%rax), %rax
	movq	%rax, (%rdx)
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3044:
	.size	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_, .-_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_
	.weak	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC1IS1_EEOT_
	.set	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC1IS1_EEOT_,_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EEC2IS1_EEOT_
	.section	.text._ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev,"axG",@progbits,_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC5Ev,comdat
	.align 2
	.weak	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev
	.type	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev, @function
_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev:
.LFB3047:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3047:
	.size	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev, .-_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev
	.weak	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC1Ev
	.set	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC1Ev,_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EEC2Ev
	.section	.text._ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_,"axG",@progbits,_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_,comdat
	.weak	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_
	.type	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_, @function
_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_:
.LFB3049:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3049:
	.size	_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_, .-_ZNSt10_Head_baseILm0EPNSt6thread6_StateELb0EE7_M_headERS3_
	.section	.text._ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_,"axG",@progbits,_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_,comdat
	.weak	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_
	.type	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_, @function
_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_:
.LFB3050:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3050:
	.size	_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_, .-_ZNSt11_Tuple_implILm1EJSt14default_deleteINSt6thread6_StateEEEE7_M_headERS4_
	.section	.text._ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_,"axG",@progbits,_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_,comdat
	.weak	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_
	.type	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_, @function
_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_:
.LFB3051:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3051:
	.size	_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_, .-_ZNSt10_Head_baseILm1ESt14default_deleteINSt6thread6_StateEELb1EE7_M_headERS4_
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
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED5Ev,comdat
	.align 2
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev
	.type	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev, @function
_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev:
.LFB3053:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	leaq	16+_ZTVNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE(%rip), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread6_StateD2Ev@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3053:
	.size	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev, .-_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED1Ev
	.set	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED1Ev,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED2Ev
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED5Ev,comdat
	.align 2
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev
	.type	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev, @function
_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev:
.LFB3055:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED1Ev
	movq	-8(%rbp), %rax
	movl	$24, %esi
	movq	%rax, %rdi
	call	_ZdlPvm@PLT
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3055:
	.size	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev, .-_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEED0Ev
	.weak	_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.section	.data.rel.ro._ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,"awG",@progbits,_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,comdat
	.align 8
	.type	_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, @object
	.size	_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, 24
_ZTINSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE:
	.quad	_ZTVN10__cxxabiv120__si_class_type_infoE+16
	.quad	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.quad	_ZTINSt6thread6_StateE
	.weak	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE
	.section	.rodata._ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,"aG",@progbits,_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE,comdat
	.align 32
	.type	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, @object
	.size	_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE, 84
_ZTSNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE:
	.string	"NSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEEE"
	.section	.rodata
	.type	_ZSt12__is_ratio_vISt5ratioILl1ELl1000000000EEE, @object
	.size	_ZSt12__is_ratio_vISt5ratioILl1ELl1000000000EEE, 1
_ZSt12__is_ratio_vISt5ratioILl1ELl1000000000EEE:
	.byte	1
	.type	_ZSt12__is_ratio_vISt5ratioILl1ELl1EEE, @object
	.size	_ZSt12__is_ratio_vISt5ratioILl1ELl1EEE, 1
_ZSt12__is_ratio_vISt5ratioILl1ELl1EEE:
	.byte	1
	.type	_ZNSt8__detail30__integer_to_chars_is_unsignedIjEE, @object
	.size	_ZNSt8__detail30__integer_to_chars_is_unsignedIjEE, 1
_ZNSt8__detail30__integer_to_chars_is_unsignedIjEE:
	.byte	1
	.type	_ZNSt8__detail30__integer_to_chars_is_unsignedImEE, @object
	.size	_ZNSt8__detail30__integer_to_chars_is_unsignedImEE, 1
_ZNSt8__detail30__integer_to_chars_is_unsignedImEE:
	.byte	1
	.type	_ZNSt8__detail30__integer_to_chars_is_unsignedIyEE, @object
	.size	_ZNSt8__detail30__integer_to_chars_is_unsignedIyEE, 1
_ZNSt8__detail30__integer_to_chars_is_unsignedIyEE:
	.byte	1
	.type	_ZSt12__is_ratio_vISt5ratioILl1000000000ELl1EEE, @object
	.size	_ZSt12__is_ratio_vISt5ratioILl1000000000ELl1EEE, 1
_ZSt12__is_ratio_vISt5ratioILl1000000000ELl1EEE:
	.byte	1
	.section	.text._ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv,"axG",@progbits,_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv,comdat
	.align 2
	.weak	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv
	.type	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv, @function
_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv:
.LFB3056:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3056:
	.size	_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv, .-_ZNSt6thread11_State_implINS_8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEEE6_M_runEv
	.section	.text._ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv,"axG",@progbits,_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv,comdat
	.align 2
	.weak	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv
	.type	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv, @function
_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv:
.LFB3057:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3057:
	.size	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv, .-_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEEclEv
	.section	.text._ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE,"axG",@progbits,_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE,comdat
	.align 2
	.weak	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE
	.type	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE, @function
_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE:
.LFB3058:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_
	movq	%rax, %rdi
	call	_ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_
	movq	%rax, %rdi
	call	_ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3058:
	.size	_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE, .-_ZNSt6thread8_InvokerISt5tupleIJPFvRiESt17reference_wrapperIiEEEE9_M_invokeIJLm0ELm1EEEEvSt12_Index_tupleIJXspT_EEE
	.section	.text._ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_,"axG",@progbits,_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_,comdat
	.weak	_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_
	.type	_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_, @function
_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_:
.LFB3060:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3060:
	.size	_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_, .-_ZSt4moveIRSt5tupleIJPFvRiESt17reference_wrapperIiEEEEONSt16remove_referenceIT_E4typeEOS9_
	.section	.text._ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_,"axG",@progbits,_ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_,comdat
	.weak	_ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_
	.type	_ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_, @function
_ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_:
.LFB3061:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE
	movq	%rax, %rdi
	call	_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3061:
	.size	_ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_, .-_ZSt3getILm0EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_
	.section	.text._ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_,"axG",@progbits,_ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_,comdat
	.weak	_ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_
	.type	_ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_, @function
_ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_:
.LFB3062:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3062:
	.size	_ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_, .-_ZSt3getILm1EJPFvRiESt17reference_wrapperIiEEEONSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeEOS9_
	.section	.text._ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_,"axG",@progbits,_ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_,comdat
	.weak	_ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_
	.type	_ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_, @function
_ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_:
.LFB3063:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3063:
	.size	_ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_, .-_ZSt8__invokeIPFvRiEJSt17reference_wrapperIiEEENSt15__invoke_resultIT_JDpT0_EE4typeEOS6_DpOS7_
	.section	.text._ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE
	.type	_ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE, @function
_ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE:
.LFB3064:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3064:
	.size	_ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE, .-_ZSt12__get_helperILm0EPFvRiEJSt17reference_wrapperIiEEERT0_RSt11_Tuple_implIXT_EJS5_DpT1_EE
	.section	.text._ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE,"axG",@progbits,_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE,comdat
	.weak	_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	.type	_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE, @function
_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE:
.LFB3065:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3065:
	.size	_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE, .-_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	.section	.text._ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE,"axG",@progbits,_ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE,comdat
	.weak	_ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE
	.type	_ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE, @function
_ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE:
.LFB3066:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3066:
	.size	_ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE, .-_ZSt12__get_helperILm1ESt17reference_wrapperIiEJEERT0_RSt11_Tuple_implIXT_EJS2_DpT1_EE
	.section	.text._ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_,"axG",@progbits,_ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_,comdat
	.weak	_ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_
	.type	_ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_, @function
_ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_:
.LFB3067:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIPFvRiEEOT_RNSt16remove_referenceIS3_E4typeE
	movq	(%rax), %rbx
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardISt17reference_wrapperIiEEOT_RNSt16remove_referenceIS2_E4typeE
	movq	%rax, %rdi
	call	_ZNKSt17reference_wrapperIiEcvRiEv
	movq	%rax, %rdi
	call	*%rbx
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3067:
	.size	_ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_, .-_ZSt13__invoke_implIvPFvRiEJSt17reference_wrapperIiEEET_St14__invoke_otherOT0_DpOT1_
	.section	.text._ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_,"axG",@progbits,_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_,comdat
	.weak	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_
	.type	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_, @function
_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_:
.LFB3068:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	addq	$8, %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3068:
	.size	_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_, .-_ZNSt11_Tuple_implILm0EJPFvRiESt17reference_wrapperIiEEE7_M_headERS5_
	.section	.text._ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_,"axG",@progbits,_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_,comdat
	.weak	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_
	.type	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_, @function
_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_:
.LFB3069:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3069:
	.size	_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_, .-_ZNSt11_Tuple_implILm1EJSt17reference_wrapperIiEEE7_M_headERS2_
	.section	.text._ZNKSt17reference_wrapperIiEcvRiEv,"axG",@progbits,_ZNKSt17reference_wrapperIiEcvRiEv,comdat
	.align 2
	.weak	_ZNKSt17reference_wrapperIiEcvRiEv
	.type	_ZNKSt17reference_wrapperIiEcvRiEv, @function
_ZNKSt17reference_wrapperIiEcvRiEv:
.LFB3070:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt17reference_wrapperIiE3getEv
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3070:
	.size	_ZNKSt17reference_wrapperIiEcvRiEv, .-_ZNKSt17reference_wrapperIiEcvRiEv
	.section	.text._ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_,"axG",@progbits,_ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_,comdat
	.weak	_ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_
	.type	_ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_, @function
_ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_:
.LFB3071:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3071:
	.size	_ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_, .-_ZNSt10_Head_baseILm0EPFvRiELb0EE7_M_headERS3_
	.section	.text._ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_,"axG",@progbits,_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_,comdat
	.weak	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_
	.type	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_, @function
_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_:
.LFB3072:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3072:
	.size	_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_, .-_ZNSt10_Head_baseILm1ESt17reference_wrapperIiELb0EE7_M_headERS2_
	.section	.text._ZNKSt17reference_wrapperIiE3getEv,"axG",@progbits,_ZNKSt17reference_wrapperIiE3getEv,comdat
	.align 2
	.weak	_ZNKSt17reference_wrapperIiE3getEv
	.type	_ZNKSt17reference_wrapperIiE3getEv, @function
_ZNKSt17reference_wrapperIiE3getEv:
.LFB3073:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3073:
	.size	_ZNKSt17reference_wrapperIiE3getEv, .-_ZNKSt17reference_wrapperIiE3getEv
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
