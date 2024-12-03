main:                                   // @main
        sub     sp, sp, #16
        mov     x8, sp
        mov     w9, #1
        strb    wzr, [sp]
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
        ldaxrb  w10, [x8]
        stxrb   w11, w9, [x8]
        cbnz    w11, .LBB0_1
        tbz     w10, #0, .LBB0_9
        mov     x8, sp
        mov     w9, #1
.LBB0_4:                                // =>This Loop Header: Depth=1
        str     xzr, [sp, #8]
        ldr     x10, [sp, #8]
        cmp     x10, #99
        b.hi    .LBB0_6
.LBB0_5:                                //   Parent Loop BB0_4 Depth=1
        ldr     x10, [sp, #8]
        add     x10, x10, #1
        str     x10, [sp, #8]
        ldr     x10, [sp, #8]
        cmp     x10, #100
        b.lo    .LBB0_5
.LBB0_6:                                //   in Loop: Header=BB0_4 Depth=1
        ldrb    w10, [sp]
        tbnz    w10, #0, .LBB0_4
.LBB0_7:                                //   Parent Loop BB0_4 Depth=1
        ldaxrb  w10, [x8]
        stxrb   w11, w9, [x8]
        cbnz    w11, .LBB0_7
        tbnz    w10, #0, .LBB0_4
.LBB0_9:
        mov     x8, sp
        mov     w0, wzr
        stlrb   wzr, [x8]
        add     sp, sp, #16
        ret
