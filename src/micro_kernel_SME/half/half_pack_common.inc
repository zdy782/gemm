#ifndef PACK_INCOPY
#error "PACK_INCOPY must be defined before including half_pack_common.inc"
#endif

#ifndef PACK_ITCOPY
#error "PACK_ITCOPY must be defined before including half_pack_common.inc"
#endif

#ifndef PACK_ONCOPY
#error "PACK_ONCOPY must be defined before including half_pack_common.inc"
#endif

#ifndef PACK_OTCOPY
#error "PACK_OTCOPY must be defined before including half_pack_common.inc"
#endif

    .macro HALF_TCOPY_IMPL func_name
    .text
    .align 4
    .global \func_name
    .type \func_name, %function
\func_name:
    cbz     w0, 9f
    cbz     w1, 9f

    mov     w9, w0
    mov     w10, w1
    mov     w11, w3
    cnth    x12
    mov     x13, x2
    mov     x14, x4

1:
    mov     x15, xzr
2:
    whilelt p0.h, x15, x10
    add     x16, x13, x15, lsl #1
    add     x17, x14, x15, lsl #1
    ld1h    { z0.h }, p0/z, [x16]
    st1h    { z0.h }, p0, [x17]
    add     x15, x15, x12
    cmp     x15, x10
    blt     2b

    add     x13, x13, x11, lsl #1
    add     x14, x14, x10, lsl #1
    subs    x9, x9, #1
    bne     1b
9:
    ret
    .size \func_name, . - \func_name
    .endm

    .macro HALF_NCOPY_IMPL func_name
    .text
    .align 4
    .global \func_name
    .type \func_name, %function
\func_name:
    cbz     w0, 9f
    cbz     w1, 9f

    mov     w9, w0
    mov     w10, w1
    mov     w11, w3
    lsl     x12, x11, #1
    cntw    x13
    index   z1.s, wzr, w12
    mov     x14, x2
    mov     x15, x4

1:
    mov     x16, xzr
2:
    whilelt p0.s, x16, x10
    madd    x17, x16, x12, x14
    ld1h    { z0.s }, p0/z, [x17, z1.s, uxtw]
    add     x18, x15, x16, lsl #1
    st1h    { z0.s }, p0, [x18]
    add     x16, x16, x13
    cmp     x16, x10
    blt     2b

    add     x14, x14, #2
    add     x15, x15, x10, lsl #1
    subs    x9, x9, #1
    bne     1b
9:
    ret
    .size \func_name, . - \func_name
    .endm

    HALF_NCOPY_IMPL PACK_INCOPY
    HALF_TCOPY_IMPL PACK_ITCOPY
    HALF_NCOPY_IMPL PACK_ONCOPY
    HALF_TCOPY_IMPL PACK_OTCOPY
