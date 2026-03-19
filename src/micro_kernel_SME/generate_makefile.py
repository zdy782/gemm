def generate_makefile(data_type="fp32"):
    if data_type == "bf16":
        PRECISION_MACRO = "BF16"
    else:
        PRECISION_MACRO = "FP32"

    code_str = ""
    code_str += f"CXX = clang++\n"
    code_str += f"CC = clang\n"
    code_str += f"CFLAGS = -march=armv9-a+sme+bf16 -O3 -std=c++14 -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm -D{PRECISION_MACRO}\n"
    code_str += f"ASFLAGS = -march=armv9-a+sme+bf16 -O3\n"
    code_str += f"all:\n"
    code_str += f"\t$(CC) $(ASFLAGS) -c kernel_asm.S -o kernel_asm.o\n"
    code_str += f"\t$(CXX) $(CFLAGS) test.cpp kernel_asm.o -o benchmark_kernel"
    return code_str