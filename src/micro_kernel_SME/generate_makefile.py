from pathlib import Path


def generate_makefile(data_type="fp32"):
    src_dir = Path(__file__).resolve().parent
    extra_pack_src = None
    if data_type == "bf16":
        PRECISION_MACRO = "BF16"
        march_flags = "-march=armv9-a+sme+bf16"
        extra_pack_src = src_dir / "sbgemm_half_pack.S"
    elif data_type == "fp16":
        PRECISION_MACRO = "FP16"
        march_flags = "-march=armv9-a+sme"
        extra_pack_src = src_dir / "shgemm_half_pack.S"
    else:
        PRECISION_MACRO = "FP32"
        march_flags = "-march=armv9-a+sme"

    code_str = ""
    code_str += f"CXX = clang++\n"
    code_str += f"CC = clang\n"
    code_str += f"CFLAGS = {march_flags} -O3 -std=c++14 -Wno-implicit-int-float-conversion -Wno-asm-operand-widths -Wno-inline-asm -D{PRECISION_MACRO}\n"
    code_str += f"ASFLAGS = {march_flags} -O3\n"
    code_str += f"all:\n"
    code_str += f"\t$(CC) $(ASFLAGS) -c kernel_asm.S -o kernel_asm.o\n"
    link_objects = "kernel_asm.o"
    if extra_pack_src is not None:
        code_str += f"\t$(CC) $(ASFLAGS) -c {extra_pack_src} -o half_pack.o\n"
        link_objects += " half_pack.o"
    code_str += f"\t$(CXX) $(CFLAGS) test.cpp driver.cpp {link_objects} -o benchmark_kernel"
    return code_str
