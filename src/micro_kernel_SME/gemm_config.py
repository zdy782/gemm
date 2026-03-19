from gemm_type_impl import *

type = "small"
transa = "N"
transb = "N"
currect_model = None

load_inst = ""

def set_type_value(gemm_type, transA, transB):
    global type, transa, transb
    type = gemm_type
    transa = transA
    transb = transB

def get_gemm_type_model():
    global currect_model
    if type == "small" and transa == "N" and transb == "N":
        currect_model = small_gemm_nn_def
    elif type == "small" and transa == "N" and transb == "T":
        currect_model = small_gemm_nt_def
    elif type == "small" and transa == "T" and transb == "N":
        currect_model = small_gemm_tn_def
    elif type == "small" and transa == "T" and transb == "T":
        currect_model = small_gemm_tt_def
    elif type == "general":
        currect_model = general_gemm_def
    else:
        raise ValueError(
            f"Unsupported GEMM config: type={type}, transA={transa}, transB={transb}"
        )

    return currect_model
