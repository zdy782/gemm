from __future__ import annotations

from typing import Dict

from micro_kernel_SME.bf16_selector.features import PACK_LABELS


FEATURE_EXPRESSIONS = {
    "M": "f.M",
    "N": "f.N",
    "K": "f.K",
    "taT": "f.taT",
    "tbT": "f.tbT",
    "pair": "f.pair",
    "M_over_N": "f.M_over_N",
    "N_over_M": "f.N_over_M",
    "M_ge_N": "f.M_ge_N",
    "M_ge_2N": "f.M_ge_2N",
    "N_ge_2M": "f.N_ge_2M",
    "M_ge_4N": "f.M_ge_4N",
    "N_ge_4M": "f.N_ge_4M",
    "K_ge_64": "f.K_ge_64",
    "K_ge_128": "f.K_ge_128",
    "K_ge_256": "f.K_ge_256",
    "verybigN": "f.verybigN",
    "verybigM": "f.verybigM",
    "verybigK": "f.verybigK",
    "shape_code": "f.shape_code",
    "shape_SLS": "f.shape_SLS",
    "shape_LSS": "f.shape_LSS",
    "shape_LLS": "f.shape_LLS",
    "shape_verybigN": "f.shape_verybigN",
    "shape_verybigM": "f.shape_verybigM",
    "shape_verybigK": "f.shape_verybigK",
}


def _emit_tree(tree: Dict[str, object], indent: int) -> str:
    prefix = " " * indent
    if tree.get("leaf"):
        return f'{prefix}return "{tree["label"]}";\n'
    feature = str(tree["feature"])
    threshold = float(tree["threshold"])
    expr = FEATURE_EXPRESSIONS[feature]
    code = f"{prefix}if ({expr} <= {threshold:.17g}) {{\n"
    code += _emit_tree(tree["left"], indent + 4)
    code += f"{prefix}}}\n"
    code += _emit_tree(tree["right"], indent)
    return code


def generate_cpp_selector(model: Dict[str, object]) -> str:
    tile_function_defs = []
    for pack in PACK_LABELS:
        tile_tree = model["tile_trees"].get(pack)
        if tile_tree is None:
            tile_code = f'    return "{model["default_tile_by_pack"][pack]}";\n'
        else:
            tile_code = _emit_tree(tile_tree, 4)
        tile_function_defs.append(
            f"""static inline const char *autogemm_predict_bf16_tile_{pack}(const AutoGemmBf16Features &f) {{
{tile_code}}}
"""
        )

    pack_code = _emit_tree(model["pack_tree"], 4)
    default_tile_by_pack = model["default_tile_by_pack"]

    return f"""
struct AutoGemmBf16Features {{
    double M;
    double N;
    double K;
    double taT;
    double tbT;
    double pair;
    double M_over_N;
    double N_over_M;
    double M_ge_N;
    double M_ge_2N;
    double N_ge_2M;
    double M_ge_4N;
    double N_ge_4M;
    double K_ge_64;
    double K_ge_128;
    double K_ge_256;
    double verybigN;
    double verybigM;
    double verybigK;
    double shape_code;
    double shape_SLS;
    double shape_LSS;
    double shape_LLS;
    double shape_verybigN;
    double shape_verybigM;
    double shape_verybigK;
}};

static inline int autogemm_bf16_shape_code(long M, long N, long K) {{
    if (K >= 256) {{
        return 5;
    }}
    if (M >= 500) {{
        return 4;
    }}
    if (N >= 500) {{
        return 3;
    }}
    if (M <= 48) {{
        return 0;
    }}
    if (N <= 128) {{
        return 1;
    }}
    return 2;
}}

static inline AutoGemmBf16Features autogemm_make_bf16_features(long M, long N, long K, char ta, char tb) {{
    const int shape_code = autogemm_bf16_shape_code(M, N, K);
    AutoGemmBf16Features f{{}};
    f.M = static_cast<double>(M);
    f.N = static_cast<double>(N);
    f.K = static_cast<double>(K);
    f.taT = ta == 'T' ? 1.0 : 0.0;
    f.tbT = tb == 'T' ? 1.0 : 0.0;
    f.pair = f.taT * 2.0 + f.tbT;
    f.M_over_N = static_cast<double>(M) / static_cast<double>(N);
    f.N_over_M = static_cast<double>(N) / static_cast<double>(M);
    f.M_ge_N = M >= N ? 1.0 : 0.0;
    f.M_ge_2N = M >= 2 * N ? 1.0 : 0.0;
    f.N_ge_2M = N >= 2 * M ? 1.0 : 0.0;
    f.M_ge_4N = M >= 4 * N ? 1.0 : 0.0;
    f.N_ge_4M = N >= 4 * M ? 1.0 : 0.0;
    f.K_ge_64 = K >= 64 ? 1.0 : 0.0;
    f.K_ge_128 = K >= 128 ? 1.0 : 0.0;
    f.K_ge_256 = K >= 256 ? 1.0 : 0.0;
    f.verybigN = N >= 500 ? 1.0 : 0.0;
    f.verybigM = M >= 500 ? 1.0 : 0.0;
    f.verybigK = K >= 256 ? 1.0 : 0.0;
    f.shape_code = static_cast<double>(shape_code);
    f.shape_SLS = shape_code == 0 ? 1.0 : 0.0;
    f.shape_LSS = shape_code == 1 ? 1.0 : 0.0;
    f.shape_LLS = shape_code == 2 ? 1.0 : 0.0;
    f.shape_verybigN = shape_code == 3 ? 1.0 : 0.0;
    f.shape_verybigM = shape_code == 4 ? 1.0 : 0.0;
    f.shape_verybigK = shape_code == 5 ? 1.0 : 0.0;
    return f;
}}

static inline const char *autogemm_predict_bf16_pack(const AutoGemmBf16Features &f) {{
{pack_code}}}

{''.join(tile_function_defs)}
struct AutoGemmBf16Choice {{
    const char *pack;
    const char *tile;
    bool pack_a;
    bool pack_b;
    int m_vl;
    int n_vl;
}};

static inline AutoGemmBf16Choice autogemm_predict_bf16_choice(long M, long N, long K, char ta, char tb) {{
    const AutoGemmBf16Features f = autogemm_make_bf16_features(M, N, K, ta, tb);
    const char *pack = autogemm_predict_bf16_pack(f);
    const char *tile = "{default_tile_by_pack['nopack']}";
    if (std::strcmp(pack, "nopack") == 0) {{
        tile = autogemm_predict_bf16_tile_nopack(f);
    }} else if (std::strcmp(pack, "packa") == 0) {{
        tile = autogemm_predict_bf16_tile_packa(f);
    }} else if (std::strcmp(pack, "packb") == 0) {{
        tile = autogemm_predict_bf16_tile_packb(f);
    }} else if (std::strcmp(pack, "packab") == 0) {{
        tile = autogemm_predict_bf16_tile_packab(f);
    }}

    AutoGemmBf16Choice choice{{}};
    choice.pack = pack;
    choice.tile = tile;
    choice.pack_a = std::strcmp(pack, "packa") == 0 || std::strcmp(pack, "packab") == 0;
    choice.pack_b = std::strcmp(pack, "packb") == 0 || std::strcmp(pack, "packab") == 0;
    if (std::strcmp(tile, "1x4") == 0) {{
        choice.m_vl = 1;
        choice.n_vl = 4;
    }} else if (std::strcmp(tile, "2x2") == 0) {{
        choice.m_vl = 2;
        choice.n_vl = 2;
    }} else {{
        choice.m_vl = 4;
        choice.n_vl = 1;
    }}
    return choice;
}}
"""
