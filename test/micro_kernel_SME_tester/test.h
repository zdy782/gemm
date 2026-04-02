#ifndef _TEST_H_
#define _TEST_H_

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <arm_neon.h>
#ifdef BF16
#include <arm_bf16.h>
#endif

#if defined(BF16)
using KML_FP = __bf16;
#elif defined(FP16)
using KML_FP = __fp16;
#else
#error "test.h supports only BF16 or FP16 builds."
#endif

#ifdef BF16
static inline float bf16_to_float(__bf16 val) {
    float result;
    uint16_t bits = *(uint16_t*)&val;
    uint32_t float_bits = ((uint32_t)bits) << 16;
    result = *(float*)&float_bits;
    return result;
}

static inline __bf16 float_to_bf16(float val) {
    __bf16 result;
    uint32_t float_bits = *(uint32_t*)&val;
    uint16_t bf16_bits = (uint16_t)(float_bits >> 16);
    result = *(__bf16*)&bf16_bits;
    return result;
}
#endif

static inline float kml_to_float(KML_FP val) {
#ifdef BF16
    return bf16_to_float(val);
#else
    return (float)val;
#endif
}

static inline KML_FP float_to_kml(float val) {
#ifdef BF16
    return float_to_bf16(val);
#else
    return (__fp16)val;
#endif
}

class test_utils {
public:
  // Column-major GEMM reference implementation
  // All matrices are stored in column-major order:
  // - A[i,k] = A[i + k*lda]
  // - B[k,j] = B[k + j*ldb]
  // - C[i,j] = C[i + j*ldc]
  static void gemm_ref(const KML_FP *A, const KML_FP *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool ACC, char transA = 'N', char transB = 'N') {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float sum = ACC ? C[i + j * ldc] : 0.0f;
        for (int k = 0; k < K; ++k) {
          float a_val;
          float b_val;
          if (transA == 'N') {
            a_val = kml_to_float(A[i + k * lda]);
          } else {
            a_val = kml_to_float(A[k + i * lda]);
          }
          if (transB == 'N') {
            b_val = kml_to_float(B[k + j * ldb]);
          } else {
            b_val = kml_to_float(B[j + k * ldb]);
          }
          sum += a_val * b_val;
        }
        C[i + j * ldc] = sum;
      }
    }
  }

  static bool is_same_matrix(const float *C1, const float *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float c1_val = C1[i + j * ldc];
        float c2_val = C2[i + j * ldc];
        float diff = fabs(c1_val - c2_val);
        if (diff > atol && (fabs(c1_val) > 1e-10f ? diff / fabs(c1_val) > rtol : diff > atol)) {
          return false;
        }
      }
    }
    return true;
  }

  static int diff_index(const float *C1, const float *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float c1_val = C1[i + j * ldc];
        float c2_val = C2[i + j * ldc];
        float diff = fabs(c1_val - c2_val);
        if (diff > atol && (fabs(c1_val) > 1e-10f ? diff / fabs(c1_val) > rtol : diff > atol)) {
          return i + j * ldc;
        }
      }
    }
    return -1;
  }

  static int print_diff(const float *C1, const float *C2, int M, int N, int ldc) {
    printf("\n=== Diff Matrix (refC vs ourC) ===\n");
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float diff = fabs(C1[i + j * ldc] - C2[i + j * ldc]);
        if (diff > 1e-6f) {
          printf("First diff: [i=%d,j=%d] ref=%.6f our=%.6f diff=%.6f\n",
                 i, j, C1[i + j * ldc], C2[i + j * ldc], diff);
          return -1;
        }
      }
    }
    return -1;
  }

  static void init(KML_FP *buf, int size, int start_value = 1) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = float_to_kml(1.0f * rand() / RAND_MAX);
    }
  }

  static void init(float *buf, int size, int start_value = 1) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = 1.0f * rand() / RAND_MAX;
    }
  }

  static void print_matrix(const KML_FP *matrix, int rows, int cols, int lda, const char* name) {
    printf("\n=== %s Matrix ===\n", name);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%.6f ", kml_to_float(matrix[i + j * lda]));
      }
      printf("\n");
    }
    printf("\n");
  }

  static void print_matrix(const float *matrix, int rows, int cols, int lda, const char* name) {
    printf("\n=== %s Matrix ===\n", name);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%.6f ", matrix[i + j * lda]);
      }
      printf("\n");
    }
    printf("\n");
  }
};

#endif
