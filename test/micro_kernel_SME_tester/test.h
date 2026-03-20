#ifndef _TEST_H_
#define _TEST_H_
#include <cstdlib>
#include <cmath>
#include <arm_neon.h>
#ifdef BF16
#include <arm_bf16.h>
#endif

#ifdef FP64
typedef double KML_FP;
#elif FP32
typedef float KML_FP;
#else
typedef __fp16 KML_FP;
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

static inline __fp16 float_to_fp16(float val) {
    return (__fp16)val;
}

class test_utils {
public:
  // Column-major GEMM reference implementation
  // All matrices are stored in column-major order:
  // - A[i,k] = A[i + k*lda]
  // - B[k,j] = B[k + j*ldb]
  // - C[i,j] = C[i + j*ldc]
  //
  // transA/transB control the operation:
  // - transA='N': A is M×K, access A[i + k*lda]
  // - transA='T': A is K×M, access A[k + i*lda]
  // - transB='N': B is K×N, access B[k + j*ldb]
  // - transB='T': B is N×K, access B[j + k*ldb]
  static void gemm_ref(const KML_FP *A, const KML_FP *B, KML_FP *C, int M, int N, int K, int lda, int ldb, int ldc, bool ACC, char transA = 'N', char transB = 'N') {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float sum = ACC ? C[i + j * ldc] : 0;
        for (int k = 0; k < K; ++k) {
          float a_val, b_val;
          // Column-major access for A
          if (transA == 'N') {
            // A is M×K: A[i,k] = A[i + k*lda]
            a_val = A[i + k * lda];
          } else {
            // A is K×M (transposed): A^T[i,k] = A[k,i] = A[k + i*lda]
            a_val = A[k + i * lda];
          }
          // Column-major access for B
          if (transB == 'N') {
            // B is K×N: B[k,j] = B[k + j*ldb]
            b_val = B[k + j * ldb];
          } else {
            // B is N×K (transposed): B^T[k,j] = B[j,k] = B[j + k*ldb]
            b_val = B[j + k * ldb];
          }
          sum += a_val * b_val;
        }
        // Column-major access for C
        C[i + j * ldc] = sum;
      }
    }
  }

  #ifdef BF16
  static void gemm_ref(const __bf16 *A, const __bf16 *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool ACC, char transA = 'N', char transB = 'N') {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float sum = ACC ? C[i + j * ldc] : 0;
        for (int k = 0; k < K; ++k) {
          float a_val, b_val;
          // Column-major access for A
          if (transA == 'N') {
            a_val = bf16_to_float(A[i + k * lda]);
          } else {
            a_val = bf16_to_float(A[k + i * lda]);
          }
          // Column-major access for B
          if (transB == 'N') {
            b_val = bf16_to_float(B[k + j * ldb]);
          } else {
            b_val = bf16_to_float(B[j + k * ldb]);
          }
          sum += a_val * b_val;
        }
        C[i + j * ldc] = sum;
      }
    }
  }
  #endif

  static void gemm_ref(const __fp16 *A, const __fp16 *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool ACC, char transA = 'N', char transB = 'N') {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float sum = ACC ? C[i + j * ldc] : 0;
        for (int k = 0; k < K; ++k) {
          float a_val, b_val;
          if (transA == 'N') {
            a_val = A[i + k * lda];
          } else {
            a_val = A[k + i * lda];
          }
          if (transB == 'N') {
            b_val = B[k + j * ldb];
          } else {
            b_val = B[j + k * ldb];
          }
          sum += a_val * b_val;
        }
        C[i + j * ldc] = sum;
      }
    }
  }

#ifndef FP32
  static void gemm_ref(const float *A, const float *B, float *C, int M, int N, int K, int lda, int ldb, int ldc, bool ACC, char transA = 'N', char transB = 'N') {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float sum = ACC ? C[i + j * ldc] : 0;
        for (int k = 0; k < K; ++k) {
          float a_val, b_val;
          // Column-major access for A
          if (transA == 'N') {
            // A is M×K: A[i,k] = A[i + k*lda]
            a_val = A[i + k * lda];
          } else {
            // A is K×M (transposed): A^T[i,k] = A[k,i] = A[k + i*lda]
            a_val = A[k + i * lda];
          }
          // Column-major access for B
          if (transB == 'N') {
            // B is K×N: B[k,j] = B[k + j*ldb]
            b_val = B[k + j * ldb];
          } else {
            // B is N×K (transposed): B^T[k,j] = B[j,k] = B[j + k*ldb]
            b_val = B[j + k * ldb];
          }
          sum += a_val * b_val;
        }
        // Column-major access for C
        C[i + j * ldc] = sum;
      }
    }
  }
#endif
  
  static bool is_same_matrix(const KML_FP *C1, const KML_FP *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float c1_val = C1[i + j * ldc];
        float c2_val = C2[i + j * ldc];
        float diff = fabs(c1_val - c2_val);
        if (diff > atol && (fabs(c1_val) > 1e-10 ? diff / fabs(c1_val) > rtol : diff > atol)) {
          return false;
        }
      }
    }
    return true;
  }

#ifndef FP32
  static bool is_same_matrix(const float *C1, const float *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float c1_val = C1[i + j * ldc];
        float c2_val = C2[i + j * ldc];
        float diff = fabs(c1_val - c2_val);
        if (diff > atol && (fabs(c1_val) > 1e-10 ? diff / fabs(c1_val) > rtol : diff > atol)) {
          return false;
        }
      }
    }
    return true;
  }
#endif

  static int diff_index(const KML_FP *C1, const KML_FP *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float c1_val = C1[i + j * ldc];
        float c2_val = C2[i + j * ldc];
        float diff = fabs(c1_val - c2_val);
        if (diff > atol && (fabs(c1_val) > 1e-10 ? diff / fabs(c1_val) > rtol : diff > atol)) {
          return i + j * ldc;
        }
      }
    }
    return -1;
  }

#ifndef FP32
  static int diff_index(const float *C1, const float *C2, int M, int N, int ldc, float rtol, float atol) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float c1_val = C1[i + j * ldc];
        float c2_val = C2[i + j * ldc];
        float diff = fabs(c1_val - c2_val);
        if (diff > atol && (fabs(c1_val) > 1e-10 ? diff / fabs(c1_val) > rtol : diff > atol)) {
          return i + j * ldc;
        }
      }
    }
    return -1;
  }
#endif

  static int print_diff(const KML_FP *C1, const KML_FP *C2, int M, int N, int ldc) {
    printf("\n=== Diff Matrix (refC vs ourC) ===\n");
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float diff = fabs(C1[i + j * ldc] - C2[i + j * ldc]);
        if (diff > 1e-6) {
          printf("First diff: [i=%d,j=%d] ref=%.6f our=%.6f diff=%.6f\n", 
                 i, j, C1[i + j * ldc], C2[i + j * ldc], diff);
          return -1;
        }
      }
    }
    return -1;
  }

#ifndef FP32
  static int print_diff(const float *C1, const float *C2, int M, int N, int ldc) {
    printf("\n=== Diff Matrix (refC vs ourC) ===\n");
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) {
        float diff = fabs(C1[i + j * ldc] - C2[i + j * ldc]);
        if (diff > 1e-6) {
          printf("First diff: [i=%d,j=%d] ref=%.6f our=%.6f diff=%.6f\n", 
                 i, j, C1[i + j * ldc], C2[i + j * ldc], diff);
          return -1;
        }
      }
    }
    return -1;
  }
#endif

  #ifndef FP16
  static void init(KML_FP *buf, int size, int start_value = 1) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = 1.0f * rand() / RAND_MAX;
    }
  }
  #endif

  #ifdef BF16
  static void init(__bf16 *buf, int size, int start_value = 1) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = float_to_bf16(1.0f * rand() / RAND_MAX);
    }
  }
  #endif

  static void init(__fp16 *buf, int size, int start_value = 1) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = float_to_fp16(1.0f * rand() / RAND_MAX);
    }
  }

#ifndef FP32
  static void init(float *buf, int size, int start_value = 1) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        buf[i] = 1.0f * rand() / RAND_MAX;
    }
  }
#endif

  #ifndef FP16
  static void print_matrix(const KML_FP *matrix, int rows, int cols, int lda, const char* name) {
    printf("\n=== %s Matrix ===\n", name);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%.6f ", matrix[i + j * lda]);
      }
      printf("\n");
    }
    printf("\n");
  }
  #endif

  #ifdef BF16
  static void print_matrix(const __bf16 *matrix, int rows, int cols, int lda, const char* name) {
    printf("\n=== %s Matrix ===\n", name);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%.6f ", bf16_to_float(matrix[i + j * lda]));
      }
      printf("\n");
    }
    printf("\n");
  }
  #endif

  static void print_matrix(const __fp16 *matrix, int rows, int cols, int lda, const char* name) {
    printf("\n=== %s Matrix ===\n", name);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        printf("%.6f ", (float)matrix[i + j * lda]);
      }
      printf("\n");
    }
    printf("\n");
  }

#ifndef FP32
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
#endif

};
#endif
