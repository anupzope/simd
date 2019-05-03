#ifndef SIMD_SIMD256X86DEF_HPP
#define SIMD_SIMD256X86DEF_HPP

#include <immintrin.h>

//#include <iostream>

namespace simd {

//void print_mm128i_epi32(__m128i var) {
//  int32_t *val = (int32_t*) &var;
//  for(int i = 0; i < 3; ++i) {
//    std::cout << val[i] << " ";
//  }
//  std::cout << val[3] << std::endl;
//}

//void print_mm256i_epi32(__m256i var) {
//  int32_t *val = (int32_t*) &var;
//  for(int i = 0; i < 7; ++i) {
//    std::cout << val[i] << " ";
//  }
//  std::cout << val[7] << std::endl;
//}

//void print_mm256i_epi64(__m256i var) {
//  int64_t *val = (int64_t*) &var;
//  for(int i = 0; i < 3; ++i) {
//    std::cout << val[i] << " ";
//  }
//  std::cout << val[3] << std::endl;
//}

#if SIMD256

static_assert(sizeof(int) == 4, "Unsupported int size");
static_assert(sizeof(long) == 8, "Unsupported long size");
static_assert(sizeof(float) == 4, "Unsupported float size");
static_assert(sizeof(double) == 8, "Unsupported double size");

#if SIMD512OFFLOAD

template<>
struct defaults<int> {
  static constexpr int nway = 16;
};

template<>
struct defaults<long> {
  static constexpr int nway = 8;
};

template<>
struct defaults<float> {
  static constexpr int nway = 16;
};

template<>
struct defaults<double> {
  static constexpr int nway = 8;
};

template<int NW>
struct type_traits<int, NW> {
  static_assert(NW%simd::defaults<int>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<int>::nway > 0, "Invalid NW");
  
  typedef int base_type;
  typedef __m256i register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = 2*(NW/simd::defaults<int>::nway);
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

template<int NW>
struct type_traits<long, NW> {
  static_assert(NW%simd::defaults<long>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<long>::nway > 0, "Invalid NW");
  
  typedef long base_type;
  typedef __m256i register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = 2*(NW/simd::defaults<long>::nway);
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

template<int NW>
struct type_traits<float, NW> {
  static_assert(NW%simd::defaults<float>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<float>::nway > 0, "Invalid NW");
  
  typedef float base_type;
  typedef __m256 register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = 2*(NW/simd::defaults<float>::nway);
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

template<int NW>
struct type_traits<double, NW> {
  static_assert(NW%simd::defaults<double>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<double>::nway > 0, "Invalid NW");
  
  typedef double base_type;
  typedef __m256d register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = 2*(NW/simd::defaults<double>::nway);
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

#else // #if SIMD512OFFLOAD

template<>
struct defaults<int> {
  static constexpr int nway = 8;
};

template<>
struct defaults<long> {
  static constexpr int nway = 4;
};

template<>
struct defaults<float> {
  static constexpr int nway = 8;
};

template<>
struct defaults<double> {
  static constexpr int nway = 4;
};

template<int NW>
struct type_traits<int, NW> {
  static_assert(NW%simd::defaults<int>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<int>::nway > 0, "Invalid NW");
  
  typedef int base_type;
  typedef __m256i register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = NW/simd::defaults<int>::nway;
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

template<int NW>
struct type_traits<long, NW> {
  static_assert(NW%simd::defaults<long>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<long>::nway > 0, "Invalid NW");
  
  typedef long base_type;
  typedef __m256i register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = NW/simd::defaults<long>::nway;
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

template<int NW>
struct type_traits<float, NW> {
  static_assert(NW%simd::defaults<float>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<float>::nway > 0, "Invalid NW");
  
  typedef float base_type;
  typedef __m256 register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = NW/simd::defaults<float>::nway;
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

template<int NW>
struct type_traits<double, NW> {
  static_assert(NW%simd::defaults<double>::nway == 0, "Invalid NW");
  static_assert(NW/simd::defaults<double>::nway > 0, "Invalid NW");
  
  typedef double base_type;
  typedef __m256d register_type;
  typedef __m256i mask_register_type;
  
  static constexpr int num_regs = NW/simd::defaults<double>::nway;
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

#endif // #if SIMD512OFFLOAD

/* Function to convert value to scalar boolean value */

/*template<typename T, int NW, int W>
inline bool to_bool(const pack<T, NW, __m256i, int, W> &op) {
  bool res = true;
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
    res = res && (bool)_mm256_testz_si256(op(i), op(i));
  }
  return !res;
}

template<typename T, int NW, int W>
inline bool to_bool(const pack<T, NW, __m256i, long, W> &op) {
  bool res = true;
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
    res  = res && (bool)_mm256_testz_si256(op(i), op(i));
  }
  return !res;
}

template<typename T, int NW, int W>
inline bool to_bool(const pack<T, NW, __m256, float, W> &op) {
  bool res = true;
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    res = res && (bool)_mm256_testz_ps(op(i), op(i));
  }
  return !res;
}

template<typename T, int NW, int W>
inline bool to_bool(const pack<T, NW, __m256d, double, W> &op) {
  bool res = true;
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    res = res && (bool)_mm256_testz_pd(op(i), op(i));
  }
  return !res;
}*/

/* Functions to get and set mask register bits */

//inline __m256i get_mask(int mask) {
//  __m256i vmask(_mm256_set1_epi32(mask));
//  const __m256i shuffle(_mm256_setr_epi64x(0x0000000000000000,
//      0x0101010101010101, 0x0202020202020202, 0x0303030303030303));
//  vmask = _mm256_shuffle_epi8(vmask, shuffle);
//  const __m256i bit_mask(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
//  vmask = _mm256_or_si256(vmask, bit_mask);
//  return _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
//}

//inline __m256i int2mask_epi32(int bitmask) {
//#if defined(SIMD_AVX2)
//  return _mm256_cmpeq_epi32(
//    _mm256_set1_epi32(-1),
//    _mm256_or_si256(
//      _mm256_set1_epi32(bitmask),
//      _mm256_set_epi32(0xffffff7f, 0xffffffbf, 0xffffffdf, 0xffffffef, 0xfffffff7, 0xfffffffb, 0xfffffffd, 0xfffffffe)
//    )
//  );
//#else
//  __m128i vmask = _mm_set1_epi32(bitmask);
//  const __m128i allset = _mm_set1_epi32(-1);
//  return _mm256_set_m128i(
//    _mm_cmpeq_epi32(allset, _mm_or_si128(vmask, _mm_set_epi32(0xffffff7f, 0xffffffbf, 0xffffffdf, 0xffffffef))),
//    _mm_cmpeq_epi32(allset, _mm_or_si128(vmask, _mm_set_epi32(0xfffffff7, 0xfffffffb, 0xfffffffd, 0xfffffffe)))
//  );
//#endif
//}

//inline __m256i int2mask_epi64(int bitmask) {
//#if defined(SIMD_AVX2)
//  return _mm256_cmpeq_epi64(
//    _mm256_set1_epi64x(-1),
//    _mm256_or_si256(
//      _mm256_set1_epi64x(bitmask),
//      _mm256_set_epi64x(0xfffffffffffffff7, 0xfffffffffffffffb, 0xfffffffffffffffd, 0xfffffffffffffffe)
//    )
//  );
//#else
//  __m128i vmask = _mm_set1_epi64x(bitmask);
//  const __m128i allset = _mm_set1_epi64x(-1);
//  return _mm256_set_m128i(
//    _mm_cmpeq_epi64(allset, _mm_or_si128(vmask, _mm_set_epi64x(0xfffffffffffffff7, 0xfffffffffffffffb))),
//    _mm_cmpeq_epi64(allset, _mm_or_si128(vmask, _mm_set_epi64x(0xfffffffffffffffd, 0xfffffffffffffffe)))
//  );
//#endif
//}

/* Functions to convert between 32 bit and 64 bit masks */

inline __m256i mask_cvtepi32lo_epi64(__m256i mask) {
  __m128i lo = _mm256_castsi256_si128(mask);
#if defined(SIMD_AVX2)
  return _mm256_cvtepi32_epi64(lo);
#else
  return _mm256_set_m128i(
    _mm_cvtepi32_epi64(_mm_srli_si128(lo, 8)), 
    _mm_cvtepi32_epi64(lo)
  );
#endif
}

inline __m256i mask_cvtepi32hi_epi64(__m256i mask) {
  __m128i hi = _mm256_extractf128_si256(mask, 1);
#if defined(SIMD_AVX2)
  return _mm256_cvtepi32_epi64(hi);
#else
  return _mm256_set_m128i(
    _mm_cvtepi32_epi64(_mm_srli_si128(hi, 8)),
    _mm_cvtepi32_epi64(hi)
  );
#endif
}

inline __m128i mask_cvtepi64_epi32(__m256i mask) {
#if defined(SIMD_AVX2)
  return _mm256_castsi256_si128(
    _mm256_permute4x64_epi64(
      _mm256_shuffle_epi32(mask, _MM_SHUFFLE(3,1,3,1)), _MM_SHUFFLE(2,0,2,0)
    )
  );
#else
  __m256d temp = _mm256_castps_pd(
    _mm256_shuffle_ps(_mm256_castsi256_ps(mask), _mm256_castsi256_ps(mask), _MM_SHUFFLE(3,1,3,1))
  );
  return _mm_castpd_si128(_mm_unpacklo_pd(_mm256_castpd256_pd128(temp), _mm256_extractf128_pd(temp, 1)));
#endif
}

/* Masked load and store functions */

inline __m128i maskload128_epi32(int const *mem, __m128i mask) {
#if defined(SIMD_AVX2)
  return _mm_maskload_epi32(mem, mask);
#else
  return _mm_castps_si128(_mm_maskload_ps(reinterpret_cast<float const*>(mem), mask));
#endif
}

inline void maskstore128_epi32(int *mem, __m128i mask, __m128i op) {
#if defined(SIMD_AVX2)
  _mm_maskstore_epi32(mem, mask, op);
#else
  _mm_maskstore_ps(reinterpret_cast<float*>(mem), mask, _mm_castsi128_ps(op));
#endif
}

inline __m256i maskload256_epi32(int const *mem, __m256i mask) {
#if defined(SIMD_AVX2)
  return _mm256_maskload_epi32(mem, mask);
#else
  return _mm256_castps_si256(_mm256_maskload_ps(reinterpret_cast<float const*>(mem), mask));
#endif
}

inline void maskstore256_epi32(int *mem, __m256i mask, __m256i op) {
#if defined(SIMD_AVX2)
  _mm256_maskstore_epi32(mem, mask, op);
#else
  _mm256_maskstore_ps(reinterpret_cast<float*>(mem), mask, _mm256_castsi256_ps(op));
#endif
}

inline __m128i maskload128_epi64(__int64 const *mem, __m128i mask) {
#if defined(SIMD_AVX2)
  return _mm_maskload_epi64(mem, mask);
#else
  return _mm_castpd_si128(_mm_maskload_pd(reinterpret_cast<double const*>(mem), mask));
#endif
}

inline void maskstore128_epi64(__int64 *mem, __m128i mask, __m128i op) {
#if defined(SIMD_AVX2)
  _mm_maskstore_epi64(mem, mask, op);
#else
  _mm_maskstore_pd(reinterpret_cast<double*>(mem), mask, _mm_castsi128_pd(op));
#endif
}

inline __m256i maskload256_epi64(__int64 const *mem, __m256i mask) {
#if defined(SIMD_AVX2)
  return _mm256_maskload_epi64(mem, mask);
#else
  return _mm256_castpd_si256(_mm256_maskload_pd(reinterpret_cast<double const*>(mem), mask));
#endif
}

inline void maskstore256_epi64(__int64 *mem, __m256i mask, __m256i op) {
#if defined(SIMD_AVX2)
  _mm256_maskstore_epi64(mem, mask, op);
#else
  _mm256_maskstore_pd(reinterpret_cast<double*>(mem), mask, _mm256_castsi256_pd(op));
#endif
}

inline __m128 maskload128_ps(float const *mem, __m128i mask) {
  return _mm_maskload_ps(mem, mask);
}

inline void maskstore128_ps(float *mem, __m128i mask, __m128 op) {
  _mm_maskstore_ps(mem, mask, op);
}

inline __m256 maskload256_ps(float const *mem, __m256i mask) {
  return _mm256_maskload_ps(mem, mask);
}

inline void maskstore256_ps(float *mem, __m256i mask, __m256 op) {
  _mm256_maskstore_ps(mem, mask, op);
}

inline __m128d maskload128_pd(double const *mem, __m128i mask) {
  return _mm_maskload_pd(mem, mask);
}

inline void maskstore128_pd(double *mem, __m128i mask, __m128d op) {
  _mm_maskstore_pd(mem, mask, op);
}

inline __m256d maskload256_pd(double const *mem, __m256i mask) {
  return _mm256_maskload_pd(mem, mask);
}

inline void maskstore256_pd(double *mem, __m256i mask, __m256d op) {
  _mm256_maskstore_pd(mem, mask, op);
}

/* Conversion functions */

// int -> long
inline __m256i cvt256_epi32_epi64(__m128i op) {
#if defined(SIMD_AVX2)
  return _mm256_cvtepi32_epi64(op);
#else
  return _mm256_set_m128i(
    _mm_cvtepi32_epi64(_mm_srli_si128(op, 8)),
    _mm_cvtepi32_epi64(op)
  );
#endif
}

// int -> float
inline __m256 cvt256_epi32_ps(__m256i op) {
  return _mm256_cvtepi32_ps(op);
}

// int -> double
inline __m256d cvt256_epi32_pd(__m128i op) {
  return _mm256_cvtepi32_pd(op);
}

// long -> int
inline __m128i cvt256_epi64_epi32(__m256i op) {
#if defined(SIMD_AVX2)
  __m256i temp = _mm256_shuffle_epi32(op, _MM_SHUFFLE(3,1,2,0));
  __m128i lo = _mm256_castsi256_si128(temp);
  __m128i hi = _mm256_extractf128_si256(temp, 1);
  return _mm_unpacklo_epi64(lo, hi);
#else
  __m128i lo = _mm_shuffle_epi32(_mm256_castsi256_si128(op), _MM_SHUFFLE(3,1,2,0));
  __m128i hi = _mm_shuffle_epi32(_mm256_extractf128_si256(op, 1), _MM_SHUFFLE(3,1,2,0));
  return _mm_unpacklo_epi64(lo, hi);
#endif
}

// long -> float
inline __m128 cvt256_epi64_ps(__m256i op) {
  __int64 a0 = _mm256_extract_epi64(op, 0);
  __int64 a1 = _mm256_extract_epi64(op, 1);
  __int64 a2 = _mm256_extract_epi64(op, 2);
  __int64 a3 = _mm256_extract_epi64(op, 3);
  __m128 t = _mm_setzero_ps();
  return _mm_shuffle_ps(
    _mm_unpacklo_ps(_mm_cvtsi64_ss(t, a0), _mm_cvtsi64_ss(t, a1)),
    _mm_unpacklo_ps(_mm_cvtsi64_ss(t, a2), _mm_cvtsi64_ss(t, a3)),
    _MM_SHUFFLE(1,0,1,0)
  );
}

// long -> double
inline __m256d cvt256_epi64_pd(__m256i op) {
  __int64 a0 = _mm256_extract_epi64(op, 0);
  __int64 a1 = _mm256_extract_epi64(op, 1);
  __int64 a2 = _mm256_extract_epi64(op, 2);
  __int64 a3 = _mm256_extract_epi64(op, 3);
  __m128d t;
  return _mm256_set_m128d(
    _mm_unpacklo_pd(_mm_cvtsi64_sd(t, a2), _mm_cvtsi64_sd(t, a3)),
    _mm_unpacklo_pd(_mm_cvtsi64_sd(t, a0), _mm_cvtsi64_sd(t, a1))
  );
}

// float -> int
inline __m256i cvtt256_ps_epi32(__m256 op) {
  return _mm256_cvttps_epi32(op);
}

// float -> int
inline __m256i cvt256_ps_epi32(__m256 op) {
  return _mm256_cvtps_epi32(op);
}

// float -> long
inline __m256i cvtt256_ps_epi64(__m128 op) {
  __int64 a0 = _mm_cvttss_si64(op);
  __int64 a1 = _mm_cvttss_si64(_mm_permute_ps(op, _MM_SHUFFLE(3,2,1,1)));
  __int64 a2 = _mm_cvttss_si64(_mm_permute_ps(op, _MM_SHUFFLE(3,2,1,2)));
  __int64 a3 = _mm_cvttss_si64(_mm_permute_ps(op, _MM_SHUFFLE(3,2,1,3)));
  return _mm256_set_epi64x(a3, a2, a1, a0);
}

// float -> long
inline __m256i cvt256_ps_epi64(__m128 op) {
  __int64 a0 = _mm_cvtss_si64(op);
  __int64 a1 = _mm_cvtss_si64(_mm_permute_ps(op, _MM_SHUFFLE(3,2,1,1)));
  __int64 a2 = _mm_cvtss_si64(_mm_permute_ps(op, _MM_SHUFFLE(3,2,1,2)));
  __int64 a3 = _mm_cvtss_si64(_mm_permute_ps(op, _MM_SHUFFLE(3,2,1,3)));
  return _mm256_set_epi64x(a3, a2, a1, a0);
}

// float -> double
inline __m256d cvt256_ps_pd(__m128 op) {
  return _mm256_cvtps_pd(op);
}

// double -> int
inline __m128i cvtt256_pd_epi32(__m256d op) {
  return _mm256_cvttpd_epi32(op);
}

// double -> int
inline __m128i cvt256_ps_epi32(__m256d op) {
  return _mm256_cvtpd_epi32(op);
}

// double -> long
inline __m256i cvtt256_pd_epi64(__m256d op) {
  __m256d op2 = _mm256_permute_pd(op, 0b1111);
  __m128d t1 = _mm256_extractf128_pd(op, 1);
  __m128d t2 = _mm256_extractf128_pd(op2, 1);
  
  __int64 a0 = _mm_cvttsd_si64x(_mm256_castpd256_pd128(op));
  __int64 a1 = _mm_cvttsd_si64x(_mm256_castpd256_pd128(op2));
  __int64 a2 = _mm_cvttsd_si64x(t1);
  __int64 a3 = _mm_cvttsd_si64x(t2);
  
  return _mm256_set_epi64x(a3, a2, a1, a0);
}

// double -> long
inline __m256i cvt256_pd_epi64(__m256d op) {
  __m256d op2 = _mm256_permute_pd(op, 0b1111);
  __m128d t1 = _mm256_extractf128_pd(op, 1);
  __m128d t2 = _mm256_extractf128_pd(op2, 1);
  
  __int64 a0 = _mm_cvtsd_si64x(_mm256_castpd256_pd128(op));
  __int64 a1 = _mm_cvtsd_si64x(_mm256_castpd256_pd128(op2));
  __int64 a2 = _mm_cvtsd_si64x(t1);
  __int64 a3 = _mm_cvtsd_si64x(t2);
  
  return _mm256_set_epi64x(a3, a2, a1, a0);
}

// double -> float
inline __m128 cvt256_pd_ps(__m256d op) {
  return _mm256_cvtpd_ps(op);
}

// generator for base type of a mask<T, NW>
template<typename T>
struct mask_base_type_generator {
  using type = typename std::conditional<
    sizeof(typename std::remove_all_extents<T>::type) == 4,
    __int32,
    __int64
  >::type;
};

/* mask_set(...) implementations  */

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, bool value) {
  const __int64 val = (value ? -1 : 0);
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    m(i) = _mm256_set1_epi64x(val);
  }
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, bool value) {
  const __int64 val = (value ? -1 : 0);
  m(ari) = _mm256_set1_epi64x(val);
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, vindex avi, bool value) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  const base_type val = (value ? -1 : 0);
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  for(int i = bstart; i < bend; ++i) {
    values[i] = val;
  }
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, bindex abi, bool value) {
  int ari = abi / type_traits<T, NW>::num_bvals_per_reg;
  int b = abi % type_traits<T, NW>::num_bvals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  const base_type val = (value ? -1 : 0);
  
  values[b] = val;
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, vindex v, bool value) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  const base_type val = (value ? -1 : 0);
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  for(int i = bstart; i < bend; ++i) {
    values[i] = val;
  }
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, bindex b, bool value) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  const base_type val = (value ? -1 : 0);
  
  values[b] = val;
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, vindex avi, bindex b, bool value) {
  int ari = avi / type_traits<T>::num_vals_per_reg;
  int v = avi % type_traits<T>::num_vals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  const base_type val = (value ? -1 : 0);
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  values[bstart+b] = val;
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, vindex v, bindex b, bool value) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  const base_type val = (value ? -1 : 0);
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  values[bstart+b] = val;
  
  return m;
}

/* mask_reset(...) implementations */

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m) {
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    m(i) = _mm256_setzero_si256();
  }
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari) {
  m(ari) = _mm256_setzero_si256();
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, vindex avi) {
  return mask_set(m, avi, false);
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, bindex abi) {
  return mask_set(m, abi, false);
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari, vindex v) {
  return mask_set(m, ari, v, false);
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari, bindex b) {
  return mask_set(m, ari, b, false);
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, vindex avi, bindex b) {
  return mask_set(m, avi, b, false);
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari, vindex v, bindex b) {
  return mask_set(m, ari, v, b, false);
}

/* mask_flip(...) implementations */

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m) {
  __m256i one = _mm256_set1_epi64x(-1);
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    m(i) = _mm256_andnot_si256(m(i), one);
#else
    m(i) = _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(m(i)), _mm256_castsi256_ps(one)));
#endif
  }
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari) {
  __m256i one = _mm256_set1_epi64x(-1);
#if defined(SIMD_AVX2)
  m(ari) = _mm256_andnot_si256(m(ari), one);
#else
  m(ari) = _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(m(ari)), _mm256_castsi256_ps(one)));
#endif
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  for(int i = bstart; i < bend; ++i) {
    values[i] = ~values[i];
  }
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, bindex abi) {
  int ari = abi / type_traits<T, NW>::num_bvals_per_reg;
  int b = abi % type_traits<T, NW>::num_bvals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  
  values[b] = ~values[b];
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, vindex v) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  for(int i = bstart; i < bend; ++i) {
    values[i] = ~values[i];
  }
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, bindex b) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  
  values[b] = ~values[b];
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, vindex avi, bindex b) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  values[bstart+b] = ~values[bstart+b];
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, vindex v, bindex b) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type *values = reinterpret_cast<base_type*>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  values[bstart+b] = ~values[bstart+b];
  
  return m;
}

/* mask_all(...) implementations */

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m) {
  bool value = true;
  __m256i one = _mm256_set1_epi64x(-1);
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    value = value && (bool)_mm256_testc_si256(m(i), one);
  }
  return value;
}

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, rindex ari) {
  __m256i one = _mm256_set1_epi64x(-1);
  return (bool)_mm256_testc_si256(m(ari), one);
}

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const *>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  
  bool value = true;
  for(int i = bstart; i < bend; ++i) {
    value = value && (bool)(values[i]);
  }
  
  return value;
}

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, rindex ari, vindex v) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const *>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  
  bool value = true;
  for(int i = bstart; i < bend; ++i) {
    value = value && (bool)(values[i]);
  }
  
  return value;
}

/* mask_any(...) implementations */

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m) {
  bool value = false;
  __m256i one = _mm256_set1_epi64x(-1);
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    value = value || !(bool)_mm256_testz_si256(m(i), one);
  }
  return value;
}

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, rindex ari) {
  __m256i one = _mm256_set1_epi64x(-1);
  return !(bool)_mm256_testz_si256(m(ari), one);
}

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const *>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  
  bool value = false;
  for(int i = bstart; i < bend; ++i) {
    value = value || (bool)(values[i]);
  }
  
  return value;
}

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, rindex ari, vindex v) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const *>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  
  bool value = false;
  for(int i = bstart; i < bend; ++i) {
    value = value || (bool)(values[i]);
  }
  
  return value;
}

/* mask_none(...) implementations */

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m) {
  bool value = true;
  __m256i one = _mm256_set1_epi64x(-1);
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    value = value && (bool)_mm256_testz_si256(m(i), one);
  }
  return value;
}

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, rindex ari) {
  __m256i one = _mm256_set1_epi64x(-1);
  return (bool)_mm256_testz_si256(m(ari), one);
}

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const *>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  
  bool value = false;
  for(int i = bstart; i < bend; ++i) {
    value = value || (bool)(values[i]);
  }
  
  return !value;
}

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, rindex ari, vindex v) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const *>(&m(ari));
  
  int bstart = v * type_traits<T, NW>::num_bvals_per_val;
  int bend = bstart + type_traits<T, NW>::num_bvals_per_val;
  
  bool value = false;
  for(int i = bstart; i < bend; ++i) {
    value = value || (bool)(values[i]);
  }
  
  return !value;
}

/* mask_test(...) implementations */

template<typename T, int NW>
bool mask_test(const mask<T, NW> &m, bindex abi) {
  int ari = abi / type_traits<T, NW>::num_bvals_per_reg;
  int b = abi % type_traits<T, NW>::num_bvals_per_reg;
  
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const*>(&m(ari));
  
  return (bool)(values[b]);
}

template<typename T, int NW>
bool mask_test(const mask<T, NW> &m, rindex ari, bindex b) {
  using base_type = typename mask_base_type_generator<T>::type;
  base_type const *values = reinterpret_cast<base_type const*>(&m(ari));
  
  return (bool)(values[b]);
}

/* Function to set value to zero */

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m256i, int, W> &op) {
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
    op(i) = _mm256_setzero_si256();
  }
}

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m256i, long, W> &op) {
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
    op(i) = _mm256_setzero_si256();
  }
}

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m256, float, W> &op) {
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    op(i) = _mm256_setzero_ps();
  }
}

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m256d, double, W> &op) {
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    op(i) = _mm256_setzero_pd();
  }
}

/* Function to set all elements of a pack to a scalar value */

template<typename T, int NW, int W, typename BT>
inline void set_scalar(pack<T, NW, __m256i, int, W> &op, BT value, 
  typename std::enable_if<std::is_arithmetic<BT>::value>::type * = nullptr
) {
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
    op(i) = _mm256_set1_epi32((int)value);
  }
}

template<typename T, int NW, int W, typename BT>
inline void set_scalar(pack<T, NW, __m256i, long, W> &op, BT value, 
  typename std::enable_if<std::is_arithmetic<BT>::value>::type * = nullptr
) {
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
    op(i) = _mm256_set1_epi64x((long)value);
  }
}

template<typename T, int NW, int W, typename BT>
inline void set_scalar(pack<T, NW, __m256, float, W> &op, BT value, 
  typename std::enable_if<std::is_arithmetic<BT>::value>::type * = nullptr
) {
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    op(i) = _mm256_set1_ps((float)value);
  }
}

template<typename T, int NW, int W, typename BT>
inline void set_scalar(pack<T, NW, __m256d, double, W> &op, BT value, 
  typename std::enable_if<std::is_arithmetic<BT>::value>::type * = nullptr
) {
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    op(i) = _mm256_set1_pd((double)value);
  }
}

/* Arithmetic operator: + */

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, int, W> operator+(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  pack<T, NW, __m256i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_add_epi32(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_add_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_add_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, long, W> operator+(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  pack<T, NW, __m256i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_add_epi64(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_add_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_add_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256, float, W> operator+(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_add_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256d, double, W> operator+(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_add_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: - */

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, int, W> operator-(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  pack<T, NW, __m256i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_sub_epi32(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_sub_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_sub_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, long, W> operator-(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  pack<T, NW, __m256i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_sub_epi64(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_sub_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_sub_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256, float, W> operator-(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_sub_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256d, double, W> operator-(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_sub_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: * */

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, int, W> operator*(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  pack<T, NW, __m256i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_mullo_epi32(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_mullo_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_mullo_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, long, W> operator*(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  pack<T, NW, __m256i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    __m256i bswap   = _mm256_shuffle_epi32(rhs(i), 0xB1);
    __m256i prod0  = _mm256_mullo_epi32(lhs(i), bswap);
    __m256i zero    = _mm256_setzero_si256();
    __m256i prod1 = _mm256_hadd_epi32(prod0, zero);
    __m256i prod2 = _mm256_shuffle_epi32(prod1, 0x73);
    __m256i prod3  = _mm256_mul_epu32(lhs(i), rhs(i));
    temp(i) = _mm256_add_epi64(prod3, prod2);
#else
    __m128i al = _mm256_castsi256_si128(lhs(i));
    __m128i ah = _mm256_extractf128_si256(lhs(i), 1);
    __m128i bl = _mm256_castsi256_si128(rhs(i));
    __m128i bh = _mm256_extractf128_si256(rhs(i), 1);
    __m128i bls = _mm_shuffle_epi32(bl, 0xB1);
    __m128i bhs = _mm_shuffle_epi32(bh, 0xB1);
    __m128i prodl0 = _mm_mullo_epi32(al, bls);
    __m128i prodh0 = _mm_mullo_epi32(ah, bhs);
    __m128i zero = _mm_setzero_si128();
    __m128i prodl1 = _mm_hadd_epi32 (prodl0, zero);
    __m128i prodh1 = _mm_hadd_epi32 (prodh0, zero);
    __m128i prodl2 = _mm_shuffle_epi32(prodl1, 0x73);
    __m128i prodh2 = _mm_shuffle_epi32(prodh1, 0x73);
    __m128i prodl3 = _mm_mul_epu32(al, bl);
    __m128i prodh3 = _mm_mul_epu32(ah, bh);
    __m128i prodl = _mm_add_epi64(prodl3, prodl2);
    __m128i prodh = _mm_add_epi64(prodh3, prodh2);
    temp(i) = _mm256_set_m128i(prodh, prodl);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256, float, W> operator*(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_mul_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256d, double, W> operator*(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_mul_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: / */

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, int, W> operator/(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  pack<T, NW, __m256i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
    temp(i) = _mm256_div_epi32(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, long, W> operator/(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  pack<T, NW, __m256i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
    temp(i) = _mm256_div_epi64(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256, float, W> operator/(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_div_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256d, double, W> operator/(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_div_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: % - only for integer types */

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, int, W> operator%(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  pack<T, NW, __m256i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
    temp(i) = _mm256_rem_epi32(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m256i, long, W> operator%(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  pack<T, NW, __m256i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
    temp(i) = _mm256_rem_epi64(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Comparison operator: == */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, int, W>::mask_type operator==(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  typename pack<T, NW, __m256i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_cmpeq_epi32(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_cmpeq_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)), 
      _mm_cmpeq_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, long, W>::mask_type operator==(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  typename pack<T, NW, __m256i, long, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_cmpeq_epi64(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_cmpeq_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_cmpeq_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256, float, W>::mask_type operator==(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  typename pack<T, NW, __m256, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_castps_si256(_mm256_cmp_ps(lhs(i), rhs(i), _CMP_EQ_US));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256d, double, W>::mask_type operator==(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  typename pack<T, NW, __m256d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_castpd_si256(_mm256_cmp_pd(lhs(i), rhs(i), _CMP_EQ_US));
  }
  
  return temp;
}

/* Comparison operator: != */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, int, W>::mask_type operator!=(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  typename pack<T, NW, __m256i, int, W>::mask_type temp;
  
#if defined(SIMD_AVX2)
  __m256i one = _mm256_set1_epi32(-1);
#else
  __m128i one = _mm_set1_epi32(-1);
#endif
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_andnot_si256(_mm256_cmpeq_epi32(lhs(i), rhs(i)), one);
#else
    temp(i) = _mm256_set_m128i(
      _mm_andnot_si128(
        _mm_cmpeq_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)), 
        one
      ), 
      _mm_andnot_si128(
        _mm_cmpeq_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i))), 
        one
      )
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, long, W>::mask_type operator!=(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  typename pack<T, NW, __m256i, long, W>::mask_type temp;
  
#if defined(SIMD_AVX2)
  __m256i one = _mm256_set1_epi64x(-1);
#else
  __m128i one = _mm_set1_epi64x(-1);
#endif
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_andnot_si256(_mm256_cmpeq_epi64(lhs(i), rhs(i)), one);
#else
    temp(i) = _mm256_set_m128i(
      _mm_andnot_si128(
        _mm_cmpeq_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)), 
        one
      ), 
      _mm_andnot_si128(
        _mm_cmpeq_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i))), 
        one
      )
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256, float, W>::mask_type operator!=(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  typename pack<T, NW, __m256, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_castps_si256(_mm256_cmp_ps(lhs(i), rhs(i), _CMP_NEQ_US));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256d, double, W>::mask_type operator!=(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  typename pack<T, NW, __m256d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_castpd_si256(_mm256_cmp_pd(lhs(i), rhs(i), _CMP_NEQ_US));
  }
  
  return temp;
}

/* Comparison operator: < */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, int, W>::mask_type operator<(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  typename pack<T, NW, __m256i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_cmpgt_epi32(rhs(i), lhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_cmplt_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)), 
      _mm_cmplt_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, long, W>::mask_type operator<(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  typename pack<T, NW, __m256i, long, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_cmpgt_epi64(rhs(i), lhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_cmpgt_epi64(_mm256_extractf128_si256(rhs(i), 1), _mm256_extractf128_si256(lhs(i), 1)),
      _mm_cmpgt_epi64(_mm256_castsi256_si128(rhs(i)), _mm256_castsi256_si128(lhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256, float, W>::mask_type operator<(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  typename pack<T, NW, __m256, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_castps_si256(_mm256_cmp_ps(lhs(i), rhs(i), _CMP_NGE_US));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256d, double, W>::mask_type operator<(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  typename pack<T, NW, __m256d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_castpd_si256(_mm256_cmp_pd(lhs(i), rhs(i), _CMP_NGE_US));
  }
  
  return temp;
}

/* Comparison operator: > */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, int, W>::mask_type operator>(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  typename pack<T, NW, __m256i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_cmpgt_epi32(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_cmpgt_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_cmpgt_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, long, W>::mask_type operator>(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  typename pack<T, NW, __m256i, long, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    temp(i) = _mm256_cmpgt_epi64(lhs(i), rhs(i));
#else
    temp(i) = _mm256_set_m128i(
      _mm_cmpgt_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)),
      _mm_cmpgt_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)))
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256, float, W>::mask_type operator>(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  typename pack<T, NW, __m256, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_castps_si256(_mm256_cmp_ps(lhs(i), rhs(i), _CMP_NLE_US));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256d, double, W>::mask_type operator>(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  typename pack<T, NW, __m256d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_castpd_si256(_mm256_cmp_pd(lhs(i), rhs(i), _CMP_NLE_US));
  }
  
  return temp;
}

/* Comparison operator: <= */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, int, W>::mask_type operator<=(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  typename pack<T, NW, __m256i, int, W>::mask_type temp;
  
#if defined(SIMD_AVX2)
  __m256i one = _mm256_set1_epi32(-1);
#else
  __m128i one = _mm_set1_epi32(-1);
#endif
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    //__m256i a = _mm256_cmpgt_epi32(lhs(i), rhs(i));
    temp(i) = _mm256_andnot_si256(_mm256_cmpgt_epi32(lhs(i), rhs(i)), one);
#else
    //__m128i hi = _mm_cmpgt_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1));
    //__m128i lo = _mm_cmpgt_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)));
    temp(i) = _mm256_set_m128i(
      _mm_andnot_si128(
        _mm_cmpgt_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)), 
        one
      ), 
      _mm_andnot_si128(
        _mm_cmpgt_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i))), 
        one
      )
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256i, long, W>::mask_type operator<=(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  typename pack<T, NW, __m256i, long, W>::mask_type temp;
  
#if defined(SIMD_AVX2)
  __m256i one = _mm256_set1_epi64x(-1);
#else
  __m128i one = _mm_set1_epi64x(-1);
#endif
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    //__m256i a = _mm256_cmpgt_epi64(lhs(i), rhs(i));
    temp(i) = _mm256_andnot_si256(_mm256_cmpgt_epi64(lhs(i), rhs(i)), one);
#else
    //__m128i hi = _mm_cmpgt_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1));
    //__m128i lo = _mm_cmpgt_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)));
    temp(i) = _mm256_set_m128i(
      _mm_andnot_si128(
        _mm_cmpgt_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)), 
        one
      ), 
      _mm_andnot_si128(
        _mm_cmpgt_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i))), 
        one
      )
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256, float, W>::mask_type operator<=(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  typename pack<T, NW, __m256, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_castps_si256(_mm256_cmp_ps(lhs(i), rhs(i), _CMP_NGT_US));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m256d, double, W>::mask_type operator<=(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  typename pack<T, NW, __m256d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_castpd_si256(_mm256_cmp_pd(lhs(i), rhs(i), _CMP_NGT_US));
  }
  
  return temp;
}

/* Comparison operator: >= */

template<typename T, int NW, int W>
typename pack<T, NW, __m256i, int, W>::mask_type operator>=(const pack<T, NW, __m256i, int, W> &lhs, const pack<T, NW, __m256i, int, W> &rhs) {
  typename pack<T, NW, __m256i, int, W>::mask_type temp;
  
#if defined(SIMD_AVX2)
  __m256i one = _mm256_set1_epi32(-1);
#else
  __m128i one = _mm_set1_epi32(-1);
#endif
  
  for(int i = 0; i < pack<T, NW, __m256i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    //__m256i a = _mm256_cmpgt_epi32(rhs(i), lhs(i));
    temp(i) = _mm256_andnot_si256(_mm256_cmpgt_epi32(rhs(i), lhs(i)), one);
#else
    //__m128i hi = _mm_cmplt_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1));
    //__m128i lo = _mm_cmplt_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)));
    temp(i) = _mm256_set_m128i(
      _mm_andnot_si128(
        _mm_cmplt_epi32(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1)), 
        one
      ), 
      _mm_andnot_si128(
        _mm_cmplt_epi32(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i))), 
        one
      )
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
typename pack<T, NW, __m256i, long, W>::mask_type operator>=(const pack<T, NW, __m256i, long, W> &lhs, const pack<T, NW, __m256i, long, W> &rhs) {
  typename pack<T, NW, __m256i, long, W>::mask_type temp;
  
#if defined(SIMD_AVX2)
  __m256i one = _mm256_set1_epi64x(-1);
#else
  __m128i one = _mm_set1_epi64x(-1);
#endif
  
  for(int i = 0; i < pack<T, NW, __m256i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX2)
    //__m256i a = _mm256_cmpgt_epi64(rhs(i), lhs(i));
    temp(i) = _mm256_andnot_si256(_mm256_cmpgt_epi64(rhs(i), lhs(i)), one);
#else
    //__m128i hi = _mm_cmplt_epi64(_mm256_extractf128_si256(lhs(i), 1), _mm256_extractf128_si256(rhs(i), 1));
    //__m128i lo = _mm_cmplt_epi64(_mm256_castsi256_si128(lhs(i)), _mm256_castsi256_si128(rhs(i)));
    temp(i) = _mm256_set_m128i(
      _mm_andnot_si128(
        _mm_cmpgt_epi64(_mm256_extractf128_si256(rhs(i), 1), _mm256_extractf128_si256(lhs(i), 1)), 
        one
      ), 
      _mm_andnot_si128(
        _mm_cmpgt_epi64(_mm256_castsi256_si128(rhs(i)), _mm256_castsi256_si128(lhs(i))), 
        one
      )
    );
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
typename pack<T, NW, __m256, float, W>::mask_type operator>=(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  typename pack<T, NW, __m256, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_castps_si256(_mm256_cmp_ps(lhs(i), rhs(i), _CMP_NLT_US));
  }
  
  return temp;
}

template<typename T, int NW, int W>
typename pack<T, NW, __m256d, double, W>::mask_type operator>=(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  typename pack<T, NW, __m256d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_castpd_si256(_mm256_cmp_pd(lhs(i), rhs(i), _CMP_NLT_US));
  }
  
  return temp;
}

/* Function: inverse */

template<typename T, int NW, int W>
pack<T, NW, __m256, float, W> inv(const pack<T, NW, __m256, float, W> &op) {
  pack<T, NW, __m256, float, W> temp;
  
  __m256 one = _mm256_set1_ps(1.0f);
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_div_ps(one, op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m256d, double, W> inv(const pack<T, NW, __m256d, double, W> &op) {
  pack<T, NW, __m256d, double, W> temp;

  __m256d one = _mm256_set1_pd(1.0);
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_div_pd(one, op(i));
  }

  return temp;
}

/* Function: sin */

template<typename T, int NW, int W>
pack<T, NW, __m256, float, W> sin(const pack<T, NW, __m256, float,  W> &op) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_sin_ps(op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m256d, double, W> sin(const pack<T, NW, __m256d, double,  W> &op) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_sin_pd(op(i));
  }
  
  return temp;
}

/* Function: cos */

template<typename T, int NW, int W>
pack<T, NW, __m256, float, W> cos(const pack<T, NW, __m256, float, W> &op) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_cos_ps(op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m256d, double, W> cos(const pack<T, NW, __m256d, double, W> &op) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_cos_pd(op(i));
  }
  
  return temp;
}

/* Function: tan */

template<typename T, int NW, int W>
pack<T, NW, __m256, float, W> tan(const pack<T, NW, __m256, float, W> &op) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_tan_ps(op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m256d, double, W> tan(const pack<T, NW, __m256d, double, W> &op) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_tan_pd(op(i));
  }
  
  return temp;
}

/* Function: add adjacent numbers and interleave results */
template<typename T, int NW, int W>
pack<T, NW, __m256, float, W> hadd_pairwise_interleave(const pack<T, NW, __m256, float, W> &lhs, const pack<T, NW, __m256, float, W> &rhs) {
  pack<T, NW, __m256, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256, float, W>::num_regs; ++i) {
    temp(i) = _mm256_permute_ps(_mm256_hadd_ps(lhs(i), rhs(i)), _MM_SHUFFLE(3,1,2,0));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m256d, double, W> hadd_pairwise_interleave(const pack<T, NW, __m256d, double, W> &lhs, const pack<T, NW, __m256d, double, W> &rhs) {
  pack<T, NW, __m256d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m256d, double, W>::num_regs; ++i) {
    temp(i) = _mm256_hadd_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Load functions */

// unaligned temporal load: int <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = _mm256_lddqu_si256(reinterpret_cast<__m256i const*>(p));
  }
}

// masked unaligned temporal load: int <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = maskload256_epi32(p, m(i));
  }
}

// unaligned temporal load: int <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m256i lo = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(p));
    p += type_traits<long>::num_bvals_per_reg;
    __m256i hi = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(p));
    p += type_traits<long>::num_bvals_per_reg;
    op(i) = _mm256_set_m128i(
      cvt256_epi64_epi32(hi), 
      cvt256_epi64_epi32(lo)
    );
  }
}

// masked unaligned temporal load: int <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    __m256i lo = maskload256_epi64(reinterpret_cast<__int64 const*>(p), m1);
    p += type_traits<long>::num_bvals_per_reg;
    __m256i hi = maskload256_epi64(reinterpret_cast<__int64 const*>(p), m2);
    p += type_traits<long>::num_bvals_per_reg;
    op(i) = _mm256_set_m128i(
      cvt256_epi64_epi32(hi), 
      cvt256_epi64_epi32(lo)
    );
  }
}

// unaligned temporal load: int <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvtt256_ps_epi32(_mm256_loadu_ps(p));
  }
}

// masked unaligned temporal load: int <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvtt256_ps_epi32(maskload256_ps(p, m(i)));
  }
}

// unaligned temporal load: int <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m128i lo = cvtt256_pd_epi32(_mm256_loadu_pd(p));
    p += type_traits<double>::num_bvals_per_reg;
    __m128i hi = cvtt256_pd_epi32(_mm256_loadu_pd(p));
    p += type_traits<double>::num_bvals_per_reg;
    op(i) = _mm256_set_m128i(hi, lo);
  }
}

// masked unaligned temporal load: int <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, int, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    __m128i lo = cvtt256_pd_epi32(maskload256_pd(p, m1));
    p += type_traits<double>::num_bvals_per_reg;
    __m128i hi = cvtt256_pd_epi32(maskload256_pd(p, m2));
    p += type_traits<double>::num_bvals_per_reg;
    op(i) = _mm256_set_m128i(hi, lo);
  }
}

// unaligned temporal load: long <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvt256_epi32_epi64(_mm_lddqu_si128(reinterpret_cast<__m128i const*>(p)));
  }
}

// masked unaligned temporal load: long <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    op(i) = cvt256_epi32_epi64(maskload128_epi32(p, mask));
  }
}

// unaligned temporal load: long <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = _mm256_lddqu_si256(reinterpret_cast<const __m256i*>(p));
  }
}

// masked unaligned temporal load: long <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = maskload256_epi64(p, m(i));
  }
}

// unaligned temporal load: long <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvtt256_ps_epi64(_mm_loadu_ps(p));
  }
}

// masked unaligned temporal load: long <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    op(i) = cvtt256_ps_epi64(maskload128_ps(p, mask));
  }
}

// unaligned temporal load: long <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvtt256_pd_epi64(_mm256_loadu_pd(p));
  }
}

// masked unaligned temporal load: long <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256i, long, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvtt256_pd_epi64(maskload256_pd(p, m(i)));
  }
}

// unaligned temporal load: float <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvt256_epi32_ps(_mm256_lddqu_si256(reinterpret_cast<__m256i const*>(p)));
  }
}

// masked unaligned temporal load: float <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvt256_epi32_ps(maskload256_epi32(p, m(i)));
  }
}

// unaligned temporal load: float <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128 lo = cvt256_epi64_ps(_mm256_lddqu_si256(reinterpret_cast<__m256i const*>(p)));
    __m128 hi = cvt256_epi64_ps(_mm256_lddqu_si256(reinterpret_cast<__m256i const*>(p+type_traits<long>::num_bvals_per_reg)));
    op(i) = _mm256_set_m128(hi, lo);
  }
}

// masked unaligned temporal load: float <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    __m128 lo = cvt256_epi64_ps(maskload256_epi64(p, m1));
    __m128 hi = cvt256_epi64_ps(maskload256_epi64(p+type_traits<long>::num_bvals_per_reg, m2));
    op(i) = _mm256_set_m128(hi, lo);
  }
}

// unaligned temporal load: float <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = _mm256_loadu_ps(p);
  }
}

// masked unaligned temporal load: float <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = maskload256_ps(p, m(i));
  }
}

// unaligned temporal load: float <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128 lo = cvt256_pd_ps(_mm256_loadu_pd(p));
    __m128 hi = cvt256_pd_ps(_mm256_loadu_pd(p+type_traits<double>::num_bvals_per_reg));
    op(i) = _mm256_set_m128(hi, lo);
  }
}

// masked unaligned temporal load: float <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256, float, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    __m128 lo = cvt256_pd_ps(maskload256_pd(p, m1));
    __m128 hi = cvt256_pd_ps(maskload256_pd(p+type_traits<double>::num_bvals_per_reg, m2));
    op(i) = _mm256_set_m128(hi, lo);
  }
}

// unaligned temporal load: double <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvt256_epi32_pd(_mm_lddqu_si128(reinterpret_cast<__m128i const*>(p)));
  }
}

// masked unaligned temporal load: double <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  int const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    op(i) = cvt256_epi32_pd(maskload128_epi32(p, mask));
  }
}

// unaligned temporal load: double <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvt256_epi64_pd(_mm256_lddqu_si256(reinterpret_cast<__m256i const*>(p)));
  }
}

// masked unaligned temporal load: double <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvt256_epi64_pd(maskload256_epi64(p, m(i)));
  }
}

// unaligned temporal load: double <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = cvt256_ps_pd(_mm_loadu_ps(p));
  }
}

// masked unaligned temporal load: double <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    op(i) = cvt256_ps_pd(maskload128_ps(p, mask));
  }
}

// unaligned temporal load: double <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = _mm256_loadu_pd(p);
  }
}

// masked unaligned temporal load: double <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m256d, double, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    op(i) = maskload256_pd(p, m(i));
  }
}

/* Store functions */

// unaligned temporal store: int -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), op(i));
  }
}

// masked unaligned temporal store: int -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_epi32(p, m(i), op(i));
  }
}

// unaligned temporal store: int -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), cvt256_epi32_epi64(_mm256_castsi256_si128(op(i))));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p+type_traits<long>::num_bvals_per_reg), cvt256_epi32_epi64(_mm256_extractf128_si256(op(i), 1)));
  }
}

// masked unaligned temporal store: int -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    maskstore256_epi64(reinterpret_cast<__int64*>(p), m1, cvt256_epi32_epi64(_mm256_castsi256_si128(op(i))));
    maskstore256_epi64(reinterpret_cast<__int64*>(p+type_traits<long>::num_bvals_per_reg), m2, cvt256_epi32_epi64(_mm256_extractf128_si256(op(i), 1)));
  }
}

// unaligned temporal store: int -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_ps(p, cvt256_epi32_ps(op(i)));
  }
}

// masked unaligned temporal store: int -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_ps(p, m(i), cvt256_epi32_ps(op(i)));
  }
}

// unaligned temporal store: int -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_pd(p, cvt256_epi32_pd(_mm256_castsi256_si128(op(i))));
    _mm256_storeu_pd(p+type_traits<double>::num_bvals_per_reg, cvt256_epi32_pd(_mm256_extractf128_si256(op(i), 1)));
  }
}

// masked unaligned temporal store: int -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, int, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, int, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    maskstore256_pd(p, m1, cvt256_epi32_pd(_mm256_castsi256_si128(op(i))));
    maskstore256_pd(p+type_traits<double>::num_bvals_per_reg, m2, cvt256_epi32_pd(_mm256_extractf128_si256(op(i), 1)));
  }
}

// unaligned temporal store: long -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), cvt256_epi64_epi32(op(i)));
  }
}

// masked unaligned temporal store: long -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    maskstore128_epi32(p, mask, cvt256_epi64_epi32(op(i)));
  }
}

// unaligned temporal store: long -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), op(i));
  }
}

// masked unaligned temporal store: long -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_epi64(reinterpret_cast<__int64*>(p), m(i), op(i));
  }
}

// unaligned temporal store: long -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm_storeu_ps(p, cvt256_epi64_ps(op(i)));
  }
}

// masked unaligned temporal store: long -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    maskstore128_ps(p, mask, cvt256_epi64_ps(op(i)));
  }
}

// unaligned temporal store: long -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_pd(p, cvt256_epi64_pd(op(i)));
  }
}

// masked unaligned temporal store: long -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256i, long, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256i, long, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_pd(p, m(i), cvt256_epi64_pd(op(i)));
  }
}

// unaligned temporal store: float -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), cvtt256_ps_epi32(op(i)));
  }
}

// masked unaligned temporal store: float -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_epi32(p, m(i), cvtt256_ps_epi32(op(i)));
  }
}

// unaligned temporal store: float -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), cvtt256_ps_epi64(_mm256_castps256_ps128(op(i))));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p+type_traits<long>::num_bvals_per_reg), cvtt256_ps_epi64(_mm256_extractf128_ps(op(i), 1)));
  }
}

// masked unaligned temporal store: float -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    maskstore256_epi64(reinterpret_cast<__int64*>(p), m1, cvtt256_ps_epi64(_mm256_castps256_ps128(op(i))));
    maskstore256_epi64(reinterpret_cast<__int64*>(p+type_traits<long>::num_bvals_per_reg), m2, cvtt256_ps_epi64(_mm256_extractf128_ps(op(i), 1)));
  }
}

// unaligned temporal store: float -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_ps(p, op(i));
  }
}

// masked unaligned temporal store: float -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_ps(p, m(i), op(i));
  }
}

// unaligned temporal store: float -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256, float, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_pd(p, cvt256_ps_pd(_mm256_castps256_ps128(op(i))));
    _mm256_storeu_pd(p+type_traits<double>::num_bvals_per_reg, cvt256_ps_pd(_mm256_extractf128_ps(op(i), 1)));
  }
}

// masked unaligned temporal store: float -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256, float, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256, float, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m256i m1 = mask_cvtepi32lo_epi64(m(i));
    __m256i m2 = mask_cvtepi32hi_epi64(m(i));
    maskstore256_pd(p, m1, cvt256_ps_pd(_mm256_castps256_ps128(op(i))));
    maskstore256_pd(p+type_traits<double>::num_bvals_per_reg, m2, cvt256_ps_pd(_mm256_extractf128_ps(op(i), 1)));
  }
}

// unaligned temporal store: double -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), cvtt256_pd_epi32(op(i)));
  }
}

// masked unaligned temporal store: double -> int
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    maskstore128_epi32(p, mask, cvtt256_pd_epi32(op(i)));
  }
}

// unaligned temporal store: double -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), cvtt256_pd_epi64(op(i)));
  }
}

// masked temporal store: double -> long
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_epi64(reinterpret_cast<__int64*>(p), m(i), cvtt256_pd_epi64(op(i)));
  }
}

// unaligned temporal store: double -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm_storeu_ps(p, cvt256_pd_ps(op(i)));
  }
}

// masked unaligned temporal store: double -> float
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    __m128i mask = mask_cvtepi64_epi32(m(i));
    maskstore128_ps(p, mask, cvt256_pd_ps(op(i)));
  }
}

// unaligned temporal store: double -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    _mm256_storeu_pd(p, op(i));
  }
}

// masked unaligned temporal store: double -> double
template<typename T, int NW, int W>
inline void store(const pack<T, NW, __m256d, double, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m256d, double, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i, p+=type_t::num_bvals_per_reg) {
    maskstore256_pd(p, m(i), op(i));
  }
}

/* Permute functions */

template<>
class permutation<permute_pattern::aacc> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(2,2,0,0));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(2,2,0,0)), 
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(2,2,0,0))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      //temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(1,0,1,0));
      temp(i) = _mm256_unpacklo_epi64(op(i), op(i));
#else
      //temp(i) = _mm256_set_m128i(
      //  _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(1,0,1,0)),
      //  _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(1,0,1,0))
      //);
      __m128i t = _mm256_extractf128_si256(op(i), 1);
      temp(i) = _mm256_set_m128i(
        _mm_unpacklo_epi64(t, t),
        _mm_unpacklo_epi64(_mm256_castsi256_si128(op(i)), _mm256_castsi256_si128(op(i)))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(2,2,0,0));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      //temp(i) = _mm256_permute_pd(op(i), 0x0);
      temp(i) = _mm256_unpacklo_pd(op(i), op(i));
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::abab> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(1,0,1,0));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(1,0,1,0)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(1,0,1,0))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute2f128_si256(op(i), op(i), 0x00);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(1,0,1,0));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute2f128_pd(op(i), op(i), 0x00);
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::bbdd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(3,3,1,1));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(3,3,1,1)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(3,3,1,1))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      //temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(3,2,3,2));
      temp(i) = _mm256_unpackhi_epi64(op(i), op(i));
#else
      //temp(i) = _mm256_set_m128i(
      //  _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(3,2,3,2)),
      //  _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(3,2,3,2))
      //);
      __m128i t = _mm256_extractf128_si256(op(i), 1);
      temp(i) = _mm256_set_m128i(
        _mm_unpackhi_epi64(t, t),
        _mm_unpackhi_epi64(_mm256_castsi256_si128(op(i)), _mm256_castsi256_si128(op(i)))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(3,3,1,1));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      //temp(i) = _mm256_permute_pd(op(i), 0xf);
      temp(i) = _mm256_unpackhi_pd(op(i), op(i));
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::cdcd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(3,2,3,2));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(3,2,3,2)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(3,2,3,2))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute2f128_si256(op(i), op(i), _MM_SHUFFLE(0,1,0,1));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(3,2,3,2));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute2f128_pd(op(i), op(i), _MM_SHUFFLE(0,1,0,1));
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::dcba> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(0,1,2,3));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(0,1,2,3)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(0,1,2,3))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_epi64(op(i), _MM_SHUFFLE(0,1,2,3));
#else
      __m256i t = _mm256_permute2f128_si256(op(i), op(i), 1);
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(t, 1), _MM_SHUFFLE(1,0,3,2)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(t), _MM_SHUFFLE(1,0,3,2))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(0,1,2,3));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_pd(op(i), _MM_SHUFFLE(0,1,2,3));
#else
      temp(i) = _mm256_permute_pd(
        _mm256_permute2f128_pd(op(i), op(i), 1), 5
      );
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::dbca> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(0,2,1,3));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(0,2,1,3)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(0,2,1,3))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_epi64(op(i), _MM_SHUFFLE(0,2,1,3));
#else
      __m128i t = _mm256_extractf128_si256(op(i), 1);
      temp(i) = _mm256_set_m128i(
        _mm_unpacklo_epi64(t, _mm256_castsi256_si128(op(i))), 
        _mm_unpackhi_epi64(t, _mm256_castsi256_si128(op(i)))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(0,2,1,3));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_pd(op(i), _MM_SHUFFLE(0,2,1,3));
#else
      __m128d t = _mm256_extractf128_pd(op(i), 1);
      temp(i) = _mm256_set_m128d(
        _mm_unpacklo_pd(t, _mm256_castpd256_pd128(op(i))), 
        _mm_unpackhi_pd(t, _mm256_castpd256_pd128(op(i)))
      );
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::acac> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(2,0,2,0));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(2,0,2,0)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(2,0,2,0))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;
    
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_epi64(op(i), _MM_SHUFFLE(2,0,2,0));
#else
      __m128i tmpreg = _mm_unpacklo_epi64(_mm256_castsi256_si128(op(i)), _mm256_extractf128_si256(op(i), 1));
      temp(i) = _mm256_set_m128i(tmpreg, tmpreg);
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(2,0,2,0));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_pd(op(i), _MM_SHUFFLE(2,0,2,0));
#else
      __m128d tmpreg = _mm_unpacklo_pd(_mm256_castpd256_pd128(op(i)), _mm256_extractf128_pd(op(i), 1));
      temp(i) = _mm256_set_m128d(tmpreg, tmpreg);
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::bdbd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(3,1,3,1));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(3,1,3,1)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(3,1,3,1))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;

    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_epi64(op(i), _MM_SHUFFLE(3,1,3,1));
#else
      __m128i tmpreg = _mm_unpackhi_epi64(_mm256_castsi256_si128(op(i)), _mm256_extractf128_si256(op(i), 1));
      temp(i) = _mm256_set_m128i(tmpreg, tmpreg);
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(3,1,3,1));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_pd(op(i), _MM_SHUFFLE(3,1,3,1));
#else
      __m128d tmpreg = _mm_unpackhi_pd(_mm256_castpd256_pd128(op(i)), _mm256_extractf128_pd(op(i), 1));
      temp(i) = _mm256_set_m128d(tmpreg, tmpreg);
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::acbd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, int, 4> permute(const pack<T, NW, __m256i, int, 4> &op) {
    typedef pack<T, NW, __m256i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_shuffle_epi32(op(i), _MM_SHUFFLE(3,1,2,0));
#else
      temp(i) = _mm256_set_m128i(
        _mm_shuffle_epi32(_mm256_extractf128_si256(op(i), 1), _MM_SHUFFLE(3,1,2,0)),
        _mm_shuffle_epi32(_mm256_castsi256_si128(op(i)), _MM_SHUFFLE(3,1,2,0))
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256i, long, 4> permute(const pack<T, NW, __m256i, long, 4> &op) {
    typedef pack<T, NW, __m256i, long, 4> type;
    type temp;

    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_epi64(op(i), _MM_SHUFFLE(3,1,2,0));
#else
      __m128i tmp = _mm256_extractf128_si256(op(i), 1);
      temp(i) = _mm256_set_m128i(
        _mm_unpackhi_epi64(_mm256_castsi256_si128(op(i)), tmp), 
        _mm_unpacklo_epi64(_mm256_castsi256_si128(op(i)), tmp)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256, float, 4> permute(const pack<T, NW, __m256, float, 4> &op) {
    typedef pack<T, NW, __m256, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm256_permute_ps(op(i), _MM_SHUFFLE(3,1,2,0));
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m256d, double, 4> permute(const pack<T, NW, __m256d, double, 4> &op) {
    typedef pack<T, NW, __m256d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX2)
      temp(i) = _mm256_permute4x64_pd(op(i), _MM_SHUFFLE(3,1,2,0));
#else
      __m128d tmp = _mm256_extractf128_pd(op(i), 1);
      temp(i) = _mm256_set_m128d(
        _mm_unpackhi_pd(_mm256_castpd256_pd128(op(i)), tmp), 
        _mm_unpacklo_pd(_mm256_castpd256_pd128(op(i)), tmp)
      );
#endif
    }
    return temp;
  }
};

/* Interleave functions */

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m256, float, 1>, VF> op) {
  typedef pack<T, NW, __m256, float, 1> type_t;
  
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m256 lo = _mm256_unpacklo_ps(op[i](ri), op[i+1](ri));
      op[i+1](ri) = _mm256_unpackhi_ps(op[i](ri), op[i+1](ri));
      op[i](ri) = lo;
    }
    
    for(int i = 0; i < NW; i+=4) {
      for(int j = 0; j < 2; ++j) {
        __m256 lo = _mm256_shuffle_ps(op[i+j](ri), op[i+j+2](ri), _MM_SHUFFLE(1,0,1,0));
        op[i+j+2](ri) = _mm256_shuffle_ps(op[i+j](ri), op[i+j+2](ri), _MM_SHUFFLE(3,2,3,2));
        op[i+j](ri) = lo;
      }
      __m256 lo = op[i+1](ri);
      op[i+1](ri) = op[i+2](ri);
      op[i+2](ri) = lo;
    }
    
    for(int i = 0; i < NW; i += 8) {
      for(int j = 0; j < 4; ++j) {
        __m256 lo = _mm256_permute2f128_ps(op[i+j](ri), op[i+j+4](ri), 0x20);
        op[i+j+4](ri) = _mm256_permute2f128_ps(op[i+j](ri), op[i+j+4](ri), 0x31);
        op[i+j](ri) = lo;
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*8;
        int m2 = j*8;
        for(int k = 0; k < 8; ++k, ++m1, ++m2) {
          __m256 lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m256, float, 2>, VF> op) {
  typedef pack<T, NW, __m256, float, 2> type_t;
  
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m256 lo = _mm256_shuffle_ps(op[i](ri), op[i+1](ri), _MM_SHUFFLE(1,0,1,0));
      op[i+1](ri) = _mm256_shuffle_ps(op[i](ri), op[i+1](ri), _MM_SHUFFLE(3,2,3,2));
      op[i](ri) = lo;
    }
    
    for(int i = 0; i < NW; i += 4) {
      for(int j = 0; j < 2; ++j) {
        __m256 lo = _mm256_permute2f128_ps(op[i+j](ri), op[i+j+2](ri), 0x20);
        op[i+j+2](ri) = _mm256_permute2f128_ps(op[i+j](ri), op[i+j+2](ri), 0x31);
        op[i+j](ri) = lo;
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*4;
        int m2 = j*4;
        for(int k = 0; k < 4; ++k, ++m1, ++m2) {
          __m256 lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m256, float, 4>, VF> op) {
  typedef pack<T, NW, __m256, float, 4> type_t;
  
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i += 2) {
      __m256 lo = _mm256_permute2f128_ps(op[i](ri), op[i+1](ri), 0x20);
      op[i+1](ri) = _mm256_permute2f128_ps(op[i](ri), op[i+1](ri), 0x31);
      op[i](ri) = lo;
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*2;
        int m2 = j*2;
        for(int k = 0; k < 2; ++k, ++m1, ++m2) {
          __m256 lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m256, float, 8>, VF> op) {
  typedef pack<T, NW, __m256, float, 8> type_t;
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        __m256 lo = op[i](j);
        op[i](j) = op[j](i);
        op[j](i) = lo;
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m256d, double, 1>, VF> op) {
  typedef pack<T, NW, __m256d, double, 1> type_t;
  
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m256d lo = _mm256_shuffle_pd(op[i](ri), op[i+1](ri), 0x0);
      op[i+1](ri) = _mm256_shuffle_pd(op[i](ri), op[i+1](ri), 0xF);
      op[i](ri) = lo;
    }
    
    for(int i = 0; i < NW; i += 4) {
      for(int j = 0; j < 2; ++j) {
        __m256d lo = _mm256_permute2f128_pd(op[i+j](ri), op[i+j+2](ri), 0x20);
        op[i+j+2](ri) = _mm256_permute2f128_pd(op[i+j](ri), op[i+j+2](ri), 0x31);
        op[i+j](ri) = lo;
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*4;
        int m2 = j*4;
        for(int k = 0; k < 4; ++k, ++m1, ++m2) {
          __m256d lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m256d, double, 2>, VF> op) {
  typedef pack<T, NW, __m256d, double, 2> type_t;
  
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i += 2) {
      __m256d lo = _mm256_permute2f128_pd(op[i](ri), op[i+1](ri), 0x20);
      op[i+1](ri) = _mm256_permute2f128_pd(op[i](ri), op[i+1](ri), 0x31);
      op[i](ri) = lo;
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*2;
        int m2 = j*2;
        for(int k = 0; k < 2; ++k, ++m1, ++m2) {
          __m256d lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m256d, double, 4>, VF> op) {
  typedef pack<T, NW, __m256d, double, 4> type_t;
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        __m256d lo = op[i](j);
        op[i](j) = op[j](i);
        op[j](i) = lo;
      }
    }
  }
}

#endif // #if SIMD256

} // namespace simd {

#endif // SIMD_SIMD256X86DEF_HPP
