#ifndef SIMD_SIMD256X86IMPL_HPP
#define SIMD_SIMD256X86IMPL_HPP

#include <immintrin.h>

namespace simd {

//------------------------------------------------------------------------------
// Memory loader implementations
//------------------------------------------------------------------------------

#if SIMD256
//------------------------------------------------------------------------------
// SimdLoader<simd::vector, 256> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::vector, 256> {
public:
  static inline simd::value<short, 256>::reg load(const short *mem) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<unsigned short, 256>::reg load(const unsigned short *mem) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<int, 256>::reg load(const int *mem) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<unsigned int, 256>::reg load(const unsigned int *mem) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<long, 256>::reg load(const long *mem) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<unsigned long, 256>::reg load(const unsigned long *mem) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<float, 256>::reg load(const float *mem) {
    return _mm256_load_ps(mem);
  }
  
  static inline simd::value<double, 256>::reg load(const double *mem) {
    return _mm256_load_pd(mem);
  }
};
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// SimdLoader<simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::vector, 512> {
public:
  static inline simd::value<short, 512>::reg load(const short *mem) {
    return simd::value<short, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_SHORT_WIDTH)
    );
  }

  static inline simd::value<unsigned short, 512>::reg load(const unsigned short *mem) {
    return simd::value<unsigned short, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_SHORT_WIDTH)
    );
  }
  
  static inline simd::value<int, 512>::reg load(const int *mem) {
    return simd::value<int, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_INT_WIDTH)
    );
  }
  
  static inline simd::value<unsigned int, 512>::reg load(const unsigned int *mem) {
    return simd::value<unsigned int, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_INT_WIDTH)
    );
  }
  
  static inline simd::value<long, 512>::reg load(const long *mem) {
    return simd::value<long, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_LONG_WIDTH)
    );
  }
  
  static inline simd::value<unsigned long, 512>::reg load(const unsigned long *mem) {
    return simd::value<unsigned long, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_LONG_WIDTH)
    );
  }
  
  static inline simd::value<float, 512>::reg load(const float *mem) {
    return simd::value<float, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_FLOAT_WIDTH)
    );
  }
  
  static inline simd::value<double, 512>::reg load(const double *mem) {
    return simd::value<double, 512>::reg(
      SimdLoader<simd::vector, 256>::load(mem), 
      SimdLoader<simd::vector, 256>::load(mem+SIMD256_DOUBLE_WIDTH)
    );
  }
};
#endif

#if SIMD256
//------------------------------------------------------------------------------
// SimdLoader<simd::unaligned|simd::vector, 256> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::unaligned|simd::vector, 256> {
public:
  static inline simd::value<short, 256>::reg load(const short *mem) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<unsigned short, 256>::reg load(const unsigned short *mem) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<int, 256>::reg load(const int *mem) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<unsigned int, 256>::reg load(const unsigned int *mem) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<long, 256>::reg load(const long *mem) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<unsigned long, 256>::reg load(const unsigned long *mem) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
  }
  
  static inline simd::value<float, 256>::reg load(const float *mem) {
    return _mm256_loadu_ps(mem);
  }
  
  static inline simd::value<double, 256>::reg load(const double *mem) {
    return _mm256_loadu_pd(mem);
  }
};
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// SimdLoader<simd::unaligned|simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::unaligned|simd::vector, 512> {
public:
  static inline simd::value<short, 512>::reg load(const short *mem) {
    typedef simd::value<short, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_SHORT_WIDTH)
    );
  }
  
  static inline simd::value<unsigned short, 512>::reg load(const unsigned short *mem) {
    typedef simd::value<unsigned short, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_SHORT_WIDTH)
    );
  }
  
  static inline simd::value<int, 512>::reg load(const int *mem) {
    typedef simd::value<int, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_INT_WIDTH)
    );
  }
  
  static inline simd::value<unsigned int, 512>::reg load(const unsigned int *mem) {
    typedef simd::value<unsigned int, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_INT_WIDTH)
    );
  }
  
  static inline simd::value<long, 512>::reg load(const long *mem) {
    typedef simd::value<long, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_LONG_WIDTH)
    );
  }
  
  static inline simd::value<unsigned long, 512>::reg load(const unsigned long *mem) {
    typedef simd::value<unsigned long, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_LONG_WIDTH)
    );
  }
  
  static inline simd::value<float, 512>::reg load(const float *mem) {
    typedef simd::value<float, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_FLOAT_WIDTH)
    );
  }
  
  static inline simd::value<double, 512>::reg load(const double *mem) {
    typedef simd::value<double, 512>::reg reg;
    return reg(
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem),
      SimdLoader<simd::unaligned|simd::vector, 256>::load(mem+SIMD256_DOUBLE_WIDTH)
    );
  }
};
#endif

#if SIMD256
//------------------------------------------------------------------------------
// SimdLoader<simd::unaligned|simd::broadcast, 256> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::unaligned|simd::broadcast, 256> {
public:
  static inline simd::value<short, 256>::reg load(const short *mem) {
    return _mm256_set1_epi16(*mem);
  }
  
  static inline simd::value<unsigned short, 256>::reg load(const unsigned short *mem) {
    return _mm256_set1_epi16(*mem);
  }
  
  static inline simd::value<int, 256>::reg load(const int *mem) {
    return _mm256_set1_epi32(*mem);
  } 
  
  static inline simd::value<unsigned int, 256>::reg load(const unsigned int *mem) {
    return _mm256_set1_epi32(*mem);
  }
  
  static inline simd::value<long, 256>::reg load(const long *mem) {
    return _mm256_set1_epi64x(*mem);
  } 
  
  static inline simd::value<unsigned long, 256>::reg load(const unsigned long *mem) {
    return _mm256_set1_epi64x(*mem);
  }
  
  static inline simd::value<float, 256>::reg load(const float *mem) {
    return _mm256_set1_ps(*mem);
  }
  
  static inline simd::value<double, 256>::reg load(const double *mem) {
    return _mm256_set1_pd(*mem);
  }
};
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
//  SimdLoader<simd::unaligned|simd::broadcast, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::unaligned|simd::broadcast, 512> {
public:
  static inline simd::value<short, 512>::reg load(const short *mem) {
    return simd::value<short, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }

  static inline simd::value<unsigned short, 512>::reg load(const unsigned short *mem) {
    return simd::value<unsigned short, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }
  
  static inline simd::value<int, 512>::reg load(const int *mem) {
    return simd::value<int, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }
  
  static inline simd::value<unsigned int, 512>::reg load(const unsigned int *mem) {
    return simd::value<unsigned int, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }
  
  static inline simd::value<long, 512>::reg load(const long *mem) {
    return simd::value<long, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }
  
  static inline simd::value<unsigned long, 512>::reg load(const unsigned long *mem) {
    return simd::value<unsigned long, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }
  
  static inline simd::value<float, 512>::reg load(const float *mem) {
    return simd::value<float, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }
  
  static inline simd::value<double, 512>::reg load(const double *mem) {
    return simd::value<double, 512>::reg(
      SimdLoader<simd::unaligned|simd::broadcast, 256>::load(mem)
    );
  }
};
#endif

//------------------------------------------------------------------------------
// Memory storer implementations
//------------------------------------------------------------------------------

#if SIMD256
//------------------------------------------------------------------------------
// SimdStorer<simd::vector, 256> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::vector, 256> {
public:
  static inline void store(simd::value<short, 256>::reg &reg, short *mem) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<unsigned short, 256>::reg &reg, unsigned short *mem) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<int, 256>::reg &reg, int *mem) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<unsigned int, 256>::reg &reg, unsigned int *mem) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<long, 256>::reg &reg, long *mem) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<unsigned long, 256>::reg &reg, unsigned long *mem) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<float, 256>::reg &reg, float *mem) {
    _mm256_store_ps(mem, reg);
  }
  
  static inline void store(simd::value<double, 256>::reg &reg, double *mem) {
    _mm256_store_pd(mem, reg);
  }
};
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
//  SimdStorer<simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::vector, 512> {
public:
  static inline void store(simd::value<short, 512>::reg &reg, short *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_SHORT_WIDTH);
  }
  
  static inline void store(simd::value<unsigned short, 512>::reg &reg, unsigned short *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_SHORT_WIDTH);
  }
  
  static inline void store(simd::value<int, 512>::reg &reg, int *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_INT_WIDTH);
  }
  
  static inline void store(simd::value<unsigned int, 512>::reg &reg, unsigned int *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_INT_WIDTH);
  }
  
  static inline void store(simd::value<long, 512>::reg &reg, long *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_LONG_WIDTH);
  }
  
  static inline void store(simd::value<unsigned long, 512>::reg &reg, unsigned long *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_LONG_WIDTH);
  }
  
  static inline void store(simd::value<float, 512>::reg &reg, float *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_FLOAT_WIDTH);
  }
  
  static inline void store(simd::value<double, 512>::reg &reg, double *mem) {
    SimdStorer<simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::vector, 256>::store(reg.r1, mem+SIMD256_DOUBLE_WIDTH);
  }
};
#endif

#if SIMD256
//------------------------------------------------------------------------------
// SimdStorer<simd::unaligned|simd::vector, 256> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::unaligned|simd::vector, 256> {
public:
  static inline void store(simd::value<short, 256>::reg &reg, short *mem) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<unsigned short, 256>::reg &reg, unsigned short *mem) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<int, 256>::reg &reg, int *mem) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<unsigned int, 256>::reg &reg, unsigned int *mem) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<long, 256>::reg &reg, long *mem) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<unsigned long, 256>::reg &reg, unsigned long *mem) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), reg);
  }
  
  static inline void store(simd::value<float, 256>::reg &reg, float *mem) {
    _mm256_storeu_ps(mem, reg);
  }
  
  static inline void store(simd::value<double, 256>::reg &reg, double *mem) {
    _mm256_storeu_pd(mem, reg);
  }
};
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// SimdStorer<simd::unaligned|simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::unaligned|simd::vector, 512> {
public:
  static inline void store(simd::value<short, 512>::reg &reg, short *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_SHORT_WIDTH);
  }
  
  static inline void store(simd::value<unsigned short, 512>::reg &reg, unsigned short *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_SHORT_WIDTH);
  }
  
  static inline void store(simd::value<int, 512>::reg &reg, int *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_INT_WIDTH);
  }
  
  static inline void store(simd::value<unsigned int, 512>::reg &reg, unsigned int *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_INT_WIDTH);
  }
  
  static inline void store(simd::value<long, 512>::reg &reg, long *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_LONG_WIDTH); 
  }
  
  static inline void store(simd::value<unsigned long, 512>::reg &reg, unsigned long *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_LONG_WIDTH);
  }
  
  static inline void store(simd::value<float, 512>::reg &reg, float *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_FLOAT_WIDTH); 
  }
  
  static inline void store(simd::value<double, 512>::reg &reg, double *mem) {
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::vector, 256>::store(reg.r1, mem+SIMD256_DOUBLE_WIDTH); 
  }
};
#endif

#if SIMD256
//------------------------------------------------------------------------------
// SimdStorer<simd::unaligned|simd::broadcast, 256> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::unaligned|simd::broadcast, 256> {
public:
  static inline void store(simd::value<short, 256>::reg &reg, short *mem) {
    __m256i temp;
#if defined(SIMD_AVX2)
    temp = _mm256_broadcastw_epi16(_mm256_castsi256_si128(reg));
#else
    __m128i mask = _mm_set_epi8(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
    __m128i temp1 = _mm_shuffle_epi8(_mm256_castsi256_si128(reg), mask);
    temp = _mm256_set_m128i(temp1, temp1);
#endif
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), temp);
  }
  
  static inline void store(simd::value<unsigned short, 256>::reg &reg, unsigned short *mem) {
    store(reg, reinterpret_cast<short*>(mem));
  }
  
  static inline void store(simd::value<int, 256>::reg &reg, int *mem) {
    __m256i temp;
#if defined(SIMD_AVX2)
    temp = _mm256_broadcastd_epi32(_mm256_castsi256_si128(reg));
#else
    __m128i mask = _mm_set_epi8(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
    __m128i temp1 = _mm_shuffle_epi8(_mm256_castsi256_si128(reg), mask);
    temp = _mm256_set_m128i(temp1, temp1);
#endif
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), temp);
  }
  
  static inline void store(simd::value<unsigned int, 256>::reg &reg, unsigned int *mem) {
    store(reg, reinterpret_cast<int*>(mem));
  }
  
  static inline void store(simd::value<long, 256>::reg &reg, long *mem) {
    __m256i temp;
#if defined(SIMD_AVX2)
    temp = _mm256_broadcastq_epi64(_mm256_castsi256_si128(reg));
#else
    __m128i mask = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0);
    __m128i temp1 = _mm_shuffle_epi8(_mm256_castsi256_si128(reg), mask);
    temp = _mm256_set_m128i(temp1, temp1);
#endif
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), temp);
  }
  
  static inline void store(simd::value<unsigned long, 256>::reg &reg, unsigned long *mem) {
    store(reg, reinterpret_cast<long*>(mem));
  }
  
  static inline void store(simd::value<float, 256>::reg &reg, float *mem) {
    __m256 temp;
#if defined(SIMD_AVX2)
    temp = _mm256_broadcastss_ps(_mm256_castps256_ps128(reg));
#else
    __m128i mask = _mm_set_epi8(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
    __m128 temp1 = _mm_castsi128_ps(_mm_shuffle_epi8(
      _mm256_castsi256_si128(_mm256_castps_si256(reg)), mask));
    temp = _mm256_set_m128(temp1, temp1);
#endif
    _mm256_storeu_ps(mem, temp);
  }
  
  static inline void store(simd::value<double, 256>::reg &reg, double *mem) {
    __m256d temp;
#if defined(SIMD_AVX2)
    temp = _mm256_broadcastsd_pd(_mm256_castpd256_pd128(reg));
#else
    __m128i mask = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0);
    __m128d temp1 = _mm_castsi128_pd(_mm_shuffle_epi8(
      _mm256_castsi256_si128(_mm256_castpd_si256(reg)), mask));
    temp = _mm256_set_m128d(temp1, temp1);
#endif
    _mm256_storeu_pd(mem, temp);
  }
};
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// SimdStorer<simd::unaligned|simd::broadcast, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::unaligned|simd::broadcast, 512> {
public:
  static inline void store(simd::value<short, 512>::reg &reg, short *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_SHORT_WIDTH);
  }
  
  static inline void store(simd::value<unsigned short, 512>::reg &reg, unsigned short *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_SHORT_WIDTH);
  }
  
  static inline void store(simd::value<int, 512>::reg &reg, int *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_INT_WIDTH);
  }
  
  static inline void store(simd::value<unsigned int, 512>::reg &reg, unsigned int *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_INT_WIDTH);
  }
  
  static inline void store(simd::value<long, 512>::reg &reg, long *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_LONG_WIDTH);
  }
  
  static inline void store(simd::value<unsigned long, 512>::reg &reg, unsigned long *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_LONG_WIDTH);
  }
  
  static inline void store(simd::value<float, 512>::reg &reg, float *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_FLOAT_WIDTH);
  }
  
  static inline void store(simd::value<double, 512>::reg &reg, double *mem) {
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem);
    SimdStorer<simd::unaligned|simd::broadcast, 256>::store(reg.r0, mem+SIMD256_DOUBLE_WIDTH);
  }
};
#endif

//------------------------------------------------------------------------------
// Prefetch implementations
//------------------------------------------------------------------------------

//template<>
//inline void prefetch<simd::pfnta>(const void *mem) {
//  _mm_prefetch(reinterpret_cast<const char*>(mem), _MM_HINT_NTA);
//}

//template<>
//inline void prefetch<simd::pf0>(const void *mem) {
//  _mm_prefetch(reinterpret_cast<const char*>(mem), _MM_HINT_T0);
//}

//template<>
//inline void prefetch<simd::pf1>(const void *mem) {
//  _mm_prefetch(reinterpret_cast<const char*>(mem), _MM_HINT_T1);
//}

//template<>
//inline void prefetch<simd::pf2>(const void *mem) {
//  _mm_prefetch(reinterpret_cast<const char*>(mem), _MM_HINT_T2);
//}

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<short, 256>
//------------------------------------------------------------------------------

inline SimdType<short, 256> operator+(const SimdType<short, 256> &lhs, const SimdType<short, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_add_epi16(lhs, rhs);
#else
  return _mm256_set_m128i(
    _mm_add_epi16(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)), 
    _mm_add_epi16(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<short, 256> operator-(const SimdType<short, 256> &lhs, const SimdType<short, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_sub_epi16(lhs, rhs);
#else
  return _mm256_set_m128i(
    _mm_sub_epi16(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)), 
    _mm_sub_epi16(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<short, 256> operator*(const SimdType<short, 256> &lhs, const SimdType<short, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_mullo_epi16(lhs, rhs);
#else
  return _mm256_set_m128i(
    _mm_mullo_epi16(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)), 
    _mm_mullo_epi16(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<short, 256> operator/(const SimdType<short, 256> &lhs, const SimdType<short, 256> &rhs) {
  return _mm256_div_epi16(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<short, 512>
//------------------------------------------------------------------------------

inline SimdType<short, 512> operator+(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
  typedef simd::value<short, 512>::reg reg;
  return reg(
    SimdType<short, 256>(reg(lhs).r0) + SimdType<short, 256>(reg(rhs).r0),
    SimdType<short, 256>(reg(lhs).r1) + SimdType<short, 256>(reg(rhs).r1)
  );
}

inline SimdType<short, 512> operator-(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
  typedef simd::value<short, 512>::reg reg;
  return reg(
    SimdType<short, 256>(reg(lhs).r0) - SimdType<short, 256>(reg(rhs).r0),
    SimdType<short, 256>(reg(lhs).r1) - SimdType<short, 256>(reg(rhs).r1)
  );
}

inline SimdType<short, 512> operator*(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
  typedef simd::value<short, 512>::reg reg;
  return reg(
    SimdType<short, 256>(reg(lhs).r0) * SimdType<short, 256>(reg(rhs).r0),
    SimdType<short, 256>(reg(lhs).r1) * SimdType<short, 256>(reg(rhs).r1)
  );
}

inline SimdType<short, 512> operator/(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
  typedef simd::value<short, 512>::reg reg;
  return reg(
    SimdType<short, 256>(reg(lhs).r0) / SimdType<short, 256>(reg(rhs).r0),
    SimdType<short, 256>(reg(lhs).r1) / SimdType<short, 256>(reg(rhs).r1)
  );
}
#endif

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned short, 256>
//------------------------------------------------------------------------------

inline SimdType<unsigned short, 256> operator+(const SimdType<unsigned short, 256> &lhs, const SimdType<unsigned short, 256> &rhs) {
  return SimdType<unsigned short, 256>(SimdType<short, 256>(lhs) + SimdType<short, 256>(rhs));
}

inline SimdType<unsigned short, 256> operator-(const SimdType<unsigned short, 256> &lhs, const SimdType<unsigned short, 256> &rhs) {
  return SimdType<unsigned short, 256>(SimdType<short, 256>(lhs) - SimdType<short, 256>(rhs));
}

inline SimdType<unsigned short, 256> operator*(const SimdType<unsigned short, 256> &lhs, const SimdType<unsigned short, 256> &rhs) {
  return SimdType<unsigned short, 256>(SimdType<short, 256>(lhs) * SimdType<short, 256>(rhs));
}

inline SimdType<unsigned short, 256> operator/(const SimdType<unsigned short, 256> &lhs, const SimdType<unsigned short, 256> &rhs) {
  return _mm256_div_epu16(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned short, 512>
//------------------------------------------------------------------------------

inline SimdType<unsigned short, 512> operator+(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short, 512> &rhs) {
  typedef simd::value<unsigned short, 512>::reg reg;
  return reg(
    SimdType<unsigned short, 256>(reg(lhs).r0) + SimdType<unsigned short, 256>(reg(rhs).r0),
    SimdType<unsigned short, 256>(reg(lhs).r1) + SimdType<unsigned short, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned short, 512> operator-(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short, 512> &rhs) {
  typedef simd::value<unsigned short, 512>::reg reg;
  return reg(
    SimdType<unsigned short, 256>(reg(lhs).r0) - SimdType<unsigned short, 256>(reg(rhs).r0),
    SimdType<unsigned short, 256>(reg(lhs).r1) - SimdType<unsigned short, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned short, 512> operator*(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short, 512> &rhs) {
  typedef simd::value<unsigned short, 512>::reg reg;
  return reg(
    SimdType<unsigned short, 256>(reg(lhs).r0) * SimdType<unsigned short, 256>(reg(rhs).r0),
    SimdType<unsigned short, 256>(reg(lhs).r1) * SimdType<unsigned short, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned short, 512> operator/(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short, 512> &rhs) {
  typedef simd::value<unsigned short, 512>::reg reg;
  return reg(
    SimdType<unsigned short, 256>(reg(lhs).r0) / SimdType<unsigned short, 256>(reg(rhs).r0),
    SimdType<unsigned short, 256>(reg(lhs).r1) / SimdType<unsigned short, 256>(reg(rhs).r1)
  );
}
#endif

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<int, 256>
//------------------------------------------------------------------------------

inline SimdType<int, 256> operator+(const SimdType<int, 256> &lhs, const SimdType<int, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_add_epi32(lhs, rhs);
#else
  return _mm256_set_m128i(
    _mm_add_epi32(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)),
    _mm_add_epi32(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<int, 256> operator-(const SimdType<int, 256> &lhs, const SimdType<int, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_sub_epi32(lhs, rhs);
#else
  return _mm256_set_m128i(
    _mm_sub_epi32(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)),
    _mm_sub_epi32(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<int, 256> operator*(const SimdType<int, 256> &lhs, const SimdType<int, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_mullo_epi32(lhs, rhs);
#else
  return _mm256_set_m128i(
    _mm_mullo_epi32(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)),
    _mm_mullo_epi32(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<int, 256> operator/(const SimdType<int, 256> &lhs, const SimdType<int, 256> &rhs) {
  return _mm256_div_epi32(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<int, 512>
//------------------------------------------------------------------------------

inline SimdType<int, 512> operator+(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  typedef simd::value<int, 512>::reg reg;
  return reg(
    SimdType<int, 256>(reg(lhs).r0) + SimdType<int, 256>(reg(rhs).r0),
    SimdType<int, 256>(reg(lhs).r1) + SimdType<int, 256>(reg(rhs).r1)
  );
}

inline SimdType<int, 512> operator-(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  typedef simd::value<int, 512>::reg reg;
  return reg(
    SimdType<int, 256>(reg(lhs).r0) - SimdType<int, 256>(reg(rhs).r0),
    SimdType<int, 256>(reg(lhs).r1) - SimdType<int, 256>(reg(rhs).r1)
  );
}

inline SimdType<int, 512> operator*(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  typedef simd::value<int, 512>::reg reg;
  return reg(
    SimdType<int, 256>(reg(lhs).r0) * SimdType<int, 256>(reg(rhs).r0),
    SimdType<int, 256>(reg(lhs).r1) * SimdType<int, 256>(reg(rhs).r1)
  );
}

inline SimdType<int, 512> operator/(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  typedef simd::value<int, 512>::reg reg;
  return reg(
    SimdType<int, 256>(reg(lhs).r0) / SimdType<int, 256>(reg(rhs).r0),
    SimdType<int, 256>(reg(lhs).r1) / SimdType<int, 256>(reg(rhs).r1)
  );
}
#endif

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned int, 256>
//------------------------------------------------------------------------------

inline SimdType<unsigned int, 256> operator+(const SimdType<unsigned int, 256> &lhs, const SimdType<unsigned int, 256> &rhs) {
  return SimdType<unsigned int, 256>(SimdType<int, 256>(lhs) + SimdType<int, 256>(rhs));
}

inline SimdType<unsigned int, 256> operator-(const SimdType<unsigned int, 256> &lhs, const SimdType<unsigned int, 256> &rhs) {
  return SimdType<unsigned int, 256>(SimdType<int, 256>(lhs) - SimdType<int, 256>(rhs));
}

inline SimdType<unsigned int, 256> operator*(const SimdType<unsigned int, 256> &lhs, const SimdType<unsigned int, 256> &rhs) {
  return SimdType<unsigned int, 256>(SimdType<int, 256>(lhs) * SimdType<int, 256>(rhs));
}

inline SimdType<unsigned int, 256> operator/(const SimdType<unsigned int, 256> &lhs, const SimdType<unsigned int, 256> &rhs) {
  return _mm256_div_epu32(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned int, 512>
//------------------------------------------------------------------------------

inline SimdType<unsigned int, 512> operator+(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  typedef simd::value<unsigned int, 512>::reg reg;
  return reg(
    SimdType<unsigned int, 256>(reg(lhs).r0) + SimdType<unsigned int, 256>(reg(rhs).r0),
    SimdType<unsigned int, 256>(reg(lhs).r1) + SimdType<unsigned int, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned int, 512> operator-(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  typedef simd::value<unsigned int, 512>::reg reg;
  return reg(
    SimdType<unsigned int, 256>(reg(lhs).r0) - SimdType<unsigned int, 256>(reg(rhs).r0),
    SimdType<unsigned int, 256>(reg(lhs).r1) - SimdType<unsigned int, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned int, 512> operator*(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  typedef simd::value<unsigned int, 512>::reg reg;
  return reg(
    SimdType<unsigned int, 256>(reg(lhs).r0) * SimdType<unsigned int, 256>(reg(rhs).r0),
    SimdType<unsigned int, 256>(reg(lhs).r1) * SimdType<unsigned int, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned int, 512> operator/(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  typedef simd::value<unsigned int, 512>::reg reg;
  return reg(
    SimdType<unsigned int, 256>(reg(lhs).r0) / SimdType<unsigned int, 256>(reg(rhs).r0),
    SimdType<unsigned int, 256>(reg(lhs).r1) / SimdType<unsigned int, 256>(reg(rhs).r1)
  );
}
#endif

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<long, 256>
//------------------------------------------------------------------------------

inline SimdType<long, 256> operator+(const SimdType<long, 256> &lhs, const SimdType<long, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_add_epi64(lhs, rhs);
#else
  return _mm256_set_m128i(
    _mm_add_epi64(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)),
    _mm_add_epi64(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<long, 256> operator-(const SimdType<long, 256> &lhs, const SimdType<long, 256> &rhs) {
#if defined(SIMD_AVX2)
  return _mm256_sub_epi64(lhs, rhs);
#else
  return _mm256_set_m128i( 
    _mm_sub_epi64(_mm256_extractf128_si256(lhs, 1), _mm256_extractf128_si256(rhs, 1)),
    _mm_sub_epi64(_mm256_castsi256_si128(lhs), _mm256_castsi256_si128(rhs))
  );
#endif
}

inline SimdType<long, 256> operator*(const SimdType<long, 256> &lhs, const SimdType<long, 256> &rhs) {
#if defined(SIMD_AVX2)
  __m256i bswap   = _mm256_shuffle_epi32(rhs, 0xB1);
  __m256i prod0  = _mm256_mullo_epi32(lhs, bswap);
  __m256i zero    = _mm256_setzero_si256();
  __m256i prod1 = _mm256_hadd_epi32(prod0, zero);
  __m256i prod2 = _mm256_shuffle_epi32(prod1, 0x73);
  __m256i prod3  = _mm256_mul_epu32(lhs, rhs);
  return _mm256_add_epi64(prod3, prod2);
#else
  __m128i al = _mm256_castsi256_si128(lhs);
  __m128i ah = _mm256_extractf128_si256(lhs, 1);
  __m128i bl = _mm256_castsi256_si128(rhs);
  __m128i bh = _mm256_extractf128_si256(rhs, 1);
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
  return _mm256_set_m128i (prodh, prodl);
#endif
}

inline SimdType<long, 256> operator/(const SimdType<long, 256> &lhs, const SimdType<long, 256> &rhs) {
  return _mm256_div_epi64(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<long, 512>
//------------------------------------------------------------------------------

inline SimdType<long, 512> operator+(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
  typedef simd::value<long, 512>::reg reg;
  return reg(
    SimdType<long, 256>(reg(lhs).r0) + SimdType<long, 256>(reg(rhs).r0),
    SimdType<long, 256>(reg(lhs).r1) + SimdType<long, 256>(reg(rhs).r1)
  );
}

inline SimdType<long, 512> operator-(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
  typedef simd::value<long, 512>::reg reg;
  return reg(
    SimdType<long, 256>(reg(lhs).r0) - SimdType<long, 256>(reg(rhs).r0),
    SimdType<long, 256>(reg(lhs).r1) - SimdType<long, 256>(reg(rhs).r1)
  );
}

inline SimdType<long, 512> operator*(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
  typedef simd::value<long, 512>::reg reg;
  return reg(
    SimdType<long, 256>(reg(lhs).r0) * SimdType<long, 256>(reg(rhs).r0),
    SimdType<long, 256>(reg(lhs).r1) * SimdType<long, 256>(reg(rhs).r1)
  );
}

inline SimdType<long, 512> operator/(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
  typedef simd::value<long, 512>::reg reg;
  return reg(
    SimdType<long, 256>(reg(lhs).r0) / SimdType<long, 256>(reg(rhs).r0),
    SimdType<long, 256>(reg(lhs).r1) / SimdType<long, 256>(reg(rhs).r1)
  );
}
#endif

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned long, 256>
//------------------------------------------------------------------------------

inline SimdType<unsigned long, 256> operator+(const SimdType<unsigned long, 256> &lhs, const SimdType<unsigned long, 256> &rhs) {
  return SimdType<unsigned long, 256>(SimdType<long, 256>(lhs) + SimdType<long, 256>(rhs));
}

inline SimdType<unsigned long, 256> operator-(const SimdType<unsigned long, 256> &lhs, const SimdType<unsigned long, 256> &rhs) {
  return SimdType<unsigned long, 256>(SimdType<long, 256>(lhs) - SimdType<long, 256>(rhs));
}

inline SimdType<unsigned long, 256> operator*(const SimdType<unsigned long, 256> &lhs, const SimdType<unsigned long, 256> &rhs) {
  return SimdType<unsigned long, 256>(SimdType<long, 256>(lhs) * SimdType<long, 256>(rhs));
}

inline SimdType<unsigned long, 256> operator/(const SimdType<unsigned long, 256> &lhs, const SimdType<unsigned long, 256> &rhs) {
  return _mm256_div_epu64(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned long, 512>
//------------------------------------------------------------------------------

inline SimdType<unsigned long, 512> operator+(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  typedef simd::value<unsigned long, 512>::reg reg;
  return reg(
    SimdType<unsigned long, 256>(reg(lhs).r0) + SimdType<unsigned long, 256>(reg(rhs).r0),
    SimdType<unsigned long, 256>(reg(lhs).r1) + SimdType<unsigned long, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned long, 512> operator-(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  typedef simd::value<unsigned long, 512>::reg reg;
  return reg(
    SimdType<unsigned long, 256>(reg(lhs).r0) - SimdType<unsigned long, 256>(reg(rhs).r0),
    SimdType<unsigned long, 256>(reg(lhs).r1) - SimdType<unsigned long, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned long, 512> operator*(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  typedef simd::value<unsigned long, 512>::reg reg;
  return reg(
    SimdType<unsigned long, 256>(reg(lhs).r0) * SimdType<unsigned long, 256>(reg(rhs).r0),
    SimdType<unsigned long, 256>(reg(lhs).r1) * SimdType<unsigned long, 256>(reg(rhs).r1)
  );
}

inline SimdType<unsigned long, 512> operator/(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  typedef simd::value<unsigned long, 512>::reg reg;
  return reg(
    SimdType<unsigned long, 256>(reg(lhs).r0) / SimdType<unsigned long, 256>(reg(rhs).r0),
    SimdType<unsigned long, 256>(reg(lhs).r1) / SimdType<unsigned long, 256>(reg(rhs).r1)
  );
}
#endif

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<float, 256>
//------------------------------------------------------------------------------

inline SimdType<float, 256> operator+(const SimdType<float, 256> &lhs, const SimdType<float, 256> &rhs) {
  return _mm256_add_ps(lhs, rhs);
}

inline SimdType<float, 256> operator-(const SimdType<float, 256> &lhs, const SimdType<float, 256> &rhs) {
  return _mm256_sub_ps(lhs, rhs);
}

inline SimdType<float, 256> operator*(const SimdType<float, 256> &lhs, const SimdType<float, 256> &rhs) {
  return _mm256_mul_ps(lhs, rhs);
}

inline SimdType<float, 256> operator/(const SimdType<float, 256> &lhs, const SimdType<float, 256> &rhs) {
  return _mm256_div_ps(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<float, 512>
//------------------------------------------------------------------------------

inline SimdType<float, 512> operator+(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  typedef simd::value<float, 512>::reg reg;
  return reg(
    SimdType<float, 256>(reg(lhs).r0) + SimdType<float, 256>(reg(rhs).r0),
    SimdType<float, 256>(reg(lhs).r1) + SimdType<float, 256>(reg(rhs).r1)
  );
}

inline SimdType<float, 512> operator-(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  typedef simd::value<float, 512>::reg reg;
  return reg(
    SimdType<float, 256>(reg(lhs).r0) - SimdType<float, 256>(reg(rhs).r0),
    SimdType<float, 256>(reg(lhs).r1) - SimdType<float, 256>(reg(rhs).r1)
  );
}

inline SimdType<float, 512> operator*(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  typedef simd::value<float, 512>::reg reg;
  return reg(
    SimdType<float, 256>(reg(lhs).r0) * SimdType<float, 256>(reg(rhs).r0),
    SimdType<float, 256>(reg(lhs).r1) * SimdType<float, 256>(reg(rhs).r1)
  );
}

inline SimdType<float, 512> operator/(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  typedef simd::value<float, 512>::reg reg;
  return reg(
    SimdType<float, 256>(reg(lhs).r0) / SimdType<float, 256>(reg(rhs).r0),
    SimdType<float, 256>(reg(lhs).r1) / SimdType<float, 256>(reg(rhs).r1)
  );
}
#endif

#if SIMD256
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<double, 256>
//------------------------------------------------------------------------------

inline SimdType<double, 256> operator+(const SimdType<double, 256> &lhs, const SimdType<double, 256> &rhs) {
  return _mm256_add_pd(lhs, rhs);
}

inline SimdType<double, 256> operator-(const SimdType<double, 256> &lhs, const SimdType<double, 256> &rhs) {
  return _mm256_sub_pd(lhs, rhs);
}

inline SimdType<double, 256> operator*(const SimdType<double, 256> &lhs, const SimdType<double, 256> &rhs) {
  return _mm256_mul_pd(lhs, rhs);
}

inline SimdType<double, 256> operator/(const SimdType<double, 256> &lhs, const SimdType<double, 256> &rhs) {
  return _mm256_div_pd(lhs, rhs);
}
#endif

#if SIMD256 && SIMD512OFFLOAD
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<double, 512>
//------------------------------------------------------------------------------

inline SimdType<double, 512> operator+(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  typedef simd::value<double, 512>::reg reg;
  return reg(
    SimdType<double, 256>(reg(lhs).r0) + SimdType<double, 256>(reg(rhs).r0),
    SimdType<double, 256>(reg(lhs).r1) + SimdType<double, 256>(reg(rhs).r1)
  );
}

inline SimdType<double, 512> operator-(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  typedef simd::value<double, 512>::reg reg;
  return reg(
    SimdType<double, 256>(reg(lhs).r0) - SimdType<double, 256>(reg(rhs).r0),
    SimdType<double, 256>(reg(lhs).r1) - SimdType<double, 256>(reg(rhs).r1)
  );
}

inline SimdType<double, 512> operator*(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  typedef simd::value<double, 512>::reg reg;
  return reg(
    SimdType<double, 256>(reg(lhs).r0) * SimdType<double, 256>(reg(rhs).r0),
    SimdType<double, 256>(reg(lhs).r1) * SimdType<double, 256>(reg(rhs).r1)
  );
}

inline SimdType<double, 512> operator/(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  typedef simd::value<double, 512>::reg reg;
  return reg(
    SimdType<double, 256>(reg(lhs).r0) / SimdType<double, 256>(reg(rhs).r0),
    SimdType<double, 256>(reg(lhs).r1) / SimdType<double, 256>(reg(rhs).r1)
  );
}
#endif

}

#endif // SIMD_SIMD256X86IMPL_HPP
