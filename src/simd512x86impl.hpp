#ifndef SIMD_SIMD512X86IMPL_HPP
#define SIMD_SIMD512X86IMPL_HPP

#include <immintrin.h>

namespace simd {

//------------------------------------------------------------------------------
// Memory loader implementations
//------------------------------------------------------------------------------

#if SIMD512
//------------------------------------------------------------------------------
// SimdLoader<simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::vector, 512> {
public:
  static inline simd::value<short, 512>::reg load(const short *mem) {
//#if defined(SIMD_KNC)
//    return simd::value<short, 512>::reg(
//      _mm256_load_si256(reinterpret_cast<__m256i const *>(mem)), 
//      _mm256_load_si256(reinterpret_cast<__m256i const *>(mem+SIMD512_SHORT_WIDTH/2))
//    );
//#else
    return _mm512_load_si512(mem);
//#endif
  }
  
  static inline simd::value<unsigned short, 512>::reg load(const unsigned short *mem) {
    return load(reinterpret_cast<const short *>(mem));
  }
  
  static inline simd::value<int, 512>::reg load(const int *mem) {
    return _mm512_load_si512(mem);
  }
  
  static inline simd::value<unsigned int, 512>::reg load(const unsigned int *mem) {
    return load(reinterpret_cast<const int *>(mem));
  }
  
  static inline simd::value<long, 512>::reg load(const long *mem) {
    return _mm512_load_si512(mem);
  }
  
  static inline simd::value<unsigned long, 512>::reg load(const unsigned long *mem) {
    return load(reinterpret_cast<const long *>(mem));
  }
  
  static inline simd::value<float, 512>::reg load(const float *mem) {
    return _mm512_load_ps(mem);
  }
  
  static inline simd::value<double, 512>::reg load(const double *mem) {
    return _mm512_load_pd(mem);
  }
};
#endif

#if SIMD512
//------------------------------------------------------------------------------
// SimdLoader<simd::unaligned|simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::unaligned|simd::vector, 512> {
public:
  static inline simd::value<short, 512>::reg load(const short *mem) {
#if defined(SIMD_KNC)
//    return simd::value<short, 512>::reg(
//      _mm256_lddqu_si256(reinterpret_cast<__m256i const *>(mem)), 
//      _mm256_lddqu_si256(reinterpret_cast<__m256i const *>(mem+SIMD512_SHORT_WIDTH/2))
//    );
    __m512i temp;
    temp = _mm512_loadunpacklo_epi32(temp, mem);
    temp = _mm512_loadunpackhi_epi32(temp, mem+SIMD512_SHORT_WIDTH);
    return temp;
#else
    return _mm512_loadu_si512(mem);
#endif
  }
  
  static inline simd::value<unsigned short, 512>::reg load(const unsigned short *mem) {
    return load(reinterpret_cast<const short *>(mem));
  }
  
  static inline simd::value<int, 512>::reg load(const int *mem) {
#if defined(SIMD_KNC)
    __m512i temp;
    temp = _mm512_loadunpacklo_epi32(temp, mem);
    temp = _mm512_loadunpackhi_epi32(temp, mem+SIMD512_INT_WIDTH);
    return temp;
#else
    return _mm512_loadu_si512(mem);
#endif
  }
  
  static inline simd::value<unsigned int, 512>::reg load(const unsigned int *mem) {
    return load(reinterpret_cast<const int *>(mem));
  }
  
  static inline simd::value<long, 512>::reg load(const long *mem) {
#if defined(SIMD_KNC)
    __m512i temp;
    temp = _mm512_loadunpacklo_epi64(temp, mem);
    temp = _mm512_loadunpackhi_epi64(temp, mem+SIMD512_LONG_WIDTH);
    return temp;
#else
    return _mm512_loadu_si512(mem);
#endif
  }
  
  static inline simd::value<unsigned long, 512>::reg load(const unsigned long *mem) {
    return load(reinterpret_cast<const long *>(mem));
  }
  
  static inline simd::value<float, 512>::reg load(const float *mem) {
#if defined(SIMD_KNC)
    __m512 temp;
    temp = _mm512_loadunpacklo_ps(temp, mem);
    temp = _mm512_loadunpackhi_ps(temp, mem+SIMD512_FLOAT_WIDTH);
    return temp;
#else
    return _mm512_loadu_ps(mem);
#endif
  }
  
  static inline simd::value<double, 512>::reg load(const double *mem) {
#if defined(SIMD_KNC)
    __m512d temp;
    temp = _mm512_loadunpacklo_pd(temp, mem);
    temp = _mm512_loadunpackhi_pd(temp, mem+SIMD512_DOUBLE_WIDTH);
    return temp;
#else
    return _mm512_loadu_pd(mem);
#endif
  }
};
#endif

#if SIMD512
//------------------------------------------------------------------------------
// SimdLoader<simd::unaligned|simd::broadcast, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdLoader<simd::unaligned|simd::broadcast, 512> {
public:
  static inline simd::value<short, 512>::reg load(const short *mem) {
#if defined(SIMD_KNC)
//    return simd::value<short, 512>::reg(
//      _mm256_set1_epi16(*mem)
//    );
    return _mm512_extload_epi32(mem, _MM_UPCONV_EPI32_UINT16, _MM_BROADCAST_1X16, _MM_HINT_NONE);
#else
    return _mm512_set1_epi16(*mem);
#endif
  }
  
  static inline simd::value<unsigned short, 512>::reg load(const unsigned short *mem) {
    return load(reinterpret_cast<const short *>(mem));
  }
  
  static inline simd::value<int, 512>::reg load(const int *mem) {
#if defined(SIMD_KNC)
    return _mm512_extload_epi32(mem, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
#else
    return _mm512_set1_epi32(*mem);
#endif
  } 
  
  static inline simd::value<unsigned int, 512>::reg load(const unsigned int *mem) {
    return load(reinterpret_cast<const int *>(mem));
  }
  
  static inline simd::value<long, 512>::reg load(const long *mem) {
#if defined(SIMD_KNC)
    return _mm512_extload_epi64(mem, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
#else
    return _mm512_set1_epi64(*mem);
#endif
  } 
  
  static inline simd::value<unsigned long, 512>::reg load(const unsigned long *mem) {
    return load(reinterpret_cast<const long *>(mem));
  }
  
  static inline simd::value<float, 512>::reg load(const float *mem) {
#if defined(SIMD_KNC)
    return _mm512_extload_ps(mem, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
#else
    return _mm512_set1_ps(*mem);
#endif
  }
  
  static inline simd::value<double, 512>::reg load(const double *mem) {
#if defined(SIMD_KNC)
    return _mm512_extload_pd(mem, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
#else
    return _mm512_set1_pd(*mem);
#endif
  }
};
#endif

//------------------------------------------------------------------------------
// Memory storer implementations
//------------------------------------------------------------------------------

#if SIMD512
//------------------------------------------------------------------------------
// SimdStorer<simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::vector, 512> {
public:
  static inline void store(simd::value<short, 512>::reg &reg, short *mem) {
//#if defined(SIMD_KNC)
//    typedef simd::value<short, 512>::reg r;
//    _mm256_store_si256(reinterpret_cast<__m256i*>(mem), r(reg).r0);
//    _mm256_store_si256(reinterpret_cast<__m256i*>(mem+SIMD512_SHORT_WIDTH/2), r(reg).r0);
//#else
    _mm512_store_si512(mem, reg);
//#endif
  }
  
  static inline void store(simd::value<unsigned short, 512>::reg &reg, unsigned short *mem) {
    store(reg, reinterpret_cast<short *>(mem));
  }
  
  static inline void store(simd::value<int, 512>::reg &reg, int *mem) {
    _mm512_store_si512(mem, reg);
  }
  
  static inline void store(simd::value<unsigned int, 512>::reg &reg, unsigned int *mem) {
    store(reg, reinterpret_cast<int *>(mem));
  }
  
  static inline void store(simd::value<long, 512>::reg &reg, long *mem) {
    _mm512_store_si512(mem, reg);
  }
  
  static inline void store(simd::value<unsigned long, 512>::reg &reg, unsigned long *mem) {
    store(reg, reinterpret_cast<long *>(mem));
  }
  
  static inline void store(simd::value<float, 512>::reg &reg, float *mem) {
    _mm512_store_ps(mem, reg);
  }
  
  static inline void store(simd::value<double, 512>::reg &reg, double *mem) {
    _mm512_store_pd(mem, reg);
  }
};
#endif

#if SIMD512
//------------------------------------------------------------------------------
// SimdStorer<simd::unaligned|simd::vector, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::unaligned|simd::vector, 512> {
public:
  static inline void store(simd::value<short, 512>::reg &reg, short *mem) {
#if defined(SIMD_KNC)
    //typedef simd::value<short, 512>::reg r;
    //_mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), r(reg).r0);
    //_mm256_storeu_si256(reinterpret_cast<__m256i*>(mem+SIMD512_SHORT_WIDTH/2), r(reg).r1);
    _mm512_packstorelo_epi32(mem, reg);
    _mm512_packstorehi_epi32(mem+SIMD512_SHORT_WIDTH, reg);
#else
    _mm512_storeu_si512(mem, reg);
#endif
  }
  
  static inline void store(simd::value<unsigned short, 512>::reg &reg, unsigned short *mem) {
    store(reg, reinterpret_cast<short *>(mem));
  }
  
  static inline void store(simd::value<int, 512>::reg &reg, int *mem) {
#if defined(SIMD_KNC)
    _mm512_packstorelo_epi32(mem, reg);
    _mm512_packstorehi_epi32(mem+SIMD512_INT_WIDTH, reg);
#else
    _mm512_storeu_si512(mem, reg);
#endif
  }
  
  static inline void store(simd::value<unsigned int, 512>::reg &reg, unsigned int *mem) {
    store(reg, reinterpret_cast<int *>(mem));
  }
  
  static inline void store(simd::value<long, 512>::reg &reg, long *mem) {
#if defined(SIMD_KNC)
    _mm512_packstorelo_epi64(mem, reg);
    _mm512_packstorehi_epi64(mem+SIMD512_LONG_WIDTH, reg);
#else
    _mm512_storeu_si512(mem, reg);
#endif
  }
  
  static inline void store(simd::value<unsigned long, 512>::reg &reg, unsigned long *mem) {
    store(reg, reinterpret_cast<long *>(mem));
  }
  
  static inline void store(simd::value<float, 512>::reg &reg, float *mem) {
#if defined(SIMD_KNC)
    _mm512_packstorelo_ps(mem, reg);
    _mm512_packstorehi_ps(mem+SIMD512_FLOAT_WIDTH, reg);
#else
    _mm512_storeu_ps(mem, reg);
#endif
  }
  
  static inline void store(simd::value<double, 512>::reg &reg, double *mem) {
#if defined(SIMD_KNC)
    _mm512_packstorelo_pd(mem, reg);
    _mm512_packstorehi_pd(mem+SIMD512_DOUBLE_WIDTH, reg);
#else
    _mm512_storeu_pd(mem, reg);
#endif
  }
};
#endif

#if SIMD512
//------------------------------------------------------------------------------
// SimdStorer<simd::unaligned|simd::broadcast, 512> implementation
//------------------------------------------------------------------------------

template<>
class SimdStorer<simd::unaligned|simd::broadcast, 512> {
public:
  static inline void store(simd::value<short, 512>::reg &reg, short *mem) {
#if defined(SIMD_KNC)
    __m512i temp1, temp2, temp3;
    temp1 = _mm512_slli_epi32(_mm512_permutevar_epi32(_mm512_xor_si512(temp1, temp1), reg), 16);
    temp2 = _mm512_srli_epi32 (temp1, 16);
    temp3 = _mm512_or_epi32(temp1, temp2);
    SimdStorer<simd::unaligned|simd::vector, 512>::store(
      temp3, 
      mem
    );
//    typedef simd::value<short, 512>::reg r;
//    __m256i temp = _mm256_broadcastw_epi16(_mm256_castsi256_si128(r(reg).r0));
//    _mm256_storeu_si256(reinterpret_cast<__m256i *>(mem), temp);
//    _mm256_storeu_si256(reinterpret_cast<__m256i *>(mem+SIMD512_SHORT_WIDTH/2), temp);
#else
    __m512i temp = _mm512_broadcastw_epi16(_mm512_castsi512_si128(reg));
    SimdStorer<simd::unaligned|simd::vector, 512>::store(
      temp, 
      mem
    );
#endif
  }
  
  static inline void store(simd::value<unsigned short, 512>::reg &reg, unsigned short *mem) {
    store(reg, reinterpret_cast<short *>(mem));
  }
  
  static inline void store(simd::value<int, 512>::reg &reg, int *mem) {
    __m512i temp;
#if defined(SIMD_KNC)
    temp = _mm512_permutevar_epi32(_mm512_xor_si512(temp, temp), reg);
#else
    temp = _mm512_broadcastd_epi32(_mm512_castsi512_si128(reg));
#endif
    SimdStorer<simd::unaligned|simd::vector, 512>::store(
      temp,                                                
      mem
    );
  }
  
  static inline void store(simd::value<unsigned int, 512>::reg &reg, unsigned int *mem) {
    store(reg, reinterpret_cast<int *>(mem));
  }
  
  static inline void store(simd::value<long, 512>::reg &reg, long *mem) {
    __m512i temp;
#if defined(SIMD_KNC)
    temp = _mm512_permute4f128_epi32(
      _mm512_swizzle_epi64(reg, _MM_SWIZ_REG_AAAA), 
      _MM_PERM_AAAA
    );
#else
    temp = _mm512_broadcastq_epi64(_mm512_castsi512_si128(reg));
#endif
    SimdStorer<simd::unaligned|simd::vector, 512>::store(
      temp,                                                
      mem
    );
  }
  
  static inline void store(simd::value<unsigned long, 512>::reg &reg, unsigned long *mem) {
    store(reg, reinterpret_cast<long *>(mem));
  }
  
  static inline void store(simd::value<float, 512>::reg &reg, float *mem) {
    __m512 temp;
#if defined(SIMD_KNC)
    temp = _mm512_permute4f128_ps(
      _mm512_swizzle_ps(reg, _MM_SWIZ_REG_AAAA), 
      _MM_PERM_AAAA
    );
#else
    temp = _mm512_broadcastss_ps(_mm512_castps512_ps128(reg));
#endif
    SimdStorer<simd::unaligned|simd::vector, 512>::store(
      temp,
      mem
    );
  }
  
  static inline void store(simd::value<double, 512>::reg &reg, double *mem) {
    __m512d temp;
#if defined(SIMD_KNC)
    temp = _mm512_castps_pd(
      _mm512_permute4f128_ps(
        _mm512_castpd_ps(_mm512_swizzle_pd(reg, _MM_SWIZ_REG_AAAA)), 
        _MM_PERM_AAAA
      )
    );
#else
    temp = _mm512_broadcastsd_pd(_mm512_castpd512_pd128(reg))
#endif
    SimdStorer<simd::unaligned|simd::vector, 512>::store(
      temp,
      mem
    );
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

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<short, 512>
//------------------------------------------------------------------------------

inline SimdType<short, 512> operator+(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
#if defined(SIMD_KNC)
//  typedef simd::value<short, 512>::reg r;
//  r res;
//  res.r0 = _mm256_add_epi16(r(lhs).r0, r(rhs).r0);
//  res.r1 = _mm256_add_epi16(r(lhs).r1, r(rhs).r1);
//  return res;
  __m512i temp1, temp2;
  temp1 = _mm512_srli_epi32(_mm512_slli_epi32(_mm512_add_epi32(lhs, rhs), 16), 16);
  temp2 = _mm512_slli_epi32(_mm512_add_epi32(_mm512_srli_epi32(lhs, 16), _mm512_srli_epi32(rhs, 16)), 16);
  return _mm512_or_epi32(temp1, temp2);
#else
  return _mm512_add_epi16(lhs, rhs);
#endif
}

inline SimdType<short, 512> operator-(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
#if defined(SIMD_KNC)
//  typedef simd::value<short, 512>::reg r;
//  r res;
//  res.r0 = _mm256_sub_epi16(r(lhs).r0, r(rhs).r0);
//  res.r1 = _mm256_sub_epi16(r(lhs).r1, r(rhs).r1);
//  return res;
  __m512i temp1, temp2;
  temp1 = _mm512_srli_epi32(_mm512_slli_epi32(_mm512_sub_epi32(lhs, rhs), 16), 16);
  temp2 = _mm512_slli_epi32(_mm512_sub_epi32(_mm512_srli_epi32(lhs, 16), _mm512_srli_epi32(rhs, 16)), 16);
  return _mm512_or_epi32(temp1, temp2);
#else
  return _mm512_sub_epi16(lhs, rhs);
#endif
}

inline SimdType<short, 512> operator*(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
#if defined(SIMD_KNC)
//  typedef simd::value<short, 512>::reg r;
//  r res;
//  res.r0 = _mm256_mullo_epi16(r(lhs).r0, r(rhs).r0);
//  res.r1 = _mm256_mullo_epi16(r(lhs).r1, r(rhs).r1);
//  return res;
  __m512i temp1, temp2;
  temp1 = _mm512_srli_epi32(_mm512_slli_epi32(_mm512_mullo_epi32(lhs, rhs), 16), 16);
  temp2 = _mm512_slli_epi32(_mm512_mullo_epi32(_mm512_srli_epi32(lhs, 16), _mm512_srli_epi32(rhs, 16)), 16);
  return _mm512_or_epi32(temp1, temp2);
#else
  return _mm512_mullo_epi16(lhs, rhs);
#endif
}

inline SimdType<short, 512> operator/(const SimdType<short, 512> &lhs, const SimdType<short, 512> &rhs) {
#if defined(SIMD_KNC)
//  typedef simd::value<short, 512>::reg r;
//  r res;
//  res.r0 = _mm256_div_epi16(r(lhs).r0, r(rhs).r0);
//  res.r1 = _mm256_div_epi16(r(lhs).r1, r(rhs).r1);
//  return res;
  return _mm512_div_epi16(lhs, rhs);
#else
  return _mm512_div_epi16(lhs, rhs);
#endif
}
#endif

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned short, 512>
//------------------------------------------------------------------------------

inline SimdType<unsigned short, 512> operator+(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short, 512> &rhs) {
  return SimdType<unsigned short, 512>(SimdType<short, 512>(lhs) + SimdType<short, 512>(rhs));
}

inline SimdType<unsigned short, 512> operator-(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short,  512> &rhs) {
  return SimdType<unsigned short, 512>(SimdType<short, 512>(lhs) - SimdType<short, 512>(rhs));
}

inline SimdType<unsigned short, 512> operator*(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short, 512> &rhs) {
  return SimdType<unsigned short, 512>(SimdType<short, 512>(lhs) * SimdType<short, 512>(rhs));
}

inline SimdType<unsigned short, 512> operator/(const SimdType<unsigned short, 512> &lhs, const SimdType<unsigned short, 512> &rhs) {
#if defined(SIMD_KNC)
//  typedef simd::value<unsigned short, 512>::reg r;
//  r res;
//  res.r0 = _mm256_div_epu16(r(lhs).r0, r(rhs).r0);
//  res.r1 = _mm256_div_epu16(r(lhs).r1, r(rhs).r1);
//  return res;
  return _mm512_div_epu16(lhs, rhs);
#else
  return _mm512_div_epu16(lhs, rhs);
#endif
}
#endif

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<int, 512>
//------------------------------------------------------------------------------

inline SimdType<int, 512> operator+(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  return _mm512_add_epi32(lhs, rhs);
}

inline SimdType<int, 512> operator-(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  return _mm512_sub_epi32(lhs, rhs);
}

inline SimdType<int, 512> operator*(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  return _mm512_mullo_epi32(lhs, rhs);
}

inline SimdType<int, 512> operator/(const SimdType<int, 512> &lhs, const SimdType<int, 512> &rhs) {
  return _mm512_div_epi32(lhs, rhs);
}
#endif

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned int, 512>
//------------------------------------------------------------------------------

inline SimdType<unsigned int, 512> operator+(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  return SimdType<unsigned int, 512>(SimdType<int, 512>(lhs) + SimdType<int, 512>(rhs));
}

inline SimdType<unsigned int, 512> operator-(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  return SimdType<unsigned int, 512>(SimdType<int, 512>(lhs) - SimdType<int, 512>(rhs));
}

inline SimdType<unsigned int, 512> operator*(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  return SimdType<unsigned int, 512>(SimdType<int, 512>(lhs) * SimdType<int, 512>(rhs));
}

inline SimdType<unsigned int, 512> operator/(const SimdType<unsigned int, 512> &lhs, const SimdType<unsigned int, 512> &rhs) {
  return _mm512_div_epu32(lhs, rhs);
}
#endif

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<long, 512>
//------------------------------------------------------------------------------

inline SimdType<long, 512> operator+(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
  return _mm512_add_epi64(lhs, rhs);
}

inline SimdType<long, 512> operator-(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
#if defined(SIMD_KNC)
  __m512i result;
  __mmask16 borrow = _mm512_int2mask(0x0000);
  __mmask16 mask1 = _mm512_int2mask(0x5555);
  
  result = _mm512_mask_subsetb_epi32(lhs, mask1, borrow, rhs, &borrow);
  borrow = _mm512_int2mask(_mm512_mask2int(borrow) << 1);
  result = _mm512_sbb_epi32(lhs, borrow, rhs, &borrow);
  
  return result;
#else
  return _mm512_sub_epi64(lhs, rhs);
#endif
}

inline SimdType<long, 512> operator*(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
#if defined(SIMD_KNC)
  __mmask16 lomask = _mm512_int2mask(0x5555);
  __mmask16 himask = _mm512_int2mask(0xAAAA);
  
  __m512i zero;
  zero = _mm512_xor_epi32(zero, zero);
  
  __m512i res0 = _mm512_mullo_epi32(lhs, rhs);
  
  __m512i res1 = _mm512_shuffle_epi32(
    _mm512_mask_mulhi_epu32(zero, lomask, lhs, rhs),
    _MM_PERM_CDAB
  );
  
  __m512i rhsswap = _mm512_shuffle_epi32(rhs, _MM_PERM_CDAB);
  __m512i res2 = _mm512_mullo_epi32(lhs, rhsswap);
  __m512i res3 = _mm512_shuffle_epi32(res2, _MM_PERM_CDAB);
  __m512i res4 = _mm512_mask_add_epi32(zero, himask, res2, res3);
  
  return _mm512_add_epi32(_mm512_add_epi32(res0, res1), res4);
#else
  return _mm512_mullo_epi64(lhs, rhs);
#endif
}

inline SimdType<long, 512> operator/(const SimdType<long, 512> &lhs, const SimdType<long, 512> &rhs) {
  return _mm512_div_epi64(lhs, rhs);
}
#endif

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<unsigned long, 512>
//------------------------------------------------------------------------------

inline SimdType<unsigned long, 512> operator+(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  return SimdType<unsigned long, 512>(SimdType<long, 512>(lhs) + SimdType<long, 512>(rhs));
}

inline SimdType<unsigned long, 512> operator-(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  return SimdType<unsigned long, 512>(SimdType<long, 512>(lhs) - SimdType<long, 512>(rhs));
}

inline SimdType<unsigned long, 512> operator*(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  return SimdType<unsigned long, 512>(SimdType<long, 512>(lhs) * SimdType<long, 512>(rhs));
}

inline SimdType<unsigned long, 512> operator/(const SimdType<unsigned long, 512> &lhs, const SimdType<unsigned long, 512> &rhs) {
  return _mm512_div_epu64(lhs, rhs);
}
#endif

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<float, 512>
//------------------------------------------------------------------------------

inline SimdType<float, 512> operator+(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  return _mm512_add_ps(lhs, rhs);
}

inline SimdType<float, 512> operator-(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  return _mm512_sub_ps(lhs, rhs);
}

inline SimdType<float, 512> operator*(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  return _mm512_mul_ps(lhs, rhs);
}

inline SimdType<float, 512> operator/(const SimdType<float, 512> &lhs, const SimdType<float, 512> &rhs) {
  return _mm512_div_ps(lhs, rhs);
}
#endif

#if SIMD512
//------------------------------------------------------------------------------
// Implementation of overloaded operators for SimdType<double, 512>
//------------------------------------------------------------------------------

inline SimdType<double, 512> operator+(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  return _mm512_add_pd(lhs, rhs);
}

inline SimdType<double, 512> operator-(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  return _mm512_sub_pd(lhs, rhs);
}

inline SimdType<double, 512> operator*(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  return _mm512_mul_pd(lhs, rhs);
}

inline SimdType<double, 512> operator/(const SimdType<double, 512> &lhs, const SimdType<double, 512> &rhs) {
  return _mm512_div_pd(lhs, rhs);
}
#endif

}

#endif // SIMD_SIMD512X86IMPL_HPP
