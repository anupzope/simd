#ifndef SIMD_SIMD512X86DEF_HPP
#define SIMD_SIMD512X86DEF_HPP

#include <immintrin.h>

#include <iostream>
#include <iomanip>

namespace simd {

//void print_mm512i_epi32(__m256i var) {
//  int32_t *val = (int32_t*) &var;
//  for(int i = 0; i < 15; ++i) {
//    std::cout << val[i] << " ";
//  }
//  std::cout << val[15] << std::endl;
//}

//void print_mm512i_epi64(__m256i var) {
//  int64_t *val = (int64_t*) &var;
//  for(int i = 0; i < 7; ++i) {
//    std::cout << val[i] << " ";
//  }
//  std::cout << val[7] << std::endl;
//}

#if SIMD512

//void print_epi64(__m512i *op) {
//  __int64* values = reinterpret_cast<__int64*>(op);
//  
//  for(int i = 0; i < 8; ++i) {
//    std::cout << std::hex << std::setw(16) << values[i] << std::endl;
//  }
//  std::cout << std::dec << std::endl;
//}

//void print_epi32(__m512i *op) {
//  __int32* values = reinterpret_cast<__int32*>(op);
//  
//  for(int i = 0; i < 16; i+=2) {
//    std::cout << std::hex << std::setw(8) << values[i] << " " << std::hex << std::setw(8) << values[i+1] << std::endl;
//  }
//  std::cout << std::dec << std::endl;
//}

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
  typedef __m512i register_type;
  typedef __mmask16 mask_register_type;
  
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
  typedef __m512i register_type;
  typedef __mmask8 mask_register_type;
  
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
  typedef __m512 register_type;
  typedef __mmask16 mask_register_type;
  
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
  typedef __m512d register_type;
  typedef __mmask8 mask_register_type;
  
  static constexpr int num_regs = NW/simd::defaults<double>::nway;
  static constexpr int num_vals = NW;
  static constexpr int num_bvals = NW;
  static constexpr int num_vals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_reg = NW/num_regs;
  static constexpr int num_bvals_per_val = 1;
};

/* Conversion functions */

// int -> long
inline __m512i cvt512_epi32lo_epi64(__m512i op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtepi32_epi64(_mm512_castsi512_si256(op));
#else
  __mmask16 k1 = _mm512_int2mask(0b0000111100001111);
  __mmask16 k2 = _mm512_int2mask(0b1010101010101010);
  
  __m512i temp = _mm512_permute4f128_epi32(op, _MM_PERM_BBAA);
  temp = _mm512_mask_shuffle_epi32(temp, k1, temp, _MM_PERM_BBAA);
  temp = _mm512_mask_shuffle_epi32(temp, _mm512_knot(k1), temp, _MM_PERM_DDCC);
  temp = _mm512_mask_srai_epi32(temp, k2, temp, 32);
  return _mm512_mask_slli_epi32(temp, k2, temp, 31);
#endif
}

inline __m512i cvt512_epi32hi_epi64(__m512i op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtepi32_epi64(_mm512_castsi512_si256(_mm512_shuffle_i32x4(op, op, _MM_SHUFFLE(1, 0, 3, 2))));
#else
  __mmask16 k1 = _mm512_int2mask(0b0000111100001111);
  __mmask16 k2 = _mm512_int2mask(0b1010101010101010);
  
  __m512i temp = _mm512_permute4f128_epi32(op, _MM_PERM_DDCC);
  temp = _mm512_mask_shuffle_epi32(temp, k1, temp, _MM_PERM_BBAA);
  temp = _mm512_mask_shuffle_epi32(temp, _mm512_knot(k1), temp, _MM_PERM_DDCC);
  temp = _mm512_mask_srai_epi32(temp, k2, temp, 32);
  return _mm512_mask_slli_epi32(temp, k2, temp, 31);
#endif
}

// int -> float
inline __m512 cvt512_epi32_ps(__m512i op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtepi32_ps(op);
#else
  return _mm512_cvtfxpnt_round_adjustepi32_ps(op, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC, _MM_EXPADJ_NONE);
#endif
}

// int -> double
inline __m512d cvt512_epi32lo_pd(__m512i op) {
  return _mm512_cvtepi32lo_pd(op);
}

inline __m512d cvt512_epi32hi_pd(__m512i op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtepi32lo_pd(_mm512_shuffle_i32x4(op, op, _MM_SHUFFLE(1, 0, 3, 2)));
#else
  return _mm512_cvtepi32lo_pd(_mm512_permute4f128_epi32(op, _MM_PERM_BADC));
#endif
}

// long -> int
inline __m512i cvt512_epi64_epi32lo(__m512i op) {
#if defined(SIMD_AVX512F)
  return _mm512_castsi256_si512(_mm512_cvtepi64_epi32(op));
#else
  __m512i zero;
  return _mm512_swizzle_epi32(
     _mm512_permute4f128_epi32(
      _mm512_swizzle_epi64(
        _mm512_swizzle_epi64(
          _mm512_mask_shuffle_epi32(
            _mm512_xor_si512(zero, zero), 
            _mm512_int2mask(0b0011001100110011), 
            op, 
            _MM_PERM_DBCA
          ), 
          _MM_SWIZ_REG_DACB
        ), 
        _MM_SWIZ_REG_DACB
      ), 
      _MM_PERM_DBCA
    ), 
    _MM_SWIZ_REG_BADC
  );
#endif
}

inline __m512i cvt512_epi64_epi32hi(__m512i op) {
#if defined(SIMD_AVX512F)
  __m512i temp = _mm512_castsi256_si512(_mm512_cvtepi64_epi32(op));
  return _mm512_shuffle_i32x4(temp, temp, _MM_SHUFFLE(1, 0, 3, 2));
#else
  __m512i zero;
  return _mm512_swizzle_epi32(
     _mm512_permute4f128_epi32(
      _mm512_swizzle_epi64(
        _mm512_swizzle_epi64(
          _mm512_mask_shuffle_epi32(
            _mm512_xor_si512(zero, zero), 
            _mm512_int2mask(0b0101010101010101), 
            op, 
            _MM_PERM_DBCA
          ), 
          _MM_SWIZ_REG_DACB
        ), 
        _MM_SWIZ_REG_DACB
      ), 
      _MM_PERM_CADB
    ), 
    _MM_SWIZ_REG_BADC
  );
#endif
}

inline __m512i cvt512_epi64x2_epi32(__m512i hi, __m512i lo) {
#if defined(SIMD_AVX512F)
  __m512i hi1 = _mm512_castsi256_si512(_mm512_cvtepi64_epi32(hi));
  __m512i lo1 = _mm512_castsi256_si512(_mm512_cvtepi64_epi32(lo));
  return _mm512_mask_blend_epi32(
    _mm512_int2mask(0b0000000011111111), 
    _mm512_shuffle_i32x4(hi1, hi1, _MM_SHUFFLE(1, 0, 3, 2)), 
    lo1
  );
#else
  __m512i temp = _mm512_permute4f128_epi32(
    _mm512_swizzle_epi64(
      _mm512_swizzle_epi64(
        _mm512_shuffle_epi32(
          _mm512_mask_blend_epi32(
            _mm512_int2mask(0b0101010101010101), 
            _mm512_swizzle_epi32(hi, _MM_SWIZ_REG_CDAB), 
            lo
          )
          , _MM_PERM_DBCA
        ), 
        _MM_SWIZ_REG_DACB
      ), 
      _MM_SWIZ_REG_DACB
    ), 
    _MM_PERM_DBCA
  );
  return _mm512_mask_swizzle_epi32(temp, _mm512_int2mask(0b0000000011111111), temp, _MM_SWIZ_REG_BADC);
#endif
}

// long -> float
inline __m512 cvt512_epi64_pslo(__m512i op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtepi64_ps(op);
#else
  __m512i temp = _mm512_swizzle_epi64(
    _mm512_swizzle_epi64(
      _mm512_shuffle_epi32(op, _MM_PERM_DBCA),
      _MM_SWIZ_REG_DACB
    ),
    _MM_SWIZ_REG_DACB
  );
  temp = _mm512_mask_shuffle_epi32(temp, _mm512_int2mask(0b0000111100001111), temp, _MM_PERM_BADC);
  
  __m512i lo = _mm512_permute4f128_epi32(temp, _MM_PERM_CACA); // unsigned part
  __m512i hi = _mm512_permute4f128_epi32(temp, _MM_PERM_DBDB); // signed part
  
  __m512 lops = _mm512_cvtfxpnt_round_adjustepu32_ps(lo, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC, _MM_EXPADJ_NONE);
  __m512 hips = _mm512_cvtfxpnt_round_adjustepi32_ps(hi, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC, _MM_EXPADJ_32);
  
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
  __mmask16 k = _mm512_int2mask(0b0000000011111111);
  
  return _mm512_mask_add_ps(_mm512_castsi512_ps(zero), k, lops, hips);
#endif
}

inline __m512 cvt512_epi64_pshi(__m512i op) {
#if defined(SIMD_AVX512F)
  __m512 temp = _mm512_castps256_ps512(_mm512_cvtepi64_ps(op));
  return _mm512_shuffle_f32x4(temp, temp, _MM_SHUFFLE(1, 0, 3, 2));
#else
  __m512i temp = _mm512_swizzle_epi64(
    _mm512_swizzle_epi64(
      _mm512_shuffle_epi32(op, _MM_PERM_DBCA),
      _MM_SWIZ_REG_DACB
    ),
    _MM_SWIZ_REG_DACB
  );
  temp = _mm512_mask_shuffle_epi32(temp, _mm512_int2mask(0b0000111100001111), temp, _MM_PERM_BADC);
  
  __m512i lo = _mm512_permute4f128_epi32(temp, _MM_PERM_CACA); // unsigned part
  __m512i hi = _mm512_permute4f128_epi32(temp, _MM_PERM_DBDB); // signed part
  
  __m512 lops = _mm512_cvtfxpnt_round_adjustepu32_ps(lo, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC, _MM_EXPADJ_NONE);
  __m512 hips = _mm512_cvtfxpnt_round_adjustepi32_ps(hi, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC, _MM_EXPADJ_32);
  
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
  __mmask16 k = _mm512_int2mask(0b1111111100000000);
  
  return _mm512_mask_add_ps(_mm512_castsi512_ps(zero), k, lops, hips);
#endif
}

inline __m512 cvt512_epi64x2_ps(__m512i hi, __m512i lo) {
  return _mm512_mask_blend_ps(_mm512_int2mask(0b0000000011111111), cvt512_epi64_pshi(hi), cvt512_epi64_pslo(lo));
}

// long -> double
inline __m512d cvt512_epi64_pd(__m512i op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtepi64_pd(op);
#else
  static const double mul = 4294967296.0;
  
  __m512i temp = _mm512_swizzle_epi64(
    _mm512_swizzle_epi64(
      _mm512_shuffle_epi32(op, _MM_PERM_DBCA), 
      _MM_SWIZ_REG_DACB
    ), 
    _MM_SWIZ_REG_DACB
  ); 
  temp = _mm512_mask_swizzle_epi32(temp, _mm512_int2mask(0b0000111100001111), temp, _MM_SWIZ_REG_BADC);
  
  __m512i lo = _mm512_permute4f128_epi32(temp, _MM_PERM_DBCA); // unsigned part
  __m512i hi = _mm512_permute4f128_epi32(temp, _MM_PERM_CADB); // signed part
  
  __m512d lopd = _mm512_cvtepu32lo_pd(lo);
  __m512d hipd = _mm512_cvtepi32lo_pd(hi);
  __m512d himl = _mm512_extload_pd(&mul, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  
  return _mm512_fmadd_pd(himl, hipd, lopd);
#endif
}

// float -> int
inline __m512i cvt512_ps_epi32(__m512 op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtps_epi32(op);
#else
  return _mm512_cvtfxpnt_round_adjustps_epi32(op, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC, _MM_EXPADJ_NONE);
#endif
}

// float -> long
#if !defined(SIMD_AVX512F)
__m512i cvt512_pshilo_epi64(__m512 op) {
  __m512i result;
  
  __mmask16 himask = _mm512_int2mask(0b1010101010101010);
  __mmask16 lomask = _mm512_int2mask(0b0101010101010101);
  
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
  
  // mask for mantisa
  static const int mant_mask_val = 0x007fffff;
  __m512i mant = _mm512_extload_epi32(&mant_mask_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  
  // mask for exponent + mantisa
  static const int mantexp_mask_val = 0x7fffffff;
  __m512i mantexp = _mm512_extload_epi32(&mantexp_mask_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  
  // obtain absolute value of op
  __m512i absop = _mm512_and_epi32(_mm512_castps_si512(op), mantexp);
  
  // obtain raw exponent of op
  __m512i expraw = _mm512_andnot_epi64(mant, absop);
  
  // obtain mantisa
  static const int implicit_mantisa_val = 0x00800000;
  __m512i implicit_mantisa = _mm512_extload_epi32(&implicit_mantisa_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  __m512i mantisa = _mm512_or_epi32(implicit_mantisa, _mm512_and_si512(absop, mant));
  
  // obtain exponent shifted to 23
  static const int exp23_base_val = 150;
  __m512i exp23_base = _mm512_extload_epi32(&exp23_base_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  __m512i exp23 = _mm512_sub_epi32(_mm512_srli_epi32(expraw, 23), exp23_base);
  
  static const int exp32_cutoff_val = 32;
  __m512i exp32_cutoff = _mm512_extload_epi32(&exp32_cutoff_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  __mmask16 exp23lt32 = _mm512_cmp_epi32_mask(exp23, exp32_cutoff, _MM_CMPINT_LT);
  
  static const int exp40_cutoff_val = 40;
  __m512i exp40_cutoff = _mm512_extload_epi32(&exp40_cutoff_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  __mmask16 exp23lt40 = _mm512_cmp_epi32_mask(exp23, exp40_cutoff, _MM_CMPINT_LT);
  
  static const int sign_val = 0x80000000;
  __m512i sign = _mm512_extload_epi32(&sign_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  
  // For numbers with exponent in range [55, 63)
  result = _mm512_mask_blend_epi32(
    himask,
    _mm512_mask_mov_epi32(zero, exp23lt40, zero),
    _mm512_mask_sllv_epi32(sign, exp23lt40, mantisa, _mm512_sub_epi32(exp23, exp32_cutoff))
  );
  
  // For numbers with exponent in range [23, 55)
  result = _mm512_mask_blend_epi32(
    himask,
    _mm512_mask_sllv_epi32(result, exp23lt32, mantisa, exp23),
    _mm512_mask_srlv_epi32(result, exp23lt32, mantisa, _mm512_sub_epi32(exp32_cutoff, exp23))
  );
  
  // number 2^23
  static const int num2to23_val = 0x4b000000;
  __m512i num2to23 = _mm512_extload_epi32(&num2to23_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  
  // For numbers with exponent less than 23
  result = _mm512_mask_mov_epi32(
    result,
    _mm512_kand(lomask, _mm512_cmplt_ps_mask(_mm512_castsi512_ps(expraw), _mm512_castsi512_ps(num2to23))),
    _mm512_sub_epi32(_mm512_castps_si512(_mm512_add_round_ps(_mm512_castsi512_ps(num2to23), _mm512_castsi512_ps(absop), _MM_FROUND_TO_ZERO)), num2to23)
  );
  
  // For negative numbers find their 2's complement
  __mmask16 negmask = _mm512_cmplt_ps_mask(op, _mm512_castsi512_ps(zero));
  __mmask16 borrow, borrow2;
  __m512i numneg = _mm512_mask_subsetb_epi32(zero, lomask, lomask, result, &borrow);
  int bint = _mm512_mask2int(borrow);
  bint += bint;
  borrow2 = _mm512_int2mask(bint);
  numneg = _mm512_mask_sbb_epi32(numneg, himask, borrow2, result, &borrow);
  result = _mm512_mask_mov_epi32(result, negmask, numneg);
  
  return result;
  
//union dtol {
//  float dbl;
//  int lng;
//};
//
//int main(int argc, char **argv) {
//  alignas(64) float src[16];
//  alignas(64) long dst[8];
//  
//  std::srand(std::time(NULL));
//  for(int i = 0; i < 16; ++i) {
//    src[i] = 1.0;
//    for(long j = 0; j < 2; ++j) {
//      src[i] *= (std::rand()%2 ? -1.0 : 1.0)*std::rand();
//    }
//  }
//  
//  //int raw = 0x4a512928; // positive number less than 2^23
//  //int raw = 0xca512928; // negative number less than 2^23
//  //int raw = 0xdb5539a8;
//  //_mm512_store_ps(src, _mm512_castsi512_ps(_mm512_extload_epi32(&raw, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE)));
//  
//  _mm512_store_si512(dst, cvt_pslo_epi64(_mm512_load_ps(src)));
//  
//  dtol t;
//  for(int i = 0; i < 8; ++i) {
//    t.dbl = src[i];
//    std::cout << std::dec << "custom cvt: 0x" << std::hex << dst[i] << ", " << std::dec << dst[i];
//    std::cout << std::dec << ", compiler cvt: 0x" << std::hex << (long)src[i] << ", " << std::dec << (long)src[i];
//    std::cout << ", original number: 0x" << std::hex << t.lng << std::dec << ", " << src[i] << std::endl;
//    assert(dst[i] == (long)src[i]);
//  }
//  
//  return 0;
//}
}
#endif

inline __m512i cvt512_pslo_epi64(__m512 op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtps_epi64(_mm512_castps512_ps256(op));
#else
  // Obtain low elements in packed float
  __m512i temp = _mm512_permute4f128_epi32(_mm512_castps_si512(op), _MM_PERM_BBAA);
  temp = _mm512_mask_swizzle_epi32(temp, _mm512_int2mask(0b1111000011110000), temp, _MM_SWIZ_REG_BADC);
  temp = _mm512_shuffle_epi32(temp, _MM_PERM_BBAA);
  return cvt512_pshilo_epi64(_mm512_castsi512_ps(temp));
#endif
}

inline __m512i cvt512_pshi_epi64(__m512 op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtps_epi64(_mm512_castps512_ps256(_mm512_shuffle_f32x4(op, op, _MM_SHUFFLE(1, 0, 3, 2))));
#else
  // Obtain hi elements in packed float
  __m512i temp = _mm512_permute4f128_epi32(_mm512_castps_si512(op), _MM_PERM_DDCC);
  temp = _mm512_mask_swizzle_epi32(temp, _mm512_int2mask(0b1111000011110000), temp, _MM_SWIZ_REG_BADC);
  temp = _mm512_shuffle_epi32(temp, _MM_PERM_BBAA);
  return cvt512_pshilo_epi64(_mm512_castsi512_ps(temp));
#endif
}

// float -> double
inline __m512d cvt512_pslo_pd(__m512 op) {
  return _mm512_cvtpslo_pd(op);
}

inline __m512d cvt512_pshi_pd(__m512 op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtpslo_pd(_mm512_shuffle_f32x4(op, op, _MM_SHUFFLE(1, 0, 3, 2)));
#else
  return _mm512_cvtpslo_pd(_mm512_permute4f128_ps(op, _MM_PERM_BADC));
#endif
}

// double -> int
inline __m512i cvt512_pd_epi32lo(__m512d op) {
#if defined(SIMD_AVX512F)
  return _mm512_castsi256_si512(_mm512_cvtpd_epi32(op));
#else
  return _mm512_cvtfxpnt_roundpd_epi32lo(op, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);
#endif
}

inline __m512i cvt512_pd_epi32hi(__m512d op) {
#if defined(SIMD_AVX512F)
  __m512i temp = _mm512_castsi256_si512(_mm512_cvtpd_epi32(op));
  return _mm512_shuffle_i32x4(temp, temp, _MM_SHUFFLE(1, 0, 3, 2));
#else
  return _mm512_permute4f128_epi32(_mm512_cvtfxpnt_roundpd_epi32lo(op, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC), _MM_PERM_BADC);
#endif
}

inline __m512i cvt512_pdx2_epi32(__m512d hi, __m512d lo) {
  return _mm512_mask_blend_epi32(_mm512_int2mask(0b0000000011111111), cvt512_pd_epi32hi(hi), cvt512_pd_epi32lo(lo));
}

// double -> long
inline __m512i cvt512_pd_epi64(__m512d op) {
#if defined(SIMD_AVX512F)
  return _mm512_cvtpd_epi64(op);
#else
  __m512i result;
  
  // mask for hi 32 bits of each double
  __mmask16 himask = _mm512_int2mask(0b1010101010101010);
  
  // mask for lo 32 bits of each double
  __mmask16 lomask = _mm512_int2mask(0b0101010101010101);
  
  // zero
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
  
  // integer bit width
  static const int ceil32_val = 32;
  __m512i ceil32 = _mm512_extload_epi32(&ceil32_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  
  // bit mask for mantisa part of a double
  static const long mant_mask_val = 0x000fffffffffffff;
  __m512i mant = _mm512_extload_epi64(&mant_mask_val, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  
  // Get absolute value of op
  static const long mantexp_mask_val = 0x7fffffffffffffff;
  __m512i mantexp = _mm512_extload_epi64(&mantexp_mask_val, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  __m512i absop = _mm512_and_epi64(_mm512_castpd_si512(op), mantexp);
  
  // Get raw exponent of op
  __m512i expraw = _mm512_andnot_epi64(mant, absop);
  
  // number 2^52
  static const long num2to52_val = 0x4330000000000000;
  __m512i num2to52 = _mm512_extload_epi64(&num2to52_val, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  
  // Get exponent shifted to 52
  static const long longexp_val = 1075;
  __m512i longexp = _mm512_extload_epi64(&longexp_val, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  __m512i exp52 = _mm512_shuffle_epi32(
    _mm512_sub_epi32(
      _mm512_srli_epi32(
        _mm512_mask_mov_epi32(zero, lomask, _mm512_shuffle_epi32(expraw, _MM_PERM_CDAB)),
        20
      ),
      longexp
    ),
    _MM_PERM_CCAA
  );
  
  // Get mantisa of each double in op
  static const long implicit_mantisa_val = 0x0010000000000000;
  __m512i implicit_mantisa = _mm512_extload_epi64(&implicit_mantisa_val, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  __m512i mantisa = _mm512_or_epi64(_mm512_and_epi64(_mm512_castpd_si512(op), mant), implicit_mantisa);
  
  // Get mask with 52-shifted exponent less than 32
  __mmask16 exp52lt32 = _mm512_cmplt_epi32_mask(exp52, ceil32);
  
  // Get mask with 52-shifted exponent less than 32 and not zero
  __mmask16 exp52lt32nz = _mm512_kand(himask, _mm512_mask_cmp_epi32_mask(exp52lt32, exp52, zero, _MM_CMPINT_NE));
  
  // Get mask for numbers with 52-shifted exponent less than 11
  static const int ceil11_val = 0x0000000b;
  __m512i ceil11 = _mm512_extload_epi32(&ceil11_val, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  __mmask16 exp52lt11 = _mm512_cmplt_epi32_mask(exp52, ceil11);
  
  // Get operand with all bits set to 1
  static const long sign_val = 0x8000000000000000;
  __m512i sign = _mm512_extload_epi64(&sign_val, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  
  // Get adjusted mantisa for numbers with exponents in range [52, 63)
  __m512i zmm13 = _mm512_sllv_epi32(mantisa, exp52);
  zmm13 = _mm512_mask_or_epi32(zmm13, exp52lt32nz, zmm13, _mm512_shuffle_epi32(_mm512_srlv_epi32(mantisa, _mm512_subr_epi32(exp52, ceil32)), _MM_PERM_CDAB));
  result = _mm512_mask_mov_epi32(sign, exp52lt11, zmm13);
  
//  // Get mask with 52-shifted exponent less than 32
//  __mmask16 exp52lt32 = _mm512_cmplt_epi32_mask(exp52, ceil32);
//  
//  // Get mask with 52-shifted exponent less than 32 and not zero
//  __mmask16 exp52lt32nz = _mm512_kand(himask, _mm512_mask_cmp_epi32_mask(exp52lt32, exp52, zero, _MM_CMPINT_NE));
//  
//  // Get mantisa of each double in op
//  static const long implicit_mantisa_val = 0x0010000000000000;
//  __m512i implicit_mantisa = _mm512_extload_epi64(&implicit_mantisa_val, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
//  __m512i mantisa = _mm512_or_epi64(_mm512_and_epi64(_mm512_castpd_si512(op), mant), implicit_mantisa);
//  
//  // Get adjusted mantisa for numbers with exponents in range [52+32, 52+64)
//  result = _mm512_sllv_epi32(_mm512_mask_mov_epi32(zero, himask, _mm512_shuffle_epi32(mantisa, _MM_PERM_CDAB)), _mm512_sub_epi32(exp52, ceil32));
//  
//  // Get adjusted mantisa for numbers with exponents in range [52, 52+32)
//  __m512i zmm13 = _mm512_sllv_epi32(mantisa, exp52);
//  zmm13 = _mm512_mask_or_epi32(zmm13, exp52lt32nz, zmm13, _mm512_shuffle_epi32(_mm512_srlv_epi32(mantisa, _mm512_subr_epi32(exp52, ceil32)), _MM_PERM_CDAB));
//  result = _mm512_mask_mov_epi32(result, exp52lt32, zmm13);
  
  // For numbers less than 2^52 in absolute value
  result = _mm512_mask_mov_epi64(
    result,
    _mm512_cmplt_pd_mask(_mm512_castsi512_pd(expraw), _mm512_castsi512_pd(num2to52)),
    _mm512_sub_epi32(_mm512_castpd_si512(_mm512_add_round_pd(_mm512_castsi512_pd(num2to52), _mm512_castsi512_pd(absop), _MM_FROUND_TO_ZERO)), num2to52)
  );
  
  // For negative numbers find their 2's complement
  __mmask16 borrow, borrow2;
  __m512i numneg = _mm512_mask_subsetb_epi32(zero, lomask, lomask, result, &borrow);
  int bint = _mm512_mask2int(borrow);
  bint += bint;
  borrow2 = _mm512_int2mask(bint);
  numneg = _mm512_mask_sbb_epi32(numneg, himask, borrow2, result, &borrow);
  result = _mm512_castpd_si512(
    _mm512_mask_mov_pd(
      _mm512_castsi512_pd(result),
      _mm512_cmplt_pd_mask(op, _mm512_castsi512_pd(zero)),
      _mm512_castsi512_pd(numneg)
    )
  );
  
  return result;
  
// Test fixture  
//union dtol {
//  double dbl;
//  long lng;
//};
//
//int main(int argc, char **argv) {
//  alignas(64) double src[8];
//  alignas(64) long dst[8];
//  
//  std::srand(std::time(NULL));
//  for(int i = 0; i < 8; ++i) {
//    src[i] = 1.0;
//    for(long j = 0; j < 10000; ++j) {
//      src[i] += (std::rand()%2 ? -1.0 : 1.0)*(double)std::rand();
//    }
//  }
//  //long raw = 0x473fffffffffffff;
//  //long raw = 0x472fffffffffffff;
//  //long raw = 0x4730000000000000;
//  //long raw = 0x4720000000000000;
//  //long raw = 0x440fffffffffffff;
//  //long raw = 0x43d7ee740579ae20;
//  //long raw = 0xc3e7ee740579ae20;
//  //long raw = 0xc3efffffffffffff;
//  //long raw = 0x43d43cd0d251fa54;
//  //_mm512_store_pd(src, _mm512_castsi512_pd(_mm512_extload_epi64(&raw, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE)));
//  
//  _mm512_store_si512(dst, cvt(_mm512_load_pd(src)));
//  
//  dtol t;
//  for(int i = 0; i < 8; ++i) {
//    t.dbl = src[i];
//    std::cout << std::dec << "custom cvt: 0x" << std::hex << dst[i] << ", " << std::dec << dst[i];
//    std::cout << std::dec << ", compiler cvt: 0x" << std::hex << (long)src[i] << ", " << std::dec << (long)src[i];
//    std::cout << ", original number: 0x" << std::hex << t.lng << std::dec << ", " << src[i] << std::endl;
//    assert(dst[i] == (long)src[i]);
//  }
//  
//  return 0;
//}
#endif
}

// double -> float
inline __m512 cvt512_pd_pslo(__m512d op) {
  return _mm512_cvtpd_pslo(op);
}

inline __m512 cvt512_pd_pshi(__m512d op) {
  __m512 temp = _mm512_cvtpd_pslo(op);
#if defined(SIMD_AVX512F)
  return _mm512_shuffle_f32x4(temp, temp, _MM_SHUFFLE(1, 0, 3, 2));
#else
  return _mm512_permute4f128_ps(temp, _MM_PERM_BADC);
#endif
}

inline __m512 cvt512_pdx2_ps(__m512d hi, __m512d lo) {
  return _mm512_mask_blend_ps(_mm512_int2mask(0b0000000011111111), cvt512_pd_pshi(hi), cvt512_pd_pslo(lo));
}

/* mask supplementary functions */
inline __mmask16 maskconvert16(__mmask16 m) {
  return m;
}

inline __mmask16 maskconvert16(__mmask8 m) {
  return _mm512_kmovlhb(m, m);
}

/* mask_set(...) implementations  */

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, bool value) {
  const int val = (value ? 0xffffffff : 0x0);
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    m(i) = _mm512_int2mask(val);
  }
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, bool value) {
  const int val = (value ? 0xffffffff : 0x0);
  m(ari) = _mm512_int2mask(val);
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, vindex avi, bool value) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  
  unsigned int filter_value = _mm512_mask2int(m(ari));
  filter_value = (value ? ~filter_value : filter_value);
  filter_value <<= left;
  filter_value >>= (left+right);
  filter_value <<= right;
  
  m(ari) = _mm512_kxor(m(ari), _mm512_mask2int(filter_value));
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, bindex abi, bool value) {
  int ari = abi / type_traits<T, NW>::num_bvals_per_reg;
  int b = abi % type_traits<T, NW>::num_bvals_per_reg;
  
  int filter_value = (0x1 << b);
  if(value) {
    m(ari) = _mm512_kor(m(ari), _mm512_int2mask(filter_value));
  } else {
    m(ari) = _mm512_kandnr(m(ari), _mm512_int2mask(filter_value));
  }
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, vindex v, bool value) {
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  
  unsigned int filter_value = _mm512_mask2int(m(ari));
  filter_value = (value ? ~filter_value : filter_value);
  filter_value <<= left;
  filter_value >>= (left+right);
  filter_value <<= right;
  
  m(ari) = _mm512_kxor(m(ari), _mm512_mask2int(filter_value));
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, bindex b, bool value) {
  int filter_value = (0x1 << b);
  if(value) {
    m(ari) = _mm512_kor(m(ari), _mm512_int2mask(filter_value));
  } else {
    m(ari) = _mm512_kandnr(m(ari), _mm512_int2mask(filter_value));
  }
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, vindex avi, bindex b, bool value) {
  int ari = avi / type_traits<T>::num_vals_per_reg;
  int v = avi % type_traits<T>::num_vals_per_reg;
  int bstart = v * type_traits<T, NW>::num_bvals_per_val + b;
  
  int filter_value = (0x1 << bstart);
  if(value) {
    m(ari) = _mm512_kor(m(ari), _mm512_int2mask(filter_value));
  } else {
    m(ari) = _mm512_kandnr(m(ari), _mm512_int2mask(filter_value));
  }
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, vindex v, bindex b, bool value) {
  int bstart = v * type_traits<T, NW>::num_bvals_per_val + b;
  
  int filter_value = (0x1 << bstart);
  if(value) {
    m(ari) = _mm512_kor(m(ari), _mm512_int2mask(filter_value));
  } else {
    m(ari) = _mm512_kandnr(m(ari), _mm512_int2mask(filter_value));
  }
  
  return m;
}

/* mask_reset(...) implementations */

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m) {
  __mmask16 zero;
  zero = _mm512_kxor(zero, zero);
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    m(i) = zero;
  }
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari) {
  __mmask16 zero;
  m(ari) = _mm512_kxor(zero, zero);
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
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    m(i) = _mm512_knot(m(i));
  }
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari) {
  m(ari) = _mm512_knot(m(ari));
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  
  unsigned int filter_value = (((0xffffffffu << left) >> (left+right)) << right);
  
  m(ari) = _mm512_kxor(m(ari), _mm512_int2mask(filter_value));
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, bindex abi) {
  int ari = abi / type_traits<T, NW>::num_bvals_per_reg;
  int b = abi % type_traits<T, NW>::num_bvals_per_reg;
  
  int filter_value = 0x1 << b;
  m(ari) = _mm512_kxor(m(ari), _mm512_int2mask(filter_value));
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, vindex v) {
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  
  unsigned int filter_value = (((0xffffffffu << left) >> (left+right)) << right);
  
  m(ari) = _mm512_kxor(m(ari), _mm512_int2mask(filter_value));
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, bindex b) {
  int filter_value = 0x1 << b;
  m(ari) = _mm512_kxor(m(ari), _mm512_int2mask(filter_value));
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, vindex avi, bindex b) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  int bstart = v * type_traits<T, NW>::num_bvals_per_val + b;
  
  int filter_value = 0x1 << bstart;
  m(ari) = _mm512_kxor(m(ari), _mm512_int2mask(filter_value));
  
  return m;
}

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, vindex v, bindex b) {
  int bstart = v * type_traits<T, NW>::num_bvals_per_val + b;
  
  int filter_value = 0x1 << bstart;
  m(ari) = _mm512_kxor(m(ari), _mm512_int2mask(filter_value));
  
  return m;
}

/* mask_all(...) implementations */

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m) {
  bool value = true;
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    __mmask16 temp = maskconvert16(m(i));
    value = value && (bool)_mm512_kortestc(temp, temp);
  }
  return value;
}

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, rindex ari) {
  __mmask16 temp = maskconvert16(m(ari));
  return (bool)_mm512_kortestc(temp, temp);
}

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  unsigned int filter_value = ~(((0xffffffffu << left) >> (right+left)) << right);
  return _mm512_kortestc(m(ari), _mm512_int2mask(filter_value));
}

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, rindex ari, vindex v) {
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  unsigned int filter_value = ~(((0xffffffffu << left) >> (right+left)) << right);
  return _mm512_kortestc(m(ari), _mm512_int2mask(filter_value));
}

/* mask_any(...) implementations */

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m) {
  bool value = false;
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    value = value || !(bool)_mm512_kortestz(m(i), m(i));
  }
  return value;
}

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, rindex ari) {
  return !(bool)_mm512_kortestz(m(ari), m(ari));
}

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  unsigned int filter_value = (((0xffffffffu << left) >> (right+left)) << right);
  __mmask16 filtered_mask = _mm512_kand(m(ari), _mm512_int2mask(filter_value));
  return !((bool)_mm512_kortestz(filtered_mask, filtered_mask));
}

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, rindex ari, vindex v) {
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  unsigned int filter_value = (((0xffffffffu << left) >> (right+left)) << right);
  __mmask16 filtered_mask = _mm512_kand(m(ari), _mm512_int2mask(filter_value));
  return !_mm512_kortestz(filtered_mask, filtered_mask);
}

/* mask_none(...) implementations */

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m) {
  bool value = true;
  for(int i = 0; i < type_traits<T, NW>::num_regs; ++i) {
    value = value && (bool)_mm512_kortestz(m(i), m(i));
  }
  return value;
}

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, rindex ari) {
  return (bool)_mm512_kortestz(m(ari), m(ari));
}

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, vindex avi) {
  int ari = avi / type_traits<T, NW>::num_vals_per_reg;
  int v = avi % type_traits<T, NW>::num_vals_per_reg;
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  unsigned int filter_value = (((0xffffffffu << left) >> (right+left)) << right);
  __mmask16 filtered_mask = _mm512_kand(m(ari), _mm512_int2mask(filter_value));
  return _mm512_kortestz(filtered_mask, filtered_mask);
}

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, rindex ari, vindex v) {
  int right = v * type_traits<T, NW>::num_bvals_per_val;
  int left = 32 - right - type_traits<T, NW>::num_bvals_per_val;
  unsigned int filter_value = (((0xffffffffu << left) >> (right+left)) << right);
  __mmask16 filtered_mask = _mm512_kand(m(ari), _mm512_int2mask(filter_value));
  return _mm512_kortestz(filtered_mask, filtered_mask);
}

/* mask_test(...) implementations */

template<typename T, int NW>
bool mask_test(const mask<T, NW> &m, bindex abi) {
  int ari = abi / type_traits<T, NW>::num_bvals_per_reg;
  int b = abi % type_traits<T, NW>::num_bvals_per_reg;
  __mmask16 filtered_mask = _mm512_kand(m(ari), _mm512_int2mask(0x1 << b));
  return !_mm512_kortestz(filtered_mask, filtered_mask);
}

template<typename T, int NW>
bool mask_test(const mask<T, NW> &m, rindex ari, bindex b) {
  __mmask16 filtered_mask = _mm512_kand(m(ari), _mm512_int2mask(0x1 << b));
  return !_mm512_kortestz(filtered_mask, filtered_mask);
}

/* Functions to set value to zero */

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m512i, int, W> &op) {
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_setzero_si512();
#else
    __m512i a;
    op(i) = _mm512_xor_si512(a, a);
#endif
  }
}

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m512i, long, W> &op) {
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_setzero_si512();
#else
    __m512i a;
    op(i) = _mm512_xor_si512(a, a);
#endif
  }
}

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m512, float, W> &op) {
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_setzero_ps();
#else
    __m512i a;
    op(i) = _mm512_castsi512_ps(_mm512_xor_si512(a, a));
#endif
  }
}

template<typename T, int NW, int W>
inline void set_zero(pack<T, NW, __m512d, double, W> &op) {
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_setzero_pd();
#else
    __m512i a;
    op(i) = _mm512_castsi512_pd(_mm512_xor_si512(a, a));
#endif
  }
}

/* Function to set all elements of a pack to a scalar value */

template<typename T, int NW, int W>
inline void set_scalar(pack<T, NW, __m512i, int, W> &op, int value) {
#if defined(SIMD_AVX512F)
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    op(i) = _mm512_set1_epi32(value);
  }
#else
  __m512i temp = _mm512_extload_epi32(&value, _MM_UPCONV_EPI32_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    op(i) = temp;
  }
#endif
}

template<typename T, int NW, int W>
inline void set_scalar(pack<T, NW, __m512i, long, W> &op, long value) {
#if defined(SIMD_AVX512F)
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
    op(i) = _mm512_set1_epi64((__int64)value);
  }
#else
  __m512i temp = _mm512_extload_epi64(&value, _MM_UPCONV_EPI64_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
    op(i) = temp;
  }
#endif
}

template<typename T, int NW, int W>
inline void set_scalar(pack<T, NW, __m512, float, W> &op, float value) {
#if defined(SIMD_AVX512F)
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    op(i) = _mm512_set1_ps(value);
  }
#else
  __m512 temp = _mm512_extload_ps(&value, _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE);
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    op(i) = temp;
  }
#endif
}
  
template<typename T, int NW, int W>
inline void set_scalar(pack<T, NW, __m512d, double, W> &op, double value) {
#if defined(SIMD_AVX512F)
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    op(i) = _mm512_set1_pd(value);
  }
#else
  __m512d temp = _mm512_extload_pd(&value, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    op(i) = temp;
  }
#endif
}

/* Arithmetic operator: + */

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, int, W> operator+(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  pack<T, NW, __m512i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_add_epi32(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, long, W> operator+(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  pack<T, NW, __m512i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
    temp(i) = _mm512_add_epi64(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512, float, W> operator+(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  pack<T, NW, __m512, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_add_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512d, double, W> operator+(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  pack<T, NW, __m512d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_add_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: - */

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, int, W> operator-(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  pack<T, NW, __m512i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_sub_epi32(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, long, W> operator-(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  pack<T, NW, __m512i, long, W> temp;
  
#if defined(SIMD_KNC)
  __mmask16 borrow;
  __mmask16 mask1 = _mm512_int2mask(0x5555);
#endif
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_sub_epi64(lhs(i), rhs(i));
#else
    temp(i) = _mm512_mask_subsetb_epi32(lhs(i), mask1, _mm512_int2mask(0x0000), rhs(i), &borrow);
    borrow = _mm512_int2mask(_mm512_mask2int(borrow) << 1);
    temp(i) = _mm512_sbb_epi32(lhs(i), borrow, rhs(i), &borrow);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512, float, W> operator-(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  pack<T, NW, __m512, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_sub_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512d, double, W> operator-(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  pack<T, NW, __m512d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_sub_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: * */

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, int, W> operator*(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  pack<T, NW, __m512i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_mullo_epi32(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, long, W> operator*(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  pack<T, NW, __m512i, long, W> temp;
  
#if defined(SIMD_KNC)
  __m512i zero;
    zero = _mm512_xor_epi32(zero, zero);
  
  __mmask16 lomask = _mm512_int2mask(0x5555);
  __mmask16 himask = _mm512_int2mask(0xAAAA);
#endif
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_mullo_epi64(lhs(i), rhs(i));
#else
    __m512i res0 = _mm512_mullo_epi32(lhs(i), rhs(i));
    
    __m512i res1 = _mm512_shuffle_epi32(
      _mm512_mask_mulhi_epu32(zero, lomask, lhs(i), rhs(i)),
      _MM_PERM_CDAB
    );
    
    __m512i rhsswap = _mm512_shuffle_epi32(rhs(i), _MM_PERM_CDAB);
    __m512i res2 = _mm512_mullo_epi32(lhs(i), rhsswap);
    __m512i res3 = _mm512_shuffle_epi32(res2, _MM_PERM_CDAB);
    __m512i res4 = _mm512_mask_add_epi32(zero, himask, res2, res3);
    
    temp(i) = _mm512_add_epi32(_mm512_add_epi32(res0, res1), res4);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512, float, W> operator*(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  pack<T, NW, __m512, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_mul_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512d, double, W> operator*(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  pack<T, NW, __m512d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_mul_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: / */

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, int, W> operator/(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  pack<T, NW, __m512i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_div_epi32(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, long, W> operator/(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  pack<T, NW, __m512i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
    temp(i) = _mm512_div_epi64(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512, float, W> operator/(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  pack<T, NW, __m512, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_div_ps(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512d, double, W> operator/(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  pack<T, NW, __m512d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_div_pd(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Arithmetic operator: % - only for integer types */

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, int, W> operator%(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  pack<T, NW, __m512i, int, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_rem_epi32(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline pack<T, NW, __m512i, long, W> operator%(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  pack<T, NW, __m512i, long, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
    temp(i) = _mm512_rem_epi64(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Comparison operator: == */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, int, W>::mask_type operator==(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  typename pack<T, NW, __m512i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpeq_epi32_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, long, W>::mask_type operator==(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  typename pack<T, NW, __m512i, long, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_cmpeq_epi64_mask(lhs(i), rhs(i));
#else
    unsigned int res = _mm512_mask2int(_mm512_cmpeq_epi32_mask(lhs(i), rhs(i)));
//    temp(i) = _mm512_int2mask(_pext_u32(res & (res >> 1), 0x00005555));
    res = (res & (res >> 1)) & 0x55555555;
    res = (res | (res >> 1)) & 0x33333333;
    res = (res | (res >> 2)) & 0x0f0f0f0f;
    res = (res | (res >> 4)) & 0x00ff00ff;
    temp(i) = _mm512_int2mask(res);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512, float, W>::mask_type operator==(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  typename pack<T, NW, __m512, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpeq_ps_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512d, double, W>::mask_type operator==(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  typename pack<T, NW, __m512d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpeq_pd_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Comparison operator: != */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, int, W>::mask_type operator!=(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  typename pack<T, NW, __m512i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpneq_epi32_mask(lhs(i), rhs(i));	
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, long, W>::mask_type operator!=(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  typename pack<T, NW, __m512i, long, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_cmpneq_epi64_mask(lhs(i), rhs(i));
#else
    unsigned int res = _mm512_mask2int(_mm512_cmpneq_epi32_mask(lhs(i), rhs(i)));
//    temp(i) = _mm512_int2mask(_pext_u32(res & (res >> 1), 0x5555));
    res = (res | (res >> 1)) & 0x55555555;
    res = (res | (res >> 1)) & 0x33333333;
    res = (res | (res >> 2)) & 0x0f0f0f0f;
    res = (res | (res >> 4)) & 0x00ff00ff;
    temp(i) = _mm512_int2mask(res);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512, float, W>::mask_type operator!=(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  typename pack<T, NW, __m512, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpneq_ps_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512d, double, W>::mask_type operator!=(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  typename pack<T, NW, __m512d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpneq_pd_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Comparison operator: < */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, int, W>::mask_type operator<(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  typename pack<T, NW, __m512i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_cmplt_epi32_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, long, W>::mask_type operator<(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  typename pack<T, NW, __m512i, long, W>::mask_type temp;
  
#if defined(SIMD_KNC)
  __mmask16 oddmask = _mm512_int2mask(0xAAAA); // 1, 3, 5, ...
  __mmask16 evenmask = _mm512_int2mask(0x5555); // 0, 2, 4, ...
#endif
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_cmplt_epi64_mask(lhs(i), rhs(i));
#else
//    unsigned int res = _mm512_mask2int(_mm512_cmplt_epi32_mask(lhs(i), rhs(i)));
//    temp(i) = _mm512_int2mask(_pext_u32(res & (res >> 1), 0x5555));
    
//    unsigned int res = _mm512_mask2int(_mm512_cmplt_epi32_mask(lhs(i), rhs(i)));
//    res = (res & (res >> 1)) & 0x55555555;
//    res = (res | (res >> 1)) & 0x33333333;
//    res = (res | (res >> 2)) & 0x0f0f0f0f;
//    res = (res | (res >> 4)) & 0x00ff00ff;
//    temp(i) = _mm512_int2mask(res);
    
//    unsigned int res1 = _mm512_mask2int(_mm512_mask_cmplt_epi32_mask(oddmask, lhs(i), rhs(i)));
//    unsigned int res2 = _mm512_mask2int(_mm512_mask_cmplt_epu32_mask(evenmask, lhs(i), rhs(i)));
//    unsigned int res3 = _mm512_mask2int(_mm512_mask_cmpeq_epi32_mask(oddmask, lhs(i), rhs(i)));
//    unsigned int res = (res1>>1) | (res2 & (res3>>1));
//    res = (res | (res >> 1)) & 0x33333333;
//    res = (res | (res >> 2)) & 0x0f0f0f0f;
//    res = (res | (res >> 4)) & 0x00ff00ff;
//    temp(i) = _mm512_int2mask(res);
    
    unsigned int res1 = _mm512_mask2int(_mm512_mask_cmpge_epi32_mask(oddmask, lhs(i), rhs(i)));
    unsigned int res2 = _mm512_mask2int(_mm512_mask_cmpge_epu32_mask(evenmask, lhs(i), rhs(i)));
    unsigned int res = (res1>>1) & res2;
    res = (res | (res >> 1)) & 0x33333333;
    res = (res | (res >> 2)) & 0x0f0f0f0f;
    res = (res | (res >> 4)) & 0x00ff00ff;
    temp(i) = _mm512_int2mask(~res);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512, float, W>::mask_type operator<(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  typename pack<T, NW, __m512, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_cmplt_ps_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512d, double, W>::mask_type operator<(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  typename pack<T, NW, __m512d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_cmplt_pd_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Comparison operator: > */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, int, W>::mask_type operator>(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  typename pack<T, NW, __m512i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpgt_epi32_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, long, W>::mask_type operator>(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  typename pack<T, NW, __m512i, long, W>::mask_type temp;
  
#if defined(SIMD_KNC)
  __mmask16 oddmask = _mm512_int2mask(0xAAAA); // 1, 3, 5, ...
  __mmask16 evenmask = _mm512_int2mask(0x5555); // 0, 2, 4, ...
#endif
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_cmpgt_epi64_mask(lhs(i), rhs(i));
#else
//    unsigned int res = _mm512_mask2int(_mm512_cmpgt_epi32_mask(lhs(i), rhs(i)));
//    res = (res & (res >> 1)) & 0x55555555;
//    res = (res | (res >> 1)) & 0x33333333;
//    res = (res | (res >> 2)) & 0x0f0f0f0f;
//    res = (res | (res >> 4)) & 0x00ff00ff;
//    temp(i) = _mm512_int2mask(res);
    
//    unsigned int res = _mm512_mask2int(_mm512_cmpgt_epi32_mask(lhs(i), rhs(i)));
//    temp(i) = _mm512_int2mask(_pext_u32(res & (res >> 1), 0x5555));
    
//    unsigned int res1 = _mm512_mask2int(_mm512_mask_cmpgt_epi32_mask(oddmask, lhs(i), rhs(i)));
//    unsigned int res2 = _mm512_mask2int(_mm512_mask_cmpgt_epu32_mask(evenmask, lhs(i), rhs(i)));
//    unsigned int res3 = _mm512_mask2int(_mm512_mask_cmpeq_epi32_mask(oddmask, lhs(i), rhs(i)));
//    unsigned int res = (res1>>1) | (res2 & (res3>>1));
//    res = (res | (res >> 1)) & 0x33333333;
//    res = (res | (res >> 2)) & 0x0f0f0f0f;
//    res = (res | (res >> 4)) & 0x00ff00ff;
//    temp(i) = _mm512_int2mask(res);
    
    unsigned int res1 = _mm512_mask2int(_mm512_mask_cmple_epi32_mask(oddmask, lhs(i), rhs(i)));
    unsigned int res2 = _mm512_mask2int(_mm512_mask_cmple_epu32_mask(evenmask, lhs(i), rhs(i)));
    unsigned int res = (res1>>1) & res2;
    res = (res | (res >> 1)) & 0x33333333;
    res = (res | (res >> 2)) & 0x0f0f0f0f;
    res = (res | (res >> 4)) & 0x00ff00ff;
    temp(i) = _mm512_int2mask(~res);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512, float, W>::mask_type operator>(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  typename pack<T, NW, __m512, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_cmplt_ps_mask(rhs(i), lhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512d, double, W>::mask_type operator>(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  typename pack<T, NW, __m512d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_cmplt_pd_mask(rhs(i), lhs(i));
  }
  
  return temp;
}

/* Comparison operator: <= */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, int, W>::mask_type operator<=(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  typename pack<T, NW, __m512i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_cmple_epi32_mask(lhs(i), rhs(i));
  }
  
  return temp;
}   
  
template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, long, W>::mask_type operator<=(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  typename pack<T, NW, __m512i, long, W>::mask_type temp;
  
#if defined(SIMD_KNC)
  __mmask16 oddmask = _mm512_int2mask(0xAAAA); // 1, 3, 5, ...
  __mmask16 evenmask = _mm512_int2mask(0x5555); // 0, 2, 4, ...
#endif
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_cmple_epi64_mask(lhs(i), rhs(i));
#else
//    unsigned int res = _mm512_mask2int(_mm512_cmple_epi32_mask(lhs(i), rhs(i)));
//    res = (res & (res >> 1)) & 0x55555555;
//    res = (res | (res >> 1)) & 0x33333333;
//    res = (res | (res >> 2)) & 0x0f0f0f0f;
//    res = (res | (res >> 4)) & 0x00ff00ff;
//    temp(i) = _mm512_int2mask(res);
    
//    unsigned int res = _mm512_mask2int(_mm512_cmple_epi32_mask(lhs(i), rhs(i)));
//    temp(i) = _mm512_int2mask(_pext_u32(res & (res >> 1), 0x5555));
    
    unsigned int res1 = _mm512_mask2int(_mm512_mask_cmple_epi32_mask(oddmask, lhs(i), rhs(i)));
    unsigned int res2 = _mm512_mask2int(_mm512_mask_cmple_epu32_mask(evenmask, lhs(i), rhs(i)));
    unsigned int res = (res1>>1) & res2;
    res = (res | (res >> 1)) & 0x33333333;
    res = (res | (res >> 2)) & 0x0f0f0f0f;
    res = (res | (res >> 4)) & 0x00ff00ff;
    temp(i) = _mm512_int2mask(res);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512, float, W>::mask_type operator<=(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  typename pack<T, NW, __m512, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_cmple_ps_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512d, double, W>::mask_type operator<=(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  typename pack<T, NW, __m512d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_cmple_pd_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

/* Comparison operator: >= */

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, int, W>::mask_type operator>=(const pack<T, NW, __m512i, int, W> &lhs, const pack<T, NW, __m512i, int, W> &rhs) {
  typename pack<T, NW, __m512i, int, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    temp(i) = _mm512_cmpge_epi32_mask(lhs(i), rhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512i, long, W>::mask_type operator>=(const pack<T, NW, __m512i, long, W> &lhs, const pack<T, NW, __m512i, long, W> &rhs) {
  typename pack<T, NW, __m512i, long, W>::mask_type temp;
  
#if defined(SIMD_KNC)
  __mmask16 oddmask = _mm512_int2mask(0xAAAA); // 1, 3, 5, ...
  __mmask16 evenmask = _mm512_int2mask(0x5555); // 0, 2, 4, ...
#endif
  
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    temp(i) = _mm512_cmpge_epi64_mask(lhs(i), rhs(i));
#else
//    unsigned int res = _mm512_mask2int(_mm512_cmpge_epi32_mask(lhs(i), rhs(i)));
//    res = (res & (res >> 1)) & 0x55555555;
//    res = (res | (res >> 1)) & 0x33333333;
//    res = (res | (res >> 2)) & 0x0f0f0f0f;
//    res = (res | (res >> 4)) & 0x00ff00ff;
//    temp(i) = _mm512_int2mask(res);
    
//    unsigned int res = _mm512_mask2int(_mm512_cmpge_epi32_mask(lhs(i), rhs(i)));
//    temp(i) = _mm512_int2mask(_pext_u32(res & (res >> 1), 0x5555));
    
    unsigned int res1 = _mm512_mask2int(_mm512_mask_cmpge_epi32_mask(oddmask, lhs(i), rhs(i)));
    unsigned int res2 = _mm512_mask2int(_mm512_mask_cmpge_epu32_mask(evenmask, lhs(i), rhs(i)));
    unsigned int res = (res1>>1) & res2;
    res = (res | (res >> 1)) & 0x33333333;
    res = (res | (res >> 2)) & 0x0f0f0f0f;
    res = (res | (res >> 4)) & 0x00ff00ff;
    temp(i) = _mm512_int2mask(res);
#endif
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512, float, W>::mask_type operator>=(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  typename pack<T, NW, __m512, float, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_cmple_ps_mask(rhs(i), lhs(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
inline typename pack<T, NW, __m512d, double, W>::mask_type operator>=(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  typename pack<T, NW, __m512d, double, W>::mask_type temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_cmple_pd_mask(rhs(i), lhs(i));
  }
  
  return temp;
}

/* Function: inverse */

template<typename T, int NW, int W>
pack<T, NW, __m512, float, W> inv(const pack<T, NW, __m512, float, W> &op) {
  pack<T, NW, __m512, float, W> temp;
  
  __m512 one = _mm512_set1_ps(1.0f);
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_div_ps(one, op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m512d, double, W> inv(const pack<T, NW, __m512d, double, W> &op) {
  pack<T, NW, __m512d, double, W> temp;
  
  __m512d one = _mm512_set1_pd(1.0);
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_div_pd(one, op(i));
  }
  
  return temp;
}

/* Function: sin */

template<typename T, int NW, int W>
pack<T, NW, __m512, float, W> sin(const pack<T, NW, __m512, float,  W> &op) {
  pack<T, NW, __m512, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_sin_ps(op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m512d, double, W> sin(const pack<T, NW, __m512d, double,  W> &op) {
  pack<T, NW, __m512d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_sin_pd(op(i));
  }
  
  return temp;
}

/* Function: cos */

template<typename T, int NW, int W>
pack<T, NW, __m512, float, W> cos(const pack<T, NW, __m512, float,  W> &op) {
  pack<T, NW, __m512, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_cos_ps(op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m512d, double, W> cos(const pack<T, NW, __m512d, double,  W> &op) {
  pack<T, NW, __m512d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_cos_pd(op(i));
  }
  
  return temp;
}

/* Function: tan */

template<typename T, int NW, int W>
pack<T, NW, __m512, float, W> tan(const pack<T, NW, __m512, float,  W> &op) {
  pack<T, NW, __m512, float, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_tan_ps(op(i));
  }
  
  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m512d, double, W> tan(const pack<T, NW, __m512d, double,  W> &op) {
  pack<T, NW, __m512d, double, W> temp;
  
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_tan_pd(op(i));
  }
  
  return temp;
}

/* Function: add adjacent numbers and interleave results */

template<typename T, int NW, int W>
pack<T, NW, __m512, float, W> hadd_pairwise_interleave(const pack<T, NW, __m512, float, W> &lhs, const pack<T, NW, __m512, float, W> &rhs) {
  pack<T, NW, __m512, float, W> temp;

  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    temp(i) = _mm512_mask_blend_ps(
      _mm512_int2mask(0b1010101010101010), 
      _mm512_add_ps(lhs(i), _mm512_swizzle_ps(lhs(i), _MM_SWIZ_REG_CDAB)), 
      _mm512_add_ps(rhs(i), _mm512_swizzle_ps(rhs(i), _MM_SWIZ_REG_CDAB))
    );
  }

  return temp;
}

template<typename T, int NW, int W>
pack<T, NW, __m512d, double, W> hadd_pairwise_interleave(const pack<T, NW, __m512d, double, W> &lhs, const pack<T, NW, __m512d, double, W> &rhs) {
  pack<T, NW, __m512d, double, W> temp;

  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    temp(i) = _mm512_mask_blend_pd(
      _mm512_int2mask(0b1010101010101010), 
      _mm512_add_pd(lhs(i), _mm512_swizzle_pd(lhs(i), _MM_SWIZ_REG_CDAB)), 
      _mm512_add_pd(rhs(i), _mm512_swizzle_pd(rhs(i), _MM_SWIZ_REG_CDAB))
    );
  }

  return temp;
}

/* Load functions */

// unaligned temporal load: int <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  int const *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 m;
  m = _mm512_kxnor(m, m);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_maskz_loadu_epi32(m, p);
    p += type_traits<int>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_epi32(op(i), p);
    p += type_traits<int>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_epi32(op(i), p);
#endif
  }
}

// masked unaligned temporal load: int <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  int const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_maskz_loadu_epi32(m(i), p);
    p += type_traits<int>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_epi32(op(i), p);
    p += type_traits<int>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_epi32(op(i), p);
    op(i) = _mm512_mask_blend_epi32(m(i), zero, op(i));
#endif
  }
}

// unaligned temporal load: int <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  long const *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 m;
  m = _mm512_kxnor(m, m);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo, hi;
#if defined(SIMD_AVX512F)
    lo = _mm512_maskz_loadu_epi64(m, p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_maskz_loadu_epi64(m, p);
    p += type_traits<long>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_epi64(lo, p);
    p += type_traits<long>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_epi64(lo, p);
    hi = _mm512_loadunpacklo_epi64(hi, p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_epi64(hi, p);
#endif
    op(i) = cvt512_epi64x2_epi32(hi, lo);
  }
}

// masked unaligned temporal load: int <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  long const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo, hi;
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    lo = _mm512_maskz_loadu_epi64(m(i), p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_maskz_loadu_epi64(himask, p);
    p += type_traits<long>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_epi64(lo, p);
    p += type_traits<long>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_epi64(lo, p);
    lo = _mm512_mask_blend_epi64(m(i), zero, lo);
    hi = _mm512_loadunpacklo_epi64(hi, p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_epi64(hi, p);
    hi = _mm512_mask_blend_epi64(himask, zero, hi);
#endif
    op(i) = cvt512_epi64x2_epi32(hi, lo);
  }
}

// unaligned temporal load: int <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_loadu_ps(p);
    p += type_traits<float>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_ps(temp, p);
    p += type_traits<float>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_ps(temp, p);
#endif
    op(i) = cvt512_ps_epi32(temp);
  }
}

// masked unaligned temporal load: int <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  float const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512 zero;
  zero = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(zero), _mm512_castps_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_ps(m(i), p);
    p += type_traits<float>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_ps(temp, p);
    p += type_traits<float>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_ps(temp, p);
    temp = _mm512_mask_blend_ps(m(i), zero, temp);
#endif
    op(i) = cvt512_ps_epi32(temp);
  }
}

// unaligned temporal load: int <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo, hi;
#if defined(SIMD_AVX512F)
    lo = _mm512_loadu_pd(p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_loadu_pd(p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_pd(lo, p);
    p += type_traits<double>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_pd(lo, p);
    hi = _mm512_loadunpacklo_pd(hi, p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_pd(hi, p);
#endif
    op(i) = cvt512_pdx2_epi32(hi, lo);
  }
}

// masked unaligned temporal load: int <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, int, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  double const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512d zero;
  zero = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(zero), _mm512_castpd_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo, hi;
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    lo = _mm512_maskz_loadu_pd(m(i), p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_maskz_loadu_pd(m(i), p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_pd(lo, p);
    p += type_traits<double>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_pd(lo, p);
    lo = _mm512_mask_blend_pd(m(i), zero, lo);
    hi = _mm512_loadunpacklo_pd(hi, p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_pd(hi, p);
    hi = _mm512_mask_blend_pd(himask, zero, hi);
#endif
    op(i) = cvt512_pdx2_epi32(hi, lo);
  }
}

// unaligned temporal load: long <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  int const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi32(lm, p);
#else
    temp = _mm512_mask_loadunpacklo_epi32(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_epi32(temp, lm, p+type_traits<int>::num_bvals_per_reg);
#endif
    op(i) = cvt512_epi32lo_epi64(temp);
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// masked unaligned temporal load: long <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  int const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi32(_mm512_kand(m(i), lm), p);
#else
    temp = _mm512_mask_loadunpacklo_epi32(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_epi32(temp, lm, p+type_traits<int>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi32(m(i), zero, temp);
#endif
    op(i) = cvt512_epi32lo_epi64(temp);
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// unaligned temporal load: long <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  long const *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_maskz_loadu_epi64(lm, p);
    p += type_traits<long>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_epi64(op(i), p);
    p += type_traits<long>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_epi64(op(i), p);
#endif
  }
}

// masked unaligned temporal load: long <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  long const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_maskz_loadu_epi64(m(i), p);
    p += type_traits<long>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_epi64(op(i), p);
    p += type_traits<long>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_epi64(op(i), p);
    op(i) = _mm512_mask_blend_epi64(m(i), zero, op(i));
#endif
  }
}

// unaligned temporal load: long <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  float const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_ps(lm, p);
#else
    temp = _mm512_mask_loadunpacklo_ps(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_ps(temp, lm, p+type_traits<float>::num_bvals_per_reg);
#endif
    op(i) = cvt512_pslo_epi64(temp);
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// masked unaligned temporal load: long <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  float const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
#if !defined(SIMD_AVX512F)
  __m512 zero;
  zero = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(zero), _mm512_castps_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_ps(_mm512_kand(m(i), lm), p);
#else
    temp = _mm512_mask_loadunpacklo_ps(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_ps(temp, lm, p+type_traits<float>::num_bvals_per_reg);
    temp = _mm512_mask_blend_ps(m(i), zero, temp);
#endif
    op(i) = cvt512_pslo_epi64(temp);
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// unaligned temporal load: long <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_loadu_pd(p);
    p += type_traits<long>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_pd(temp, p);
    p += type_traits<long>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_pd(temp, p);
#endif
    op(i) = cvt512_pd_epi64(temp);
  }
}

// masked unaligned temporal load: long <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  double const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512d zero;
  zero = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(zero), _mm512_castpd_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_pd(m(i), p);
    p += type_traits<long>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_pd(temp, p);
    p += type_traits<long>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_pd(temp, p);
    temp = _mm512_mask_blend_pd(m(i), zero, temp);
#endif
    op(i) = cvt512_pd_epi64(temp);
  }
}

// unaligned temporal load: float <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  int const *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 lm;
  lm = _mm512_kxnor(lm lm);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi32(lm, p);
    p += type_traits<float>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_epi32(temp, p);
    p += type_traits<float>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_epi32(temp, p);
#endif
    op(i) = cvt512_epi32_ps(temp);
  }
}

// masked unaligned temporal load: float <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  int const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi32(m(i), p);
    p += type_traits<float>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_epi32(temp, p);
    p += type_traits<float>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_epi32(temp, p);
    temp = _mm512_mask_blend_epi32(m(i), zero, temp);
#endif
    op(i) = cvt512_epi32_ps(temp);
  }
}

// unaligned temporal load: float <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  long const *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 lm;
  lm = _mm512_kxnor(lm, lm);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo, hi;
#if defined(SIMD_AVX512F)
    lo = _mm512_maskz_loadu_epi64(lm, p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_maskz_loadu_epi64(lm, p)
    p += type_traits<long>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_epi64(lo, p);
    p += type_traits<long>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_epi64(lo, p);
    hi = _mm512_loadunpacklo_epi64(hi, p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_epi64(hi, p);
#endif
    op(i) = cvt512_epi64x2_ps(hi, lo);
  }
}

// masked unaligned temporal load: float <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  long const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo, hi;
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    lo = _mm512_maskz_loadu_epi64(m(i), p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_maskz_laodu_epi64(himask, p);
    p += type_traits<long>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_epi64(lo, p);
    p += type_traits<long>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_epi64(lo, p);
    lo = _mm512_mask_blend_epi64(m(i), zero, lo);
    hi = _mm512_loadunpacklo_epi64(hi, p);
    p += type_traits<long>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_epi64(hi, p);
    hi = _mm512_mask_blend_epi64(himask, zero, hi);
#endif
    op(i) = cvt512_epi64x2_ps(hi, lo);
  }
}

// unaligned temporal load: float <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  float const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_loadu_ps(p);
    p += type_traits<float>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_ps(op(i), p);
    p += type_traits<float>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_ps(op(i), p);
#endif
  }
}

// masked unaligned temporal load: float <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  float const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512 zero;
  zero = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(zero), _mm512_castps_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_maskz_loadu_ps(m(i), p);
    p += type_traits<float>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_ps(op(i), p);
    p += type_traits<float>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_ps(op(i), p);
    op(i) = _mm512_mask_blend_ps(m(i), zero, op(i));
#endif
  }
}

// unaligned temporal load: float <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo, hi;
#if defined(SIMD_AVX512F)
    lo = _mm512_loadu_pd(p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_loadu_pd(p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_pd(lo, p);
    p += type_traits<double>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_pd(lo, p);
    hi = _mm512_loadunpacklo_pd(hi, p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_pd(hi, p);
#endif
    op(i) = cvt512_pdx2_ps(hi, lo);
  }
}

// masked unaligned temporal load: float <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  double const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512d zero;
  zero = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(zero), _mm512_castpd_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo, hi;
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    lo = _mm512_maskz_loadu_pd(m(i), p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_maskz_loadu_pd(himask, p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    lo = _mm512_loadunpacklo_pd(lo, p);
    p += type_traits<double>::num_bvals_per_reg;
    lo = _mm512_loadunpackhi_pd(lo, p);
    lo = _mm512_mask_blend_pd(m(i), zero, lo);
    hi = _mm512_loadunpacklo_pd(hi, p);
    p += type_traits<double>::num_bvals_per_reg;
    hi = _mm512_loadunpackhi_pd(hi, p);
    hi = _mm512_mask_blend_pd(himask, zero, hi);
#endif
    op(i) = cvt512_pdx2_ps(hi, lo);
  }
}

// unaligned temporal load: double <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  int const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi32(lm, p);
#else
    temp = _mm512_mask_loadunpacklo_epi32(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_epi32(temp, lm, p+type_traits<int>::num_bvals_per_reg);
#endif
    op(i) = cvt512_epi32lo_pd(temp);
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// masked unaligned temporal load: double <- int
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  int const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero = _mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi32(_mm512_kand(m(i), lm), p);
#else
    temp = _mm512_mask_loadunpacklo_epi32(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_epi32(temp, lm, p+type_traits<int>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi32(m(i), zero, temp);
#endif
    op(i) = cvt512_epi32lo_pd(temp);
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// unaligned temporal load: double <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  long const *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 lm;
  lm = _mm512_kxnor(lm, lm);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi64(lm, p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_epi64(temp, p);
    p += type_traits<double>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_epi64(temp, p);
#endif
    op(i) = cvt512_epi64_pd(temp);
  }
}

// masked unaligned temporal load: double <- long
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  long const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512i zero;
  zero  =_mm512_xor_si512(zero, zero);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_epi64(m(i), p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    temp = _mm512_loadunpacklo_epi64(temp, p);
    p += type_traits<double>::num_bvals_per_reg;
    temp = _mm512_loadunpackhi_epi64(temp, p);
    temp = _mm512_mask_blend_epi64(m(i), zero, temp);
#endif
    op(i) = cvt512_epi64_pd(temp);
  }
}

// unaligned temporal load: double <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  float const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_ps(lm, p);
#else
    temp = _mm512_mask_loadunpacklo_ps(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_ps(temp, lm, p+type_traits<float>::num_bvals_per_reg);
#endif
    op(i) = cvt512_pslo_pd(temp);
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// masked unaligned temporal load: double <- float
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  float const *p = ptr;
  __mmask16 lm = _mm512_int2mask(0b0000000011111111);
#if !defined(SIMD_AVX512F)
  __m512 zero;
  zero = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(zero), _mm512_castps_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp;
#if defined(SIMD_AVX512F)
    temp = _mm512_maskz_loadu_ps(_mm512_kand(m(i), lm), p);
#else
    temp = _mm512_mask_loadunpacklo_ps(temp, lm, p);
    temp = _mm512_mask_loadunpackhi_ps(temp, lm, p+type_traits<float>::num_bvals_per_reg);
    temp = _mm512_mask_blend_ps(m(i), zero, temp);
#endif
    op(i) = cvt512_pslo_pd(temp);
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// unaligned temporal load: double <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  double const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_loadu_pd(p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_pd(op(i), p);
    p += type_traits<double>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_pd(op(i), p);
#endif
  }
}

// masked unaligned temporal load: double <- double
template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  double const *p = ptr;
#if !defined(SIMD_AVX512F)
  __m512d zero;
  zero = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(zero), _mm512_castpd_si512(zero)));
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    op(i) = _mm512_maskz_loadu_pd(m(i), p);
    p += type_traits<double>::num_bvals_per_reg;
#else
    op(i) = _mm512_loadunpacklo_pd(op(i), p);
    p += type_traits<double>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_pd(op(i), p);
    op(i) = _mm512_mask_blend_pd(m(i), zero, op(i));
#endif
  }
}

/* Store functions */

// unaligned temporal store: int -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  int *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 m;
  m = _mm512_kxnor(m, m);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, m, op(i));
    p += type_traits<int>::num_bvals_per_reg;
#else
    _mm512_packstorelo_epi32(p, op(i));
    p += type_traits<int>::num_bvals_per_reg;
    _mm512_packstorehi_epi32(p, op(i));
#endif
  }
}

// masked unaligned temporal store: int -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, m(i), op(i));
    p += type_traits<int>::num_bvals_per_reg;
#else
    __m512i temp;
    temp = _mm512_loadunpacklo_epi32(temp, p);
    temp = _mm512_loadunpackhi_epi32(temp, p+type_traits<int>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi32(m(i), temp, op(i));
    _mm512_packstorelo_epi32(p, temp);
    p += type_traits<int>::num_bvals_per_reg;
    _mm512_packstorehi_epi32(p, temp);
#endif
  }
}

// unaligned temporal store: int -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  long *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 m;
  m = _mm512_kxnor(m, m);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo = cvt512_epi32lo_epi64(op(i));
    __m512i hi = cvt512_epi32hi_epi64(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, m, lo);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_mask_storeu_epi64(p, m, hi);
    p += type_traits<long>::num_bvals_per_reg;
#else
    _mm512_packstorelo_epi64(p, lo);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, lo);
    _mm512_packstorelo_epi64(p, hi);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, hi);
#endif
  }
}

// masked unaligned temporal store: int -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo = cvt512_epi32lo_epi64(op(i));
    __m512i hi = cvt512_epi32hi_epi64(op(i));
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, m(i), lo);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_mask_storeu_epi64(p, himask, hi);
    p += type_traits<long>::num_bvals_per_reg;
#else
    __m512i temp, temp1;
    temp = _mm512_loadunpacklo_epi64(temp, p);
    temp = _mm512_loadunpackhi_epi64(temp, p+type_traits<long>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi64(m(i), temp, lo);
    _mm512_packstorelo_epi64(p, temp);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, temp);
    temp1 = _mm512_loadunpacklo_epi64(temp1, p);
    temp1 = _mm512_loadunpackhi_epi64(temp1, p+type_traits<long>::num_bvals_per_reg);
    temp1 = _mm512_mask_blend_epi64(himask, temp1, hi);
    _mm512_packstorelo_epi64(p, temp1);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, temp1);
#endif
  }
}

// unaligned temporal store: int -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp = cvt512_epi32_ps(op(i));
#if defined(SIMD_AVX512F)
    _mm512_storeu_ps(p, temp);
    p += type_traits<float>::num_bvals_per_reg;
#else
    _mm512_packstorelo_ps(p, temp);
    p += type_traits<float>::num_bvals_per_reg;
    _mm512_packstorehi_ps(p, temp);
#endif
  }
}

// masked unaligned temporal store: int -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp = cvt512_epi32_ps(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_ps(p, m(i), temp);
    p += type_traits<float>::num_bvals_per_reg;
#else
    __m512 temp1;
    temp1 = _mm512_loadunpacklo_ps(temp1, p);
    temp1 = _mm512_loadunpackhi_ps(temp1, p+type_traits<float>::num_bvals_per_reg);
    temp = _mm512_mask_blend_ps(m(i), temp1, temp);
    _mm512_packstorelo_ps(p, temp);
    p += type_traits<float>::num_bvals_per_reg;
    _mm512_packstorehi_ps(p, temp);
#endif
  }
}

// unaligned temporal store: int -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo = cvt512_epi32lo_pd(op(i));
    __m512d hi = cvt512_epi32hi_pd(op(i));
#if defined(SIMD_AVX512F)
    _mm512_storeu_pd(p, lo);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_storeu_pd(p, hi);
    p += type_traits<double>::num_bvals_per_reg;
#else
    _mm512_packstorelo_pd(p, lo);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, lo);
    _mm512_packstorelo_pd(p, hi);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, hi);
#endif
  }
}

// masked unaligned temporal store: int -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, int, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo = cvt512_epi32lo_pd(op(i));
    __m512d hi = cvt512_epi32hi_pd(op(i));
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_pd(p, m(i), lo);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_mask_storeu_pd(p, himask, hi);
#else
    __m512d temp, temp1;
    temp = _mm512_loadunpacklo_pd(temp, p);
    temp = _mm512_loadunpackhi_pd(temp, p+type_traits<double>::num_bvals_per_reg);
    temp = _mm512_mask_blend_pd(m(i), temp, lo);
    _mm512_packstorelo_pd(p, temp);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, temp);
    temp1 = _mm512_loadunpacklo_pd(temp1, p);
    temp1 = _mm512_loadunpackhi_pd(temp1, p+type_traits<double>::num_bvals_per_reg);
    temp1 = _mm512_mask_blend_pd(himask, temp1, hi);
    _mm512_packstorelo_pd(p, temp1);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, temp1);
#endif
  }
}

// unaligned temporal store: long -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  int *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_epi64_epi32lo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, sm, temp);
#else
    _mm512_mask_packstorelo_epi32(p, sm, temp);
    _mm512_mask_packstorehi_epi32(p+type_traits<int>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// masked unaligned temporal store: long -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  int *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_epi64_epi32lo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, _mm512_kand(sm, m(i)), temp);
#else
    __m512i temp1;
    temp1 = _mm512_mask_loadunpacklo_epi32(temp1, sm, p);
    temp1 = _mm512_mask_loadunpackhi_epi32(temp1, sm, p+type_traits<int>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi32(m(i), temp1, temp);
    _mm512_mask_packstorelo_epi32(p, sm, temp);
    _mm512_mask_packstorehi_epi32(p+type_traits<int>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// unaligned temporal store: long -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  long *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, sm, op(i));
    p += type_traits<long>::num_bvals_per_reg;
#else
    _mm512_packstorelo_epi64(p, op(i));
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, op(i));
#endif
  }
}

// masked unaligned temporal store: long -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, m(i), op(i));
    p += type_traits<long>::num_bvals_per_reg;
#else
    __m512i temp;
    temp = _mm512_loadunpacklo_epi64(temp, p);
    temp = _mm512_loadunpackhi_epi64(temp, p+type_traits<long>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi64(m(i), temp, op(i));
    _mm512_packstorelo_epi64(p, temp);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, temp);
#endif
  }
}

// unaligned temporal store: long -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  float *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp = cvt512_epi64_pslo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_ps(p, sm, temp);
#else
    _mm512_mask_packstorelo_ps(p, sm, temp);
    _mm512_mask_packstorehi_ps(p+type_traits<float>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// masked unaligned temporal store: long -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  float *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp = cvt512_epi64_pslo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_ps(p, _mm512_kand(sm, m(i)), temp);
#else
    __m512 temp1;
    temp1 = _mm512_mask_loadunpacklo_ps(temp1, sm, p);
    temp1 = _mm512_mask_loadunpackhi_ps(temp1, sm, p+type_traits<float>::num_bvals_per_reg);
    temp = _mm512_mask_blend_ps(m(i), temp1, temp);
    _mm512_mask_packstorelo_ps(p, sm, temp);
    _mm512_mask_packstorehi_ps(p+type_traits<float>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<long>::num_bvals_per_reg;
  }
}

// unaligned temporal store: long -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d temp = cvt512_epi64_pd(op(i));
#if defined(SIMD_AVX512F)
    _mm512_storeu_pd(p, temp);
    p += type_traits<long>::num_bvals_per_reg;
#else
    _mm512_packstorelo_pd(p, temp);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, temp);
#endif
  }
}

// masked unaligned temporal store: long -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d temp = cvt512_epi64_pd(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_pd(p, m(i), temp);
    p += type_traits<long>::num_bvals_per_reg;
#else
    __m512d temp1;
    temp1 = _mm512_loadunpacklo_pd(temp1, p);
    temp1 = _mm512_loadunpackhi_pd(temp1, p+type_traits<double>::num_bvals_per_reg);
    temp = _mm512_mask_blend_pd(m(i), temp1, temp);
    _mm512_packstorelo_pd(p, temp);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, temp);
#endif
  }
}

// unaligned temporal store: float -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  int *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 sm;
  sm = _mm512_kxnor(sm, sm);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_ps_epi32(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, sm, temp);
    p += type_traits<float>::num_bvals_per_reg;
#else
    _mm512_packstorelo_epi32(p, temp);
    p += type_traits<float>::num_bvals_per_reg;
    _mm512_packstorehi_epi32(p, temp);
#endif
  }
}

// masked unaligned temporal store: float -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  int *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_ps_epi32(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, m(i), temp);
    p += type_traits<float>::num_bvals_per_reg;
#else
    __m512i temp1;
    temp1 = _mm512_loadunpacklo_epi32(temp1, p);
    temp1 = _mm512_loadunpackhi_epi32(temp1, p+type_traits<int>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi32(m(i), temp1, temp);
    _mm512_packstorelo_epi32(p, temp);
    p += type_traits<float>::num_bvals_per_reg;
    _mm512_packstorehi_epi32(p, temp);
#endif
  }
}

// unaligned temporal store: float -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  long *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 sm;
  sm = _mm512_kxnor(sm, sm);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo = cvt512_pslo_epi64(op(i));
    __m512i hi = cvt512_pshi_epi64(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, sm, lo);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_mask_storeu_epi64(p, sm, hi);
    p += type_traits<long>::num_bvals_per_reg;
#else
    _mm512_packstorelo_epi64(p, lo);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, lo);
    _mm512_packstorelo_epi64(p, hi);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, hi);
#endif
  }
}

// masked unaligned temporal store: float -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i lo = cvt512_pslo_epi64(op(i));
    __m512i hi = cvt512_pshi_epi64(op(i));
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, m(i), lo);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_mask_storeu_epi64(p, himask, hi);
    p += type_traits<long>::num_bvals_per_reg;
#else
    __m512i temp, temp1;
    temp = _mm512_loadunpacklo_epi64(temp, p);
    temp = _mm512_loadunpackhi_epi64(temp, p+type_traits<long>::num_bvals_per_reg);
    lo = _mm512_mask_blend_epi64(m(i), temp, lo);
    _mm512_packstorelo_epi64(p, lo);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, lo);
    
    temp1 = _mm512_loadunpacklo_epi64(temp1, p);
    temp1 = _mm512_loadunpackhi_epi64(temp1, p+type_traits<long>::num_bvals_per_reg);
    hi = _mm512_mask_blend_epi64(himask, temp1, hi);
    _mm512_packstorelo_epi64(p, hi);
    p += type_traits<long>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, hi);
#endif
  }
}

// unaligned temporal store: float -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_storeu_ps(p, op(i));
    p += type_traits<float>::num_bvals_per_reg;
#else
    _mm512_packstorelo_ps(p, op(i));
    p += type_traits<float>::num_bvals_per_reg;
    _mm512_packstorehi_ps(p, op(i));
#endif
  }
}

// masked unaligned temporal store: float -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  float *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_ps(p, m(i), op(i));
#else
    __m512 temp;
    temp = _mm512_loadunpacklo_ps(temp, p);
    temp = _mm512_loadunpackhi_ps(temp, p+type_traits<float>::num_bvals_per_reg);
    temp = _mm512_mask_blend_ps(m(i), temp, op(i));
    _mm512_packstorelo_ps(p, temp);
    p += type_traits<float>::num_bvals_per_reg;
    _mm512_packstorehi_ps(p, temp);
#endif
  }
}

// unaligned temporal store: float -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512, float, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo = cvt512_pslo_pd(op(i));
    __m512d hi = cvt512_pshi_pd(op(i));
#if defined(SIMD_AVX512F)
    _mm512_storeu_pd(p, lo);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_storeu_pd(p, hi);
    p += type_traits<double>::num_bvals_per_reg;
#else
    _mm512_packstorelo_pd(p, lo);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, lo);
    _mm512_packstorelo_pd(p, hi);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, hi);
#endif
  }
}

// masked unaligned temporal store: float -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512, float, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512d lo = cvt512_pslo_pd(op(i));
    __m512d hi = cvt512_pshi_pd(op(i));
    __mmask16 himask = _mm512_kmerge2l1h(m(i), m(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_pd(p, m(i), lo);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_mask_storeu_pd(p, himask, hi);
    p += type_traits<double>::num_bvals_per_reg;
#else
    __m512d temp, temp1;
    temp = _mm512_loadunpacklo_pd(temp, p);
    temp = _mm512_loadunpackhi_pd(temp, p+type_traits<double>::num_bvals_per_reg);
    lo = _mm512_mask_blend_pd(m(i), temp, lo);
    temp1 = _mm512_loadunpacklo_pd(temp1, p+type_traits<double>::num_bvals_per_reg);
    temp1 = _mm512_loadunpackhi_pd(temp1, p+type_traits<double>::num_bvals_per_reg+type_traits<double>::num_bvals_per_reg);
    hi = _mm512_mask_blend_pd(himask, temp1, hi);
    _mm512_packstorelo_pd(p, lo);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, lo);
    _mm512_packstorelo_pd(p, hi);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, hi);
#endif
  }
}

// unaligned temporal store: double -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  int *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_pd_epi32lo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, sm, temp);
#else
    _mm512_mask_packstorelo_epi32(p, sm, temp);
    _mm512_mask_packstorehi_epi32(p+type_traits<int>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// masked unaligned temporal store: double -> int
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<int, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  int *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_pd_epi32lo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi32(p, _mm512_kand(m(i), sm), temp);
#else
    __m512i temp1;
    temp1 = _mm512_mask_loadunpacklo_epi32(temp1, sm, p);
    temp1 = _mm512_mask_loadunpackhi_epi32(temp1, sm, p+type_traits<int>::num_bvals_per_reg);
    temp  =_mm512_mask_blend_epi32(m(i), temp1, temp);
    _mm512_mask_packstorelo_epi32(p, sm, temp);
    _mm512_mask_packstorehi_epi32(p+type_traits<int>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// unaligned temporal store: double -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  long *p = ptr;
#if defined(SIMD_AVX512F)
  __mmask16 sm;
  sm = _mm512_kxnor(sm, sm);
#endif
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_pd_epi64(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, sm, temp);
    p += type_traits<double>::num_bvals_per_reg;
#else
    _mm512_packstorelo_epi64(p, temp);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, temp);
#endif
  }
}

// masked unaligned temporal store: double -> long
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<long, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  long *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512i temp = cvt512_pd_epi64(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_epi64(p, m(i), temp);
    p += type_traits<double>::num_bvals_per_reg;
#else
    __m512i temp1;
    temp1 = _mm512_loadunpacklo_epi64(temp1, p);
    temp1 = _mm512_loadunpackhi_epi64(temp1, p+type_traits<long>::num_bvals_per_reg);
    temp = _mm512_mask_blend_epi64(m(i), temp1, temp);
    _mm512_packstorelo_epi64(p, temp);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, temp);
#endif
  }
}

// unaligned temporal store: double -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  float *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp = cvt512_pd_pslo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_ps(p, sm, temp);
#else
    _mm512_mask_packstorelo_ps(p, sm, temp);
    _mm512_mask_packstorehi_ps(p+type_traits<float>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// masked unaligned temporal store: double -> float
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<float, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  float *p = ptr;
  __mmask16 sm = _mm512_int2mask(0b0000000011111111);
  for(int i = 0; i < type_t::num_regs; ++i) {
    __m512 temp = cvt512_pd_pslo(op(i));
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_ps(p, _mm512_kand(m(i), sm), temp);
#else
    __m512 temp1;
    temp1 = _mm512_mask_loadunpacklo_ps(temp1, sm, p);
    temp1 = _mm512_mask_loadunpackhi_ps(temp1, sm, p+type_traits<float>::num_bvals_per_reg);
    temp = _mm512_mask_blend_ps(m(i), temp1, temp);
    _mm512_mask_packstorelo_ps(p, sm, temp);
    _mm512_mask_packstorehi_ps(p+type_traits<float>::num_bvals_per_reg, sm, temp);
#endif
    p += type_traits<double>::num_bvals_per_reg;
  }
}

// unaligned temporal store: double -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_storeu_pd(p, op(i));
    p += type_traits<double>::num_bvals_per_reg;
#else
    _mm512_packstorelo_pd(p, op(i));
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, op(i));
#endif
  }
}

// masked unaligned temporal store: double -> double
template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> &op, mptr<double, memhint::temporal|memhint::unaligned> ptr, const mask<T, NW> &m) {
  typedef pack<T, NW, __m512d, double, W> type_t;
  double *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
#if defined(SIMD_AVX512F)
    _mm512_mask_storeu_pd(p, m(i), op(i));
    p += type_traits<double>::num_bvals_per_reg;
#else
    __m512d temp;
    temp = _mm512_loadunpacklo_pd(temp, p);
    temp = _mm512_loadunpackhi_pd(temp, p+type_traits<double>::num_bvals_per_reg);
    temp = _mm512_mask_blend_pd(m(i), temp, op(i));
    _mm512_packstorelo_pd(p, temp);
    p += type_traits<double>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, temp);
#endif
  }
}

/*template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512i, long, W> &op, const_mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  typedef pack<T, NW, __m512i, long, W> type_t;
  long const *p = ptr;
  for(int i = 0; i < type_t::num_regs; ++i) {
    op(i) = _mm512_loadunpacklo_epi64(op(i), p);
    p += type_t::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_epi64(op(i), p);
  }
}*/

/*template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512, float, W> &op, const_mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  float const *p = ptr;
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    op(i) = _mm512_loadunpacklo_ps(op(i), p);
    p += pack<T, NW, __m512, float, W>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_ps(op(i), p);
  }
}*/

/*template<typename T, int NW, int W>
inline void load(pack<T, NW, __m512d, double, W> &op, const_mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  double const *p = ptr;
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    op(i) = _mm512_loadunpacklo_pd(op(i), p);
    p += pack<T, NW, __m512d, double, W>::num_bvals_per_reg;
    op(i) = _mm512_loadunpackhi_pd(op(i), p);
  }
}*/

/*template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, int, W> const &op, mptr<int, memhint::temporal|memhint::unaligned> ptr) {
  int *p = ptr;
  for(int i = 0; i < pack<T, NW, __m512i, int, W>::num_regs; ++i) {
    _mm512_packstorelo_epi32(p, op(i));
    p += pack<T, NW, __m512i, int, W>::num_bvals_per_reg;
    _mm512_packstorehi_epi32(p, op(i));
  }
}*/

/*template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512i, long, W> const &op, mptr<long, memhint::temporal|memhint::unaligned> ptr) {
  long *p = ptr;
  for(int i = 0; i < pack<T, NW, __m512i, long, W>::num_regs; ++i) {
    _mm512_packstorelo_epi64(p, op(i));
    p += pack<T, NW, __m512i, long, W>::num_bvals_per_reg;
    _mm512_packstorehi_epi64(p, op(i));
  }
}*/

/*template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512, float, W> const &op, mptr<float, memhint::temporal|memhint::unaligned> ptr) {
  float *p = ptr;
  for(int i = 0; i < pack<T, NW, __m512, float, W>::num_regs; ++i) {
    _mm512_packstorelo_ps(p, op(i));
    p += pack<T, NW, __m512, float, W>::num_bvals_per_reg;
    _mm512_packstorehi_ps(p, op(i));
  }
}*/

/*template<typename T, int NW, int W>
inline void store(pack<T, NW, __m512d, double, W> const &op, mptr<double, memhint::temporal|memhint::unaligned> ptr) {
  double *p = ptr;
  for(int i = 0; i < pack<T, NW, __m512d, double, W>::num_regs; ++i) {
    _mm512_packstorelo_pd(p, op(i));
    p += pack<T, NW, __m512d, double, W>::num_bvals_per_reg;
    _mm512_packstorehi_pd(p, op(i));
  }
}*/

/* Permute functions */

template<>
class permutation<permute_pattern::aacc> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_CCAA);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_BABA);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_PERM_CCAA);
#else
      temp(i) = _mm512_castsi512_ps(_mm512_shuffle_epi32(_mm512_castps_si512(op(i)), _MM_PERM_CCAA));
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_pd(op(i), 0x0);
#else
      temp(i) = _mm512_castsi512_pd(_mm512_shuffle_epi32(_mm512_castpd_si512(op(i)), _MM_PERM_BABA));
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::abab> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_BABA);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_permute4f128_epi32(op(i), _MM_PERM_CCAA);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_PERM_BABA);
#else
      temp(i) = _mm512_castsi512_ps(_mm512_shuffle_epi32(_mm512_castps_si512(op(i)), _MM_PERM_BABA));
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(op(i)), _MM_PERM_CCAA));
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::bbdd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_DDBB);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_DCDC);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_PERM_DDBB);
#else
      temp(i) = _mm512_castsi512_ps(_mm512_shuffle_epi32(_mm512_castps_si512(op(i)), _MM_PERM_DDBB));
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_castsi512_pd(_mm512_shuffle_epi32(_mm512_castpd_si512(op(i)), _MM_PERM_DCDC));
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::cdcd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_DCDC);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_permute4f128_epi32(op(i), _MM_PERM_DDBB);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_PERM_DCDC);
#else
      temp(i) = _mm512_castsi512_ps(_mm512_shuffle_epi32(_mm512_castps_si512(op(i)), _MM_PERM_DCDC));
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_castsi512_pd(_mm512_permute4f128_epi32(_mm512_castpd_si512(op(i)), _MM_PERM_DDBB));
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::dcba> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_ABCD);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_epi64(op(i), _MM_SHUFFLE(0,1,2,3));
#else
      temp(i) = _mm512_swizzle_epi64(_mm512_swizzle_epi64(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC);
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_SHUFFLE(0,1,2,3));
#else
      temp(i) = _mm512_swizzle_ps(_mm512_swizzle_ps(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC);
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_pd(op(i), _MM_SHUFFLE(0,1,2,3));
#else
      temp(i) = _mm512_swizzle_pd(_mm512_swizzle_pd(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC);
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::dbca> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_ACBD);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_epi64(op(i), _MM_SHUFFLE(0,2,1,3));
#else
      temp(i) = _mm512_mask_blend_epi64(
        _mm512_int2mask(0b1001100110011001), 
        op(i), 
        _mm512_swizzle_epi64(_mm512_swizzle_epi64(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_SHUFFLE(0,2,1,3));
#else
      temp(i) = _mm512_mask_blend_ps(
        _mm512_int2mask(0b1001100110011001), 
        op(i), 
        _mm512_swizzle_ps(_mm512_swizzle_ps(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_pd(op(i), _MM_SHUFFLE(0,2,1,3));
#else
      temp(i) = _mm512_mask_blend_pd(
        _mm512_int2mask(0b1001100110011001), 
        op(i), 
        _mm512_swizzle_pd(_mm512_swizzle_pd(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC)
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
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_CACA);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_epi64(op(i), _MM_SHUFFLE(2,0,2,0));
#else
      temp(i) = _mm512_mask_blend_epi64(
        _mm512_int2mask(0b1010101010101010), 
        _mm512_swizzle_epi64(op(i), _MM_SWIZ_REG_AAAA), 
        _mm512_swizzle_epi64(op(i), _MM_SWIZ_REG_CCCC)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_SHUFFLE(2,0,2,0));
#else
      temp(i) = _mm512_mask_blend_ps(
        _mm512_int2mask(0b1010101010101010), 
        _mm512_swizzle_ps(op(i), _MM_SWIZ_REG_AAAA), 
        _mm512_swizzle_ps(op(i), _MM_SWIZ_REG_CCCC)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_pd(op(i), _MM_SHUFFLE(2,0,2,0));
#else
      temp(i) = _mm512_mask_blend_pd(
        _mm512_int2mask(0b1010101010101010), 
        _mm512_swizzle_pd(op(i), _MM_SWIZ_REG_AAAA), 
        _mm512_swizzle_pd(op(i), _MM_SWIZ_REG_CCCC)
      );
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::bdbd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_DBDB);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_epi64(op(i), _MM_SHUFFLE(3,1,3,1));
#else
      temp(i) = _mm512_mask_blend_epi64(
        _mm512_int2mask(0b1010101010101010),
        _mm512_swizzle_epi64(op(i), _MM_SWIZ_REG_BBBB),
        _mm512_swizzle_epi64(op(i), _MM_SWIZ_REG_DDDD)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_SHUFFLE(3,1,3,1));
#else
      temp(i) = _mm512_mask_blend_ps(
        _mm512_int2mask(0b1010101010101010), 
        _mm512_swizzle_ps(op(i), _MM_SWIZ_REG_BBBB), 
        _mm512_swizzle_ps(op(i), _MM_SWIZ_REG_DDDD)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_pd(op(i), _MM_SHUFFLE(3,1,3,1));
#else
      temp(i) = _mm512_mask_blend_pd(
        _mm512_int2mask(0b1010101010101010),
        _mm512_swizzle_pd(op(i), _MM_SWIZ_REG_BBBB),
        _mm512_swizzle_pd(op(i), _MM_SWIZ_REG_DDDD)
      );
#endif
    }
    return temp;
  }
};

template<>
class permutation<permute_pattern::acbd> {
public:
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, int, 4> permute(const pack<T, NW, __m512i, int, 4> &op) {
    typedef pack<T, NW, __m512i, int, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
      temp(i) = _mm512_shuffle_epi32(op(i), _MM_PERM_DBCA);
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512i, long, 4> permute(const pack<T, NW, __m512i, long, 4> &op) {
    typedef pack<T, NW, __m512i, long, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_epi64(op(i), _MM_SHUFFLE(3,1,2,0));
#else
      temp(i) = _mm512_mask_blend_epi64(
        _mm512_int2mask(0b0110011001100110), 
        op(i), 
        _mm512_swizzle_epi64(_mm512_swizzle_epi64(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512, float, 4> permute(const pack<T, NW, __m512, float, 4> &op) {
    typedef pack<T, NW, __m512, float, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permute_ps(op(i), _MM_SHUFFLE(3,1,2,0));
#else
      temp(i) = _mm512_mask_blend_ps(
        _mm512_int2mask(0b0110011001100110), 
        op(i), 
        _mm512_swizzle_ps(_mm512_swizzle_ps(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC)
      );
#endif
    }
    return temp;
  }
  
  template<typename T, int NW>
  inline static pack<T, NW, __m512d, double, 4> permute(const pack<T, NW, __m512d, double, 4> &op) {
    typedef pack<T, NW, __m512d, double, 4> type;
    type temp;
    for(int i = 0; i < type::num_regs; ++i) {
#if defined(SIMD_AVX512F)
      temp(i) = _mm512_permutex_pd(op(i), _MM_SHUFFLE(3,1,2,0));
#else
      temp(i) = _mm512_mask_blend_pd(
        _mm512_int2mask(0b0110011001100110), 
        op(i), 
        _mm512_swizzle_pd(_mm512_swizzle_pd(op(i), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC)
      );
#endif
    }
    return temp;
  }
};

/* Interleave functions */

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512, float, 1>, VF> op) {
  typedef pack<T, NW, __m512, float, 1> type_t;
  
  __mmask16 m1 = _mm512_int2mask(0b1010101010101010);
  __mmask16 m2 = _mm512_int2mask(0b1100110011001100);
  __mmask16 m3 = _mm512_int2mask(0b1111000011110000);
  __mmask16 m4 = _mm512_int2mask(0b1111111100000000);
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m512 op1 = _mm512_swizzle_ps(op[i](ri), _MM_SWIZ_REG_CDAB);
      __m512 op2 = _mm512_swizzle_ps(op[i+1](ri), _MM_SWIZ_REG_CDAB);
      op[i](ri) = _mm512_mask_blend_ps(m1, op[i](ri), op2);
      op[i+1](ri) = _mm512_mask_blend_ps(m1, op1, op[i+1](ri));
    }
    
    for(int i = 0; i < NW; i+=4) {
      for(int j = 0; j < 2; ++j) {
        __m512 op1 = _mm512_swizzle_ps(op[i+j](ri), _MM_SWIZ_REG_BADC);
        __m512 op2 = _mm512_swizzle_ps(op[i+j+2](ri), _MM_SWIZ_REG_BADC);
        op[i+j](ri) = _mm512_mask_blend_ps(m2, op[i+j](ri), op2);
        op[i+j+2](ri) = _mm512_mask_blend_ps(m2, op1, op[i+j+2](ri));
      }
    }
    
    for(int i = 0; i < NW; i += 8) {
      for(int j = 0; j < 4; ++j) {
        __m512 op1 = _mm512_permute4f128_ps(op[i+j](ri), _MM_PERM_CDAB);
        __m512 op2 = _mm512_permute4f128_ps(op[i+j+4](ri), _MM_PERM_CDAB);
        op[i+j](ri) = _mm512_mask_blend_ps(m3, op[i+j](ri), op2);
        op[i+j+4](ri) = _mm512_mask_blend_ps(m3, op1, op[i+j+4](ri));
      }
    }
    
    for(int i = 0; i < NW; i += 16) {
      for(int j = 0; j < 8; ++j) {
        __m512 op1 = _mm512_permute4f128_ps(op[i+j](ri), _MM_PERM_BADC);
        __m512 op2 = _mm512_permute4f128_ps(op[i+j+8](ri), _MM_PERM_BADC);
        op[i+j](ri) = _mm512_mask_blend_ps(m4, op[i+j](ri), op2);
        op[i+j+8](ri) = _mm512_mask_blend_ps(m4, op1, op[i+j+8](ri));
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*16;
        int m2 = j*16;
        for(int k = 0; k < 16; ++k, ++m1, ++m2) {
          __m512 lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512, float, 2>, VF> op) {
  typedef pack<T, NW, __m512, float, 2> type_t;
  
  __mmask16 m2 = _mm512_int2mask(0b1100110011001100);
  __mmask16 m3 = _mm512_int2mask(0b1111000011110000);
  __mmask16 m4 = _mm512_int2mask(0b1111111100000000);
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m512 op1 = _mm512_swizzle_ps(op[i](ri), _MM_SWIZ_REG_BADC);
      __m512 op2 = _mm512_swizzle_ps(op[i+1](ri), _MM_SWIZ_REG_BADC);
      op[i](ri) = _mm512_mask_blend_ps(m2, op[i](ri), op2);
      op[i+1](ri) = _mm512_mask_blend_ps(m2, op1, op[i+1](ri));
    }
    
    for(int i = 0; i < NW; i+=4) {
      for(int j = 0; j < 2; ++j) {
        __m512 op1 = _mm512_permute4f128_ps(op[i+j](ri), _MM_PERM_CDAB);
        __m512 op2 = _mm512_permute4f128_ps(op[i+j+2](ri), _MM_PERM_CDAB);
        op[i+j](ri) = _mm512_mask_blend_ps(m3, op[i+j](ri), op2);
        op[i+j+2](ri) = _mm512_mask_blend_ps(m3, op1, op[i+j+2](ri));
      }
    }
    
    for(int i = 0; i < NW; i += 8) {
      for(int j = 0; j < 4; ++j) {
        __m512 op1 = _mm512_permute4f128_ps(op[i+j](ri), _MM_PERM_BADC);
        __m512 op2 = _mm512_permute4f128_ps(op[i+j+4](ri), _MM_PERM_BADC);
        op[i+j](ri) = _mm512_mask_blend_ps(m4, op[i+j](ri), op2);
        op[i+j+4](ri) = _mm512_mask_blend_ps(m4, op1, op[i+j+4](ri));
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*8;
        int m2 = j*8;
        for(int k = 0; k < 8; ++k, ++m1, ++m2) {
          __m512 lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512, float, 4>, VF> op) {
  typedef pack<T, NW, __m512, float, 4> type_t;
  
  __mmask16 m3 = _mm512_int2mask(0b1111000011110000);
  __mmask16 m4 = _mm512_int2mask(0b1111111100000000);
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m512 op1 = _mm512_permute4f128_ps(op[i](ri), _MM_PERM_CDAB);
      __m512 op2 = _mm512_permute4f128_ps(op[i+1](ri), _MM_PERM_CDAB);
      op[i](ri) = _mm512_mask_blend_ps(m3, op[i](ri), op2);
      op[i+1](ri) = _mm512_mask_blend_ps(m3, op1, op[i+1](ri));
    }
    
    for(int i = 0; i < NW; i+=4) {
      for(int j = 0; j < 2; ++j) {
        __m512 op1 = _mm512_permute4f128_ps(op[i+j](ri), _MM_PERM_BADC);
        __m512 op2 = _mm512_permute4f128_ps(op[i+j+2](ri), _MM_PERM_BADC);
        op[i+j](ri) = _mm512_mask_blend_ps(m4, op[i+j](ri), op2);
        op[i+j+2](ri) = _mm512_mask_blend_ps(m4, op1, op[i+j+2](ri));
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*4;
        int m2 = j*4;
        for(int k = 0; k < 4; ++k, ++m1, ++m2) {
          __m512 lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512, float, 8>, VF> op) {
  typedef pack<T, NW, __m512, float, 8> type_t;
  
  __mmask16 m4 = _mm512_int2mask(0b1111111100000000);
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m512 op1 = _mm512_permute4f128_ps(op[i](ri), _MM_PERM_BADC);
      __m512 op2 = _mm512_permute4f128_ps(op[i+1](ri), _MM_PERM_BADC);
      op[i](ri) = _mm512_mask_blend_ps(m4, op[i](ri), op2);
      op[i+1](ri) = _mm512_mask_blend_ps(m4, op1, op[i+1](ri));
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*2;
        int m2 = j*2;
        for(int k = 0; k < 2; ++k, ++m1, ++m2) {
          __m512 lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512, float, 16>, VF> op) {
  typedef pack<T, NW, __m512, float, 16> type_t;
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        __m512 lo = op[i](j);
        op[i](j) = op[j](i);
        op[j](i) = lo;
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512d, double, 1>, VF> op) {
  typedef pack<T, NW, __m512d, double, 1> type_t;
  
  __mmask8 m1 = _mm512_int2mask(0b10101010);
  __mmask8 m2 = _mm512_int2mask(0b11001100);
  __mmask8 m3 = _mm512_int2mask(0b11110000);
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m512d op1 = _mm512_swizzle_pd(op[i](ri), _MM_SWIZ_REG_CDAB);
      __m512d op2 = _mm512_swizzle_pd(op[i+1](ri), _MM_SWIZ_REG_CDAB);
      op[i](ri) = _mm512_mask_blend_pd(m1, op[i](ri), op2);
      op[i+1](ri) = _mm512_mask_blend_pd(m1, op1, op[i+1](ri));
    }
    
    for(int i = 0; i < NW; i += 4) {
      for(int j = 0; j < 2; ++j) {
        __m512d op1 = _mm512_swizzle_pd(op[i+j](ri), _MM_SWIZ_REG_BADC);
        __m512d op2 = _mm512_swizzle_pd(op[i+j+2](ri), _MM_SWIZ_REG_BADC);
        op[i+j](ri) = _mm512_mask_blend_pd(m2, op[i+j](ri), op2);
        op[i+j+2](ri) = _mm512_mask_blend_pd(m2, op1, op[i+j+2](ri));
      }
    }
    
    for(int i = 0; i < NW; i += 8) {
      for(int j = 0; j < 4; ++j) {
        __m512d op1 = _mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(op[i+j](ri)), _MM_PERM_BADC));
        __m512d op2 = _mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(op[i+j+4](ri)), _MM_PERM_BADC));
        op[i+j](ri) = _mm512_mask_blend_pd(m3, op[i+j](ri), op2);
        op[i+j+4](ri) = _mm512_mask_blend_pd(m3, op1, op[i+j+4](ri));
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*8;
        int m2 = j*8;
        for(int k = 0; k < 8; ++k, ++m1, ++m2) {
          __m512d lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512d, double, 2>, VF> op) {
  typedef pack<T, NW, __m512d, double, 2> type_t;
  
  __mmask8 m2 = _mm512_int2mask(0b11001100);
  __mmask8 m3 = _mm512_int2mask(0b11110000);
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m512d op1 = _mm512_swizzle_pd(op[i](ri), _MM_SWIZ_REG_BADC);
      __m512d op2 = _mm512_swizzle_pd(op[i+1](ri), _MM_SWIZ_REG_BADC);
      op[i](ri) = _mm512_mask_blend_pd(m2, op[i](ri), op2);
      op[i+1](ri) = _mm512_mask_blend_pd(m2, op1, op[i+1](ri));
    }
    
    for(int i = 0; i < NW; i += 4) {
      for(int j = 0; j < 2; ++j) {
        __m512d op1 = _mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(op[i+j](ri)), _MM_PERM_BADC));
        __m512d op2 = _mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(op[i+j+2](ri)), _MM_PERM_BADC));
        op[i+j](ri) = _mm512_mask_blend_pd(m3, op[i+j](ri), op2);
        op[i+j+2](ri) = _mm512_mask_blend_pd(m3, op1, op[i+j+2](ri));
      }
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*4;
        int m2 = j*4;
        for(int k = 0; k < 4; ++k, ++m1, ++m2) {
          __m512d lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512d, double, 4>, VF> op) {
  typedef pack<T, NW, __m512d, double, 4> type_t;
  
  __mmask8 m3 = _mm512_int2mask(0b11110000);
  for(int ri = 0; ri < type_t::num_regs; ++ri) {
    for(int i = 0; i < NW; i+=2) {
      __m512d op1 = _mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(op[i](ri)), _MM_PERM_BADC));
      __m512d op2 = _mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(op[i+1](ri)), _MM_PERM_BADC));
      op[i](ri) = _mm512_mask_blend_pd(m3, op[i](ri), op2);
      op[i+1](ri) = _mm512_mask_blend_pd(m3, op1, op[i+1](ri));
    }
  }
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        int m1 = i*2;
        int m2 = j*2;
        for(int k = 0; k < 2; ++k, ++m1, ++m2) {
          __m512d lo = op[m1](j);
          op[m1](j) = op[m2](i);
          op[m2](i) = lo;
        }
      }
    }
  }
}

template<typename T, int NW, int VF>
void interleave(varray<pack<T, NW, __m512d, double, 8>, VF> op) {
  typedef pack<T, NW, __m512d, double, 8> type_t;
  
  if(type_t::num_regs > 1) {
    for(int i = 0; i < type_t::num_regs; ++i) {
      for(int j = i+1; j < type_t::num_regs; ++j) {
        __m512d lo = op[i](j);
        op[i](j) = op[j](i);
        op[j](i) = lo;
      }
    }
  }
}

#endif // SIMD512

}

/*#if SIMD512

namespace simd {

//------------------------------------------------------------------------------
// Macros that define SIMD width of various types
//------------------------------------------------------------------------------

#define SIMD512_WIDTH 64

#ifdef __SIZEOF_SHORT__
  #if __SIZEOF_SHORT__ == 2
    CHECK_TYPE_SIZE(short, 2, simd)
    #define SIMD512_SHORT_WIDTH 32
  #else
    #error "Unsupported short size"
  #endif
#else
  #error "Macro __SIZEOF_SHORT__ not defined"
#endif

#ifdef __SIZEOF_INT__
  #if __SIZEOF_INT__ == 4
    CHECK_TYPE_SIZE(int, 4, simd)
    #define SIMD512_INT_WIDTH 16
  #else
    #error "Unsupported int size"
  #endif
#else
  #error "Macro __SIZEOF_INT__ not defined"
#endif

#ifdef __SIZEOF_LONG__
  #if __SIZEOF_LONG__ == 8
    CHECK_TYPE_SIZE(long, 8, simd)
    #define SIMD512_LONG_WIDTH 8
  #else
    #error "Unsupported long size"
  #endif
#else
  #error "Macro __SIZEOF_LONG__ not defined"
#endif

#ifdef __SIZEOF_FLOAT__
  #if __SIZEOF_FLOAT__ == 4
    CHECK_TYPE_SIZE(float, 4, simd)
    #define SIMD512_FLOAT_WIDTH 16
  #else
    #error "Unsupported float size"
  #endif
#else
  #error "Macro __SIZEOF_FLOAT__ not defined"
#endif

#ifdef __SIZEOF_DOUBLE__
  #if __SIZEOF_DOUBLE__ == 8
    CHECK_TYPE_SIZE(double, 8, simd)
    #define SIMD512_DOUBLE_WIDTH 8
  #else
    #error "Unsupported double size"
  #endif
#else
  #error "Macro __SIZEOF_DOUBLE__ not defined"
#endif

//------------------------------------------------------------------------------
// SIMD value types
//------------------------------------------------------------------------------

template<typename ctype, int x>
struct value;

template<>
struct value<short, 512> {
//#if defined(SIMD_KNC)
//  typedef reg2x<__m256i> reg;
//#else
  typedef __m512i reg;
//#endif
  static const size_t width = SIMD512_SHORT_WIDTH;
};

template<>
struct value<unsigned short, 512> {
//#if defined(SIMD_KNC)
//  typedef reg2x<__m256i> reg;
//#else
  typedef __m512i reg;
//#endif
  static const size_t width = SIMD512_SHORT_WIDTH;
};

template<>
struct value<int, 512> {
  typedef __m512i reg;
  static const size_t width = SIMD512_INT_WIDTH;
};

template<>
struct value<unsigned int, 512> {
  typedef __m512i reg;
  static const size_t width = SIMD512_INT_WIDTH;
};

template<>
struct value<long, 512> {
  typedef __m512i reg;
  static const size_t width = SIMD512_LONG_WIDTH;
};

template<>
struct value<unsigned long, 512> {
  typedef __m512i reg;
  static const size_t width = SIMD512_LONG_WIDTH;
};

template<>
struct value<float, 512> {
  typedef __m512 reg;
  static const size_t width = SIMD512_FLOAT_WIDTH;
};

template<>
struct value<double, 512> {
  typedef __m512d reg;
  static const size_t width = SIMD512_DOUBLE_WIDTH;
};

}

#endif // #if SIMD512
*/
#endif // SIMD_SIMD512X86DEF_HPP
