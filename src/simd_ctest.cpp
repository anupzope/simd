#include <simd_generic_ctest.hpp>

#if SIMD256

/* TEST - sanity check */

TEST_CASE("simd::pack<int> - sanity check", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<int> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<int[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<int[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 16);
  REQUIRE(type1::num_regs == 2);
  REQUIRE(type1::num_bvals == 16);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 8);
  REQUIRE(type1::num_vals_per_reg == 8);
  
  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 8);
  REQUIRE(type2::num_regs == 2);
  REQUIRE(type2::num_bvals == 16);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 8);
  REQUIRE(type2::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type3) == 64);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 2);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 8);
  REQUIRE(type3::num_vals_per_reg == 2);
#else // #if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<int> type1; 

  /* pack for 1D array and default nways */
  typedef simd::pack<int[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<int[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 32);
  REQUIRE(alignof(type1) == 32);
  REQUIRE(type1::num_vals == 8);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 8);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 8);
  REQUIRE(type1::num_vals_per_reg == 8);
  
  REQUIRE(sizeof(type2) == 32);
  REQUIRE(alignof(type2) == 32);
  REQUIRE(type2::num_vals == 4);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 8);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 8);
  REQUIRE(type2::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type3) == 64);
  REQUIRE(alignof(type3) == 32);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 2);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 8);
  REQUIRE(type3::num_vals_per_reg == 2);
#endif // #if SIMD512OFFLOAD
}

TEST_CASE("simd::pack<long> - sanity check", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<long> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<long[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<long[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 8);
  REQUIRE(type1::num_regs == 2);
  REQUIRE(type1::num_bvals == 8);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 4);
  REQUIRE(type1::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 4);
  REQUIRE(type2::num_regs == 2);
  REQUIRE(type2::num_bvals == 8);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 4);
  REQUIRE(type2::num_vals_per_reg == 2);
  
  REQUIRE(sizeof(type3) == 128);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 4);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 4);
  REQUIRE(type3::num_vals_per_reg == 1);
#else // #if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<long> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<long[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<long[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 32);
  REQUIRE(alignof(type1) == 32);
  REQUIRE(type1::num_vals == 4);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 4);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 4);
  REQUIRE(type1::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type2) == 32);
  REQUIRE(alignof(type2) == 32);
  REQUIRE(type2::num_vals == 2);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 4);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 4);
  REQUIRE(type2::num_vals_per_reg == 2);
  
  REQUIRE(sizeof(type3) == 128);
  REQUIRE(alignof(type3) == 32);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 4);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 4);
  REQUIRE(type3::num_vals_per_reg == 1);
#endif // #if SIMD512OFFLOAD
}

TEST_CASE("simd::pack<float> - sanity check", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<float> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<float[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<float[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 16);
  REQUIRE(type1::num_regs == 2);
  REQUIRE(type1::num_bvals == 16);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 8);
  REQUIRE(type1::num_vals_per_reg == 8);

  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 8);
  REQUIRE(type2::num_regs == 2);
  REQUIRE(type2::num_bvals == 16);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 8);
  REQUIRE(type2::num_vals_per_reg == 4);

  REQUIRE(sizeof(type3) == 64);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 2);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 8);
  REQUIRE(type3::num_vals_per_reg == 2);
#else // #if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<float> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<float[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<float[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 32);
  REQUIRE(alignof(type1) == 32);
  REQUIRE(type1::num_vals == 8);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 8);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 8);
  REQUIRE(type1::num_vals_per_reg == 8);
  
  REQUIRE(sizeof(type2) == 32);
  REQUIRE(alignof(type2) == 32);
  REQUIRE(type2::num_vals == 4);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 8);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 8);
  REQUIRE(type2::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type3) == 64);
  REQUIRE(alignof(type3) == 32);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 2);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 8);
  REQUIRE(type3::num_vals_per_reg == 2);
#endif // #if SIMD512OFFLOAD
}

TEST_CASE("simd::pack<double> - sanity check", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<double> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<double[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<double[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 8);
  REQUIRE(type1::num_regs == 2);
  REQUIRE(type1::num_bvals == 8);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 4);
  REQUIRE(type1::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 4);
  REQUIRE(type2::num_regs == 2);
  REQUIRE(type2::num_bvals == 8);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 4);
  REQUIRE(type2::num_vals_per_reg == 2);
  
  REQUIRE(sizeof(type3) == 128);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 4);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 4);
  REQUIRE(type3::num_vals_per_reg == 1);
#else // #if SIMD512OFFLOAD
  /* pack for scalar type and default nways */
  typedef simd::pack<double> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<double[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<double[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 32);
  REQUIRE(alignof(type1) == 32);
  REQUIRE(type1::num_vals == 4);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 4);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 4);
  REQUIRE(type1::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type2) == 32);
  REQUIRE(alignof(type2) == 32);
  REQUIRE(type2::num_vals == 2);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 4);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 4);
  REQUIRE(type2::num_vals_per_reg == 2);
  
  REQUIRE(sizeof(type3) == 128);
  REQUIRE(alignof(type3) == 32);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 4);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 4);
  REQUIRE(type3::num_vals_per_reg == 1);
#endif // #if SIMD512OFFLOAD
}

/* TEST - ctor: pack(register_type&) */

TEST_CASE("simd::pack<int> - ctor: pack(register_type&)", "[pack:avx/avx2/offload]") {
  simd::pack<int> a(_mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1));
  
  for(int i = 0; i < simd::pack<int>::num_bvals; ++i) {
    REQUIRE(a[i] == i%8+1);
  }
}

TEST_CASE("simd::pack<long> - ctor: pack(register_type&)", "[pack:avx/avx2/offload]") {
  simd::pack<long> a(_mm256_set_epi64x(4, 3, 2, 1));
  
  for(int i = 0; i < simd::pack<long>::num_bvals; ++i) {
    REQUIRE(a[i] == i%4+1);
  }
}

TEST_CASE("simd::pack<float> - ctor: pack(register_type&)", "[pack:avx/avx2/offload]") {
  simd::pack<float> a(_mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0));
  
  for(int i = 0; i < simd::pack<float>::num_bvals; ++i) {
    REQUIRE(a[i] == i%8+1);
  }
}

TEST_CASE("simd::pack<double> - ctor: pack(register_type&)", "[pack:avx/avx2/offload]") {
  simd::pack<double> a(_mm256_set_pd(4.0, 3.0, 2.0, 1.0));
  
  for(int i = 0; i < simd::pack<double>::num_bvals; ++i) {
    REQUIRE(a[i] == i%4+1);
  }
}

/* TEST - ctor: pack(std::initializer_list<base_type>) */

TEST_CASE("simd::pack<int> - ctor: pack(std::initializer_list<base_type>)", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  simd::pack<int> a1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  
  for(int i = 0; i < simd::pack<int>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#else // #if SIMD512OFFLOAD
  simd::pack<int> a1{1, 2, 3, 4, 5, 6, 7, 8};
  
  for(int i = 0; i < simd::pack<int>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#endif // #if SIMD512OFFLOAD
}

TEST_CASE("simd::pack<long> - ctor: pack(std::initializer_list<base_type>)", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  simd::pack<long> a1{1, 2, 3, 4, 5, 6, 7, 8};
  
  for(int i = 0; i < simd::pack<long>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#else // #if SIMD512OFFLOAD
  simd::pack<long> a1{1, 2, 3, 4};
  
  for(int i = 0; i < simd::pack<long>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#endif // #if SIMD512OFFLOAD
}

TEST_CASE("simd::pack<float> - ctor: pack(std::initializer_list<base_type>)", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  simd::pack<float> a1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  
  for(int i = 0; i < simd::pack<float>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#else // #if SIMD512OFFLOAD
  simd::pack<float> a1{1, 2, 3, 4, 5, 6, 7, 8};
  
  for(int i = 0; i < simd::pack<float>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#endif // #if SIMD512OFFLOAD
}

TEST_CASE("simd::pack<double> - ctor: pack(std::initializer_list<base_type>)", "[pack:avx/avx2/offload]") {
#if SIMD512OFFLOAD
  simd::pack<double> a1{1, 2, 3, 4, 5, 6, 7, 8};
  
  for(int i = 0; i < simd::pack<double>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#else // #if SIMD512OFFLOAD
  simd::pack<double> a1{1, 2, 3, 4};
  
  for(int i = 0; i < simd::pack<double>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
#endif // #if SIMD512OFFLOAD
}

#endif // #if SIMD256

#if SIMD512

/* TEST - sanity check */

TEST_CASE("simd::pack<int> - sanity check", "[pack:knc/avx512]") {
  /* pack for scalar type and default nways */
  typedef simd::pack<int> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<int[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<int[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 16);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 16);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 16);
  REQUIRE(type1::num_vals_per_reg == 16);
  
  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 8);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 16);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 16);
  REQUIRE(type2::num_vals_per_reg == 8);
  
  REQUIRE(sizeof(type3) == 64);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 1);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 16);
  REQUIRE(type3::num_vals_per_reg == 4);
}

TEST_CASE("simd::pack<long> - sanity check", "[pack:knc/avx512]") {
  /* pack for scalar type and default nways */
  typedef simd::pack<long> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<long[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<long[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 8);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 8);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 8);
  REQUIRE(type1::num_vals_per_reg == 8);
  
  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 4);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 8);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 8);
  REQUIRE(type2::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type3) == 128);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 2);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 8);
  REQUIRE(type3::num_vals_per_reg == 2);
}

TEST_CASE("simd::pack<float> - sanity check", "[pack:knc/avx512]") {
  /* pack for scalar type and default nways */
  typedef simd::pack<float> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<float[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<float[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 16);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 16);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 16);
  REQUIRE(type1::num_vals_per_reg == 16);
  
  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 8);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 16);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 16);
  REQUIRE(type2::num_vals_per_reg == 8);
  
  REQUIRE(sizeof(type3) == 64);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 1);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 16);
  REQUIRE(type3::num_vals_per_reg == 4);
}

TEST_CASE("simd::pack<double> - sanity check", "[pack:knc/avx512]") {
  /* pack for scalar type and default nways */
  typedef simd::pack<double> type1;
  
  /* pack for 1D array and default nways */
  typedef simd::pack<double[2]> type2;
  
  /* pack for 1D array and specified nways */
  typedef simd::pack<double[4], 4> type3;
  
  REQUIRE(sizeof(type1) == 64);
  REQUIRE(alignof(type1) == 64);
  REQUIRE(type1::num_vals == 8);
  REQUIRE(type1::num_regs == 1);
  REQUIRE(type1::num_bvals == 8);
  REQUIRE(type1::num_bvals_per_val == 1);
  REQUIRE(type1::num_bvals_per_reg == 8);
  REQUIRE(type1::num_vals_per_reg == 8);
  
  REQUIRE(sizeof(type2) == 64);
  REQUIRE(alignof(type2) == 64);
  REQUIRE(type2::num_vals == 4);
  REQUIRE(type2::num_regs == 1);
  REQUIRE(type2::num_bvals == 8);
  REQUIRE(type2::num_bvals_per_val == 2);
  REQUIRE(type2::num_bvals_per_reg == 8);
  REQUIRE(type2::num_vals_per_reg == 4);
  
  REQUIRE(sizeof(type3) == 128);
  REQUIRE(alignof(type3) == 64);
  REQUIRE(type3::num_vals == 4);
  REQUIRE(type3::num_regs == 2);
  REQUIRE(type3::num_bvals == 16);
  REQUIRE(type3::num_bvals_per_val == 4);
  REQUIRE(type3::num_bvals_per_reg == 8);
  REQUIRE(type3::num_vals_per_reg == 2);
}

/* TEST - ctor: pack(register_type&) */

TEST_CASE("simd::pack<int> - ctor: pack(register_type&)", "[pack:knc/avx512]") {
  simd::pack<int> a(_mm512_set_epi32(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
  
  for(int i = 0; i < simd::pack<int>::num_bvals; ++i) {
    REQUIRE(a[i] == i%16+1);
  }
}

TEST_CASE("simd::pack<long> - ctor: pack(register_type&)", "[pack:knc/avx512]") {
  alignas(simd::pack<long>) long data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  simd::pack<long> a(_mm512_load_epi64(data));
  
  for(int i = 0; i < simd::pack<long>::num_bvals; ++i) {
    REQUIRE(a[i] == i%8+1);
  }
}

TEST_CASE("simd::pack<float> - ctor: pack(register_type&)", "[pack:knc/avx512]") {
  simd::pack<float> a(_mm512_set_ps(16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0));
  
  for(int i = 0; i < simd::pack<float>::num_bvals; ++i) {
    REQUIRE(a[i] == i%16+1);
  }
}

TEST_CASE("simd::pack<double> - ctor: pack(register_type&)", "[pack:knc/avx512]") {
  simd::pack<double> a(_mm512_set_pd(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0));
  
  for(int i = 0; i < simd::pack<double>::num_bvals; ++i) {
    REQUIRE(a[i] == i%8+1);
  }
}

/* TEST - ctor: pack(std::initializer_list<base_type>) */

TEST_CASE("simd::pack<int> - ctor: pack(std::initializer_list<base_type>)", "[pack:knc/avx512]") {
  simd::pack<int> a1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  
  for(int i = 0; i < simd::pack<int>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
}

TEST_CASE("simd::pack<long> - ctor: pack(std::initializer_list<base_type>)", "[pack:knc/avx512]") {
  simd::pack<long> a1{1, 2, 3, 4, 5, 6, 7, 8};
  
  for(int i = 0; i < simd::pack<long>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
}

TEST_CASE("simd::pack<float> - ctor: pack(std::initializer_list<base_type>)", "[pack:knc/avx512]") {
  simd::pack<float> a1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  
  for(int i = 0; i < simd::pack<float>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
}

TEST_CASE("simd::pack<double> - ctor: pack(std::initializer_list<base_type>)", "[pack:knc/avx512]") {
  simd::pack<double> a1{1, 2, 3, 4, 5, 6, 7, 8};
  
  for(int i = 0; i < simd::pack<double>::num_bvals; ++i) {
    REQUIRE(a1[i] == i+1);
  }
}

#endif // #if SIMD512

int main(int argc, char **argv) {
  int result = Catch::Session().run(argc, argv);
}
