#ifndef SIMD_SIMD_GENERIC_CTEST_HPP
#define SIMD_SIMD_GENERIC_CTEST_HPP

#include <simd.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <ctime>
#include <cstdlib>
#include <bitset>

namespace general {

/* TEST - ctor: pack(const_mptr<base_type, hint>) */

template<typename T>
void test_ctor_cvt_mptr() {
  typedef simd::pack<T> ttype;
  
  typename ttype::base_type *a1data = new typename ttype::base_type[ttype::num_bvals];
  
  std::srand(std::time(NULL));
  for(int i = 0; i < ttype::num_bvals; ++i) {
    a1data[i] = (typename ttype::base_type)std::rand();
  }
  
  ttype a1 = simd::const_mptr<typename ttype::base_type>(a1data);
  
  for(int i = 0; i < ttype::num_bvals; ++i) {
    CHECK(a1[i] == a1data[i]);
  }
  
  delete[] a1data;
}

/* TEST - ctor: pack(const value_type &rhs) */

TEST_CASE("simd::pack<T> - ctor: pack(const_mptr<base_type, hint>)", "[pack:generic]") {
  test_ctor_cvt_mptr<int>();
  test_ctor_cvt_mptr<long>();
  test_ctor_cvt_mptr<float>();
  test_ctor_cvt_mptr<double>();
}

template<typename T>
void test_ctor_value_type() {
  simd::pack<T> a1(T(2));
  for(int i = 0; i < simd::pack<T>::num_vals; ++i) {
    CHECK(a1[i] == T(2));
  }
  
  T value_a2[2] = {T(1), T(2)};
  simd::pack<T[2]> a2(value_a2);
  for(int i = 0; i < simd::pack<T[2]>::num_vals; ++i) {
    CHECK(a2[i][0] == value_a2[0]);
    CHECK(a2[i][1] == value_a2[1]);
  }
  
  T value_a3[2][2] = {{T(1), T(2)}, {T(3), T(4)}};
  simd::pack<T[2][2]> a3(value_a3);
  for(int i = 0; i < simd::pack<T[2][2]>::num_vals; ++i) {
    CHECK(a3[i][0][0] == value_a3[0][0]);
    CHECK(a3[i][0][1] == value_a3[0][1]);
    CHECK(a3[i][1][0] == value_a3[1][0]);
    CHECK(a3[i][1][1] == value_a3[1][1]);
  }
}

TEST_CASE("simd::pack<T> - ctor: pack(const value_type &)", "[pack::generic]") {
  test_ctor_value_type<int>();
  test_ctor_value_type<long>();
  test_ctor_value_type<float>();
  test_ctor_value_type<double>();
}

/* TEST - ctor: pack(const base_type &rhs) */

template<typename T>
void test_ctor_base_type() {
  simd::pack<T> a1(T(2));
  for(int i = 0; i < simd::pack<T>::num_vals; ++i) {
    CHECK(a1[i] == T(2));
  }
  
  simd::pack<T[2]> a2(T(3));
  for(int i = 0; i < simd::pack<T[2]>::num_vals; ++i) {
    CHECK(a2[i][0] == T(3));
    CHECK(a2[i][1] == T(3));
  }
  
  simd::pack<T[2][2]> a3(T(4));
  for(int i = 0; i < simd::pack<T[2][2]>::num_vals; ++i) {
    CHECK(a3[i][0][0] == T(4));
    CHECK(a3[i][0][1] == T(4));
    CHECK(a3[i][1][0] == T(4));
    CHECK(a3[i][1][1] == T(4));
  }
}

TEST_CASE("simd::pack<T> - ctor: pack(const base_type &", "[pack::generic]") {
  test_ctor_base_type<int>();
  test_ctor_base_type<long>();
  test_ctor_base_type<float>();
  test_ctor_base_type<double>();
}

/* TEST - operator=(const value_type &rhs) */

template<typename T>
void test_assignment_value_type() {
  simd::pack<T> a1;
  a1 = T(2);
  for(int i = 0; i < simd::pack<T>::num_vals; ++i) {
    CHECK(a1[i] == T(2));
  }
  
  simd::pack<T[2]> a2;
  T value_a2[2] = {T(1), T(2)};
  a2 = value_a2;
  for(int i = 0; i < simd::pack<T[2]>::num_vals; ++i) {
    CHECK(a2[i][0] == value_a2[0]);
    CHECK(a2[i][1] == value_a2[1]);
  }
  
  simd::pack<T[2][2]> a3;
  T value_a3[2][2] = {{T(1), T(2)}, {T(3), T(4)}};
  a3 = value_a3;
  for(int i = 0; i < simd::pack<T[2][2]>::num_vals; ++i) {
    CHECK(a3[i][0][0] == value_a3[0][0]);
    CHECK(a3[i][0][1] == value_a3[0][1]);
    CHECK(a3[i][1][0] == value_a3[1][0]);
    CHECK(a3[i][1][1] == value_a3[1][1]);
  }
}

TEST_CASE("simd::pack<T, NW> - operator=(const value_type &)", "[pack::generic]") {
  test_assignment_value_type<int>();
  test_assignment_value_type<long>();
  test_assignment_value_type<float>();
  test_assignment_value_type<double>();
}

/* TEST - operator=(const base_type &rhs) */

template<typename T>
void test_assignment_base_type() {
  simd::pack<T> a1;
  a1 = T(2);
  for(int i = 0; i < simd::pack<T>::num_vals; ++i) {
    CHECK(a1[i] == T(2));
  }
  
  simd::pack<T[2]> a2;
  a2 = T(3);
  for(int i = 0; i < simd::pack<T[2]>::num_vals; ++i) {
    CHECK(a2[i][0] == T(3));
    CHECK(a2[i][1] == T(3));
  }
  
  simd::pack<T[2][2]> a3;
  a3 = T(4);
  for(int i = 0; i < simd::pack<T[2][2]>::num_vals; ++i) {
    CHECK(a3[i][0][0] == T(4));
    CHECK(a3[i][0][1] == T(4));
    CHECK(a3[i][1][0] == T(4));
    CHECK(a3[i][1][1] == T(4));
  }
}

TEST_CASE("simd::pack<T, NW> - operator=(const base_type &)", "[pack::generic]") {
  test_assignment_base_type<int>();
  test_assignment_base_type<long>();
  test_assignment_base_type<float>();
  test_assignment_base_type<double>();
}

/* TEST - overloaded new and delete operators */

template<typename T>
void test_aligned_new_delete() {
  {
    // PLAIN NEW
    T *ptr = new T;
    if(ptr != nullptr) {
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      delete ptr;
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // ARRAY NEW
    T *ptr = new T[10];
    if(ptr != nullptr) {
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      delete[] ptr;
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // PLACEMENT PLAIN NEW - Success
    void *space = _mm_malloc(sizeof(T), alignof(T));
    if(space != nullptr) {
      T *ptr = new (space) T;
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      ptr->~T();
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // PLACEMENT ARRAY NEW - Success
    void *space = _mm_malloc(sizeof(T)*10, alignof(T));
    if(space != nullptr) {
      T *ptr = new (space) T[10];
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      for(int i = 0; i < 10; ++i) {
        ptr[i].~T();
      }
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // PLACEMENT PLAIN NEW - Failure
    void *space = _mm_malloc(sizeof(T), 1);
    if(space != nullptr) {
      REQUIRE_THROWS_AS(new (space) T, std::bad_alloc);
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // PLACEMENT ARRAY NEW - Failure
    void *space = _mm_malloc(sizeof(T)*10, 1);
    if(space != nullptr) {
      REQUIRE_THROWS_AS(new (space) T[10], std::bad_alloc);
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // NOTHROW PLAIN NEW
    T *ptr = new(std::nothrow) T;
    if(ptr != nullptr) {
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      delete ptr;
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // NOTHROW ARRAY NEW
    T *ptr = new(std::nothrow) T[10];
    if(ptr != nullptr) {
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      delete[] ptr;
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // NOTHROW+PLACEMENT PLAIN NEW - Success
    void *space = _mm_malloc(sizeof(T), alignof(T));
    if(space != nullptr) {
      T *ptr = new(std::nothrow, space) T;
      REQUIRE(ptr != 0);
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      ptr->~T();
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // NOTHROW+PLACEMENT ARRAY NEW - Success
    void *space = _mm_malloc(sizeof(T)*10, alignof(T));
    if(space != nullptr) {
      T *ptr = new(std::nothrow, space) T[10];
      REQUIRE(ptr != 0);
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignof(T) == 0);
      for(int i = 0; i < 10; ++i) {
        ptr[i].~T();
      }
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // NOTHROW+PLACEMENT PLAIN NEW - Failure
    void *space = _mm_malloc(sizeof(T), 1);
    if(space != nullptr) {
      T *ptr = new(std::nothrow, space) T;
      REQUIRE(ptr == 0);
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
  
  {
    // NOTHROW+PLACEMENT ARRAY NEW - Failure
    void *space = _mm_malloc(sizeof(T)*10, 1);
    if(space != nullptr) {
      T *ptr = new(std::nothrow, space) T[10];
      REQUIRE(ptr == 0);
      _mm_free(space);
    } else {
      REQUIRE(false);
    }
  }
}

TEST_CASE("simd::pack<T, NW> - new and delete operators", "[pack:generic]") {
  test_aligned_new_delete< simd::pack<int, simd::defaults<int>::nway> >();
  test_aligned_new_delete< simd::pack<int, simd::defaults<int>::nway*2> >();
  
  test_aligned_new_delete< simd::pack<long, simd::defaults<long>::nway> >();
  test_aligned_new_delete< simd::pack<long, simd::defaults<long>::nway*2> >();
  
  test_aligned_new_delete< simd::pack<float, simd::defaults<float>::nway> >();
  test_aligned_new_delete< simd::pack<float, simd::defaults<float>::nway*2> >();
  
  test_aligned_new_delete< simd::pack<double, simd::defaults<double>::nway> >();
  test_aligned_new_delete< simd::pack<double, simd::defaults<double>::nway*2> >();
}

/* TEST - mask. set(...), reset(...), flip(...), all(..), any(...), none(...), test(...) */

template<typename T, int NW>
void test_mask_bits(const simd::mask<T, NW> &m1, const std::vector<bool> &expbval) {
  // test for mask.all/any/none()
  CHECK(m1.all() == std::all_of(expbval.begin(), expbval.end(), [](bool a)->bool { return a; }));
  CHECK(m1.any() == std::any_of(expbval.begin(), expbval.end(), [](bool a)->bool { return a; }));
  CHECK(m1.none() == std::none_of(expbval.begin(), expbval.end(), [](bool a)->bool { return a; }));
  
  // test for mask.all/any/none(rindex)
  for(int i = 0; i < simd::type_traits<T, NW>::num_regs; ++i) {
    int bstart = i*simd::type_traits<T, NW>::num_bvals_per_reg;
    int bend = bstart + simd::type_traits<T, NW>::num_bvals_per_reg;
    CHECK(m1.all(simd::rindex(i)) == std::all_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
    CHECK(m1.any(simd::rindex(i)) == std::any_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
    CHECK(m1.none(simd::rindex(i)) == std::none_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
  }
  
  // test for mask.all/any/none(vindex)
  for(int i = 0; i < simd::type_traits<T, NW>::num_vals; ++i) {
    int bstart = i*simd::type_traits<T, NW>::num_bvals_per_val;
    int bend = bstart + simd::type_traits<T, NW>::num_bvals_per_val;
    CHECK(m1.all(simd::vindex(i)) == std::all_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
    CHECK(m1.any(simd::vindex(i)) == std::any_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
    CHECK(m1.none(simd::vindex(i)) == std::none_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
  }
  
  // test for mask.all/any/none(rindex, vindex)
  for(int i = 0; i < simd::type_traits<T, NW>::num_regs; ++i) {
    for(int j = 0; j < simd::type_traits<T, NW>::num_vals_per_reg; ++j) {
      int bstart = i*simd::type_traits<T, NW>::num_bvals_per_reg + j*simd::type_traits<T, NW>::num_bvals_per_val;
      int bend = bstart + simd::type_traits<T, NW>::num_bvals_per_val;
      CHECK(m1.all(simd::rindex(i), simd::vindex(j)) == std::all_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
      CHECK(m1.any(simd::rindex(i), simd::vindex(j)) == std::any_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
      CHECK(m1.none(simd::rindex(i), simd::vindex(j)) == std::none_of(expbval.begin()+bstart, expbval.begin()+bend, [](bool a)->bool { return a; }));
    }
  }
  
  // test for mask.test(bindex)
  for(int i = 0; i < simd::type_traits<T, NW>::num_bvals; ++i) {
    CHECK(m1.test(simd::bindex(i)) == expbval[i]);
  }
  
  // test for mask.test(rindex, bindex)
  for(int i = 0; i < simd::type_traits<T, NW>::num_regs; ++i) {
    for(int j = 0; j < simd::type_traits<T, NW>::num_bvals_per_reg; ++j) {
      int bindex = i*simd::type_traits<T, NW>::num_bvals_per_reg + j;
      CHECK(m1.test(simd::rindex(i), simd::bindex(j)) == expbval[bindex]);
    }
  }
}

template<typename T, int NW>
void test_mask() {
  typedef simd::mask<T, NW> mask_t;
  
  mask_t *obj = new mask_t;
  mask_t &m1 = *obj;
  
  std::srand(std::time(NULL));
  
  // test for mask.set(), mask.flip(), mask.reset()
  {
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    m1.set();
    for(int i = 0; i < simd::type_traits<T, NW>::num_bvals; ++i) {
      expbval[i] = true;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset();
    for(int i = 0; i < simd::type_traits<T, NW>::num_bvals; ++i) {
      expbval[i] = false;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.flip();
    for(int i = 0; i < simd::type_traits<T, NW>::num_bvals; ++i) {
      expbval[i] = !expbval[i];
    }
    test_mask_bits<T, NW>(m1, expbval);
  }
  
  // test for mask.set(rindex), mask.flip(rindex), mask.reset(rindex)
  {
    int ri, bstart, bend;
    m1.reset();
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg;
    bend = bstart + simd::type_traits<T, NW>::num_bvals_per_reg;
    
    m1.set(simd::rindex(ri));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = true;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset(simd::rindex(ri));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = false;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg;
    bend = bstart + simd::type_traits<T, NW>::num_bvals_per_reg;
    
    m1.flip(simd::rindex(ri));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = !expbval[i];
    }
    test_mask_bits<T, NW>(m1, expbval);
  }
  
  // test for mask.set(vindex), mask.flip(vindex), mask.reset(vindex)
  {
    int vi, bstart, bend;
    m1.reset();
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    vi = std::rand()%simd::type_traits<T, NW>::num_vals;
    bstart = vi*simd::type_traits<T, NW>::num_bvals_per_val;
    bend = bstart + simd::type_traits<T, NW>::num_bvals_per_val;
    
    m1.set(simd::vindex(vi));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = true;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset(simd::vindex(vi));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = false;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    vi = std::rand()%simd::type_traits<T, NW>::num_vals;
    bstart = vi*simd::type_traits<T, NW>::num_bvals_per_val;
    bend = bstart + simd::type_traits<T, NW>::num_bvals_per_val;
    
    m1.flip(simd::vindex(vi));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = !expbval[i];
    }
    test_mask_bits<T, NW>(m1, expbval);
  }
  
  // test for mask.set(bindex), mask.flip(bindex), mask.reset(bindex)
  {
    int bi;
    m1.reset();
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    bi = std::rand()%simd::type_traits<T, NW>::num_bvals;
    
    m1.set(simd::bindex(bi));
    expbval[bi] = true;
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset(simd::bindex(bi));
    expbval[bi] = false;
    test_mask_bits<T, NW>(m1, expbval);
    
    bi = std::rand()%simd::type_traits<T, NW>::num_vals;
    
    m1.flip(simd::bindex(bi));
    expbval[bi] = !expbval[bi];
    test_mask_bits<T, NW>(m1, expbval);
  }
  
  // test for set(rindex, vindex), flip(rindex, vindex), reset(rindex, vindex)
  {
    int ri, vi, bstart, bend;
    m1.reset();
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    vi = std::rand()%simd::type_traits<T, NW>::num_vals_per_reg;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg+vi*simd::type_traits<T, NW>::num_bvals_per_val;
    bend = bstart + simd::type_traits<T, NW>::num_bvals_per_val;
    
    m1.set(simd::rindex(ri), simd::vindex(vi));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = true;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset(simd::rindex(ri), simd::vindex(vi));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = false;
    }
    test_mask_bits<T, NW>(m1, expbval);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    vi = std::rand()%simd::type_traits<T, NW>::num_vals_per_reg;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg+vi*simd::type_traits<T, NW>::num_bvals_per_val;
    bend = bstart + simd::type_traits<T, NW>::num_bvals_per_val;
    
    m1.flip(simd::rindex(ri), simd::vindex(vi));
    for(int i = bstart; i < bend; ++i) {
      expbval[i] = !expbval[i];
    }
    test_mask_bits<T, NW>(m1, expbval);
  }
  
  // test for set(rindex, bindex), flip(rindex, bindex), reset(rindex, bindex)
  {
    int ri, bi, bstart;
    m1.reset();
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    bi = std::rand()%simd::type_traits<T, NW>::num_bvals_per_reg;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg+bi;
    
    m1.set(simd::rindex(ri), simd::bindex(bi));
    expbval[bstart] = true;
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset(simd::rindex(ri), simd::bindex(bi));
    expbval[bstart] = false;
    test_mask_bits<T, NW>(m1, expbval);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    bi = std::rand()%simd::type_traits<T, NW>::num_bvals_per_reg;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg+bi;
    
    m1.flip(simd::rindex(ri), simd::bindex(bi));
    expbval[bstart] = !expbval[bstart];
    test_mask_bits<T, NW>(m1, expbval);
  }
  
  // test for set(vindex, bindex), flip(vindex, bindex), reset(vindex, bindex)
  {
    int vi, bi, bstart;
    m1.reset();
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    vi = std::rand()%simd::type_traits<T, NW>::num_vals;
    bi = std::rand()%simd::type_traits<T, NW>::num_bvals_per_val;
    bstart = vi*simd::type_traits<T, NW>::num_bvals_per_val+bi;
    
    m1.set(simd::vindex(vi), simd::bindex(bi));
    expbval[bstart] = true;
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset(simd::vindex(vi), simd::bindex(bi));
    expbval[bstart] = false;
    test_mask_bits<T, NW>(m1, expbval);
    
    vi = std::rand()%simd::type_traits<T, NW>::num_vals;
    bi = std::rand()%simd::type_traits<T, NW>::num_bvals_per_val;
    bstart = vi*simd::type_traits<T, NW>::num_bvals_per_val+bi;
    
    m1.flip(simd::vindex(vi), simd::bindex(bi));
    expbval[bstart] = !expbval[bstart];
    test_mask_bits<T, NW>(m1, expbval);
  }
  
  // test for set(rindex, vindex, bindex), flip(rindex, vindex, bindex), reset(rindex, vindex, bindex)
  {
    int ri, vi, bi, bstart;
    m1.reset();
    std::vector<bool> expbval(simd::type_traits<T, NW>::num_bvals, false);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    vi = std::rand()%simd::type_traits<T, NW>::num_vals_per_reg;
    bi = std::rand()%simd::type_traits<T, NW>::num_bvals_per_val;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg+vi*simd::type_traits<T, NW>::num_bvals_per_val+bi;
    
    m1.set(simd::rindex(ri), simd::vindex(vi), simd::bindex(bi));
    expbval[bstart] = true;
    test_mask_bits<T, NW>(m1, expbval);
    
    m1.reset(simd::rindex(ri), simd::vindex(vi), simd::bindex(bi));
    expbval[bstart] = false;
    test_mask_bits<T, NW>(m1, expbval);
    
    ri = std::rand()%simd::type_traits<T, NW>::num_regs;
    vi = std::rand()%simd::type_traits<T, NW>::num_vals_per_reg;
    bi = std::rand()%simd::type_traits<T, NW>::num_bvals_per_val;
    bstart = ri*simd::type_traits<T, NW>::num_bvals_per_reg+vi*simd::type_traits<T, NW>::num_bvals_per_val+bi;
    
    m1.flip(simd::rindex(ri), simd::vindex(vi), simd::bindex(bi));
    expbval[bstart] = !expbval[bstart];
    test_mask_bits<T, NW>(m1, expbval);
  }
}

TEST_CASE("simd::mask<T, NW> - test_all_true(), test_all_false()", "[pack:generic]") {
  test_mask<int, simd::defaults<int>::nway>();
  test_mask<int, simd::defaults<int>::nway*2>();
  
  test_mask<int[2], simd::defaults<int>::nway>();
  test_mask<int[2], simd::defaults<int>::nway*2>();
  
  test_mask<long, simd::defaults<long>::nway>();
  test_mask<long, simd::defaults<long>::nway*2>();
  
  test_mask<long[2], simd::defaults<long>::nway>();
  test_mask<long[2], simd::defaults<long>::nway*2>();
  
  test_mask<float, simd::defaults<float>::nway>();
  test_mask<float, simd::defaults<float>::nway*2>();
  
  test_mask<float[2], simd::defaults<float>::nway>();
  test_mask<float[2], simd::defaults<float>::nway*2>();
  
  test_mask<double, simd::defaults<double>::nway>();
  test_mask<double, simd::defaults<double>::nway*2>();
  
  test_mask<double[2], simd::defaults<double>::nway>();
  test_mask<double[2], simd::defaults<double>::nway*2>();
}

/* TEST - arithmetic operator: + */

template<typename T>
void test_addition() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  T *a2data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a2data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a2 = simd::const_mptr<T>(a2data);
  simd::pack<T> a3 = a1+a2;
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == a1data[i] + a2data[i]);
  }
  
  delete[] a1data;
  delete[] a2data;
}

TEST_CASE("simd::pack<int> - arithmetic operator: +", "[pack:generic]") {
  test_addition<int>();
  test_addition<long>();
  test_addition<float>();
  test_addition<double>();
}

/* TEST - arithmetic operator: - */

template<typename T>
void test_subtraction() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  T *a2data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a2data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a2 = simd::const_mptr<T>(a2data);
  simd::pack<T> a3 = a1-a2;
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == a1data[i] - a2data[i]);
  }
  
  delete[] a1data;
  delete[] a2data;
}

TEST_CASE("simd::pack<T> - arithmetic operator -", "[pack:generic]") {
  test_subtraction<int>();
  test_subtraction<long>();
  test_subtraction<float>();
  test_subtraction<double>();
}

/* TEST - arithmetic operator: * */

template<typename T>
void test_multiplication() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  T *a2data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand() / (typename simd::pack<T>::base_type)std::rand();
  }
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a2data[i] = (typename simd::pack<T>::base_type)std::rand() / (typename simd::pack<T>::base_type)std::rand();
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a2 = simd::const_mptr<T>(a2data);
  simd::pack<T> a3 = a1*a2;
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == a1data[i] * a2data[i]);
  }
  
  delete[] a1data;
  delete[] a2data;
}


TEST_CASE("simd::pack<T> - arithmetic operator *", "[pack:generic") {
  test_multiplication<int>();
  test_multiplication<long>();
  test_multiplication<float>();
  test_multiplication<double>();
}

/* TEST - arithmetic operator: / */

template<typename T>
void test_division() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  T *a2data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    typename simd::pack<T>::base_type temp = (typename simd::pack<T>::base_type)std::rand() / (typename simd::pack<T>::base_type)std::rand();
    while(std::fabs(temp) < 1e-10) {
      temp = (typename simd::pack<T>::base_type)std::rand() / (typename simd::pack<T>::base_type)std::rand();
    }
    a2data[i] = temp;
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a2 = simd::const_mptr<T>(a2data);
  simd::pack<T> a3 = a1/a2;
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == a1data[i] / a2data[i]);
  }
  
  delete[] a1data;
  delete[] a2data;
}

TEST_CASE("simd::pack<T> - arithmetic operator /", "[pack:generic]") {
  test_division<int>();
  test_division<long>();
  test_division<float>();
  test_division<double>();
}

/* TEST - arithmetic operator: % */

template<typename T>
void test_modulus() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  T *a2data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    typename simd::pack<T>::base_type temp = (typename simd::pack<T>::base_type)std::rand();
    while(temp == 0 ) {
      temp = (typename simd::pack<T>::base_type)std::rand();
    }
    a2data[i] = temp;
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a2 = simd::const_mptr<T>(a2data);
  simd::pack<T> a3 = a1%a2;
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == a1data[i] % a2data[i]);
  }
  
  delete[] a1data;
  delete[] a2data;
}

TEST_CASE("simd::pack<T> - arithmetic operator %", "[pack:generic]") {
  test_modulus<int>();
  test_modulus<long>();
}

/* TEST - comparison operator: ==, != */

template<typename T, int NW>
void test_compare_eq_neq() {
  T *a1data = new T[simd::pack<T, NW>::num_bvals];
  T *a2data = new T[simd::pack<T, NW>::num_bvals];
  
  std::srand(std::time(NULL));
  for(int i = 0; i < simd::pack<T, NW>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T, NW>::base_type)std::rand();
    if(std::rand()%2) {
      a2data[i] = a1data[i];
    } else {
      a2data[i] = (typename simd::pack<T, NW>::base_type)std::rand();
    }
  }
  
  simd::pack<T, NW> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T, NW> a2 = simd::const_mptr<T>(a2data);
  typename simd::pack<T, NW>::mask_type eq, neq;
  
  eq = a1 == a2;
  neq = a1 != a2;
  
  for(int i = 0; i < simd::pack<T, NW>::num_bvals; ++i) {
    if(a1data[i] == a2data[i]) {
      CHECK(eq[i]);
    } else {
      CHECK(!eq[i]);
    }
    
    if(a1data[i] != a2data[i]) {
      CHECK(neq[i]);
    } else {
      CHECK(!neq[i]);
    }
  }
  
  delete[] a1data;
  delete[] a2data;
}

TEST_CASE("simd::pack<T> - comparison operator ==, !=", "[pack:generic]") {
  test_compare_eq_neq<int, simd::defaults<int>::nway>();
  test_compare_eq_neq<int, simd::defaults<int>::nway*2>();
  
  test_compare_eq_neq<long, simd::defaults<long>::nway>();
  test_compare_eq_neq<long, simd::defaults<long>::nway*2>();
  
  test_compare_eq_neq<float, simd::defaults<float>::nway>();
  test_compare_eq_neq<float, simd::defaults<float>::nway*2>();
  
  test_compare_eq_neq<double, simd::defaults<double>::nway>();
  test_compare_eq_neq<double, simd::defaults<double>::nway*2>();
}

/* TEST - comparison operator: <, >, <=, >= */

template<typename T, int NW>
void test_compare_lt_gt_le_ge() {
  //simd::pack<int[5], 5> temp;
  
  T *a1data = new T[simd::pack<T, NW>::num_bvals];
  T *a2data = new T[simd::pack<T, NW>::num_bvals];
  
  std::srand(std::time(NULL));
  for(int i = 0; i < simd::pack<T, NW>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T, NW>::base_type)std::rand();
    if(std::rand()%2) {
      a2data[i] = a1data[i];
    } else {
      a2data[i] = (typename simd::pack<T, NW>::base_type)std::rand();
    }
  }
  
  simd::pack<T, NW> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T, NW> a2 = simd::const_mptr<T>(a2data);
  typename simd::pack<T, NW>::mask_type lt, gt, le, ge;
  
  lt = a1 < a2;
  gt = a1 > a2;
  le = a1 <= a2;
  ge = a1 >= a2;
  //lt[1] = !lt[1];
  for(int i = 0; i < simd::pack<T, NW>::num_bvals; ++i) {
    if(a1data[i] < a2data[i]) {
      CHECK(lt[i]);
    } else {
      CHECK(!lt[i]);
    }
    
    if(a1data[i] > a2data[i]) {
      CHECK(gt[i]);
    } else {
      CHECK(!gt[i]);
    }
    
    if(a1data[i] <= a2data[i]) {
      CHECK(le[i]);
    } else {
      CHECK(!le[i]);
    }
    
    if(a1data[i] >= a2data[i]) {
      CHECK(ge[i]);
    } else {
      CHECK(!ge[i]);
    }
  }
  
  delete[] a1data;
  delete[] a2data;
}

TEST_CASE("simd::pack<T> - comparison operator <, >, <=, >=", "[pack:generic]") {
  test_compare_lt_gt_le_ge<int, simd::defaults<int>::nway>();
  test_compare_lt_gt_le_ge<int, simd::defaults<int>::nway*2>();
  
  test_compare_lt_gt_le_ge<long, simd::defaults<long>::nway>();
  test_compare_lt_gt_le_ge<long, simd::defaults<long>::nway*2>();
  
  test_compare_lt_gt_le_ge<float, simd::defaults<float>::nway>();
  test_compare_lt_gt_le_ge<float, simd::defaults<float>::nway*2>();
  
  test_compare_lt_gt_le_ge<double, simd::defaults<double>::nway>();
  test_compare_lt_gt_le_ge<double, simd::defaults<double>::nway*2>();
}

/* TEST - function: sin */

template<typename T>
void test_sin() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a3 = sin(a1);
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == Approx(std::sin(a1data[i])));
  }
  
  delete[] a1data;
}

TEST_CASE("simd::pack<T> - function: sin", "[pack:generic]") {
  test_sin<float>();
  test_sin<double>();
}

/* TEST - function: cos */

template<typename T>
void test_cos() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a3 = cos(a1);
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == Approx(std::cos(a1data[i])));
  }
  
  delete[] a1data;
}

TEST_CASE("simd::pack<T> - function: cos", "[pack:generic]") {
  test_cos<float>();
  test_cos<double>();
}

/* TEST - function: tan */

template<typename T>
void test_tan() {
  T *a1data = new T[simd::pack<T>::num_bvals];
  
  std::srand(time(NULL));
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    a1data[i] = (typename simd::pack<T>::base_type)std::rand();
  }
  
  simd::pack<T> a1 = simd::const_mptr<T>(a1data);
  simd::pack<T> a3 = tan(a1);
  
  for(int i = 0; i < simd::pack<T>::num_bvals; ++i) {
    CHECK(a3[i] == Approx(std::tan(a1data[i])));
  }
  
  delete[] a1data;
}

TEST_CASE("simd::pack<T> - function: tan", "[pack:generic]") {
  test_tan<float>();
  test_tan<double>();
}

/* TEST - function: load */

template<typename T, typename ST, int NW>
void test_load_store() {
  {
    {
      typedef simd::pack<T, NW> type_t;
      typedef simd::mask<T, NW> mask_t;
      
      ST *idata = new ST[type_t::num_bvals];
      ST *odata = new ST[type_t::num_bvals];
      
      for(int i = 0; i < type_t::num_bvals; ++i) {
        idata[i] = (ST)(i+1);
      }
      
      type_t op1, op2, op3;
      mask_t m1;
      
      std::srand(time(NULL));
      for(int i = 0; i < type_t::num_bvals; ++i) {
        op2[i] = (T)(std::rand()%11+1);
      }
      op3 = (T)5;
      m1 = op2 > op3;
      
      simd::load(op1, simd::const_mptr<ST>(idata));
      for(int i = 0; i < type_t::num_bvals; ++i) {
        CHECK(op1[i] == idata[i]);
      }
      
      simd::load(op1, simd::const_mptr<ST>(idata), m1);
      for(int i = 0; i < type_t::num_bvals; ++i) {
        if(m1[i]) {
          CHECK(op1[i] == idata[i]);
        } else {
          CHECK(op1[i] == ST(0));
        }
      }
      
      simd::store(op2, simd::mptr<ST>(odata));
      for(int i = 0; i < type_t::num_bvals; ++i) {
        CHECK(odata[i] == op2[i]);
      }
      
      for(int i = 0; i < type_t::num_bvals; ++i) {
        odata[i] = (ST)-1;
      }
      simd::store(op2, simd::mptr<ST>(odata), m1);
      for(int i = 0; i < type_t::num_bvals; ++i) {
        if(m1[i]) {
          CHECK(odata[i] == op2[i]);
        } else {
          CHECK(odata[i] == ST(-1));
        }
      }
      
      delete[] idata;
      delete[] odata;
    }
    
    {
      typedef simd::pack<T[2], NW> type_t;
      typedef simd::mask<T[2], NW> mask_t;
      
      ST *idata = new ST[type_t::num_bvals];
      ST *odata = new ST[type_t::num_bvals];
      
      for(int i = 0, k = 0; i < type_t::num_vals; ++i) {
        for(int j = 0; j < type_t::num_bvals_per_val; ++j, ++k) {
          idata[k] = (ST)(k+1);
        }
      }
      
      type_t op1, op2, op3;
      mask_t m1;
      
      for(int i = 0; i < type_t::num_vals; ++i) {
        for(int j = 0; j < type_t::num_bvals_per_val; ++j) {
          op2[i][j] = (T)(std::rand()%11+1);
        }
      }
      op3 = (T)5;
      m1 = op2 > op3;
      
      simd::load(op1, simd::const_mptr<ST>(idata));
      for(int i = 0, k = 0; i < type_t::num_vals; ++i) {
        for(int j = 0; j < type_t::num_bvals_per_val; ++j, ++k) {
          CHECK(op1[i][j] == idata[k]);
        }
      }
      
      simd::load(op1, simd::const_mptr<ST>(idata), m1);
      for(int i = 0, k = 0; i < type_t::num_vals; ++i) {
        for(int j = 0; j < type_t::num_bvals_per_val; ++j, ++k) {
          if(m1[k]) {
            CHECK(op1[i][j] == idata[k]);
          } else {
            CHECK(op1[i][j] == ST(0));
          }
        }
      }
      
      simd::store(op2, simd::mptr<ST>(odata));
      for(int i = 0, k = 0; i < type_t::num_vals; ++i) {
        for(int j = 0; j < type_t::num_bvals_per_val; ++j, ++k) {
          CHECK(odata[k] == op2[i][j]);
        }
      }
      
      for(int i = 0; i < type_t::num_bvals; ++i) {
        odata[i] = (ST)-1;
      }
      simd::store(op2, simd::mptr<ST>(odata), m1);
      for(int i = 0, k = 0; i < type_t::num_vals; ++i) {
        for(int j = 0; j < type_t::num_bvals_per_val; ++j, ++k) {
          if(m1[k]) {
            CHECK(odata[k] == op2[i][j]);
          } else {
            CHECK(odata[k] == ST(-1));
          }
        }
      }
      
      delete[] idata;
      delete[] odata;
    }
  }
}

TEST_CASE("simd::pack<T, NW> - function: load, store", "[pack:generic]") {
  test_load_store<int, int, simd::defaults<int>::nway>();
  test_load_store<int, int, simd::defaults<int>::nway*2>();
  test_load_store<int, long, simd::defaults<int>::nway>();
  test_load_store<int, long, simd::defaults<int>::nway*2>();
  test_load_store<int, float, simd::defaults<int>::nway>();
  test_load_store<int, float, simd::defaults<int>::nway*2>();
  test_load_store<int, double, simd::defaults<int>::nway>();
  test_load_store<int, double, simd::defaults<int>::nway*2>();
  
  test_load_store<long, int, simd::defaults<long>::nway>();
  test_load_store<long, int, simd::defaults<long>::nway*2>();
  test_load_store<long, long, simd::defaults<long>::nway>();
  test_load_store<long, long, simd::defaults<long>::nway*2>();
  test_load_store<long, float, simd::defaults<long>::nway>();
  test_load_store<long, float, simd::defaults<long>::nway*2>();
  test_load_store<long, double, simd::defaults<long>::nway>();
  test_load_store<long, double, simd::defaults<long>::nway*2>();
  
  test_load_store<float, int, simd::defaults<float>::nway>();
  test_load_store<float, int, simd::defaults<float>::nway*2>();
  test_load_store<float, long, simd::defaults<float>::nway>();
  test_load_store<float, long, simd::defaults<float>::nway*2>();
  test_load_store<float, float, simd::defaults<float>::nway>();
  test_load_store<float, float, simd::defaults<float>::nway*2>();
  test_load_store<float, double, simd::defaults<float>::nway>();
  test_load_store<float, double, simd::defaults<float>::nway*2>();
  
  test_load_store<double, int, simd::defaults<double>::nway>();
  test_load_store<double, int, simd::defaults<double>::nway*2>();
  test_load_store<double, long, simd::defaults<double>::nway>();
  test_load_store<double, long, simd::defaults<double>::nway*2>();
  test_load_store<double, float, simd::defaults<double>::nway>();
  test_load_store<double, float, simd::defaults<double>::nway*2>();
  test_load_store<double, double, simd::defaults<double>::nway>();
  test_load_store<double, double, simd::defaults<double>::nway*2>();
}

/* TEST - function: permute */

template<typename T, int NW>
void test_permute_w4() {
  typedef simd::pack<T[4], NW> ttype;
  
  typename ttype::base_type *a1data = new typename ttype::base_type[ttype::num_bvals];
  std::srand(std::time(NULL));
  for(int i = 0; i < ttype::num_bvals; ++i) {
    a1data[i] = (typename ttype::base_type)std::rand();
  }
  
  ttype a1 = simd::const_mptr<typename ttype::base_type>(a1data);
  
  {
    ttype aacc = simd::permutation<simd::permute_pattern::aacc>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(aacc[i][0] == a1[i][0]);
      CHECK(aacc[i][1] == a1[i][0]);
      CHECK(aacc[i][2] == a1[i][2]);
      CHECK(aacc[i][3] == a1[i][2]);
    }
  }
  
  {
    ttype abab = simd::permutation<simd::permute_pattern::abab>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(abab[i][0] == a1[i][0]);
      CHECK(abab[i][1] == a1[i][1]);
      CHECK(abab[i][2] == a1[i][0]);
      CHECK(abab[i][3] == a1[i][1]);
    }
  }
  
  {
    ttype bbdd = simd::permutation<simd::permute_pattern::bbdd>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(bbdd[i][0] == a1[i][1]);
      CHECK(bbdd[i][1] == a1[i][1]);
      CHECK(bbdd[i][2] == a1[i][3]);
      CHECK(bbdd[i][3] == a1[i][3]);
    }
  }
  
  {
    ttype cdcd = simd::permutation<simd::permute_pattern::cdcd>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(cdcd[i][0] == a1[i][2]);
      CHECK(cdcd[i][1] == a1[i][3]);
      CHECK(cdcd[i][2] == a1[i][2]);
      CHECK(cdcd[i][3] == a1[i][3]);
    }
  }
  
  {
    ttype dcba = simd::permutation<simd::permute_pattern::dcba>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(dcba[i][0] == a1[i][3]);
      CHECK(dcba[i][1] == a1[i][2]);
      CHECK(dcba[i][2] == a1[i][1]);
      CHECK(dcba[i][3] == a1[i][0]);
    }
  }
  
  {
    ttype dbca = simd::permutation<simd::permute_pattern::dbca>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(dbca[i][0] == a1[i][3]);
      CHECK(dbca[i][1] == a1[i][1]);
      CHECK(dbca[i][2] == a1[i][2]);
      CHECK(dbca[i][3] == a1[i][0]);
    }
  }
  
  {
    ttype acac = simd::permutation<simd::permute_pattern::acac>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(acac[i][0] == a1[i][0]);
      CHECK(acac[i][1] == a1[i][2]);
      CHECK(acac[i][2] == a1[i][0]);
      CHECK(acac[i][3] == a1[i][2]);
    }
  }
  
  {
    ttype bdbd = simd::permutation<simd::permute_pattern::bdbd>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(bdbd[i][0] == a1[i][1]);
      CHECK(bdbd[i][1] == a1[i][3]);
      CHECK(bdbd[i][2] == a1[i][1]);
      CHECK(bdbd[i][3] == a1[i][3]);
    }
  }
  
  {
    ttype acbd = simd::permutation<simd::permute_pattern::acbd>::permute(a1);
    for(int i = 0; i < ttype::num_vals; ++i) {
      CHECK(acbd[i][0] == a1[i][0]);
      CHECK(acbd[i][1] == a1[i][2]);
      CHECK(acbd[i][2] == a1[i][1]);
      CHECK(acbd[i][3] == a1[i][3]);
    }
  }
}

TEST_CASE() {
  test_permute_w4<int, simd::defaults<int>::nway>();
  test_permute_w4<int, simd::defaults<int>::nway*2>();
  
  test_permute_w4<long, simd::defaults<long>::nway>();
  test_permute_w4<long, simd::defaults<long>::nway*2>();
  
  test_permute_w4<float, simd::defaults<float>::nway>();
  test_permute_w4<float, simd::defaults<float>::nway*2>();
  
  test_permute_w4<double, simd::defaults<double>::nway>();
  test_permute_w4<double, simd::defaults<double>::nway*2>();
}

/* TEST - function: interleave */

template<typename T>
void test_interleave_w1() {
  typedef simd::pack<T> type1_t;
  type1_t arr1[type1_t::num_vals];
  T p = 1;
  for(int i = 0; i < type1_t::num_vals; ++i) {
    for(int j = 0; j < type1_t::num_vals; ++j, ++p) {
      arr1[i][j] = p;
    }
  }
  
  interleave(varray<type1_t, varray_tags<varray_tag::readwrite>::mask>(arr1));
  
  for(int i = 0; i < type1_t::num_vals; ++i) {
    p = i+1;
    for(int j = 0; j < type1_t::num_vals; ++j, p+=type1_t::num_vals) {
      //fprintf(stdout, "%f - %f\n", (T)arr[i][j], p);
      CHECK(p == arr1[i][j]);
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T, type1_t::num_vals*2> type2_t;
  type2_t arr2[type2_t::num_vals];
  p = 1;
  for(int i = 0; i < type2_t::num_vals; ++i) {
    for(int j = 0; j < type2_t::num_vals; ++j, ++p) {
      arr2[i][j] = p;
    }
  }
  
  interleave(varray<type2_t, varray_tags<varray_tag::readwrite>::mask>(arr2));
  
  for(int i = 0; i < type2_t::num_vals; ++i) {
    p = i+1;
    for(int j = 0; j < type2_t::num_vals; ++j, p+=type2_t::num_vals) {
      //fprintf(stdout, "%f - %f\n", (T)arr2[i][j], p);
      CHECK(p == arr2[i][j]);
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T, type1_t::num_vals*3> type3_t;
  type3_t arr3[type3_t::num_vals];
  p = 1;
  for(int i = 0; i < type3_t::num_vals; ++i) {
    for(int j = 0; j < type3_t::num_vals; ++j, ++p) {
      arr3[i][j] = p;
    }
  }
  
  interleave(varray<type3_t, varray_tags<varray_tag::readwrite>::mask>(arr3));
  
  for(int i = 0; i < type3_t::num_vals; ++i) {
    p = i+1;
    for(int j = 0; j < type3_t::num_vals; ++j, p+=type3_t::num_vals) {
      //fprintf(stdout, "%f - %f\n", (T)arr3[i][j], p);
      CHECK(p == arr3[i][j]);
    }
  }
}

template<typename T>
void test_interleave_w2() {
  typedef simd::pack<T[2]> type1_t;
  type1_t arr1[type1_t::num_vals];
  T p = 1;
  for(int i = 0; i < type1_t::num_vals; ++i) {
    for(int j = 0; j < type1_t::num_vals; ++j) {
      for(int k = 0; k < 2; ++k, ++p) {
        arr1[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type1_t, varray_tags<varray_tag::readwrite>::mask>(arr1));
  
  for(int i = 0; i < type1_t::num_vals; ++i) {
    p = 2*i+1;
    for(int j = 0; j < type1_t::num_vals; ++j, p+=type1_t::num_bvals) {
      //fprintf(stdout, "{%f, %f} - {%f, %f}\n", (T)arr1[i][j][0], (T)arr1[i][j][1], p, (p+1));
      CHECK(p == (T)(arr1[i][j][0]));
      CHECK((p+1) == (T)(arr1[i][j][1]));
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[2], type1_t::num_vals*2> type2_t;
  type2_t arr2[type2_t::num_vals];
  p = 1;
  for(int i = 0; i < type2_t::num_vals; ++i) {
    for(int j = 0; j < type2_t::num_vals; ++j) {
      for(int k = 0; k < 2; ++k, ++p) {
        arr2[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type2_t, varray_tags<varray_tag::readwrite>::mask>(arr2));
  
  for(int i = 0; i < type2_t::num_vals; ++i) {
    p = 2*i+1;
    for(int j = 0; j < type2_t::num_vals; ++j, p+=type2_t::num_bvals) {
      //fprintf(stdout, "{%f, %f} - {%f, %f}\n", (T)arr2[i][j][0], (T)arr2[i][j][1], p, (p+1));
      CHECK(p == arr2[i][j][0]);
      CHECK((p+1) == arr2[i][j][1]);
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[2], type1_t::num_vals*3> type3_t;
  type3_t arr3[type3_t::num_vals];
  p = 1;
  for(int i = 0; i < type3_t::num_vals; ++i) {
    for(int j = 0; j < type3_t::num_vals; ++j) {
      for(int k = 0; k < 2; ++k, ++p) {
        arr3[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type3_t, varray_tags<varray_tag::readwrite>::mask>(arr3));
  
  for(int i = 0; i < type3_t::num_vals; ++i) {
    p = 2*i+1;
    for(int j = 0; j < type3_t::num_vals; ++j, p+=type3_t::num_bvals) {
      //fprintf(stdout, "{%f, %f} - {%f, %f}\n", (T)arr3[i][j][0], (T)arr3[i][j][1], p, (p+1));
      CHECK(p == arr3[i][j][0]);
      CHECK((p+1) == arr3[i][j][1]);
    }
  }
}

template<typename T>
void test_interleave_w4() {
  typedef simd::pack<T[2][2]> type1_t;
  type1_t arr1[type1_t::num_vals];
  T p = 1;
  for(int i = 0; i < type1_t::num_vals; ++i) {
    for(int j = 0; j < type1_t::num_vals; ++j) {
      for(int k = 0; k < 2; ++k) {
        for(int l = 0; l < 2; ++l, ++p) {
          arr1[i][j][k][l] = p;
        }
      }
    }
  }
  
  interleave(varray<type1_t, varray_tags<varray_tag::readwrite>::mask>(arr1));
  
  for(int i = 0; i < type1_t::num_vals; ++i) {
    p = 4*i+1;
    for(int j = 0; j < type1_t::num_vals; ++j, p+=type1_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f} - {%f, %f, %f, %f}\n", 
      // (T)arr1[i][j][0][0], (T)arr1[i][j][0][1], (T)arr1[i][j][1][0], (T)arr1[i][j][1][1], 
      // p, (p+1), (p+2), (p+3));
      CHECK(p == (T)(arr1[i][j][0][0]));
      CHECK((p+1) == (T)(arr1[i][j][0][1]));
      CHECK((p+2) == (T)(arr1[i][j][1][0]));
      CHECK((p+3) == (T)(arr1[i][j][1][1]));
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[2][2], type1_t::num_vals*2> type2_t;
  type2_t arr2[type2_t::num_vals];
  p = 1;
  for(int i = 0; i < type2_t::num_vals; ++i) {
    for(int j = 0; j < type2_t::num_vals; ++j) {
      for(int k = 0; k < 2; ++k) {
        for(int l = 0; l < 2; ++l, ++p) {
          arr2[i][j][k][l] = p;
        }
      }
    }
  }
  
  interleave(varray<type2_t, varray_tags<varray_tag::readwrite>::mask>(arr2));
  
  for(int i = 0; i < type2_t::num_vals; ++i) {
    p = 4*i+1;
    for(int j = 0; j < type2_t::num_vals; ++j, p+=type2_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f} - {%f, %f, %f, %f}\n", 
      // (T)arr2[i][j][0][0], (T)arr2[i][j][0][1], (T)arr2[i][j][1][0], (T)arr2[i][j][1][1], 
      // p, (p+1), (p+2), (p+3));
      CHECK(p == arr2[i][j][0][0]);
      CHECK((p+1) == arr2[i][j][0][1]);
      CHECK((p+2) == arr2[i][j][1][0]);
      CHECK((p+3) == arr2[i][j][1][1]);
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[2][2], type1_t::num_vals*3> type3_t;
  type3_t arr3[type3_t::num_vals];
  p = 1;
  for(int i = 0; i < type3_t::num_vals; ++i) {
    for(int j = 0; j < type3_t::num_vals; ++j) {
      for(int k = 0; k < 2; ++k) {
        for(int l = 0; l < 2; ++l, ++p) {
          arr3[i][j][k][l] = p;
        }
      }
    }
  }
  
  interleave(varray<type3_t, varray_tags<varray_tag::readwrite>::mask>(arr3));
  
  for(int i = 0; i < type3_t::num_vals; ++i) {
    p = 4*i+1;
    for(int j = 0; j < type3_t::num_vals; ++j, p+=type3_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f} - {%f, %f, %f, %f}\n", 
      // (T)arr3[i][j][0][0], (T)arr3[i][j][0][1], (T)arr3[i][j][1][0], (T)arr3[i][j][1][1], 
      // p, (p+1), (p+2), (p+3));
      CHECK(p == arr3[i][j][0][0]);
      CHECK((p+1) == arr3[i][j][0][1]);
      CHECK((p+2) == arr3[i][j][1][0]);
      CHECK((p+3) == arr3[i][j][1][1]);
    }
  }
}

template<typename T>
void test_interleave_w8() {
  typedef simd::pack<T[8]> type1_t;
  type1_t arr1[type1_t::num_vals];
  T p = 1;
  for(int i = 0; i < type1_t::num_vals; ++i) {
    for(int j = 0; j < type1_t::num_vals; ++j) {
      for(int k = 0; k < 8; ++k, ++p) {
        arr1[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type1_t, varray_tags<varray_tag::readwrite>::mask>(arr1));
  
  for(int i = 0; i < type1_t::num_vals; ++i) {
    p = 8*i+1;
    for(int j = 0; j < type1_t::num_vals; ++j, p+=type1_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f, %f, %f, %f, %f} - {%f, %f, %f, %f, %f, %f, %f, %f}\n", 
      // (T)arr1[i][j][0], (T)arr1[i][j][1], (T)arr1[i][j][2], (T)arr1[i][j][3], 
      // (T)arr1[i][j][4], (T)arr1[i][j][5], (T)arr1[i][j][6], (T)arr1[i][j][7], 
      // p, (p+1), (p+2), (p+3), (p+4), (p+5), (p+6), (p+7));
      CHECK(p == (T)(arr1[i][j][0]));
      CHECK((p+1) == (T)(arr1[i][j][1]));
      CHECK((p+2) == (T)(arr1[i][j][2]));
      CHECK((p+3) == (T)(arr1[i][j][3]));
      CHECK((p+4) == (T)(arr1[i][j][4]));
      CHECK((p+5) == (T)(arr1[i][j][5]));
      CHECK((p+6) == (T)(arr1[i][j][6]));
      CHECK((p+7) == (T)(arr1[i][j][7]));
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[8], type1_t::num_vals*2> type2_t;
  type2_t arr2[type2_t::num_vals];
  p = 1;
  for(int i = 0; i < type2_t::num_vals; ++i) {
    for(int j = 0; j < type2_t::num_vals; ++j) {
      for(int k = 0; k < 8; ++k, ++p) {
        arr2[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type2_t, varray_tags<varray_tag::readwrite>::mask>(arr2));
  
  for(int i = 0; i < type2_t::num_vals; ++i) {
    p = 8*i+1;
    for(int j = 0; j < type2_t::num_vals; ++j, p+=type2_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f, %f, %f, %f, %f} - {%f, %f, %f, %f, %f, %f, %f, %f}\n", 
      // (T)arr2[i][j][0], (T)arr2[i][j][1], (T)arr2[i][j][2], (T)arr2[i][j][3], 
      // (T)arr2[i][j][4], (T)arr2[i][j][5], (T)arr2[i][j][6], (T)arr2[i][j][7], 
      // p, (p+1), (p+2), (p+3), (p+4), (p+5), (p+6), (p+7));
      CHECK(p == arr2[i][j][0]);
      CHECK((p+1) == arr2[i][j][1]);
      CHECK((p+2) == arr2[i][j][2]);
      CHECK((p+3) == arr2[i][j][3]);
      CHECK((p+4) == arr2[i][j][4]);
      CHECK((p+5) == arr2[i][j][5]);
      CHECK((p+6) == arr2[i][j][6]);
      CHECK((p+7) == arr2[i][j][7]);
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[8], type1_t::num_vals*3> type3_t;
  type3_t arr3[type3_t::num_vals];
  p = 1;
  for(int i = 0; i < type3_t::num_vals; ++i) {
    for(int j = 0; j < type3_t::num_vals; ++j) {
      for(int k = 0; k < 8; ++k, ++p) {
        arr3[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type3_t, varray_tags<varray_tag::readwrite>::mask>(arr3));
  
  for(int i = 0; i < type3_t::num_vals; ++i) {
    p = 8*i+1;
    for(int j = 0; j < type3_t::num_vals; ++j, p+=type3_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f, %f, %f, %f, %f} - {%f, %f, %f, %f, %f, %f, %f, %f}\n", 
      // (T)arr3[i][j][0], (T)arr3[i][j][1], (T)arr3[i][j][2], (T)arr3[i][j][3], 
      // (T)arr3[i][j][4], (T)arr3[i][j][5], (T)arr3[i][j][6], (T)arr3[i][j][7], 
      // p, (p+1), (p+2), (p+3), (p+4), (p+5), (p+6), (p+7));
      CHECK(p == arr3[i][j][0]);
      CHECK((p+1) == arr3[i][j][1]);
      CHECK((p+2) == arr3[i][j][2]);
      CHECK((p+3) == arr3[i][j][3]);
      CHECK((p+4) == arr3[i][j][4]);
      CHECK((p+5) == arr3[i][j][5]);
      CHECK((p+6) == arr3[i][j][6]);
      CHECK((p+7) == arr3[i][j][7]);
    }
  }
}

template<typename T>
void test_interleave_w16() {
  typedef simd::pack<T[16]> type1_t;
  type1_t arr1[type1_t::num_vals];
  T p = 1;
  for(int i = 0; i < type1_t::num_vals; ++i) {
    for(int j = 0; j < type1_t::num_vals; ++j) {
      for(int k = 0; k < 16; ++k, ++p) {
        arr1[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type1_t, varray_tags<varray_tag::readwrite>::mask>(arr1));
  
  for(int i = 0; i < type1_t::num_vals; ++i) {
    p = 16*i+1;
    for(int j = 0; j < type1_t::num_vals; ++j, p+=type1_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}"
      // " - {%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}\n", 
      // (T)arr1[i][j][0], (T)arr1[i][j][1], (T)arr1[i][j][2], (T)arr1[i][j][3], 
      // (T)arr1[i][j][4], (T)arr1[i][j][5], (T)arr1[i][j][6], (T)arr1[i][j][7], 
      // (T)arr1[i][j][8], (T)arr1[i][j][9], (T)arr1[i][j][10], (T)arr1[i][j][11], 
      // (T)arr1[i][j][12], (T)arr1[i][j][13], (T)arr1[i][j][14], (T)arr1[i][j][15], 
      // p, (p+1), (p+2), (p+3), (p+4), (p+5), (p+6), (p+7), 
      // (p+8), (p+9), (p+10), (p+11), (p+12), (p+13), (p+14), (p+15));
      CHECK(p == (T)(arr1[i][j][0]));
      CHECK((p+1) == (T)(arr1[i][j][1]));
      CHECK((p+2) == (T)(arr1[i][j][2]));
      CHECK((p+3) == (T)(arr1[i][j][3]));
      CHECK((p+4) == (T)(arr1[i][j][4]));
      CHECK((p+5) == (T)(arr1[i][j][5]));
      CHECK((p+6) == (T)(arr1[i][j][6]));
      CHECK((p+7) == (T)(arr1[i][j][7]));
      CHECK((p+8) == (T)(arr1[i][j][8]));
      CHECK((p+9) == (T)(arr1[i][j][9]));
      CHECK((p+10) == (T)(arr1[i][j][10]));
      CHECK((p+11) == (T)(arr1[i][j][11]));
      CHECK((p+12) == (T)(arr1[i][j][12]));
      CHECK((p+13) == (T)(arr1[i][j][13]));
      CHECK((p+14) == (T)(arr1[i][j][14]));
      CHECK((p+15) == (T)(arr1[i][j][15]));
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[16], type1_t::num_vals*2> type2_t;
  type2_t arr2[type2_t::num_vals];
  p = 1;
  for(int i = 0; i < type2_t::num_vals; ++i) {
    for(int j = 0; j < type2_t::num_vals; ++j) {
      for(int k = 0; k < 16; ++k, ++p) {
        arr2[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type2_t, varray_tags<varray_tag::readwrite>::mask>(arr2));
  
  for(int i = 0; i < type2_t::num_vals; ++i) {
    p = 16*i+1;
    for(int j = 0; j < type2_t::num_vals; ++j, p+=type2_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}"
      // " - {%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}\n", 
      // (T)arr2[i][j][0], (T)arr2[i][j][1], (T)arr2[i][j][2], (T)arr2[i][j][3], 
      // (T)arr2[i][j][4], (T)arr2[i][j][5], (T)arr2[i][j][6], (T)arr2[i][j][7], 
      // (T)arr2[i][j][8], (T)arr2[i][j][9], (T)arr2[i][j][10], (T)arr2[i][j][11], 
      // (T)arr2[i][j][12], (T)arr2[i][j][13], (T)arr2[i][j][14], (T)arr2[i][j][15], 
      // p, (p+1), (p+2), (p+3), (p+4), (p+5), (p+6), (p+7), 
      // (p+8), (p+9), (p+10), (p+11), (p+12), (p+13), (p+14), (p+15));
      CHECK(p == arr2[i][j][0]);
      CHECK((p+1) == arr2[i][j][1]);
      CHECK((p+2) == arr2[i][j][2]);
      CHECK((p+3) == arr2[i][j][3]);
      CHECK((p+4) == arr2[i][j][4]);
      CHECK((p+5) == arr2[i][j][5]);
      CHECK((p+6) == arr2[i][j][6]);
      CHECK((p+7) == arr2[i][j][7]);
      CHECK((p+8) == arr2[i][j][8]);
      CHECK((p+9) == arr2[i][j][9]);
      CHECK((p+10) == arr2[i][j][10]);
      CHECK((p+11) == arr2[i][j][11]);
      CHECK((p+12) == arr2[i][j][12]);
      CHECK((p+13) == arr2[i][j][13]);
      CHECK((p+14) == arr2[i][j][14]);
      CHECK((p+15) == arr2[i][j][15]);
    }
  }
  
  //---------------------------------------------------------------------------
  
  typedef simd::pack<T[16], type1_t::num_vals*3> type3_t;
  type3_t arr3[type3_t::num_vals];
  p = 1;
  for(int i = 0; i < type3_t::num_vals; ++i) {
    for(int j = 0; j < type3_t::num_vals; ++j) {
      for(int k = 0; k < 16; ++k, ++p) {
        arr3[i][j][k] = p;
      }
    }
  }
  
  interleave(varray<type3_t, varray_tags<varray_tag::readwrite>::mask>(arr3));
  
  for(int i = 0; i < type3_t::num_vals; ++i) {
    p = 16*i+1;
    for(int j = 0; j < type3_t::num_vals; ++j, p+=type3_t::num_bvals) {
      //fprintf(stdout, "{%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}"
      // " - {%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f}\n", 
      // (T)arr3[i][j][0], (T)arr3[i][j][1], (T)arr3[i][j][2], (T)arr3[i][j][3], 
      // (T)arr3[i][j][4], (T)arr3[i][j][5], (T)arr3[i][j][6], (T)arr3[i][j][7], 
      // (T)arr3[i][j][8], (T)arr3[i][j][9], (T)arr3[i][j][10], (T)arr3[i][j][11], 
      // (T)arr3[i][j][12], (T)arr3[i][j][13], (T)arr3[i][j][14], (T)arr3[i][j][15], 
      // p, (p+1), (p+2), (p+3), (p+4), (p+5), (p+6), (p+7), 
      // (p+8), (p+9), (p+10), (p+11), (p+12), (p+13), (p+14), (p+15));
      CHECK(p == arr3[i][j][0]);
      CHECK((p+1) == arr3[i][j][1]);
      CHECK((p+2) == arr3[i][j][2]);
      CHECK((p+3) == arr3[i][j][3]);
      CHECK((p+4) == arr3[i][j][4]);
      CHECK((p+5) == arr3[i][j][5]);
      CHECK((p+6) == arr3[i][j][6]);
      CHECK((p+7) == arr3[i][j][7]);
      CHECK((p+8) == arr3[i][j][8]);
      CHECK((p+9) == arr3[i][j][9]);
      CHECK((p+10) == arr3[i][j][10]);
      CHECK((p+11) == arr3[i][j][11]);
      CHECK((p+12) == arr3[i][j][12]);
      CHECK((p+13) == arr3[i][j][13]);
      CHECK((p+14) == arr3[i][j][14]);
      CHECK((p+15) == arr3[i][j][15]);
    }
  }
}

TEST_CASE("simd::pack<T, NW> - function: interleave", "[interleave]") {
#if SIMD256
  test_interleave_w1<float>();
  test_interleave_w2<float>();
  test_interleave_w4<float>();
  test_interleave_w8<float>();
  
  test_interleave_w1<double>();
  test_interleave_w2<double>();
  test_interleave_w4<double>();
#elif SIMD512
  test_interleave_w1<float>();
  test_interleave_w2<float>();
  test_interleave_w4<float>();
  test_interleave_w8<float>();
  test_interleave_w16<float>();
  
  test_interleave_w1<double>();
  test_interleave_w2<double>();
  test_interleave_w4<double>();
  test_interleave_w8<double>();
#endif
}

} // namespace general {

#endif // #ifndef SIMD_SIMD_GENERIC_CTEST_HPP
