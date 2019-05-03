#ifndef SIMD_SIMDDEF_HPP
#define SIMD_SIMDDEF_HPP

#include <varray.hpp>

#include <type_traits>
#include <typeinfo>
#include <cstddef>
#include <initializer_list>

#include <ostream>

namespace simd {

class rindex {
public:
  explicit rindex(int i) : index(i) {}
  operator int() { return index; }
  
private:
  int index;
};

class vindex {
public:
  explicit vindex(int i) : index(i) {}
  operator int() { return index; }
  
private:
  int index;
};

class bindex {
public:
  explicit bindex(int i) : index(i) {}
  operator int() { return index; }
  
private:
  int index;
};

// Conversion function for SIMD pack type casts
// fromBT - base type for from value
// fromRT - register type for from value
// toBT   - base type for to value
// toRT   - register_type for to value
template<typename fromBT, typename fromRT, typename toBT, typename toRT>
void convert(const fromRT **from, toRT **to) {
  static_assert(false, "Not implemented");
}

// Sample covert function specialization
//inline void convert<float, __m256i, double, __m256i>(__m256i **from, __m256i **to) {
//  (*to)[0] = _mm256_cvtps_pd(_mm256_castps256_ps128((*from)[0]));
//  (*to)[1] = _mm256_cvtps_pd(_mm256_extractf128_ps((*from)[0], 1));
//  (*to) += 2;
//  (*from) += 1;
//}

template<typename T>
struct defaults {
  static_assert(
    simd::defaults<typename std::remove_all_extents<T>::type>::nway*
    sizeof(typename std::remove_all_extents<T>::type)/sizeof(T)*sizeof(T)
    ==
    simd::defaults<typename std::remove_all_extents<T>::type>::nway*
    sizeof(typename std::remove_all_extents<T>::type), "Invalid T"
  );
  
  static const int nway = simd::defaults<typename std::remove_all_extents<T>::type>::nway*sizeof(typename std::remove_all_extents<T>::type)/sizeof(T);
};

template<typename T, int NW = simd::defaults<T>::nway>
struct type_traits {
};

template<typename T, size_t N, int NW>
struct type_traits<T[N], NW> {
  typedef typename simd::type_traits<T, N*NW>::base_type base_type;
  typedef typename simd::type_traits<T, N*NW>::register_type register_type;
  typedef typename simd::type_traits<T, N*NW>::mask_register_type mask_register_type;
  
  static constexpr int num_regs = simd::type_traits<T, N*NW>::num_regs;
  static constexpr int num_vals = simd::type_traits<T, N*NW>::num_vals/N;
  static constexpr int num_bvals = simd::type_traits<T, N*NW>::num_bvals;
  static constexpr int num_vals_per_reg = simd::type_traits<T, N*NW>::num_vals_per_reg/N;
  static constexpr int num_bvals_per_reg = simd::type_traits<T, N*NW>::num_bvals_per_reg;
  static constexpr int num_bvals_per_val = simd::type_traits<T, N*NW>::num_bvals_per_val*N;
  
  static_assert(num_vals*N == simd::type_traits<T, N*NW>::num_vals, "Invalid num_vals");
  static_assert(num_vals_per_reg*N == simd::type_traits<T, N*NW>::num_vals_per_reg, "Invalid num_vals_per_reg");
  static_assert(num_bvals_per_val/N == simd::type_traits<T, N*NW>::num_bvals_per_val, "Invalid num_bvals_per_val");
};

template<typename T, int NW = simd::defaults<T>::nway>
class mask;

template<
  typename T,
  int NW = simd::defaults<T>::nway,
  typename RT = typename simd::type_traits<T, NW>::register_type,
  typename BT = typename simd::type_traits<T, NW>::base_type,
  int W = simd::type_traits<T, NW>::num_bvals_per_val
>
class pack;

/* mask_set(...) declarations */

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, bool value = true);

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, bool value = true);

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, vindex avi, bool value = true);

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, bindex abi, bool value = true);

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, vindex v, bool value = true);

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, bindex b, bool value = true);

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, vindex avi, bindex b, bool value = true);

template<typename T, int NW>
mask<T, NW>& mask_set(mask<T, NW> &m, rindex ari, vindex v, bindex b, bool value = true);

/* mask_reset(...) declarations */

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m);

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari);

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, vindex avi);

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, bindex abi);

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari, vindex v);

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari, bindex b);

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, vindex avi, bindex b);

template<typename T, int NW>
mask<T, NW>& mask_reset(mask<T, NW> &m, rindex ari, vindex v, bindex b);

/* mask_flip(...) declarations */

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m);

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari);

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, vindex avi);

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, bindex abi);

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, vindex v);

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, bindex b);

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, vindex avi, bindex b);

template<typename T, int NW>
mask<T, NW>& mask_flip(mask<T, NW> &m, rindex ari, vindex v, bindex b);

/* mask_all(...) declarations */

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m);

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, rindex ari);

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, vindex avi);

template<typename T, int NW>
bool mask_all(const mask<T, NW> &m, rindex ari, vindex v);

/* mask_any(...) declarations */

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m);

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, rindex ari);

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, vindex avi);

template<typename T, int NW>
bool mask_any(const mask<T, NW> &m, rindex ari, vindex v);

/* mask_none(...) implementations */

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m);

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, rindex ari);

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, vindex avi);

template<typename T, int NW>
bool mask_none(const mask<T, NW> &m, rindex ari, vindex v);

/* mask_test(...) declarations */

template<typename T, int NW>
bool mask_test(const mask<T, NW> &m, bindex abi);

template<typename T, int NW>
bool mask_test(const mask<T, NW> &m, rindex ari, bindex b);

/* Function to set value to zero */
template<typename T, int NW, typename RT, typename BT, int W>
void set_zero(pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Function to set all elements of a pack to a scalar value */
template<typename T, int NW, typename RT, typename BT, int W, typename BT1>
void set_scalar(pack<T, NW, RT, BT, W> &op, BT1 value, 
  typename std::enable_if<std::is_arithmetic<BT1>::value>::type * = nullptr
) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: unary - */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator-(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not mplemented");
}

/* Arithmetic operator: unary + */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator+(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: prefix ++ */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W>& operator++(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: postfix ++ */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator++(const pack<T, NW, RT, BT, W> &op, int) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: prefix -- */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W>& operator--(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: postfix -- */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator--(const pack<T, NW, RT, BT, W> &op, int) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: + */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator+(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: - */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator-(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: * */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator*(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: / */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator/(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Arithmetic operator: % - only for integer types */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator%(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Bitwise operator: & - only for integer types */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator&(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Bitwise operator: | - only for integer types */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator|(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Bitwise operator: ^ - only for integer types */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator^(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Bitwise operator:  - only for integer types */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator~(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Bitwise operator: << - only for integer types */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator<<(const pack<T, NW, RT, BT, W> &op, int num) {
  static_assert(false, "Not implemented");
}

/* Bitwise operator: >> - only for integer types */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> operator>>(const pack<T, NW, RT, BT, W> &op, int num) {
  static_assert(false, "Not implemented");
}

/* Comparison operator: == */
template<typename T, int NW, typename RT, typename BT, int W>
typename pack<T, NW, RT, BT, W>::mask_type operator==(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Comparison operator: != */
template<typename T, int NW, typename RT, typename BT, int W>
typename pack<T, NW, RT, BT, W>::mask_type operator!=(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Comparison operator: < */
template<typename T, int NW, typename RT, typename BT, int W>
typename pack<T, NW, RT, BT, W>::mask_type operator<(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Comparison operator: > */
template<typename T, int NW, typename RT, typename BT, int W>
typename pack<T, NW, RT, BT, W>::mask_type operator>(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Comparison operator: <= */
template<typename T, int NW, typename RT, typename BT, int W>
typename pack<T, NW, RT, BT, W>::mask_type operator<=(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Comparison operator: >= */
template<typename T, int NW, typename RT, typename BT, int W>
typename pack<T, NW, RT, BT, W>::mask_type operator>=(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Logical operators are not overloaded. This is to allow conditional checks 
 * such as if(a == b). The conversion to bool triggers in such circumstances.
 */
/* Logical operator: ! */
template<typename T, int NW, typename RT, typename BT, int W>
inline bool operator!(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Logical operator: && */
template<typename T, int NW, typename RT, typename BT, int W>
inline bool operator&&(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Logical operator: || */
template<typename T, int NW, typename RT, typename BT, int W>
inline bool operator||(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Function: square root */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> sqrt(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Function: inverse */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> inv(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Function: sin */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> sin(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Function: cos */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> cos(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Function: tan */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> tan(const pack<T, NW, RT, BT, W> &op) {
  static_assert(false, "Not implemented");
}

/* Function: add adjacent numbers and interleave results */
template<typename T, int NW, typename RT, typename BT, int W>
pack<T, NW, RT, BT, W> hadd_pairwise_interleave(const pack<T, NW, RT, BT, W> &lhs, const pack<T, NW, RT, BT, W> &rhs) {
  static_assert(false, "Not implemented");
}

/* Input and output operators: << */

template<typename T, int NW, typename RT, typename BT, int W>
std::ostream& operator<<(std::ostream &strm, const pack<T, NW, RT, BT, W> &op) {
  std::streamsize w = strm.width();
  for(int i = 0; i < pack<T, NW, RT, BT, W>::num_bvals; ++i) {
    strm.width(w);
    strm << op.belm[i] << " ";
  }
  return strm;
}

/* Load and store functions */

enum memhint {
  temporal = 0x0,
  unaligned = 0x1,
  aligned = 0x2,
  nontemporal = 0x4
};

template<typename T, int hint = memhint::temporal|memhint::unaligned>
class const_mptr {
public:
  const_mptr(const T *p) : ptr(p) {}
  const_mptr(const const_mptr<T, hint> &src) : ptr(src.ptr) {}
  operator const T*() const { return ptr; }
  
private:
  const T *ptr;
};

template<typename T, int hint = memhint::temporal|memhint::unaligned>
class mptr {
public:
  mptr(T *p) : ptr(p) {}
  mptr(const mptr<T, hint> &src) : ptr(src.ptr) {}
  operator T*() { return ptr; }
  
private:
  T *ptr;
};

template<typename T, int NW, typename RT, typename BT, int W, typename MBT, int hint>
void load(pack<T, NW, RT, BT, W> &op, const_mptr<MBT, hint> ptr) {
  static_assert(false, "Not implemented");
}

template<typename T, int NW, typename RT, typename BT, int W, typename MBT, int hint>
void load(pack<T, NW, RT, BT, W> &op, const_mptr<MBT, hint> ptr, const mask<T, NW> &m) {
  static_assert(false, "Not implemented");
}

template<typename T, int NW, typename RT, typename BT, int W, typename MBT, int hint>
void store(const pack<T, NW, RT, BT, W> &op, mptr<MBT, hint> ptr) {
  static_assert(false, "Not implemented");
}

template<typename T, int NW, typename RT, typename BT, int W, typename MBT, int hint>
void store(const pack<T, NW, RT, BT, W> &op, mptr<MBT, hint> ptr, const mask<T, NW> &m) {
  static_assert(false, "Not implemented");
}

/* Permutation operations */

enum permute_pattern {
  aacc,
  abab,
  bbdd,
  cdcd, 
  dcba, 
  dbca, 
  acac, 
  bdbd, 
  acbd
};

template<permute_pattern pattern>
class permutation {
};

template<typename T, int NW, typename RT, typename BT, int W, int VF>
void interleave(varray<pack<T, NW, RT, BT, W>, VF> op) {
  static_assert(false, "Not implemented");
}

//void print128_num(__m128i var)
//{
//    int32_t *val = (int32_t*) &var;
//    std::cout << val[0] << " " << val[1] << " " << val[2] << " " << val[3] << std::endl;
//}

//void print256_num(__m256i var)
//{
//    int32_t *val = (int32_t*) &var;
//    std::cout << val[0] << " " << val[1] << " " << val[2] << " " << val[3] << " " << val[4] << " " << val[5] << " " << val[6] << " " << val[7] << std::endl;
//}

template<
  typename T, 
  int NW
>
class alignas(alignof(typename simd::type_traits<T, NW>::mask_register_type)) mask {
public:
  typedef mask<T, NW> self_type;
  typedef typename simd::type_traits<T, NW>::base_type base_type;
  typedef typename simd::type_traits<T, NW>::mask_register_type register_type;
  
private:
  class mask_bit {
    friend class mask<T, NW>;
    
  public:
    operator bool() {
      return mask_test(m_mask, rindex(m_reg_index), bindex(m_index));
    }
    
    operator bool() const {
      return mask_test(m_mask, rindex(m_reg_index), bindex(m_index));
    }
    
    mask_bit& operator=(bool value) {
      mask_set(m_mask, rindex(m_reg_index), bindex(m_index), value);
      return *this;
    }
    
  private:
    mask_bit(mask<T, NW> &mask, int index) : m_mask(mask), 
      m_reg_index(index / simd::type_traits<T, NW>::num_bvals_per_reg), 
      m_index(index % simd::type_traits<T, NW>::num_bvals_per_reg) {
    }
    
  private:
    mask<T, NW> &m_mask;
    int m_reg_index, m_index;
  };
  
public:
  inline operator bool () {
    return mask_all(*this);
  }
  
  inline operator bool () const {
    return mask_all(*this);
  }
  
  inline operator register_type & () {
    return reg[0];
  }
  
  inline operator const register_type & () const {
    return reg[0];
  }
  
  inline register_type & operator()(int index = 0) {
    return reg[index];
  }
  
  inline const register_type & operator()(int index = 0) const {
    return reg[index];
  }
  
  inline mask_bit operator[](int index) {
    return mask_bit(*this, index);
  }
  
  inline const mask_bit operator[](int index) const {
    return mask_bit(*this, index);
  }
  
  /* set(...) methods */
  
  mask<T, NW>& set(bool value = true) {
    return mask_set(*this, value);
  }
  
  mask<T, NW>& set(rindex ari, bool value = true) {
    return mask_set(*this, ari, value);
  }
  
  mask<T, NW>& set(vindex avi, bool value = true) {
    return mask_set(*this, avi, value);
  }
  
  mask<T, NW>& set(bindex abi, bool value = true) {
    return mask_set(*this, abi, value);
  }
  
  mask<T, NW>& set(rindex ari, vindex v, bool value = true) {
    return mask_set(*this, ari, v, value);
  }
  
  mask<T, NW>& set(rindex ari, bindex b, bool value = true) {
    return mask_set(*this, ari, b, value);
  }
  
  mask<T, NW>& set(vindex avi, bindex b, bool value = true) {
    return mask_set(*this, avi, b, value);
  }
  
  mask<T, NW>& set(rindex ari, vindex v, bindex b, bool value = true) {
    return mask_set(*this, ari, v, b, value);
  }
  
  /* reset(...) methods */
  
  mask<T, NW>& reset() {
    return mask_reset(*this);
  }
  
  mask<T, NW>& reset(rindex ari) {
    return mask_reset(*this, ari);
  }
  
  mask<T, NW>& reset(vindex avi) {
    return mask_reset(*this, avi);
  }
  
  mask<T, NW>& reset(bindex abi) {
    return mask_reset(*this, abi);
  }
  
  mask<T, NW>& reset(rindex ari, vindex v) {
    return mask_reset(*this, ari, v);
  }
  
  mask<T, NW>& reset(rindex ari, bindex b) {
    return mask_reset(*this, ari, b);
  }
  
  mask<T, NW>& reset(vindex avi, bindex b) {
    return mask_reset(*this, avi, b);
  }
  
  mask<T, NW>& reset(rindex ari, vindex v, bindex b) {
    return mask_reset(*this, ari, v, b);
  }
  
  /* flip(...) methods */
  
  mask<T, NW>& flip() {
    return mask_flip(*this);
  }
  
  mask<T, NW>& flip(rindex ari) {
    return mask_flip(*this, ari);
  }
  
  mask<T, NW>& flip(vindex avi) {
    return mask_flip(*this, avi);
  }
  
  mask<T, NW>& flip(bindex abi) {
    return mask_flip(*this, abi);
  }
  
  mask<T, NW>& flip(rindex ari, vindex v) {
    return mask_flip(*this, ari, v);
  }
  
  mask<T, NW>& flip(rindex ari, bindex b) {
    return mask_flip(*this, ari, b);
  }
  
  mask<T, NW>& flip(vindex avi, bindex b) {
    return mask_flip(*this, avi, b);
  }
  
  mask<T, NW>& flip(rindex ari, vindex v, bindex b) {
    return mask_flip(*this, ari, v, b);
  }
  
  /* all(...) methods */
  
  bool all() const {
    return mask_all(*this);
  }
  
  bool all(rindex ari) const {
    return mask_all(*this, ari);
  }
  
  bool all(vindex avi) const {
    return mask_all(*this, avi);
  }
  
  bool all(rindex ari, vindex v) const {
    return mask_all(*this, ari, v);
  }
  
  /* any(...) methods */
  
  bool any() const {
    return mask_any(*this);
  }
  
  bool any(rindex ari) const {
    return mask_any(*this, ari);
  }
  
  bool any(vindex avi) const {
    return mask_any(*this, avi);
  }
  
  bool any(rindex ari, vindex v) const {
    return mask_any(*this, ari, v);
  }
  
  /* none(...) methods */
  
  bool none() const {
    return mask_none(*this);
  }
  
  bool none(rindex ari) const {
    return mask_none(*this, ari);
  }
  
  bool none(vindex avi) const {
    return mask_none(*this, avi);
  }
  
  bool none(rindex ari, vindex v) const {
    return mask_none(*this, ari, v);
  }
  
  /* test(...) methods */
  
  bool test(bindex abi) const {
    return mask_test(*this, abi);
  }
  
  bool test(rindex ari, bindex b) const {
    return mask_test(*this, ari, b);
  }
  
  DEFINE_ALIGNED_NEW_DELETE_OPERATORS(self_type)
  
private:
  union {
    typename simd::type_traits<T, NW>::mask_register_type reg[simd::type_traits<T, NW>::num_regs];
  };
};

/*
 * T  - type for which this SIMD pack is instantiated
 * RT - register type
 * NW - number of ways of interleaving
 * BT - base type of type T
 * W  - width of type T in terms of number of objects of type BT
 */
template<typename T, int NW, typename RT, typename BT, int W>
class alignas(SIMD_MAX_WIDTH) pack {
  friend std::ostream& operator<< <T, NW, RT, BT, W>(std::ostream &strm, const pack<T, NW, RT, BT, W> &op);
  
  /*static_assert(sizeof(T) <= sizeof(RT), "T must fit in RT");
  static_assert(sizeof(RT)/sizeof(T)*sizeof(T) == sizeof(RT), "RT must accomodate an integer number of T");
  static_assert(sizeof(BT) <= sizeof(RT), "BT must fit in RT");
  static_assert(sizeof(RT)/sizeof(BT)*sizeof(BT) == sizeof(RT), "RT must accomodate an integer number of BT");
  static_assert(sizeof(BT) <= sizeof(T), "BT must fit in T");
  static_assert(sizeof(T)/sizeof(BT)*sizeof(BT) == sizeof(T), "T must accomodate an integer number of BT");
  static_assert(std::is_same<typename std::remove_all_extents<T>::type, BT>::value, "BT must be a base type of T");
  static_assert(NW > 0, "NW must be greater than zero");
  static_assert(sizeof(BT)*simd::arch<BT>::width/sizeof(RT)*sizeof(RT) == sizeof(BT)*simd::arch<BT>::width, "Invalid simd::arch<BT>::width");
  //static_assert(sizeof(T)*NW/sizeof(RT)*sizeof(RT) == sizeof(T)*NW, "Invalid NW");
  static_assert(sizeof(T)*NW/(sizeof(BT)*simd::arch<BT>::width)*(sizeof(BT)*simd::arch<BT>::width) == sizeof(T)*NW, "Invalid NW");
  static_assert(sizeof(T) == sizeof(BT)*W, "Invalid W");*/
  
  //static_assert(sizeof(T) <= sizeof(RT), "T must fit in RT");
  //static_assert(sizeof(RT)/sizeof(T)*sizeof(T) == sizeof(RT), "RT must accomodate an integer number of T");
  //static_assert(sizeof(BT) <= sizeof(RT), "BT must fit in RT");
  //static_assert(sizeof(RT)/sizeof(BT)*sizeof(BT) == sizeof(RT), "RT must accomodate an integer number of BT");
  //static_assert(sizeof(BT) <= sizeof(T), "BT must fit in T");
  //static_assert(sizeof(T)/sizeof(BT)*sizeof(BT) == sizeof(T), "T must accomodate an integer number of BT");
  //static_assert(std::is_same<typename std::remove_all_extents<T>::type, BT>::value, "BT must be a base type of T");
  //static_assert(NW > 0, "NW must be greater than zero");
  //static_assert(sizeof(BT)*simd::type_traits<BT>::width/sizeof(RT)*sizeof(RT) == sizeof(BT)*simd::type_traits<BT>::width, "Invalid simd::type_traits<BT>::width");
  ////static_assert(sizeof(T)*NW/sizeof(RT)*sizeof(RT) == sizeof(T)*NW, "Invalid NW");
  //static_assert(sizeof(T)*NW/(sizeof(BT)*simd::type_traits<BT>::width)*(sizeof(BT)*simd::type_traits<BT>::width) == sizeof(T)*NW, "Invalid NW");
  //static_assert(sizeof(T) == sizeof(BT)*W, "Invalid W");
  
  template<typename U, typename V>
  static void copy_value(U &dst, const V &src, 
    typename std::enable_if<std::is_arithmetic<U>::value && std::is_arithmetic<V>::value>::type * = nullptr
  ) {
    dst = src;
  }
  
  template<typename U, typename V, size_t N>
  static void copy_value(U (&dst)[N], const V (&src)[N]) {
    for(int i = 0; i < N; ++i) {
      copy_value(dst[i], src[i]);
    }
  }
  
  template<typename U, typename V, 
    typename = typename std::enable_if<std::is_arithmetic<U>::value && std::is_arithmetic<V>::value>::type
  >
  struct array_base_type_transform {
    using type = V;
  };
  
  template<typename U, size_t N, typename V>
  struct array_base_type_transform<U[N], V> {
    using type = typename array_base_type_transform<U, V>::type[N];
  };
  
  template<typename U, typename V>
  struct has_same_extents {
    static constexpr bool value = (std::rank<U>::value == 0 && std::rank<V>::value == 0);
  };
  
  template<typename U, size_t NU, typename V, size_t NV>
  struct has_same_extents<U[NU], V[NV]> {
    static constexpr bool value = ((NU == NV) && has_same_extents<U, V>::value);
  };
  
  template<typename U>
  struct is_arithmetic_array {
    static constexpr bool value = (std::is_array<U>::value && std::is_arithmetic<typename std::remove_all_extents<U>::type>::value);
  };
  
public:
  typedef pack<T, NW, RT, BT, W> self_type;
  
  //typedef T value_type;
  class value_type {
  public:
    value_type() {};
    
    template<typename DT, typename std::enable_if<std::is_arithmetic<DT>::value && has_same_extents<DT, T>::value, int>::type = 0>
    inline value_type(const DT value) {
      static_assert(std::is_arithmetic<DT>::value && has_same_extents<DT, T>::value, "DT must be an arithmetic type");
      m_value = value;
    }
    
    template<typename DT, typename std::enable_if<is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, int>::type = 0>
    inline value_type(const DT &value) {
      static_assert(is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, "DT must be an array of an arithmetic type of same extents as T");
      copy_value(m_value, value);
    }
    
    inline value_type(const value_type &src) {
      copy_value(m_value, src.m_value);
    }
    
    /*template<typename DT, typename std::enable_if<std::is_same<T, BT>::value && std::is_same<DT, BT>::value, int>::type = 0>
    inline operator DT&() {
      static_assert(std::is_same<T, BT>::value && std::is_same<DT, BT>::value, "");
      return m_value;
    }
    
    template<typename DT, typename std::enable_if<std::is_same<T, BT>::value && std::is_same<DT, BT>::value, int>::type = 0>
    inline operator const DT&() const {
      static_assert(std::is_same<T, BT>::value && std::is_same<DT, BT>::value, "");
      return m_value;
    }
    
    template<typename DT, typename std::enable_if<!std::is_same<T, BT>::value && std::is_same<DT, T>::value, int>::type = 0>
    inline operator DT&() {
      static_assert(!std::is_same<T, BT>::value && std::is_same<DT, T>::value, "");
      return m_value;
    }
    
    template<typename DT, typename std::enable_if<!std::is_same<T, BT>::value && std::is_same<DT, T>::value, int>::type = 0>
    inline operator const DT&() const {
      static_assert(!std::is_same<T, BT>::value && std::is_same<DT, T>::value, "");
      return m_value;
    }*/
    
    inline operator T&() {
      return m_value;
    }
    
    inline operator const T&() const {
      return m_value;
    }
    
    template<typename DT, typename std::enable_if<std::is_arithmetic<DT>::value && has_same_extents<DT, T>::value, int>::type = 0>
    inline value_type& operator=(const DT value) {
      static_assert(std::is_arithmetic<DT>::value && has_same_extents<DT, T>::value, "DT must be an arithmetic type");
      m_value = value;
      return *this;
    }
    
    template<typename DT, typename std::enable_if<is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, int>::type = 0>
    inline value_type& operator=(const DT &value) {
      static_assert(is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, "DT must be an array of an arithmetic type of same extents as T");
      copy_value(m_value, value);
      return *this;
    }
    
    inline value_type& operator=(const value_type &rhs) {
      copy_value(m_value, rhs.m_value);
      return *this;
    }
    
  private:
    T m_value;
  };
  
  typedef RT register_type;
  
  typedef BT base_type;
  
  typedef mask<T, NW> mask_type;
  
  typedef pack<T, NW, RT, BT, W>* pointer;
  
  typedef SIMD_WIN_PTR_ALIGN pack<T, NW, RT, BT, W>* SIMD_LINUX_PTR_ALIGN aligned_pointer;
  
  /* number of RT per pack */
  static constexpr int num_regs = simd::type_traits<T, NW>::num_regs;
  
  /* number of T per pack */
  static constexpr int num_vals = simd::type_traits<T, NW>::num_vals;
  
  /* number of BT per pack */
  static constexpr int num_bvals = simd::type_traits<T, NW>::num_bvals;
  
  /* number of T per RT */
  static constexpr int num_vals_per_reg = simd::type_traits<T, NW>::num_vals_per_reg;
  
  /* number of BT per RT */
  static constexpr int num_bvals_per_reg = simd::type_traits<T, NW>::num_bvals_per_reg;
  
  /* number of BT per T */
  static constexpr int num_bvals_per_val = simd::type_traits<T, NW>::num_bvals_per_val;
  
  inline pack() {}
  
  inline pack(const register_type &r) {
    for(int i = 0; i < num_regs; ++i) {
      reg[i] = r;
    }
  }
  
  inline pack(const self_type &rhs) {
    for(int i = 0; i < num_regs; ++i) {
      reg[i] = rhs.reg[i];
    }
  }
  
  inline pack(std::initializer_list<base_type> init) {
    load(*this, const_mptr<base_type>(init.begin()));
  }
  
  template<int hint>
  inline pack(const_mptr<base_type, hint> ptr) {
    load(*this, ptr);
  }
  
  template<typename fromT, typename fromRT, typename fromBT>
  inline pack(const pack<fromT, NW, fromRT, fromBT, W> &rhs) {
    typedef pack<fromT, NW, fromRT, fromBT, W> fromtype;
    
    //const fromtype::register_type *fromregp = &rhs.reg[0];
    //register_type *toregp = &reg[0];
    
    for(int i = 0; i < fromtype::num_regs; ++i) {
      convert<fromtype::base_type, base_type>(&rhs.reg[i], &reg[i]);
    }
  }
  
  template<typename DT, typename std::enable_if<std::is_arithmetic<DT>::value, int>::type = 0>
  inline pack(const DT &rhs) {
    static_assert(std::is_arithmetic<DT>::value, "DT must be an arithmetic type");
    set_scalar(*this, (BT)rhs);
  }

  template<typename DT, typename std::enable_if<is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, int>::type = 0>
  inline pack(const DT &rhs) {
    static_assert(is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, "DT must be an array of an arithmetic type of same extents as T");
    for(int i = 0; i < NW; ++i) {
      elm[i] = rhs;
    }
  }
  
  inline self_type & operator=(const register_type &r) {
    for(int i = 0; i < num_regs; ++i) {
      reg[i] = r;
    }
    return *this;
  }
  
  inline self_type & operator=(const self_type &rhs) {
    for(int i = 0; i < num_regs; ++i) {
      reg[i] = rhs.reg[i];
    }
    return *this;
  }
  
  inline self_type& operator=(std::initializer_list<base_type> init) {
    load(*this, const_mptr<base_type>(init.begin()));
  }
  
  template<int hint>
  inline self_type& operator=(const_mptr<base_type, hint> ptr) {
    load(*this, ptr);
  }
  
  template<typename fromT, typename fromRT, typename fromBT>
  inline self_type& operator=(const pack<fromT, NW, fromRT, fromBT, W> &rhs) {
    typedef pack<fromT, NW, fromRT, fromBT, W> fromtype;
    
    //const fromtype::register_type *fromregp = &rhs.reg[0];
    //register_type *toregp = &reg[0];
    
    for(int i = 0; i < fromtype::num_regs; ++i) {
      convert<fromtype::base_type, base_type>(&rhs.reg[i], &reg[i]);
    }
    
    return *this;
  }
  
  template<typename DT, typename std::enable_if<std::is_arithmetic<DT>::value, int>::type = 0>
  inline self_type& operator=(const DT &rhs) {
    static_assert(std::is_arithmetic<DT>::value, "DT must be an arithmetic type");
    set_scalar(*this, (BT)rhs);
    return *this;
  }
  
  template<typename DT, typename std::enable_if<is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, int>::type = 0>
  inline self_type& operator=(const DT &rhs) {
    static_assert(is_arithmetic_array<DT>::value && has_same_extents<DT, T>::value, "DT must be an array of an arithmetic type of same extents as T");
    for(int i = 0; i < NW; ++i) {
      elm[i] = rhs;
    }
    return *this;
  }
  
  inline operator register_type & () {
    return reg[0];
  }
  
  inline operator const register_type & () const {
    return reg[0];
  }
  
  inline operator value_type * () {
    return &elm[0];
  }
  
  inline operator const value_type * () const {
    return &elm[0];
  }
  
  template<typename toT, typename toRT, typename toBT>
  inline operator pack<toT, NW, toRT, toBT, W> () {
    typedef pack<toT, NW, toRT, toBT, W> totype;
    
    totype to;
    
    //const register_type *fromregp = &reg[0];
    //totype::register_type *toregp = &to.reg[0];
    
    for(int i = 0; i < num_regs; ++i) {
      convert<base_type, totype::base_type>(&reg[i], &to.reg[i]);
    }
    
    return to;
  }
  
  inline register_type & operator()(int index = 0) {
    return reg[index];
  }
  
  inline const register_type & operator()(int index = 0) const {
    return reg[index];
  }
  
  inline value_type & operator[](int index) {
    return elm[index];
  }
  
  inline const value_type & operator[](int index) const {
    return elm[index];
  }
  
  DEFINE_ALIGNED_NEW_DELETE_OPERATORS(self_type)
  
  inline self_type& operator+=(const self_type &rhs) {
    *this = *this + rhs;
    return *this;
  }
  
  inline self_type& operator-=(const self_type &rhs) {
    *this = *this - rhs;
    return *this;
  }
  
  inline self_type& operator*=(const self_type &rhs) {
    *this = *this * rhs;
    return *this;
  }
  
  inline self_type& operator/=(const self_type &rhs) {
    *this = *this / rhs;
    return *this;
  }
  
  inline self_type& operator%=(const self_type &rhs) {
    *this = *this % rhs;
    return *this;
  }
  
  inline self_type& operator&=(const self_type &rhs) {
    *this = *this & rhs;
    return *this;
  }
  
  inline self_type& operator|=(const self_type &rhs) {
    *this = *this | rhs;
    return *this;
  }
  
  inline self_type& operator^=(const self_type &rhs) {
    *this  = *this ^ rhs;
    return *this;
  }
  
  inline self_type& operator<<=(const self_type &rhs) {
    *this = *this << rhs;
    return *this;
  }
  
  inline self_type& operator>>=(const self_type &rhs) {
    *this = *this >> rhs;
    return *this;
  }
  
private:
  union {
    //register_type reg[sizeof(T)*NW/sizeof(RT)];
    register_type reg[simd::type_traits<T, NW>::num_regs];
    value_type elm[NW];
    base_type belm[simd::type_traits<T, NW>::num_bvals];
  };
};

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_regs = simd::type_traits<T, NW>::num_regs;

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_vals = simd::type_traits<T, NW>::num_vals;

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_bvals = simd::type_traits<T, NW>::num_bvals;

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_vals_per_reg = simd::type_traits<T, NW>::num_vals_per_reg;

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_bvals_per_reg = simd::type_traits<T, NW>::num_bvals_per_reg;

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_bvals_per_val = simd::type_traits<T, NW>::num_bvals_per_val;



//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_vals = NW;

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_regs = sizeof(T)*NW/sizeof(RT);

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_bvals = NW*W;

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_bvals_per_val = sizeof(T)/sizeof(BT);

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_bvals_per_reg = sizeof(RT)/sizeof(BT);

//template<typename T, int NW, typename RT, typename BT, int W>
//const int pack<T, NW, RT, BT, W>::num_vals_per_reg = sizeof(RT)/sizeof(T);

}

#endif // SIMD_SIMDDEF_HPP
