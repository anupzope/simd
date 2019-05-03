#ifndef SIMD_SIMDIF_HPP
#define SIMD_SIMDIF_HPP

namespace simd {

//------------------------------------------------------------------------------
// Memory load, store, allocate/deallocate hints
//------------------------------------------------------------------------------

enum MemoryHint {
  vector = 0x00,
  scalar = 0x01,
  broadcast = 0x02,
  nontemporal = 0x04,
  unaligned = 0x08,
  reversed = 0x10
};

//------------------------------------------------------------------------------
// Memory loader
//------------------------------------------------------------------------------

template<int hint, int x>
class SimdLoader {
public:
  SimdLoader() {
    char temp[-1];
  }
};

// Loader specialization for aligned vector load of size 256 bits
template<>
class SimdLoader<simd::vector, 256>;

// Loader specialization for aligned vector load of size 512 bits
template<>
class SimdLoader<simd::vector, 512>;

// Loader specialization for unaligned vector load of size 256 bits
template<>
class SimdLoader<simd::unaligned|simd::vector, 256>;

// Loader specialization for unaligned vector load of size 512 bits
template<>
class SimdLoader<simd::unaligned|simd::vector, 512>;

// Loader specialization for unaligned broadcast load of size 256 bits
template<>
class SimdLoader<simd::unaligned|simd::broadcast, 256>;

// Loader specialization for unaligned broadcast load of size 512 bits
template<>
class SimdLoader<simd::unaligned|simd::broadcast, 512>;

//------------------------------------------------------------------------------
// Memory storer
//------------------------------------------------------------------------------

template<int hint, int x>
class SimdStorer {
public:
  SimdStorer() {
    char temp[-1];
  }
};

// Storer specialization for aligned vector store of size 256 bits
template<>
class SimdStorer<simd::vector, 256>;

// Storer specialization for aligned vector store of size 512 bits
template<>
class SimdStorer<simd::vector, 512>;

// Storer specialization for unaligned vector store of size 256 bits
template<>
class SimdStorer<simd::unaligned|simd::vector, 256>;

// Storer specialization for unaligned vector store of size 512 bits
template<>
class SimdStorer<simd::unaligned|simd::vector, 512>;

// Storer specialization for unaligned broadcast store of size 256 bits
template<>
class SimdStorer<simd::unaligned|simd::broadcast, 256>;

// Storer specialization for unaligned broadcast store of size 512 bits
template<>
class SimdStorer<simd::unaligned|simd::broadcast, 512>;

//------------------------------------------------------------------------------
// Data allocation deallocation functions
//------------------------------------------------------------------------------

template<typename ctype, int x, int hint>
ctype * allocate(size_t n);

template<typename ctype, int x, int hint>
void deallocate(ctype *p);

//------------------------------------------------------------------------------
// Prefetch hints
//------------------------------------------------------------------------------

enum PrefetchHint {
  pfnta = 0x1,
  pf0 = 0x2,
  pf1 = 0x4,
  pf2 = 0x8
};

//------------------------------------------------------------------------------
// Declaration of prefetch instructions
//------------------------------------------------------------------------------

template<int hint>
inline void prefetch(const void *mem) {
  char temp[-1];
}

template<>
void prefetch<simd::pfnta>(const void *mem);

template<>
void prefetch<simd::pf0>(const void *mem);

template<>
void prefetch<simd::pf1>(const void *mem);

template<>
void prefetch<simd::pf2>(const void *mem);

//------------------------------------------------------------------------------
// Declaration of SimdType
//------------------------------------------------------------------------------

template<typename ctype, int x>
class SimdType;

//------------------------------------------------------------------------------
// Declaration of overloaded operators for SimdType
//------------------------------------------------------------------------------

// Unary plus and minus operators

template<typename ctype, int x>
SimdType<ctype, x> operator+(const SimdType<ctype, x> &operand);

template<typename ctype, int x>
SimdType<ctype, x> operator-(const SimdType<ctype, x> &operand);

// Arithmetic operators

template<typename ctype, int x>
SimdType<ctype, x> operator+(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
SimdType<ctype, x> operator-(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
SimdType<ctype, x> operator*(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
SimdType<ctype, x> operator/(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

// This one only for integer types
template<typename ctype, int x>
SimdType<ctype, x> operator%(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

// Increment/decrement operators

template<typename ctype, int x>
SimdType<ctype, x>& operator++(SimdType<ctype, x> &operand);

template<typename ctype, int x>
SimdType<ctype, x> operator++(SimdType<ctype, x> &operand, int);

template<typename ctype, int x>
SimdType<ctype, x>& operator--(SimdType<ctype, x> &operand);

template<typename ctype, int x>
SimdType<ctype, x> operator--(SimdType<ctype, x> &operand, int);

// Comparison operators

template<typename ctype, int x>
bool operator==(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
bool operator!=(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
bool operator>(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
bool operator<(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
bool operator>=(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
bool operator<=(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

// Bitwise operators - only for integer types

template<typename ctype, int x>
SimdType<ctype, x> operator&(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
SimdType<ctype, x> operator|(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
SimdType<ctype, x> operator^(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs);

template<typename ctype, int x>
SimdType<ctype, x> operator~(const SimdType<ctype, x> &operand);

template<typename ctype, int x>
SimdType<ctype, x> operator<<(const SimdType<ctype, x> &operand, unsigned int num);

template<typename ctype, int x>
SimdType<ctype, x> operator>>(const SimdType<ctype, x> &operand, unsigned int num);

//------------------------------------------------------------------------------
// Generic implementation of SimdType
//------------------------------------------------------------------------------

template<typename ctype, int x>
class SimdType {
public:
  static const size_t width = simd::value<ctype, x>::width;
  
public:
  inline SimdType() {
    simd::value<ctype, x> temp;
  }
  
  inline SimdType(const simd::value<ctype, x> &value) : m_value(value) {}
  
  inline operator typename simd::value<ctype, x>::reg_t&() { return m_value.reg; }
  
  inline operator const typename simd::value<ctype, x>::reg_t&() const { return m_value; }
  
  inline SimdType<ctype, x>& operator=(const simd::value<ctype, x> &value) {
    m_value.reg = value.reg;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator+=(const SimdType<ctype, x> &rhs) {
    *this = *this + rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator-=(const SimdType<ctype, x> &rhs) {
    *this = *this - rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator*=(const SimdType<ctype, x> &rhs) {
    *this = *this * rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator/=(const SimdType<ctype, x> &rhs) {
    *this = *this / rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator%=(const SimdType<ctype, x> &rhs) {
    *this = *this % rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator&=(const SimdType<ctype, x> &rhs) {
    *this = *this & rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator|=(const SimdType<ctype, x> &rhs) {
    *this = *this | rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator^=(const SimdType<ctype, x> &rhs) {
    *this = *this ^ rhs;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator<<=(unsigned int num) {
    *this = *this << num;
    return *this;
  }
  
  inline SimdType<ctype, x>& operator>>=(unsigned int num) {
    *this = *this >> num;
    return *this;
  }
  
  template<int hint>
  inline void load(const ctype *mem) {
    m_value = SimdLoader<hint, x>::load(mem);
  }
  
  template<int hint>
  inline void store(ctype *mem) {
    SimdStorer<hint, x>::store(m_value, mem);
  }
  
private:
  simd::value<ctype, x>::reg_t m_value;
};

//------------------------------------------------------------------------------
// Default implementation of overloaded operators for SIMD type
//------------------------------------------------------------------------------

template<typename ctype, int x>
SimdType<ctype, x> operator+(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs) {
  static_assert(false, "operator+ not defined");
  //char temp[-1];
}

template<typename ctype, int x>
SimdType<ctype, x> operator-(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs) {
  static_assert(false, "operator- not defined");
  //char temp[-1];
}

template<typename ctype, int x>
SimdType<ctype, x> operator*(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs) {
  static_assert(false, "operator* not defined");
  //char temp[-1];
}

template<typename ctype, int x>
SimdType<ctype, x> operator/(const SimdType<ctype, x> &lhs, const SimdType<ctype, x> &rhs) {
  static_assert(false, "operator/ not defined");
  //char temp[-1];
}

//------------------------------------------------------------------------------
// SIMD traits
//------------------------------------------------------------------------------

template<typename ctype, int x=SIMD_MAX_BIT_WIDTH>
struct traits {
  typedef SimdType<ctype, x> type;
  static const size_t width = type::width;
};

}

#endif // SIMD_SIMDIF_HPP
