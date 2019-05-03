#ifndef SIMD_VARRAY_HPP
#define SIMD_VARRAY_HPP

//------------------------------------------------------------------------------

enum varray_tag {
  values,
  pointers,
  readonly,
  readwrite,
  streaming,
  scattering,
  incremental,
  decremental
};

//------------------------------------------------------------------------------

template<int tags_mask>
constexpr bool is_varray_of_values() {
  return ((tags_mask & 0x1) == 0x0);
}

template<int tags_mask>
constexpr bool is_varray_of_pointeres() {
  return ((tags_mask & 0x1) == 0x1);
}

template<int tags_mask>
constexpr bool is_varray_readonly() {
  return ((tags_mask & 0x2) == 0x0);
}

template<int tags_mask>
constexpr bool is_varray_readwrite() {
  return ((tags_mask & 0x2) == 0x2);
}

template<int tags_mask>
constexpr bool is_varray_streaming() {
  return ((tags_mask & 0x4) == 0x0);
}

template<int tags_mask>
constexpr bool is_varray_scattering() {
  return ((tags_mask & 0x4) == 0x4);
}

template<int tags_mask>
constexpr bool is_varray_incremental() {
  return ((tags_mask & 0x8) == 0x0);
}

template<int tags_mask>
constexpr bool is_varray_decremental() {
  return ((tags_mask & 0x8) == 0x8);
}

//------------------------------------------------------------------------------

template<varray_tag...>
struct varray_tags;

template<>
struct varray_tags<> {
  static constexpr int mask = 0x0;
};

struct varray_tags<varray_tag::values> {
  static constexpr int mask = 0x0;
};

struct varray_tags<varray_tag::pointers> {
  static constexpr int mask = 0x1;
};

template<>
struct varray_tags<varray_tag::readonly> {
  static constexpr int mask = 0x0;
};

template<>
struct varray_tags<varray_tag::readwrite> {
  static constexpr int mask = 0x2;
};

template<>
struct varray_tags<varray_tag::streaming> {
  static constexpr int mask = 0x0;
};

template<>
struct varray_tags<varray_tag::scattering> {
  static constexpr int mask = 0x4;
};

template<>
struct varray_tags<varray_tag::incremental> {
  static constexpr int mask = 0x0;
};

struct varray_tags<varray_tag::decremental> {
  static constexpr int mask = 0x8;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::values, tags...> {
  static constexpr int mask = (~0x1) & varray_tags<tags...>::mask;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::pointers, tags...> {
  static constexpr int mask = 0x1 | varray_tags<tags...>::mask;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::readonly, tags...> {
  static constexpr int mask = (~0x2) & varray_tags<tags...>::mask;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::readwrite, tags...> {
  static constexpr int mask = 0x2 | varray_tags<tags...>::mask;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::streaming, tags...> {
  static constexpr int mask = (~0x4) & varray_tags<tags...>::mask;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::scattering, tags...> {
  static constexpr int mask = 0x4 | varray_tags<tags...>::mask;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::incremental, tags...> {
  static constexpr int mask = (~0x8) & varray_tags<tags...>::mask;
};

template<varray_tag... tags>
struct varray_tags<varray_tag::decremental, tags...> {
  static constexpr int mask = 0x8 & varray_tags<tags...>::mask;
};

//------------------------------------------------------------------------------

template<typename T, int tags_mask = varray_tags<>::mask>
class varray {
};

//------------------------------------------------------------------------------

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readonly, varray_tag::streaming, varray_tag::incremental>::mask> {
public:
  inline varray(T const * ptr) : m_ptr(ptr) {
  }
  
  inline T const & operator[](int index) {
    return *(m_ptr+index);
  }
  
private:
  T const * m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readonly, varray_tag::streaming, varray_tag::decremental>::mask> {
public:
  inline varray(T const * ptr) : m_ptr(ptr) {
  }
  
  inline T const & operator[](int index) {
    return *(m_ptr-index);
  }
  
private:
  T const * m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readonly, varray_tag::scattering, varray_tag::incremental>::mask> {
public:
  inline varray(T const * ptr, int const * ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T const & operator[](int index) {
    return *(m_ptr+m_ip[index]);
  }
  
private:
  T const * m_ptr;
  int const * m_ip;
};

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readonly, varray_tag::scattering, varray_tag::decremental>::mask> {
public:
  inline varray(T const * ptr, int const * ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T const & operator[](int index) {
    return *(m_ptr-m_ip[index]);
  }
  
private:
  T const * m_ptr;
  int const * m_ip;
};

//------------------------------------------------------------------------------

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readwrite, varray_tag::streaming, varray_tag::incremental>::mask> {
public:
  inline varray(T* ptr) : m_ptr(ptr) {
  }
  
  inline T& operator[](int index) {
    return *(m_ptr+index);
  }
  
private:
  T* m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readwrite, varray_tag::streaming, varray_tag::decremental>::mask> {
public:
  inline varray(T* ptr) : m_ptr(ptr) {
  }
  
  inline T& operator[](int index) {
    return *(m_ptr-index);
  }
  
private:
  T* m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readwrite, varray_tag::scattering, varray_tag::incremental>::mask> {
public:
  inline varray(T* ptr, const int* ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T& operator[](int index) {
    return *(m_ptr+m_ip[index]);
  }
  
private:
  T* m_ptr;
  int const * m_ip;
};

template<typename T>
class varray<T, varray_tags<varray_tag::values, varray_tag::readwrite, varray_tag::scattering, varray_tag::decremental>::mask> {
public:
  inline varray(T* ptr, const int* ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T& operator[](int index) {
    return *(m_ptr-m_ip[index]);
  }
  
private:
  T* m_ptr;
  int const * m_ip;
};

//------------------------------------------------------------------------------

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readonly, varray_tag::streaming, varray_tag::incremental>::mask> {
public:
  inline varray(T const** ptr) : m_ptr(ptr) {
  }
  
  inline T const& operator[](int index) {
    return *(*(m_ptr+index));
  }
  
private:
  T const** m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readonly, varray_tag::streaming, varray_tag::decremental>::mask> {
public:
  inline varray(T const** ptr) : m_ptr(ptr) {
  }
  
  inline T const& operator[](int index) {
    return *(*(m_ptr-index));
  }
  
private:
  T const** m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readonly, varray_tag::scattering, varray_tag::incremental>::mask> {
public:
  inline varray(T const** ptr, int const* ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T const& operator[](int index) {
    return *(*(m_ptr+m_ip[index]));
  }
  
private:
  T const** m_ptr;
  int const* m_ip;
};

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readonly, varray_tag::scattering, varray_tag::decremental>::mask> {
public:
  inline varray(T const** ptr, int const* ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T const& operator[](int index) {
    return *(*(m_ptr-m_ip[index]));
  }
  
private:
  T const** m_ptr;
  int const* m_ip;
};

//------------------------------------------------------------------------------

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readwrite, varray_tag::streaming, varray_tag::incremental>::mask> {
public:
  inline varray(T** ptr) : m_ptr(ptr) {
  }
  
  inline T& operator[](int index) {
    return *(*(m_ptr+index));
  }
  
private:
  T** m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readwrite, varray_tag::streaming, varray_tag::decremental>::mask> {
public:
  inline varray(T** ptr) : m_ptr(ptr) {
  }
  
  inline T& operator[](int index) {
    return *(*(m_ptr-index));
  }
  
private:
  T** m_ptr;
};

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readwrite, varray_tag::scattering, varray_tag::incremental>::mask> {
public:
  inline varray(T** ptr, int const* ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T& operator[](int index) {
    return *(*(m_ptr+m_ip[index]));
  }
  
private:
  T** m_ptr;
  int const* m_ip;
};

template<typename T>
class varray<T, varray_tags<varray_tag::pointers, varray_tag::readwrite, varray_tag::scattering, varray_tag::decremental>::mask> {
public:
  inline varray(T** ptr, int const* ip) : m_ptr(ptr), m_ip(ip) {
  }
  
  inline T& operator[](int index) {
    return *(*(m_ptr-m_ip[index]));
  }
  
private:
  T** m_ptr;
};

//------------------------------------------------------------------------------

#endif // #ifndef SIMD_VARRAY_HPP
