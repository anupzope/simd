#ifndef SIMD_HPP
#define SIMD_HPP

//#include <common.hpp>
//#include <coi.hpp>


/**
 * Description:
 * This header file detects features of compilation target
 * 
 * Supported compilers:
 *   1. GCC on Linux
 *   2. ICC on Linux and Windows
 * 
 * Supported Architectures:
 *   1. x86
 *   2. x86_64
 */

#ifndef _COI_H_
#define _COI_H_

// 1. Detect compiler
// 2. Detect operating system
// 3. Detect instruction set architecture
#if defined(__ICC) || defined(__INTEL_COMPILER) || defined(__ICL)
  #define SIMD_COMPILER_INTEL
  #pragma message("SIMD_COMPILER_INTEL")
  
  // Detect GNU compiler compliance
  #if defined(__GNUC__) || defined(__GNUG__)
    #define SIMD_COMPILER_GCC
    #pragma message("SIMD_COMPILER_GCC")
  #endif
  
  // Detect operating system
  #if defined(_WIN64)
    #define SIMD_OS_WINDOWS64
    #pragma message("SIMD_OS_WINDOWS64")
  #elif defined(_WIN32)
    #define SIMD_OS_WINDOWS32
    #pragma message("SIMD_OS_WINDOWS32")
  #elif defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix) || defined(__gnu_linux__)
    #if defined(__linux) || defined(__linux__) || defined(linux)
      #define SIMD_OS_LINUX
      #pragma message("SIMD_OS_LINUX")
    #endif
    #if defined(__unix) || defined(__unix__) || defined(unix)
      #define SIMD_OS_UNIX
      #pragma message("SIMD_OS_UNIX")
    #endif
    #if defined(__gnu_linux__)
      #define SIMD_OS_GNULINUX
      #pragma message("SIMD_OS_GNULINUX")
    #endif
  #else
    #error "Unsupported operating system"
  #endif
  
  // Detect instruction set (architecture)
  #if defined(_M_X64) || defined(_M_AMD64) || defined(__amd64) || defined(__amd64__) || defined(__x86_64) || defined(__x86_64__)
    #define SIMD_ARCH_X86_64
    #pragma message("SIMD_ARCH_X86_64")
  #elif defined(_M_IX86) || defined(__i386) || defined(__i386__) || defined(i386)
    #define SIMD_ARCH_X86
    #pragma message("SIMD_ARCH_X86")
  #else
    #error "Unsupported architecture"
  #endif
  
  // Define common macros
  #if defined(OS_WINDOWS32) || defined(OS_WINDOWS64)
    #define SIMD_FORCE_INLINE inline __declspec((always_inline))
  #else
    #define SIMD_FORCE_INLINE inline __attribute__((always_inline))
  #endif
  
#elif defined(__GNUC__) || defined(__GNUG__)
  #define SIMD_COMPILER_GCC
  #pragma message("SIMD_COMPILER_GCC")
  
  // Detect operating system
  #if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix) || defined(__gnu_linux__)
    #if defined(__linux) || defined(__linux__) || defined(linux)
      #define SIMD_OS_LINUX
      #pragma message("SIMD_OS_LINUX")
    #endif
    #if defined(__unix) || defined(__unix__) || defined(unix)
      #define SIMD_OS_UNIX
      #pragma message("SIMD_OS_UNIX")
    #endif
    #if defined(__gnu_linux__)
      #define SIMD_OS_GNULINUX
      #pragma message("SIMD_OS_GNULINUX")
    #endif
  #else
    #error "Unsupported operating system"
  #endif
  
  // Detect instruction set (architecture
  #if defined(__amd64) || defined(__amd64__) || defined(__x86_64) || defined(__x86_64__)
    #define SIMD_ARCH_X86_64
    #pragma message("SIMD_ARCH_X86_64")
  #elif defined(__i386) || defined(__i386__) || defined(i386)
    #define SIMD_ARCH_X86
    #pragma message("SIMD_ARCH_X86")
  #else
    #error "Unsupported architecture"
  #endif
  
  // Define common macros
  #define SIMD_FORCE_INLINE inline __attribute__((always_inline))
  
#else
  #error "Unsupported compiler"
#endif




// Macro to check size of various types at compile time
// TODO: This macro can be replaced with static_assert() in C++11
//#define CHECK_TYPE_SIZE(type, expected_size, id) \
//bool check_type_size_ ##type ##expected_size ##id() { \
//  char temp[1-2*(sizeof(type) != expected_size)]; \
//  return (temp != 0); \
//}

// Detect highest level of SIMD instruction set
#if defined(SIMD_COMPILER_INTEL) || defined(SIMD_COMPILER_GCC)
  #if defined(SIMD_ARCH_X86) || defined(SIMD_ARCH_X86_64)
    
    #if defined(__AVX512F__)
      #define SIMD_AVX512F
      #pragma message("SIMD_AVX512F")
    #endif
    
    // defined by intel-15 compiler when building code 
    // for native execution
    #if defined(__KNC__) || defined(__MIC__)
      #define SIMD_KNC
      #pragma message("SIMD_KNC")
    #endif
    
    // defined by intel-15 compiler when building code 
    // for host which uses offload model
    #if defined(__INTEL_OFFLOAD)
      #define SIMD_OFFLOAD_KNC
      #pragma message("SIMD_OFFLOAD_KNC")
    #endif
    
    #if defined(__AVX2__)
      #define SIMD_AVX2
      #pragma message("SIMD_AVX2")
    #endif
    
    #if defined(__AVX__)
      #define SIMD_AVX
      #pragma message("SIMD_AVX")
    #endif
    
    #if defined(__SSE4_2__)
      #define SIMD_SSE4_2
      #pragma message("SIMD_SSE4_2")
    #endif
    
    #if defined(__SSE4_1__)
      #define SIMD_SSE4_1
      #pragma message("SIMD_SSE4_1")
    #endif
    
    #if defined(__SSSE3__)
      #define SIMD_SSSE3
      #pragma message("SIMD_SSSE3")
    #endif
    
    #if defined(__SSE3__)
      #define SIMD_SSE3
      #pragma message("SIMD_SSE3")
    #endif
    
    #if defined(__SSE2__)
      #define SIMD_SSE2
      #pragma message("SIMD_SSE2")
    #endif
    
    #if defined(__SSE__)
      #define SIMD_SSE
      #pragma message("SIMD_SSE")
    #endif
    
    #if defined(__MMX__)
      #define SIMD_MMX
      #pragma message("SIMD_MMX")
    #endif
    
    #define SIMD256 (defined(SIMD_AVX) || defined(SIMD_AVX2))
    #define SIMD512 (defined(SIMD_AVX512F) || defined(SIMD_KNC))
    //#define SIMD512OFFLOAD (defined(SIMD_OFFLOAD_KNC) && !(defined(SIMD_AVX512F) || defined(SIMD_KNC)))
    #define SIMD512OFFLOAD (defined(SIMD_OFFLOAD_KNC))
    
    #if SIMD256
      #pragma message("COMPILING 256")
    #endif
    
    #if SIMD512OFFLOAD
      #pragma message("COMPILING FOR OFFLOAD")
    #endif
    
    #if SIMD512
      #pragma message("COMPILING 512")
    #endif
    
    #if SIMD512 || SIMD512OFFLOAD
      #define SIMD_MAX_WIDTH 64
      #pragma message("SIMD_MAX_WIDTH = 64")
    #elif SIMD256
      #define SIMD_MAX_WIDTH 32
      #pragma message("SIMD_MAX_WIDTH = 32")
    #endif
    
    #define SIMD_MAX_BIT_WIDTH (SIMD_MAX_WIDTH*8)
    
    #if defined(SIMD_OS_WINDOWS64) || defined(SIMD_OS_WINDOWS32)
      #define SIMD_WIN_ALIGN __declspec(align(SIMD_MAX_WIDTH))
      #define SIMD_WIN_PTR_ALIGN __declspec(align_value(SIMD_MAX_WIDTH))
    #endif
    
    #if defined(SIMD_OS_LINUX) || defined(SIMD_OS_UNIX) || defined(SIMD_OS_GNULINUX)
      #define SIMD_LINUX_ALIGN __attribute__((aligned(SIMD_MAX_WIDTH)))
      #define SIMD_LINUX_PTR_ALIGN __attribute__((align_value(SIMD_MAX_WIDTH)))
    #endif
    
    #if !defined(SIMD_WIN_ALIGN) && !defined(SIMD_LINUX_ALIGN)
      #error "Must define alignment clause for either Windows OS or Linux OS"
    #endif
    
    #if defined(SIMD_WIN_ALIGN) && defined(SIMD_LINUX_ALIGN)
      #error "Cannot define alignment clause for both Windows OS and Linux OS"
    #endif
    
    #if !defined(SIMD_WIN_PTR_ALIGN) && !defined(SIMD_LINUX_PTR_ALIGN)
      #error "Must define pointer alignment clause for either Windows OS or Linux OS"
    #endif
    
    #if defined(SIMD_WIN_PTR_ALIGN) && defined(SIMD_LINUX_PTR_ALIGN)
      #error "Cannot define pointer alignment clause for both Windows OS and Linux OS"
    #endif
    
    #if !defined(SIMD_WIN_ALIGN)
      #define SIMD_WIN_ALIGN
    #endif
    
    #if !defined(SIMD_LINUX_ALIGN)
      #define SIMD_LINUX_ALIGN
    #endif
    
    #if !defined(SIMD_WIN_PTR_ALIGN)
      #define SIMD_WIN_PTR_ALIGN
    #endif
    
    #if !defined(SIMD_LINUX_PTR_ALIGN)
      #define SIMD_LINUX_PTR_ALIGN
    #endif
    
    #if defined(SIMD_OFFLOAD_KNC)
      #pragma offload_attribute(push, target(mic))
    #endif
    #include <simddef.hpp>
    #include <simd256x86def.hpp>
    #include <simd512x86def.hpp>
    //#include <simdif.h>
    //#include <simd256x86impl.h>
    //#include <simd512x86impl.h>
    #if defined(SIMD_OFFLOAD_KNC)
      #pragma offload_attribute(pop)
    #endif
    
  #else
    #error "Could not detect SIMD instruction set: Unsupported architecture"
  #endif
#else
  #error "Could not detect SIMD instruction set: Unsupported compiler"
#endif

#endif // SIMD_HPP
