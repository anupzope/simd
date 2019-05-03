cmake_minimum_required(VERSION 3.9.6)

project(simd VERSION 0.1.0 LANGUAGES C CXX)

find_package(catch 1.9.3)

# Installation directories
if(UNIX AND NOT APPLE)
  set(INSTALL_INCLUDE_DIR "simd/0.1.0/include")
  set(INSTALL_CONFIG_DIR "simd/0.1.0/lib/cmake/simd")
else()
  message(ERROR "Unsupported operating system")
endif()

# Interface library
add_library(simd INTERFACE)
target_include_directories(
  simd INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}/simd>
)
target_compile_features(simd INTERFACE cxx_std_11)
target_compile_definitions(simd INTERFACE USING_SIMD)

# Unit test
add_executable(testsimd src/simd_ctest.cpp)
target_link_libraries(testsimd PUBLIC simd catch)

# Generate configuration files
configure_file("modulefile.lua.in"
"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_VERSION}.lua" @ONLY)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "simd-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/simd-config.cmake"
  INSTALL_DESTINATION "${INSTALL_CONFIG_DIR}"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/simd-config-version.cmake"
  COMPATIBILITY SameMajorVersion
)

# Install configuration files
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/simd-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/simd-config-version.cmake"
  DESTINATION "${INSTALL_CONFIG_DIR}"
)

# Install header files
install(
  FILES
    simd256x86def.hpp simd256x86impl.hpp 
    simd512x86def.hpp simd512x86impl.hpp
    simddef.hpp simd.hpp simdif.hpp
  DESTINATION "${INSTALL_INCLUDE_DIR}/simd"
)

# Install modulefile
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_VERSION}.lua" 
  DESTINATION "${MODULEFILE_PREFIX}/simd"
)

# Install targets
install(TARGETS simd EXPORT simd-targets)

# Install export
install(EXPORT simd-targets DESTINATION "${INSTALL_CONFIG_DIR}")


