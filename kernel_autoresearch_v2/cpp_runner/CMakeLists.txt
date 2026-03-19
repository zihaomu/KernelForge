cmake_minimum_required(VERSION 3.18)

project(kc_gemm_runner_v2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(kc_gemm_runner
  src/main.cc
  src/gemm_reference.cc
  src/gemm_naive.cc
  src/gemm_blocked.cc
  src/gemm_blocked_pack.cc
)

target_include_directories(kc_gemm_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(kc_gemm_runner PRIVATE -O3)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(kc_gemm_runner PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(kc_gemm_runner PRIVATE KC_HAS_OPENMP=1)
else()
  target_compile_definitions(kc_gemm_runner PRIVATE KC_HAS_OPENMP=0)
endif()

