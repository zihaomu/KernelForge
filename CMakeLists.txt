cmake_minimum_required(VERSION 3.18)

project(kernel_codex LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(KC_BUILD_TESTS "Build accuracy tests" ON)
option(KC_BUILD_BENCH "Build benchmarks" ON)
option(KC_BUILD_EXAMPLES "Build examples" ON)

add_subdirectory(kernel)

