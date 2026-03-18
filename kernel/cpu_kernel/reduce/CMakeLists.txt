add_library(kc_reduce_cpu
  src/kc_reduce.cc
)

target_include_directories(kc_reduce_cpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/xsimd/include
)

if(KC_BUILD_TESTS)
  add_executable(kc_reduce_accuracy_test
    test/test_reduce_accuracy.cc
  )
  target_include_directories(kc_reduce_accuracy_test PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/libnpy)
  target_link_libraries(kc_reduce_accuracy_test PRIVATE kc_reduce_cpu gtest_main)
endif()

if(KC_BUILD_BENCH)
  add_executable(kc_reduce_bench
    perf/bench_reduce.cc
  )
  target_link_libraries(kc_reduce_bench PRIVATE kc_reduce_cpu)
endif()

if(KC_BUILD_EXAMPLES)
  add_executable(kc_reduce_example
    example/run_reduce.cc
  )
  target_link_libraries(kc_reduce_example PRIVATE kc_reduce_cpu)
endif()

