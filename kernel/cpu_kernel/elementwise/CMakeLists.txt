add_library(kc_elementwise_cpu
  src/kc_elementwise.cc
)

target_include_directories(kc_elementwise_cpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/xsimd/include
)

if(KC_BUILD_TESTS)
  add_executable(kc_elementwise_accuracy_test
    test/test_elementwise_accuracy.cc
  )
  target_include_directories(kc_elementwise_accuracy_test PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/libnpy)
  target_link_libraries(kc_elementwise_accuracy_test PRIVATE kc_elementwise_cpu gtest_main)
endif()

if(KC_BUILD_BENCH)
  add_executable(kc_elementwise_bench
    perf/bench_elementwise.cc
  )
  target_link_libraries(kc_elementwise_bench PRIVATE kc_elementwise_cpu)
endif()

if(KC_BUILD_EXAMPLES)
  add_executable(kc_elementwise_example
    example/run_elementwise.cc
  )
  target_link_libraries(kc_elementwise_example PRIVATE kc_elementwise_cpu)
endif()

