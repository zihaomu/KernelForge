add_library(kc_softmax_cpu
  src/kc_softmax.cc
)

target_include_directories(kc_softmax_cpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/xsimd/include
)

if(KC_BUILD_TESTS)
  add_executable(kc_softmax_accuracy_test
    test/test_softmax_accuracy.cc
  )
  target_include_directories(kc_softmax_accuracy_test
    PRIVATE
      ${PROJECT_SOURCE_DIR}/3rdparty/libnpy
  )
  target_link_libraries(kc_softmax_accuracy_test PRIVATE kc_softmax_cpu gtest_main)
endif()

if(KC_BUILD_BENCH)
  add_executable(kc_softmax_bench
    perf/bench_softmax.cc
  )
  target_link_libraries(kc_softmax_bench PRIVATE kc_softmax_cpu)
endif()

if(KC_BUILD_EXAMPLES)
  add_executable(kc_softmax_example
    example/run_softmax.cc
  )
  target_link_libraries(kc_softmax_example PRIVATE kc_softmax_cpu)
endif()
