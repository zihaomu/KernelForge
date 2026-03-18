add_library(kc_conv3x3
  src/kc_conv3x3.cc
)

target_include_directories(kc_conv3x3
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/xsimd/include
)

if(KC_BUILD_TESTS)
  add_executable(kc_conv3x3_accuracy_test
    test/test_conv3x3_accuracy.cc
  )
  target_include_directories(kc_conv3x3_accuracy_test
    PRIVATE
      ${PROJECT_SOURCE_DIR}/3rdparty/libnpy
  )
  target_link_libraries(kc_conv3x3_accuracy_test PRIVATE kc_conv3x3 gtest_main)
endif()

if(KC_BUILD_BENCH)
  add_executable(kc_conv3x3_bench
    perf/bench_conv3x3.cc
  )
  target_include_directories(kc_conv3x3_bench PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/libnpy)
  target_link_libraries(kc_conv3x3_bench PRIVATE kc_conv3x3)
endif()

if(KC_BUILD_EXAMPLES)
  add_executable(kc_conv3x3_example
    example/run_conv3x3.cc
  )
  target_link_libraries(kc_conv3x3_example PRIVATE kc_conv3x3)
endif()
