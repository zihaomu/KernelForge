add_library(kc_rmsnorm_cpu
  src/kc_rmsnorm.cc
)

target_include_directories(kc_rmsnorm_cpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rdparty/xsimd/include
)

if(KC_BUILD_TESTS)
  add_executable(kc_rmsnorm_accuracy_test
    test/test_rmsnorm_accuracy.cc
  )
  target_include_directories(kc_rmsnorm_accuracy_test PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/libnpy)
  target_link_libraries(kc_rmsnorm_accuracy_test PRIVATE kc_rmsnorm_cpu gtest_main)
endif()

if(KC_BUILD_BENCH)
  add_executable(kc_rmsnorm_bench
    perf/bench_rmsnorm.cc
  )
  target_link_libraries(kc_rmsnorm_bench PRIVATE kc_rmsnorm_cpu)
endif()

if(KC_BUILD_EXAMPLES)
  add_executable(kc_rmsnorm_example
    example/run_rmsnorm.cc
  )
  target_link_libraries(kc_rmsnorm_example PRIVATE kc_rmsnorm_cpu)
endif()

