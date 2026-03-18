add_library(kc_topk_cpu
  src/kc_topk.cc
)

target_include_directories(kc_topk_cpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(KC_BUILD_TESTS)
  add_executable(kc_topk_accuracy_test
    test/test_topk_accuracy.cc
  )
  target_include_directories(kc_topk_accuracy_test PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/libnpy)
  target_link_libraries(kc_topk_accuracy_test PRIVATE kc_topk_cpu gtest_main)
endif()

if(KC_BUILD_BENCH)
  add_executable(kc_topk_bench
    perf/bench_topk.cc
  )
  target_link_libraries(kc_topk_bench PRIVATE kc_topk_cpu)
endif()

if(KC_BUILD_EXAMPLES)
  add_executable(kc_topk_example
    example/run_topk.cc
  )
  target_link_libraries(kc_topk_example PRIVATE kc_topk_cpu)
endif()

