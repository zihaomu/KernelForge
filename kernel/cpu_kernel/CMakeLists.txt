add_subdirectory(conv_3x3)
add_subdirectory(softmax)
add_subdirectory(elementwise)
add_subdirectory(reduce)
add_subdirectory(topk)
add_subdirectory(rmsnorm)

if(KC_BUILD_TESTS)
  # Single global googletest instance. Individual ops link against gtest_main.
  add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/googletest googletest EXCLUDE_FROM_ALL)
  enable_testing()
endif()
