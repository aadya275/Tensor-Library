cmake_minimum_required(VERSION 3.16)
project(MiniTensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(MT_NATIVE "Use -march=native" ON)
find_package(OpenMP)

add_library(minitensor INTERFACE)
target_include_directories(minitensor INTERFACE include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(minitensor INTERFACE OpenMP::OpenMP_CXX)
endif()

target_compile_options(minitensor INTERFACE -O3)
if(MT_NATIVE)
    target_compile_options(minitensor INTERFACE -march=native)
endif()

add_executable(tests tests/test_tensor.cpp)
target_link_libraries(tests minitensor)

add_executable(demo examples/demo.cpp)
target_link_libraries(demo minitensor)

add_executable(bench benchmarks/bench.cpp)
target_link_libraries(bench minitensor)

enable_testing()
add_test(NAME all_tests COMMAND tests)
