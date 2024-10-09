project(GraphActionRule CXX)
cmake_minimum_required(VERSION 3.16)
add_compile_options(-Wall -Wextra -g -fexceptions -std=c++17 -mfma -march=native -pthread)

# include_directories(include)

add_subdirectory(lib/graph)
add_subdirectory(lib/matching)
add_subdirectory(lib/utility)

set(CMAKE_C_COMPILER "/usr/bin/clang-18")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-18")
set(CMAKE_LINKER "/usr/bin/lld-18")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)