
cmake_minimum_required(VERSION 3.12)

project(ANN-seq-vs-parallel)

set(CMAKE_CXX_STANDARD 17)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp")

    execute_process(
            COMMAND brew --prefix
            OUTPUT_VARIABLE HOMEBREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    include_directories(${HOMEBREW_PREFIX}/opt/libomp/include)
    link_directories(${HOMEBREW_PREFIX}/opt/libomp/lib)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Brute Force
add_executable(
        brute_force_comparing
        experiments/brute_force_comparing.cpp
)

# Partial Sort Test
add_executable(
        partial_sort_test
        test/partial_sort.cpp
)

# Hash Function Generator
add_executable(
        hash_function_generator
        test/hash_function_generator.cpp
)
