
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
        benchmarks/brute_force_comparing.cpp
)

# Shared ANN implementation and dataset loaders.
add_library(ann STATIC
        src/ann.cpp
        src/fvecs_loader.cpp
        src/ground_truth_loader.cpp
)
target_include_directories(ann PUBLIC include)

# Sequential Hash Function Generator
add_executable(
        hash_function_generator
        benchmarks/sequential_hash_function_generator.cpp
)

# Parallel Hash Function Generator
add_executable(
        parallel_hash_function_generator
        benchmarks/parallel_hash_function_generator.cpp
)

target_link_libraries(hash_function_generator PRIVATE ann)
target_link_libraries(parallel_hash_function_generator PRIVATE ann)

target_compile_definitions(
        parallel_hash_function_generator
        PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
        hash_function_generator
        PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include(CTest)
if(BUILD_TESTING)
    add_executable(
        fvecs_loader_test
            test/fvecs_loader_test.cpp
        src/fvecs_loader.cpp
    )
    target_include_directories(fvecs_loader_test PRIVATE include)
    add_test(NAME fvecs_loader_test COMMAND fvecs_loader_test)
    add_executable(
            ann_correctness_test
            test/search_correctness_test.cpp
    )

    target_link_libraries(ann_correctness_test PRIVATE ann)

    target_compile_definitions(
            ann_correctness_test
            PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(NAME ann_correctness_test COMMAND ann_correctness_test)
endif()
