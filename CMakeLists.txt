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

# Codice condiviso: algoritmo ANN e caricamento dei dataset.
add_library(ann STATIC src/ann.cpp src/dataset.cpp)
target_include_directories(ann PUBLIC include)

# Benchmark: tempi, scalabilita', statistiche e recall sul dataset reale.
add_executable(brute_force_comparing benchmarks/brute_force_comparing.cpp)
add_executable(strong_scaling benchmarks/strong_scaling.cpp)
add_executable(weak_scaling benchmarks/weak_scaling.cpp)

foreach(target brute_force_comparing strong_scaling weak_scaling)
    target_link_libraries(${target} PRIVATE ann)
    target_compile_definitions(${target} PRIVATE PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

# Test di correttezza su piccoli dataset controllati.
include(CTest)
if(BUILD_TESTING)
    add_executable(fvecs_loader_test test/fvecs_loader_test.cpp)
    add_executable(optimized_search_test test/optimized_search_test.cpp)

    foreach(target fvecs_loader_test optimized_search_test)
        target_link_libraries(${target} PRIVATE ann)
        add_test(NAME ${target} COMMAND ${target})
    endforeach()
endif()
