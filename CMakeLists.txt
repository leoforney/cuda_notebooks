cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(cuda_notebooks CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Find CUDA
find_package(CUDA REQUIRED)

include(FetchContent)

set(BOOST_INCLUDE_LIBRARIES thread filesystem system program_options)
set(BOOST_ENABLE_CMAKE ON)

include(FetchContent)
FetchContent_Declare(
        Boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.84.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Boost)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=compute_86")

add_executable(cuda_notebooks main.cu
        vector-operations/VectorOps.cu
        vector-operations/VectorOps.cuh)

target_link_libraries(cuda_notebooks PUBLIC Boost::program_options curand)

set_target_properties(cuda_notebooks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
