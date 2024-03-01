cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(cuda_notebooks CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(Boost 1.74.0 COMPONENTS program_options REQUIRED )
set(BOOST_INCLUDE_LIBRARIES thread filesystem system program_options)
set(BOOST_ENABLE_CMAKE ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=compute_86")

# Check for Java
find_package(Java COMPONENTS Development)
if(Java_FOUND)
    find_package(JNI REQUIRED)
    include_directories(${JNI_INCLUDE_DIRS})

    add_library(list_sort SHARED
            list-sort/ListSort.cu
            list-sort/ListSort.cuh
            list-sort/LibListSort.cu
            list-sort/LibListSort.cuh)
    target_link_libraries(list_sort PUBLIC ${JNI_LIBRARIES} Boost::program_options)
    set_target_properties(list_sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "Java not found. Skipping JNI-related targets")
endif()

add_executable(cuda_notebooks main.cu
        vector-operations/VectorOps.cu
        vector-operations/VectorOps.cuh
        image-processing/ImageProcessing.cu
        image-processing/ImageProcessing.cuh
        list-sort/ListSort.cu
        CImg.h)

target_link_libraries(cuda_notebooks PUBLIC Boost::program_options curand X11)

set_target_properties(cuda_notebooks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
