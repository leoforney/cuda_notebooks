cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(list_sort CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=compute_86")

find_package(Thrust REQUIRED)
thrust_create_target(Thrust FROM_OPTIONS)
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

add_library(list_sort SHARED LibListSort.cu LibListSort.cuh)
target_link_libraries(list_sort PUBLIC ${JNI_LIBRARIES} Thrust::Thrust)
set_target_properties(list_sort PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(list_sort_exec LibListSort.cu LibListSort.cuh)
target_link_libraries(list_sort_exec PUBLIC ${JNI_LIBRARIES} Thrust::Thrust)
set_target_properties(list_sort_exec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)