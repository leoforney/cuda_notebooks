cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(cuda_notebooks CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

find_package(Qt5 COMPONENTS Widgets Multimedia REQUIRED)

#[[set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)]]

get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )

include(FetchContent)

find_package(Thrust REQUIRED CONFIG)

find_package(Boost 1.74.0 COMPONENTS program_options REQUIRED )
set(BOOST_INCLUDE_LIBRARIES thread filesystem system program_options)
set(BOOST_ENABLE_CMAKE ON)

# Explicitly add -fPIC for CUDA compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=compute_86")

# Continue with the rest of your setup...
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

FetchContent_Declare(
        libsndfile
        GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
        GIT_TAG        master
)

# Make libsndfile available for building
FetchContent_MakeAvailable(libsndfile)

add_executable(cuda_notebooks main.cpp
        vector-operations/VectorOps.cu
        vector-operations/VectorOps.cuh
        image-processing/ImageProcessing.cu
        image-processing/ImageProcessing.cuh
        list-sort/ListSort.cu
        CImg.h
        matrix-operations/MatrixOps.cu
        matrix-operations/MatrixOps.cuh
        audio-visualizer/AudioVisualizer.cpp
        audio-visualizer/AudioVisualizer.h
        audio-visualizer/VisualizerUI.h
        audio-visualizer/VisualizerConstant.h
        audio-visualizer/AudioProcessor.cuh)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(cuda_notebooks PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} sndfile Boost::program_options curand X11 cufft cudart Qt5::Widgets Qt5::Multimedia)
set_target_properties(cuda_notebooks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
