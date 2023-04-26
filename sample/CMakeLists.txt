cmake_minimum_required(VERSION 2.6)

project(TRTInferSample)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/../TRTInfer/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/data/cuda/cuda-11.8/cuda/include)
link_directories(/data/cuda/cuda-11.8/cuda/lib64)
# cudnn
include_directories(/data/cuda/cuda-11.8/cudnn/v8.0.4/include)
link_directories(/data/cuda/cuda-11.8/cudnn/v8.0.4/lib64)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TRTInferSample ${PROJECT_SOURCE_DIR}/main.cpp
                              ${PROJECT_SOURCE_DIR}/../TRTInfer/src/Inference.cpp)
target_link_libraries(TRTInferSample nvinfer)
target_link_libraries(TRTInferSample cudart)
target_link_libraries(TRTInferSample ${OpenCV_LIBS})

add_definitions(-O3 -pthread)