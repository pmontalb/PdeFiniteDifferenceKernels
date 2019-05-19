cmake_minimum_required(VERSION 3.14)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc CACHE PATH "" FORCE)

include(CheckLanguage)
check_language(CUDA)
#
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(PdeFiniteDifferenceKernels)

include_directories(.)
include_directories(../CudaLightKernels)

add_library(
        PdeFiniteDifferenceKernels

        FiniteDifferenceTypes.h

        PdeFiniteDifference.cu
        PdeFiniteDifference.cuh

        PdeFiniteDifference1D.cu
        PdeFiniteDifference1D.cuh

        PdeFiniteDifference2D.cu
        PdeFiniteDifference2D.cuh
)
target_link_libraries(PdeFiniteDifferenceKernels -lcudart -lcublas -lcusolver -lcusparse)

# computational capability
target_compile_options(PdeFiniteDifferenceKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -gencode arch=compute_61,code=sm_61>)

# force 64 bits
target_compile_options(PdeFiniteDifferenceKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -m64>)

# no aliasing
target_compile_options(PdeFiniteDifferenceKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -restrict>)

# host warnings
target_compile_options(PdeFiniteDifferenceKernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --compiler-options -Wall>)