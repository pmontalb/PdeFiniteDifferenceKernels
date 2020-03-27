cmake_minimum_required(VERSION 3.14)

include_guard()

project(PdeFiniteDifferenceKernels)

include(${CMAKE_SOURCE_DIR}/cmake/All.cmake)
set(LANGUAGES_USE_CUDA ON CACHE BOOL "" FORCE)

create_cuda_library(
    NAME
        PdeFiniteDifferenceKernels
    SOURCES
        PdeFiniteDifference.cu
        PdeFiniteDifference1D.cu
        PdeFiniteDifference2D.cu
    PUBLIC_INCLUDE_DIRECTORIES
        ../CudaLightKernels .
)