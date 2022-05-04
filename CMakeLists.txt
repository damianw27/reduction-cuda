cmake_minimum_required(VERSION 3.21)
project(mandelbrot_reduction_cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(reduction src/main.cu src/cuda_check_errors.cu)

set_target_properties(reduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(reduction PROPERTIES CUDA_ARCHITECTURES 52 61 75)
