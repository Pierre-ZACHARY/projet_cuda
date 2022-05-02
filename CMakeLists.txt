cmake_minimum_required(VERSION 3.10)
project(projet_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(projet_cuda main.cu)

set_target_properties(projet_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
