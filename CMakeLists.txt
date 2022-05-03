cmake_minimum_required(VERSION 3.10)
project(projet_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(projet_cuda main.cu)
target_link_libraries( projet_cuda ${OpenCV_LIBS} )

set_target_properties(projet_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


