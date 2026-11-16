cmake_minimum_required(VERSION 3.18)
project(decoder LANGUAGES CXX CUDA)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# gera uma DLL (SHARED)
add_library(${PROJECT_NAME} SHARED decoder.cu)

# decoder.dll sem prefixo lib (nome que o train.py procura) e runtimes do MSVC e do CUDA em DLL
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    CUDA_SEPARABLE_COMPILATION ON
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    CUDA_RUNTIME_LIBRARY "Shared"
)
