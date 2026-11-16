cmake_minimum_required(VERSION 3.28)
project(tfird LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# executáveis e a decoder.dll vão para bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
# geradores com várias configurações (Visual Studio) usam a mesma bin/ em Debug e Release
foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${cfg}" cfg_up)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_up} "${CMAKE_SOURCE_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg_up} "${CMAKE_SOURCE_DIR}/bin")
endforeach()

enable_testing()

# chess.hpp e encoding.hpp, usados no projeto
add_subdirectory(src/shared/include)

add_subdirectory(src/train/miner)
add_subdirectory(src/train/decoder)
add_subdirectory(src/train/benchmark)

add_subdirectory(src/engine)
add_subdirectory(src/validation)
