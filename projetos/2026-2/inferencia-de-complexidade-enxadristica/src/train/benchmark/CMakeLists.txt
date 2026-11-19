cmake_minimum_required(VERSION 3.15)
project(model_benchmark CXX)

# C++23, como o resto do projeto
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET chess)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include ${CMAKE_CURRENT_BINARY_DIR}/_chess)
endif()

include(FetchContent)

# baixa o ONNX Runtime com suporte a GPU
set(ORT_VERSION "1.24.1" CACHE STRING "ONNX Runtime version to download")
set(ORT_CUDA_VARIANT "cuda13" CACHE STRING "CUDA variant: 'cuda13'")
message(STATUS "Baixando/Configurando ONNX Runtime versão: ${ORT_VERSION} (GPU ${ORT_CUDA_VARIANT})")

if(ORT_CUDA_VARIANT STREQUAL "cuda13")
    set(ORT_PKG "onnxruntime-win-x64-gpu_cuda13-${ORT_VERSION}")
else()
    set(ORT_PKG "onnxruntime-win-x64-gpu-${ORT_VERSION}")
endif()

FetchContent_Declare(
    onnxruntime
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PKG}.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

# encodeBoard do motor --> o benchmark codifica o tabuleiro igual ao motor
add_executable(${PROJECT_NAME} model_benchmark.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../engine/src/board_encoder.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE "${onnxruntime_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../engine/include")
target_link_directories(${PROJECT_NAME} PRIVATE "${onnxruntime_SOURCE_DIR}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime chess)

# depois do build, copia as DLLs do ONNX Runtime para a pasta do executável
# a versão com GPU precisa também das DLLs dos provedores (shared e CUDA)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}>
    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}>
    
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}>
)
