cmake_minimum_required(VERSION 3.15)
project(tfird CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# alvo chess (shared/include) quando este CMake roda sem o CMake raiz
if(NOT TARGET chess)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../shared/include ${CMAKE_CURRENT_BINARY_DIR}/_chess)
endif()

include(FetchContent)
include_directories(include)
find_package(CUDAToolkit REQUIRED)

# ONNX Runtime --> mesmo pacote do benchmark; com o mesmo nome, o FetchContent baixa uma vez só
set(ORT_VERSION "1.24.1" CACHE STRING "ONNX Runtime version to download")
set(ORT_CUDA_VARIANT "cuda13" CACHE STRING "CUDA variant: 'cuda13'")

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

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/uci.cpp
    src/board_encoder.cpp
    src/compact_position.cpp
    src/search.cpp
    src/search_metrics.cpp
    src/inference_onnx.cpp
    src/inference_service.cpp
    src/inference_cuda.cpp
    src/inference_cuda_kernels.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${onnxruntime_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_directories(${PROJECT_NAME} PRIVATE "${onnxruntime_SOURCE_DIR}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime chess CUDA::cudart)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)

# copia as DLLs do ONNX Runtime e do provedor CUDA para a pasta do executável
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_cuda.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# testes do motor (ctest)
include(CTest)
set(TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tests")
if(BUILD_TESTING)
    add_executable(tfird_cuda_encoding_tests
        ${TESTS_DIR}/integracao/cuda_encoding_test.cpp
        src/board_encoder.cpp
        src/compact_position.cpp
        src/inference_cuda_kernels.cu
    )
    target_include_directories(tfird_cuda_encoding_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(tfird_cuda_encoding_tests PRIVATE chess CUDA::cudart)
    set_target_properties(tfird_cuda_encoding_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )
    add_test(NAME tfird_cuda_encoding_tests COMMAND tfird_cuda_encoding_tests)
    # código 77 --> teste pulado (máquina sem GPU)
    set_tests_properties(tfird_cuda_encoding_tests PROPERTIES SKIP_RETURN_CODE 77)

    # sem add_test, pois precisa do caminho do modelo .onnx; roda à mão
    add_executable(tfird_cuda_inference_tests
        ${TESTS_DIR}/integracao/cuda_inference_test.cpp
        src/board_encoder.cpp
        src/compact_position.cpp
        src/inference_onnx.cpp
        src/inference_service.cpp
        src/inference_cuda.cpp
        src/inference_cuda_kernels.cu
    )
    target_include_directories(tfird_cuda_inference_tests PRIVATE
        "${onnxruntime_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_directories(tfird_cuda_inference_tests PRIVATE
        "${onnxruntime_SOURCE_DIR}/lib"
    )
    target_link_libraries(tfird_cuda_inference_tests PRIVATE
        onnxruntime chess CUDA::cudart
    )
    set_target_properties(tfird_cuda_inference_tests PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION OFF
    )

    add_executable(tfird_search_tests
        ${TESTS_DIR}/unitarios/search_test.cpp
        src/board_encoder.cpp
        src/compact_position.cpp
        src/inference_service.cpp
        src/search.cpp
        src/search_metrics.cpp
    )
    target_include_directories(tfird_search_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(tfird_search_tests PRIVATE chess)
    add_test(NAME tfird_search_tests COMMAND tfird_search_tests)
endif()
