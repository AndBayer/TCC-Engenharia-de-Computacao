cmake_minimum_required(VERSION 3.15)
project(run_puzzles CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# alvo chess (shared/include) quando este CMake roda sem o CMake raiz
if(NOT TARGET chess)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../shared/include ${CMAKE_CURRENT_BINARY_DIR}/_chess)
endif()

include(FetchContent)
find_package(CUDAToolkit REQUIRED)

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

# run_puzzles --> busca do motor numa amostra de puzzles do Lichess, com os fontes do motor
add_executable(${PROJECT_NAME}
    run_puzzles.cpp
    puzzle_csv.cpp
    puzzle_sampling.cpp
    puzzle_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/board_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/compact_position.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/search.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/search_metrics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_service.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_cuda_kernels.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "${onnxruntime_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include"
)
target_link_directories(${PROJECT_NAME} PRIVATE "${onnxruntime_SOURCE_DIR}/lib")
target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime chess CUDA::cudart)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)

# run_positions --> busca do motor em posições lidas do stdin (position_id,fen), sem amostragem por faixa
add_executable(run_positions
    run_positions.cpp
    puzzle_csv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/board_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/compact_position.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/search.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/search_metrics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_onnx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_service.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_cuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_cuda_kernels.cu
)
target_include_directories(run_positions PRIVATE
    "${onnxruntime_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include"
)
target_link_directories(run_positions PRIVATE "${onnxruntime_SOURCE_DIR}/lib")
target_link_libraries(run_positions PRIVATE onnxruntime chess CUDA::cudart)
set_target_properties(run_positions PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)

# testes da validação (ctest)
include(CTest)
set(TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tests")
if(BUILD_TESTING)
    add_executable(puzzle_csv_tests
        ${TESTS_DIR}/unitarios/puzzle_csv_test.cpp
        puzzle_csv.cpp
    )
    target_include_directories(puzzle_csv_tests PRIVATE
        "${onnxruntime_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include"
    )
    target_link_libraries(puzzle_csv_tests PRIVATE chess)
    add_test(NAME puzzle_csv_tests COMMAND puzzle_csv_tests)

    add_executable(puzzle_validation_tests
        ${TESTS_DIR}/unitarios/puzzle_validation_test.cpp
        puzzle_validation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/board_encoder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/compact_position.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/inference_service.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/search.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../engine/src/search_metrics.cpp
    )
    target_include_directories(puzzle_validation_tests PRIVATE
        "${onnxruntime_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../engine/include"
    )
    target_link_libraries(puzzle_validation_tests PRIVATE chess)
    add_test(NAME puzzle_validation_tests COMMAND puzzle_validation_tests)
endif()
