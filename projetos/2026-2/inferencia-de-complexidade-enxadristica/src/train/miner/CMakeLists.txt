cmake_minimum_required(VERSION 3.28)
project(miner_chess_data LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# alvo chess (shared/include) quando este CMake roda sem o CMake raiz
if(NOT TARGET chess)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../shared/include ${CMAKE_CURRENT_BINARY_DIR}/_chess)
endif()

add_executable(miner miner.cpp)
add_executable(puzzle_hasher puzzle_hasher.cpp)
add_executable(filter_dataset filter_dataset.cpp)
add_executable(build_histogram build_histogram.cpp)
add_executable(balance_dataset balance_dataset.cpp)

# só o miner e o puzzle_hasher usam a chess.hpp
target_link_libraries(miner PRIVATE chess)
target_link_libraries(puzzle_hasher PRIVATE chess)
