# biblioteca só de headers (chess.hpp e encoding.hpp)
add_library(chess INTERFACE)
target_include_directories(chess INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(chess INTERFACE cxx_std_23)
