cmake_minimum_required(VERSION 3.14)
project(DEEPFIN VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python3")


find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


find_package(pybind11 CONFIG REQUIRED)


pybind11_add_module(calculate_data src/dataGenius/calculate_data.cpp)

add_custom_command(TARGET calculate_data POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:calculate_data>
    "${CMAKE_SOURCE_DIR}/src/dataGenius/calculate_data.so"
)