cmake_minimum_required(VERSION 3.15)
project(sinsum VERSION 1.0 LANGUAGES CXX)

option(USE_FLOAT "Use float instead of double" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_FLOAT)
    add_compile_definitions(USE_FLOAT)
    message(STATUS "Using float type")
else()
    message(STATUS "Using double type (default)")
endif()

add_executable(sinsum sinsum.cpp)