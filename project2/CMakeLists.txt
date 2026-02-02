# Ding, Junrui
# January 2026
#
# CMake build configuration for Project 1 (CS 5330)
# Builds imgDisplay, vidDisplay, and timeBlur using OpenCV and ONNX Runtime.

cmake_minimum_required(VERSION 3.16)
project(project1 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)



# --------  build_db --------
add_executable(build_db
        src/build_db.cpp)
        
target_include_directories(build_db PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(build_db PRIVATE ${OpenCV_LIBS})
        
# --------  query_db --------
add_executable(query_db
        src/query_db.cpp)

target_include_directories(query_db PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(query_db PRIVATE ${OpenCV_LIBS})


