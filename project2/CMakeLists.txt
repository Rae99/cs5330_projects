# Ding, Junrui
# January 2026
#
# CMake build configuration for Project 1 (CS 5330)
# Builds imgDisplay, vidDisplay, and timeBlur using OpenCV and ONNX Runtime.

cmake_minimum_required(VERSION 3.16)
project(project2 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)


# common helper library (features, csv I/O, dir scan)
add_library(common STATIC
        src/csv_io.cpp
        src/features.cpp
        src/dir_scan.cpp
        src/ranking.cpp
        src/utils.cpp
        src/task_registry.cpp)

target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(common PUBLIC ${OpenCV_LIBS})



# --------  build_db --------
add_executable(build_db
        src/build_db.cpp)
        
target_include_directories(build_db PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(build_db PRIVATE ${OpenCV_LIBS} common)
        
# --------  query_db --------
add_executable(query_db
        src/query_db.cpp)

target_include_directories(query_db PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(query_db PRIVATE ${OpenCV_LIBS} common)

# --------  query_task5 --------
add_executable(query_task5
        src/query_task5.cpp)

target_include_directories(query_task5 PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(query_task5 PRIVATE ${OpenCV_LIBS} common)


# --------  query_task7 --------
add_executable(query_task7_grass 
        src/query_task7_grass.cpp)
target_include_directories(query_task7_grass PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(query_task7_grass common ${OpenCV_LIBS})