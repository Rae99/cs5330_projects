cmake_minimum_required(VERSION 3.16)
project(project1 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# -------- imgDisplay (image display app) --------
add_executable(imgDisplay
    src/imgDisplay.cpp
)
target_link_libraries(imgDisplay PRIVATE ${OpenCV_LIBS})

# -------- vid (main app) --------
add_executable(vid
        src/vidDisplay.cpp
        src/filter.cpp
        src/faceDetect.cpp
        src/effects_face.cpp
        )

target_include_directories(vid PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vid PRIVATE ${OpenCV_LIBS})

# -------- timeBlur (timing app) --------
add_executable(timeBlur
        src/timeBlur.cpp
        src/filter.cpp
)

target_include_directories(timeBlur PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(timeBlur PRIVATE ${OpenCV_LIBS})

# 

# ONNX Runtime (Homebrew)
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS
        /opt/homebrew/opt/onnxruntime/include
        /opt/homebrew/include
        /usr/local/include
        PATH_SUFFIXES onnxruntime
)

find_library(ORT_LIB onnxruntime
        PATHS
        /opt/homebrew/opt/onnxruntime/lib
        /opt/homebrew/lib
        /usr/local/lib
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR "Could not find ONNX Runtime. Check: brew install onnxruntime")
endif()

target_include_directories(vid PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(vid PRIVATE ${ORT_LIB})

