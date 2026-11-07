cmake_minimum_required(VERSION 3.20)
project(pool_ball_vision VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /EHsc)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# Locate OpenCV 4.12.0. Prefer an existing OpenCV_DIR/OpenCVConfig.cmake, then
# the adjacent TIPS project, then the vcpkg installation used on this machine.
if(NOT OpenCV_DIR)
    set(_opencv_candidates
        "$ENV{OpenCV_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../OpenCVExample"
        "C:/vcpkg/installed/x64-windows/share/opencv4"
    )
    foreach(_candidate IN LISTS _opencv_candidates)
        if(_candidate AND EXISTS "${_candidate}/OpenCVConfig.cmake")
            set(OpenCV_DIR "${_candidate}" CACHE PATH "OpenCV config directory" FORCE)
            break()
        endif()
    endforeach()
endif()

find_package(OpenCV 4.12 REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")
if(NOT OpenCV_VERSION VERSION_EQUAL "4.12.0")
    message(WARNING "Expected OpenCV 4.12.0, found ${OpenCV_VERSION}")
endif()

add_executable(pool_ball_vision myApplication.cpp)

target_include_directories(pool_ball_vision PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pool_ball_vision PRIVATE ${OpenCV_LIBS})
