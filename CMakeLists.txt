cmake_minimum_required(VERSION 3.16)
project(pool_ball_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

add_executable(pool_ball_vision myApplication.cpp)
target_link_libraries(pool_ball_vision PRIVATE ${OpenCV_LIBS})
target_include_directories(pool_ball_vision PRIVATE ${OpenCV_INCLUDE_DIRS})

