cmake_minimum_required(VERSION 2.8.12)

project(pinball-mask)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pinball-mask pinball-mask.cpp)
target_link_libraries(pinball-mask ${OpenCV_LIBS})
