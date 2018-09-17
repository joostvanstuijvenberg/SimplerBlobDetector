cmake_minimum_required(VERSION 3.12)
project(SimplerBlobDetector)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SimplerBlobDetector SimplerBlobDetector.cpp)
target_link_libraries (SimplerBlobDetector ${OpenCV_LIBS})