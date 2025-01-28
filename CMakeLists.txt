cmake_minimum_required(VERSION 3.10)
project(MultiThreadConvolution)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(convolution multi_thread_convolution.cpp)
target_link_libraries(convolution PRIVATE ${OpenCV_LIBS} Threads::Threads)