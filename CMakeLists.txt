cmake_minimum_required(VERSION 3.0.0)
project(Convolutions VERSION 0.1.0)

include(CTest)
enable_testing()

set(SOURCE_FILES 
main.cpp
naive_conv.cpp 
threaded_conv.cpp)

find_package(Threads REQUIRED)

add_executable(Convolutions ${SOURCE_FILES})

target_link_libraries(Convolutions PRIVATE Threads::Threads)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
