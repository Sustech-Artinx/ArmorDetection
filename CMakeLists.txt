cmake_minimum_required(VERSION 3.0)
project(ArmorDetection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/3rd-party/tclap/include)

set(SOURCE_FILES
        src/main.cpp
        src/Detector.cpp
        src/Detector.h
        src/ArmorDetectionApp.cpp
        src/ArmorDetectionApp.h
        src/utils.cpp
        src/utils.h
        src/Smoother.cpp
        src/Smoother.h
        src/FixedQueue.h)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
