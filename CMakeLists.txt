cmake_minimum_required(VERSION 3.18)
project(defect_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 최적화: baseline의 진짜 성능을 보려면 -O3 필수 (자동 벡터화 활성)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(defect_pipeline
    src/app/main.cpp
)

target_link_libraries(defect_pipeline PRIVATE ${OpenCV_LIBS})

# 헤더 온리 모듈 경로
target_include_directories(defect_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
