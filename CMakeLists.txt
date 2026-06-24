cmake_minimum_required(VERSION 3.18)
project(defect_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 최적화: baseline의 진짜 성능을 보려면 -O3 필수 (자동 벡터화 활성)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# CUDA 가속 검출기(Phase 2). 끄면 CPU baseline만 빌드된다(이식성 유지).
option(USE_CUDA "Build CUDA defect detector" ON)

find_package(OpenCV REQUIRED)

set(SOURCES src/app/main.cpp)
if(USE_CUDA)
    enable_language(CUDA)
    list(APPEND SOURCES src/detect/CudaDefectDetector.cu)
endif()

add_executable(defect_pipeline ${SOURCES})

target_include_directories(defect_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(defect_pipeline PRIVATE ${OpenCV_LIBS})

if(USE_CUDA)
    target_compile_definitions(defect_pipeline PRIVATE USE_CUDA)
    # RTX A5000 = Ampere sm_86
    set_target_properties(defect_pipeline PROPERTIES CUDA_ARCHITECTURES 86)
endif()
