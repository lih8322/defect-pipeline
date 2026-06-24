#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>

// 한 장의 입력 프레임. 가상 카메라/실제 카메라 공통 단위.
struct Frame {
    uint64_t index = 0;        // 프레임 순번 (드롭 검출용)
    cv::Mat  image;            // 원본 이미지 (그레이스케일 또는 컬러)
    std::string label;         // MVTec 결함 유형명 (good/crack/scratch...) — 평가용

    bool empty() const { return image.empty(); }
};

// 검출 결과: 결함 영역 박스 + 마스크
struct DetectionResult {
    uint64_t frame_index = 0;
    std::vector<cv::Rect> defects;  // 검출된 결함 bounding box
    cv::Mat mask;                   // 이진 결함 마스크 (CV_8UC1)
    bool has_defect = false;
};
