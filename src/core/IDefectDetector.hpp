#pragma once
#include "Frame.hpp"

// 결함 검출기 추상화.
// 핵심 설계: detect()는 "프레임당 1회" 호출 → 가상 함수 OK.
// 픽셀 단위 연산은 구현체 내부의 직선 루프/커널에서 처리 (다형성 없음).
// 같은 인터페이스로 CPU baseline → CUDA → 추후 ONNX 모델까지 교체 가능.
class IDefectDetector {
public:
    virtual ~IDefectDetector() = default;

    // 기준 템플릿 학습 (정상 이미지들의 평균 등). 1회 호출.
    virtual void train(const std::vector<cv::Mat>& normal_images) = 0;

    // 한 프레임에서 결함 검출.
    virtual DetectionResult detect(const Frame& frame) = 0;

    virtual std::string name() const = 0;
};
