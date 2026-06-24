#pragma once
#include "../core/IDefectDetector.hpp"

// CUDA 가속 결함 검출기 (Phase 2 — naive 포팅).
//
// 설계 의도:
//  - 이 헤더는 순수 C++만 노출한다. CUDA 문법(__global__, device 포인터 등)은
//    전부 .cu 안의 Impl(pImpl)로 숨긴다. 덕분에 main.cpp는 g++로,
//    커널 구현은 nvcc로 분리 컴파일된다(인터페이스/구현 분리, DIP 유지).
//  - 알고리즘은 CpuDefectDetector와 1:1 대응(absdiff→blur→threshold→morphology).
//    같은 입력에 대해 비슷한 결과를 내며, 처리속도만 GPU로 가속해 baseline과 비교한다.
//  - Connected Component Labeling은 GPU 구현이 복잡해 Phase 2에선 CPU에 남긴다
//    (전체 파이프라인 중 라벨링이 차지하는 비중은 작음).
//  - 매 프레임 동기 cudaMemcpy(H2D/D2H). pinned/stream 최적화는 Phase 3~4.
class CudaDefectDetector : public IDefectDetector {
public:
    struct Params {
        int    blur_ksize  = 5;
        double threshold   = 30.0;
        int    morph_ksize = 5;
        int    min_area    = 50;
    };

    CudaDefectDetector();
    explicit CudaDefectDetector(Params p);
    ~CudaDefectDetector() override;

    // 복사 금지(디바이스 자원 소유). 이동만 허용.
    CudaDefectDetector(const CudaDefectDetector&) = delete;
    CudaDefectDetector& operator=(const CudaDefectDetector&) = delete;

    void train(const std::vector<cv::Mat>& normal_images) override;
    DetectionResult detect(const Frame& frame) override;
    std::string name() const override;

private:
    struct Impl;   // CUDA 자원/커널 호출을 숨기는 pImpl
    Impl* impl_;
};
