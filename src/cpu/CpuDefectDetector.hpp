#pragma once
#include "../core/IDefectDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>

// CPU 단일스레드 baseline 검출기.
// 알고리즘 (고전 비전):
//   1. 정상 이미지들의 평균 → 기준 템플릿 생성 (train)
//   2. 입력 - 기준 차분 (absdiff)
//   3. 가우시안 블러로 노이즈 완화
//   4. 임계값 이진화 (threshold)
//   5. 모폴로지 opening (작은 노이즈 제거) + closing (구멍 메움)
//   6. Connected Component Labeling → 결함 bbox 추출
//
// 이 모든 픽셀 연산은 OpenCV CPU 구현. 이후 CUDA 버전과 1:1 비교 대상이 된다.
class CpuDefectDetector : public IDefectDetector {
public:
    struct Params {
        int   blur_ksize   = 5;     // 가우시안 커널 크기 (홀수)
        double threshold   = 30.0;  // 차분 이진화 임계값
        int   morph_ksize  = 5;     // 모폴로지 커널 크기
        int   min_area     = 50;    // 이보다 작은 영역은 노이즈로 무시
    };

    // 주의: Params를 default 인자(= {})로 두면 GCC가 "default member initializer
    // required before the end of its enclosing class" 에러를 낸다(중첩 aggregate의
    // 멤버 초기화자를 클래스 정의 완료 전에 default 인자에서 쓸 수 없음).
    // → 기본 생성자와 Params 받는 생성자로 분리해 우회.
    CpuDefectDetector() = default;
    explicit CpuDefectDetector(Params p) : params_(p) {}

    void train(const std::vector<cv::Mat>& normal_images) override {
        if (normal_images.empty()) return;

        // 모든 정상 이미지를 float로 누적 평균 → 기준 템플릿
        cv::Size sz = normal_images.front().size();
        cv::Mat acc = cv::Mat::zeros(sz, CV_32FC1);
        int n = 0;
        for (const auto& img : normal_images) {
            cv::Mat gray, f;
            to_gray(img, gray);
            if (gray.size() != sz) cv::resize(gray, gray, sz);
            gray.convertTo(f, CV_32FC1);
            acc += f;
            ++n;
        }
        acc /= std::max(1, n);
        acc.convertTo(reference_, CV_8UC1);
        ref_size_ = sz;
        trained_ = true;
    }

    DetectionResult detect(const Frame& frame) override {
        DetectionResult res;
        res.frame_index = frame.index;
        if (frame.image.empty() || !trained_) return res;

        cv::Mat gray;
        to_gray(frame.image, gray);
        if (gray.size() != ref_size_) cv::resize(gray, gray, ref_size_);

        // 2. 차분
        cv::Mat diff;
        cv::absdiff(gray, reference_, diff);

        // 3. 블러
        cv::Mat blurred;
        cv::GaussianBlur(diff, blurred, cv::Size(params_.blur_ksize, params_.blur_ksize), 0);

        // 4. 이진화
        cv::Mat bin;
        cv::threshold(blurred, bin, params_.threshold, 255, cv::THRESH_BINARY);

        // 5. 모폴로지 (opening → closing)
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(params_.morph_ksize, params_.morph_ksize));
        cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);

        // 6. Connected Component Labeling
        cv::Mat labels, stats, centroids;
        int ncomp = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);

        for (int i = 1; i < ncomp; ++i) {  // 0은 배경
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < params_.min_area) continue;
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            res.defects.emplace_back(x, y, w, h);
        }

        res.mask = bin;
        res.has_defect = !res.defects.empty();
        return res;
    }

    std::string name() const override { return "CpuDefectDetector"; }

private:
    static void to_gray(const cv::Mat& in, cv::Mat& out) {
        if (in.channels() == 1) out = in;
        else cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    }

    Params  params_;
    cv::Mat reference_;
    cv::Size ref_size_;
    bool    trained_ = false;
};
