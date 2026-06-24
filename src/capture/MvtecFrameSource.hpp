#pragma once
#include "../core/IFrameSource.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>

namespace fs = std::filesystem;

// MVTec AD 카테고리의 test/ 하위 이미지들을 "가상 카메라 스트림"으로 공급.
// 카메라가 없어도 전체 파이프라인을 개발/측정할 수 있게 해준다.
//
// 디렉토리 구조 가정:
//   <root>/test/good/*.png
//   <root>/test/crack/*.png
//   <root>/test/scratch/*.png  ...
class MvtecFrameSource : public IFrameSource {
public:
    // root: 카테고리 경로 (예: /root/mvtec/capsule)
    // grayscale: true면 단일 채널로 로드 (결함검출 단순화)
    // loop_count: 스트림을 몇 바퀴 반복할지 (벤치마크용 프레임 수 확보)
    MvtecFrameSource(const std::string& root, bool grayscale = true, int loop_count = 1)
        : root_(root), grayscale_(grayscale), loop_count_(loop_count) {
        collect_files();
    }

    std::optional<Frame> grab() override {
        if (files_.empty()) return std::nullopt;

        size_t total = files_.size() * static_cast<size_t>(loop_count_);
        if (cursor_ >= total) return std::nullopt;

        size_t file_idx = cursor_ % files_.size();
        const auto& entry = files_[file_idx];

        Frame f;
        f.index = cursor_;
        f.label = entry.label;
        int flag = grayscale_ ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
        f.image = cv::imread(entry.path, flag);

        ++cursor_;
        if (f.image.empty()) {
            // 읽기 실패한 경우에도 스트림은 계속 (드롭과 구분하기 위해 로그용)
            return f;  // empty() == true 로 호출측에서 판단
        }
        return f;
    }

    size_t size() const override {
        return files_.size() * static_cast<size_t>(loop_count_);
    }

    std::string name() const override {
        return "MvtecFrameSource(" + root_ + ")";
    }

    // 정상 이미지(train/good)를 따로 로드 — 기준 템플릿 학습용
    std::vector<cv::Mat> load_normal_images() const {
        std::vector<cv::Mat> out;
        fs::path good = fs::path(root_) / "train" / "good";
        if (!fs::exists(good)) return out;

        std::vector<std::string> paths;
        for (const auto& e : fs::directory_iterator(good)) {
            if (is_image(e.path())) paths.push_back(e.path().string());
        }
        std::sort(paths.begin(), paths.end());
        int flag = grayscale_ ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
        for (const auto& p : paths) {
            cv::Mat m = cv::imread(p, flag);
            if (!m.empty()) out.push_back(m);
        }
        return out;
    }

private:
    struct Entry {
        std::string path;
        std::string label;  // 하위 폴더명 (good/crack/...)
    };

    static bool is_image(const fs::path& p) {
        auto ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp";
    }

    void collect_files() {
        fs::path test_dir = fs::path(root_) / "test";
        if (!fs::exists(test_dir)) return;

        // 결함 유형 폴더별로 정렬해 순회 (재현성 위해 정렬)
        std::vector<std::string> labels;
        for (const auto& d : fs::directory_iterator(test_dir)) {
            if (d.is_directory()) labels.push_back(d.path().filename().string());
        }
        std::sort(labels.begin(), labels.end());

        for (const auto& label : labels) {
            fs::path sub = test_dir / label;
            std::vector<std::string> paths;
            for (const auto& e : fs::directory_iterator(sub)) {
                if (is_image(e.path())) paths.push_back(e.path().string());
            }
            std::sort(paths.begin(), paths.end());
            for (const auto& p : paths) {
                files_.push_back({p, label});
            }
        }
    }

    std::string root_;
    bool grayscale_;
    int loop_count_;
    std::vector<Entry> files_;
    size_t cursor_ = 0;
};
