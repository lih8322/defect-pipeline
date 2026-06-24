#pragma once
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>

// 단순 스코프 타이머: 생성~소멸 사이 경과 시간(ms)을 측정
class ScopedTimer {
public:
    using clock = std::chrono::steady_clock;
    explicit ScopedTimer(double& out_ms) : out_ms_(out_ms), t0_(clock::now()) {}
    ~ScopedTimer() {
        auto t1 = clock::now();
        out_ms_ = std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    double& out_ms_;
    clock::time_point t0_;
};

// 프레임별 처리시간(ms)을 누적해 통계를 내는 수집기
class LatencyStats {
public:
    void add(double ms) { samples_.push_back(ms); }

    size_t count() const { return samples_.size(); }

    double mean() const {
        if (samples_.empty()) return 0.0;
        return std::accumulate(samples_.begin(), samples_.end(), 0.0) / samples_.size();
    }

    double min() const {
        if (samples_.empty()) return 0.0;
        return *std::min_element(samples_.begin(), samples_.end());
    }

    double max() const {
        if (samples_.empty()) return 0.0;
        return *std::max_element(samples_.begin(), samples_.end());
    }

    double stddev() const {
        if (samples_.size() < 2) return 0.0;
        double m = mean();
        double acc = 0.0;
        for (double s : samples_) acc += (s - m) * (s - m);
        return std::sqrt(acc / (samples_.size() - 1));
    }

    // p in [0,100]
    double percentile(double p) const {
        if (samples_.empty()) return 0.0;
        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        double idx = (p / 100.0) * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(idx));
        size_t hi = static_cast<size_t>(std::ceil(idx));
        double frac = idx - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    // 평균 처리시간 기반 처리량(FPS)
    double fps() const {
        double m = mean();
        return m > 0.0 ? 1000.0 / m : 0.0;
    }

    const std::vector<double>& samples() const { return samples_; }

private:
    std::vector<double> samples_;
};
