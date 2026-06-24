#include "../core/Timer.hpp"
#include "../core/IFrameSource.hpp"
#include "../core/IDefectDetector.hpp"
#include "../capture/MvtecFrameSource.hpp"
#include "../cpu/CpuDefectDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>

// 사용법:
//   ./defect_pipeline <mvtec_category_root> [loop_count] [out_csv]
// 예:
//   ./defect_pipeline /root/mvtec/capsule 1 bench/cpu_baseline.csv
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <mvtec_category_root> [loop_count] [out_csv]\n";
        return 1;
    }

    std::string root = argv[1];
    int loop_count   = (argc >= 3) ? std::stoi(argv[2]) : 1;
    std::string csv  = (argc >= 4) ? argv[3] : "";

    // --- 소스/검출기 구성 (인터페이스로 추상화) ---
    auto source = std::make_unique<MvtecFrameSource>(root, /*grayscale=*/true, loop_count);
    auto detector = std::make_unique<CpuDefectDetector>();

    std::cout << "Source:   " << source->name() << "\n";
    std::cout << "Detector: " << detector->name() << "\n";
    std::cout << "Frames:   " << source->size() << "\n";

    // --- 기준 템플릿 학습 (정상 이미지) ---
    auto normals = source->load_normal_images();
    std::cout << "Training on " << normals.size() << " normal images...\n";
    if (normals.empty()) {
        std::cerr << "WARNING: no normal images found at " << root
                  << "/train/good — check path.\n";
    }
    detector->train(normals);

    // --- 파이프라인 루프 + 측정 ---
    LatencyStats stats;
    uint64_t expected_index = 0;
    uint64_t dropped = 0;
    uint64_t defect_frames = 0;
    uint64_t read_fail = 0;

    std::ofstream fout;
    if (!csv.empty()) {
        fout.open(csv);
        fout << "frame_index,label,proc_ms,num_defects\n";
    }

    while (auto fopt = source->grab()) {
        Frame& frame = *fopt;

        // 드롭 검출: 인덱스 연속성 확인
        if (frame.index != expected_index) {
            dropped += (frame.index - expected_index);
        }
        expected_index = frame.index + 1;

        if (frame.empty()) { ++read_fail; continue; }

        double ms = 0.0;
        DetectionResult result;
        {
            ScopedTimer t(ms);                 // detect()만 측정
            result = detector->detect(frame);
        }
        stats.add(ms);
        if (result.has_defect) ++defect_frames;

        if (fout.is_open()) {
            fout << frame.index << "," << frame.label << ","
                 << ms << "," << result.defects.size() << "\n";
        }
    }

    // --- 결과 출력 ---
    std::cout << "\n===== RESULTS (" << detector->name() << ") =====\n";
    std::cout << "Processed frames : " << stats.count() << "\n";
    std::cout << "Read failures    : " << read_fail << "\n";
    std::cout << "Dropped frames   : " << dropped << "\n";
    std::cout << "Defect frames    : " << defect_frames << "\n";
    std::cout << "----- latency (ms/frame) -----\n";
    std::cout << "mean   : " << stats.mean()   << "\n";
    std::cout << "min    : " << stats.min()    << "\n";
    std::cout << "max    : " << stats.max()    << "\n";
    std::cout << "stddev : " << stats.stddev() << "\n";
    std::cout << "p50    : " << stats.percentile(50) << "\n";
    std::cout << "p95    : " << stats.percentile(95) << "\n";
    std::cout << "p99    : " << stats.percentile(99) << "\n";
    std::cout << "----- throughput -----\n";
    std::cout << "FPS    : " << stats.fps() << "\n";

    if (fout.is_open()) {
        std::cout << "\nCSV written to: " << csv << "\n";
    }
    return 0;
}
