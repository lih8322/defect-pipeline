#include "../core/Timer.hpp"
#include "../core/IFrameSource.hpp"
#include "../core/IDefectDetector.hpp"
#include "../capture/MvtecFrameSource.hpp"
#include "../cpu/CpuDefectDetector.hpp"
#ifdef USE_CUDA
#include "../detect/CudaDefectDetector.hpp"
#endif

#include "../core/SpscQueue.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// 사용법:
//   ./defect_pipeline <mvtec_category_root> [detector] [loop_count] [out_csv]
//     detector: cpu | cuda | cuda-pageable | cuda-pipe
//       (기본 cpu; cuda=pinned 직렬; cuda-pipe=GPU∥CPU lock-free 파이프라인)
// 예:
//   ./defect_pipeline /root/mvtec/capsule cpu       1  bench/cpu_baseline.csv
//   ./defect_pipeline /root/mvtec/capsule cuda      10 bench/cuda.csv
//   ./defect_pipeline /root/mvtec/capsule cuda-pipe 10 bench/cuda_pipe.csv

#ifdef USE_CUDA
// Phase 4: lock-free SPSC로 grab → GPU 스테이지 → CPU CCL 스테이지를 분리해
// 서로 다른 프레임에 대해 동시 실행. 처리량이 sum이 아니라 max(GPU, CPU)로 결정된다.
static int run_pipeline(MvtecFrameSource& source, const std::string& csv) {
    // 고프레임 파이프라인에서 CCL을 매 프레임 호출할 때 OpenCV 기본 스레드 수(=논리코어,
    // 여기선 128)는 parallel_for의 thread-pool 오버헤드로 오히려 치명적이다(측정: 117 FPS).
    // CV_THREADS 미지정 시 실제 가용 코어 수로 cap → out-of-the-box로 동작.
    if (!std::getenv("CV_THREADS")) cv::setNumThreads(cv::getNumberOfCPUs());

    CudaDefectDetector det;  // pinned 경로. 파이프라인은 GPU/CPU 스테이지를 직접 호출.

    std::cout << "Source:   " << source.name() << "\n";
    std::cout << "Detector: " << det.name() << " [pipeline GPU||CPU]\n";
    std::cout << "Frames:   " << source.size() << "\n";

    auto normals = source.load_normal_images();
    std::cout << "Training on " << normals.size() << " normal images...\n";
    det.train(normals);

    // 1) 전 프레임 선적재 → 측정 구간에서 디스크 I/O 제외(직렬 detect()와 동일 조건).
    std::vector<Frame> frames;
    frames.reserve(source.size());
    while (auto f = source.grab())
        if (!f->empty()) frames.push_back(std::move(*f));
    const std::size_t N = frames.size();

    struct MaskJob { std::uint64_t index; std::string label; cv::Mat mask; };
    SpscQueue<Frame>   in_q(16);    // feeder → GPU
    SpscQueue<MaskJob> mid_q(16);   // GPU → CCL
    std::atomic<bool> feed_done{false}, gpu_done{false};

    // 결과는 CCL 스레드만 기록(단독 writer) → 입력 순서 보존.
    std::vector<std::uint64_t> out_idx;
    std::vector<std::string>   out_label;
    std::vector<int>           out_nd;
    out_idx.reserve(N); out_label.reserve(N); out_nd.reserve(N);
    double ccl_ms_total = 0.0;
    std::uint64_t defect_frames = 0;

    auto t0 = std::chrono::steady_clock::now();

    // GPU 워커: in_q → run_gpu_stage → mid_q
    std::thread gpu_thr([&] {
        Frame f;
        while (in_q.pop(f, feed_done)) {
            cv::Mat mask = det.run_gpu_stage(f);
            mid_q.push(MaskJob{f.index, f.label, std::move(mask)});
        }
        gpu_done.store(true, std::memory_order_release);
    });

    // CCL 워커: mid_q → label_mask → 기록
    std::thread ccl_thr([&] {
        MaskJob j;
        while (mid_q.pop(j, gpu_done)) {
            double ms = 0.0;
            DetectionResult r;
            { ScopedTimer t(ms); r = det.label_mask(j.mask, j.index); }
            ccl_ms_total += ms;
            out_idx.push_back(j.index);
            out_label.push_back(j.label);
            out_nd.push_back(static_cast<int>(r.defects.size()));
            if (r.has_defect) ++defect_frames;
        }
    });

    for (auto& f : frames) in_q.push(f);     // feeder(main 스레드)
    feed_done.store(true, std::memory_order_release);
    gpu_thr.join();
    ccl_thr.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double fps = N ? 1000.0 * N / elapsed_ms : 0.0;

    if (!csv.empty()) {
        std::ofstream fout(csv);
        fout << "frame_index,label,proc_ms,num_defects\n";
        for (std::size_t i = 0; i < out_idx.size(); ++i)
            fout << out_idx[i] << "," << out_label[i] << ","
                 << (elapsed_ms / N) << "," << out_nd[i] << "\n";
    }

    std::cout << "\n===== RESULTS (pipeline GPU||CPU) =====\n";
    std::cout << "Processed frames : " << out_idx.size() << "\n";
    std::cout << "Dropped frames   : 0  (bounded SPSC backpressure)\n";
    std::cout << "Defect frames    : " << defect_frames << "\n";
    std::cout << "----- throughput -----\n";
    std::cout << "wall time      : " << elapsed_ms << " ms\n";
    std::cout << "per-frame(amrt): " << (elapsed_ms / N) << " ms\n";
    std::cout << "FPS            : " << fps << "\n";
    std::cout << "CCL stage(CPU) : " << (ccl_ms_total / N) << " ms/frame\n";
    if (!csv.empty()) std::cout << "\nCSV written to: " << csv << "\n";
    return 0;
}
#endif
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <mvtec_category_root> [detector=cpu|cuda] [loop_count] [out_csv]\n";
        return 1;
    }

    std::string root     = argv[1];
    std::string det_kind = (argc >= 3) ? argv[2] : "cpu";
    int loop_count       = (argc >= 4) ? std::stoi(argv[3]) : 1;
    std::string csv      = (argc >= 5) ? argv[4] : "";

    // OpenCV parallel_for 스레드 수: 환경변수 CV_THREADS로 조절(미설정 시 기본).
    // 코어 제한 컨테이너에서 oversubscription을 막아 CCL/파이프라인 스케줄링 안정화에 사용.
    if (const char* t = std::getenv("CV_THREADS")) {
        int nt = std::atoi(t);
        if (nt > 0) cv::setNumThreads(nt);
    }

    // Phase 4 파이프라인 모드는 별도 경로(구체 GPU/CPU 스테이지 직접 호출).
#ifdef USE_CUDA
    if (det_kind == "cuda-pipe") {
        MvtecFrameSource src(root, /*grayscale=*/true, loop_count);
        return run_pipeline(src, csv);
    }
#endif

    // --- 소스/검출기 구성 (인터페이스로 추상화) ---
    auto source = std::make_unique<MvtecFrameSource>(root, /*grayscale=*/true, loop_count);

    std::unique_ptr<IDefectDetector> detector;
    if (det_kind == "cuda" || det_kind == "cuda-pageable") {
#ifdef USE_CUDA
        CudaDefectDetector::Params p;
        p.pinned = (det_kind == "cuda");  // cuda-pageable = Phase 2 비교 경로
        detector = std::make_unique<CudaDefectDetector>(p);
#else
        std::cerr << "ERROR: built without CUDA (configure -DUSE_CUDA=ON)\n";
        return 1;
#endif
    } else {
        detector = std::make_unique<CpuDefectDetector>();
    }

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
