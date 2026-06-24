#include "CudaDefectDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

// ───────────────────────── 유틸 ─────────────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,\
                         cudaGetErrorString(err__));                          \
        }                                                                     \
    } while (0)

// 분리형 가우시안 가중치(1D)와 모폴로지 SE를 constant memory에 둔다.
// 모든 스레드가 동일하게 읽는 작은 읽기전용 데이터 → constant memory가 적합
// (브로드캐스트 캐시로 글로벌 접근 대비 빠름).
__constant__ float         c_gauss[64];   // 1D 가우시안 가중치
__constant__ int           c_ghalf;       // 가우시안 반폭
__constant__ unsigned char c_se[1024];    // 모폴로지 SE (최대 32x32)
__constant__ int           c_se_n;        // SE 한 변 크기
__constant__ int           c_se_half;     // SE 반폭

// ───────────────────────── 커널 ─────────────────────────

// 1) 차분: |input - reference|. 픽셀당 독립 → 1D 그리드.
__global__ void k_absdiff(const unsigned char* a, const unsigned char* b,
                          unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int d = int(a[i]) - int(b[i]);
        out[i] = (unsigned char)(d < 0 ? -d : d);
    }
}

// 2) 가우시안 블러(분리형): 수평 패스 → float 임시버퍼.
//    경계는 clamp(replicate). 2D 그리드.
__global__ void k_blur_h(const unsigned char* in, float* tmp, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int half = c_ghalf;
    float s = 0.f;
    for (int k = -half; k <= half; ++k) {
        int xx = min(max(x + k, 0), w - 1);
        s += c_gauss[k + half] * float(in[y * w + xx]);
    }
    tmp[y * w + x] = s;
}

//    수직 패스 → uchar 출력(반올림·clamp).
__global__ void k_blur_v(const float* tmp, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int half = c_ghalf;
    float s = 0.f;
    for (int k = -half; k <= half; ++k) {
        int yy = min(max(y + k, 0), h - 1);
        s += c_gauss[k + half] * tmp[yy * w + x];
    }
    int v = int(s + 0.5f);
    out[y * w + x] = (unsigned char)min(max(v, 0), 255);
}

// 3) 이진화: src > thresh ? maxval : 0 (OpenCV THRESH_BINARY).
__global__ void k_threshold(const unsigned char* in, unsigned char* out,
                            int n, unsigned char t, unsigned char maxv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > t ? maxv : 0;
}

// 4) 모폴로지 erode/dilate. 이진(0/255)에서 erode=SE영역 min, dilate=max.
//    경계는 clamp(replicate)로 단순화(naive 포팅). 2D 그리드.
__global__ void k_morph(const unsigned char* in, unsigned char* out,
                        int w, int h, bool dilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int half = c_se_half, n = c_se_n;
    unsigned char v = dilate ? 0 : 255;
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            if (!c_se[(dy + half) * n + (dx + half)]) continue;
            int xx = min(max(x + dx, 0), w - 1);
            int yy = min(max(y + dy, 0), h - 1);
            unsigned char p = in[yy * w + xx];
            v = dilate ? max(v, p) : min(v, p);
        }
    }
    out[y * w + x] = v;
}

// ───────────────────────── pImpl ─────────────────────────
struct CudaDefectDetector::Impl {
    CudaDefectDetector::Params params;

    int w = 0, h = 0, n = 0;
    bool trained = false;
    cv::Size ref_size;

    // 디바이스 버퍼
    unsigned char *d_in = nullptr, *d_ref = nullptr, *d_diff = nullptr;
    unsigned char *d_blur = nullptr, *d_bin = nullptr, *d_m1 = nullptr;
    float* d_tmp = nullptr;

    // 타이밍 누적(전송 vs 연산 비중 측정 — Phase 3 동기부여 근거)
    cudaEvent_t e0, e1, e2, e3;
    double ms_h2d = 0, ms_kernel = 0, ms_d2h = 0;
    long calls = 0;

    Impl() {
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventCreate(&e2); cudaEventCreate(&e3);
    }
    ~Impl() {
        free_buffers();
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaEventDestroy(e2); cudaEventDestroy(e3);
    }

    void free_buffers() {
        for (auto p : {d_in, d_ref, d_diff, d_blur, d_bin, d_m1})
            if (p) cudaFree(p);
        if (d_tmp) cudaFree(d_tmp);
        d_in = d_ref = d_diff = d_blur = d_bin = d_m1 = nullptr;
        d_tmp = nullptr;
    }

    void alloc(int width, int height) {
        free_buffers();
        w = width; h = height; n = w * h;
        size_t bytes = size_t(n);
        for (auto pp : {&d_in, &d_ref, &d_diff, &d_blur, &d_bin, &d_m1})
            CUDA_CHECK(cudaMalloc(pp, bytes));
        CUDA_CHECK(cudaMalloc(&d_tmp, bytes * sizeof(float)));
    }

    static void to_gray(const cv::Mat& in, cv::Mat& out) {
        if (in.channels() == 1) out = in;
        else cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    }
};

// ───────────────────────── 인터페이스 ─────────────────────────
CudaDefectDetector::CudaDefectDetector() : impl_(new Impl()) {}
CudaDefectDetector::CudaDefectDetector(Params p) : impl_(new Impl()) {
    impl_->params = p;
}
CudaDefectDetector::~CudaDefectDetector() {
    if (impl_ && impl_->calls > 0) {
        double c = double(impl_->calls);
        std::printf(
            "[CUDA breakdown] per-frame avg: H2D %.3f ms | kernels %.3f ms | "
            "D2H %.3f ms  (transfer %.1f%%)\n",
            impl_->ms_h2d / c, impl_->ms_kernel / c, impl_->ms_d2h / c,
            100.0 * (impl_->ms_h2d + impl_->ms_d2h) /
                (impl_->ms_h2d + impl_->ms_kernel + impl_->ms_d2h));
    }
    delete impl_;
}

void CudaDefectDetector::train(const std::vector<cv::Mat>& normal_images) {
    if (normal_images.empty()) return;
    auto& im = *impl_;

    // 기준 템플릿: 정상 이미지 평균(CPU에서 1회 계산 — 핫패스 아님).
    cv::Size sz = normal_images.front().size();
    cv::Mat acc = cv::Mat::zeros(sz, CV_32FC1);
    int cnt = 0;
    for (const auto& img : normal_images) {
        cv::Mat gray, f;
        Impl::to_gray(img, gray);
        if (gray.size() != sz) cv::resize(gray, gray, sz);
        gray.convertTo(f, CV_32FC1);
        acc += f; ++cnt;
    }
    acc /= std::max(1, cnt);
    cv::Mat reference;
    acc.convertTo(reference, CV_8UC1);
    im.ref_size = sz;

    im.alloc(sz.width, sz.height);
    CUDA_CHECK(cudaMemcpy(im.d_ref, reference.data, size_t(im.n),
                          cudaMemcpyHostToDevice));

    // 가우시안 1D 가중치 → constant memory (OpenCV와 동일 계수).
    cv::Mat g = cv::getGaussianKernel(im.params.blur_ksize, 0, CV_32F);
    int ghalf = im.params.blur_ksize / 2;
    std::vector<float> gw(g.begin<float>(), g.end<float>());
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, gw.data(), gw.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ghalf, &ghalf, sizeof(int)));

    // 모폴로지 SE(타원) → constant memory.
    cv::Mat se = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(im.params.morph_ksize, im.params.morph_ksize));
    int se_n = se.cols, se_half = se_n / 2;
    std::vector<unsigned char> sev(se.begin<unsigned char>(),
                                   se.end<unsigned char>());
    CUDA_CHECK(cudaMemcpyToSymbol(c_se, sev.data(), sev.size()));
    CUDA_CHECK(cudaMemcpyToSymbol(c_se_n, &se_n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_se_half, &se_half, sizeof(int)));

    im.trained = true;
}

DetectionResult CudaDefectDetector::detect(const Frame& frame) {
    auto& im = *impl_;
    DetectionResult res;
    res.frame_index = frame.index;
    if (frame.image.empty() || !im.trained) return res;

    // 입력 그레이스케일·리사이즈는 CPU(프레임당 1회, 핫패스 아님).
    cv::Mat gray;
    Impl::to_gray(frame.image, gray);
    if (gray.size() != im.ref_size) cv::resize(gray, gray, im.ref_size);
    if (!gray.isContinuous()) gray = gray.clone();

    float t_h2d = 0, t_kernel = 0, t_d2h = 0;

    // H2D
    cudaEventRecord(im.e0);
    CUDA_CHECK(cudaMemcpy(im.d_in, gray.data, size_t(im.n),
                          cudaMemcpyHostToDevice));
    cudaEventRecord(im.e1);

    // 커널 파이프라인
    int n = im.n;
    int threads = 256;
    int blocks1d = (n + threads - 1) / threads;
    dim3 b2(16, 16);
    dim3 g2((im.w + b2.x - 1) / b2.x, (im.h + b2.y - 1) / b2.y);

    k_absdiff<<<blocks1d, threads>>>(im.d_in, im.d_ref, im.d_diff, n);
    k_blur_h<<<g2, b2>>>(im.d_diff, im.d_tmp, im.w, im.h);
    k_blur_v<<<g2, b2>>>(im.d_tmp, im.d_blur, im.w, im.h);
    k_threshold<<<blocks1d, threads>>>(
        im.d_blur, im.d_bin, n,
        (unsigned char)im.params.threshold, 255);
    // open = erode → dilate
    k_morph<<<g2, b2>>>(im.d_bin, im.d_m1, im.w, im.h, /*dilate=*/false);
    k_morph<<<g2, b2>>>(im.d_m1, im.d_bin, im.w, im.h, /*dilate=*/true);
    // close = dilate → erode
    k_morph<<<g2, b2>>>(im.d_bin, im.d_m1, im.w, im.h, /*dilate=*/true);
    k_morph<<<g2, b2>>>(im.d_m1, im.d_bin, im.w, im.h, /*dilate=*/false);
    cudaEventRecord(im.e2);

    // D2H
    cv::Mat bin(im.ref_size, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(bin.data, im.d_bin, size_t(im.n),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(im.e3);
    CUDA_CHECK(cudaEventSynchronize(im.e3));

    cudaEventElapsedTime(&t_h2d, im.e0, im.e1);
    cudaEventElapsedTime(&t_kernel, im.e1, im.e2);
    cudaEventElapsedTime(&t_d2h, im.e2, im.e3);
    im.ms_h2d += t_h2d; im.ms_kernel += t_kernel; im.ms_d2h += t_d2h;
    ++im.calls;

    // Connected Component Labeling은 CPU 유지(Phase 2 범위).
    cv::Mat labels, stats, centroids;
    int ncomp = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);
    for (int i = 1; i < ncomp; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < im.params.min_area) continue;
        res.defects.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT),
                                 stats.at<int>(i, cv::CC_STAT_TOP),
                                 stats.at<int>(i, cv::CC_STAT_WIDTH),
                                 stats.at<int>(i, cv::CC_STAT_HEIGHT));
    }
    res.mask = bin;
    res.has_defect = !res.defects.empty();
    return res;
}

std::string CudaDefectDetector::name() const { return "CudaDefectDetector"; }
