# GPU 가속 실시간 결함 검출 파이프라인

MVTec AD 데이터셋을 "가상 카메라 스트림"으로 공급하여, 산업 표면 결함을 실시간 검출하는 end-to-end 파이프라인. CPU baseline → CUDA 가속 → 멀티스레드 비동기화 단계로 발전시키며 각 단계의 처리속도(ms/frame, FPS)를 정량 측정한다.

## 현재 단계: Phase 4 — lock-free 파이프라인 (GPU∥CPU)

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | CPU 단일스레드 baseline (OpenCV) + 처리시간 측정 | ✅ |
| 2 | CUDA 커널 포팅 (absdiff·blur·threshold·morphology) | ✅ |
| 3 | pinned(page-locked) memory + async 전송 | ✅ |
| 4 | lock-free SPSC 파이프라인 (grab→GPU→CPU 스테이지 분리) | ✅ |
| 5 | CCL GPU 이식(잔여 CPU 병목 제거) / 해상도 스윕 | 예정 |

## 측정 결과 — Phase 1 CPU baseline

`detect()`만 측정 (이미지 로드 제외), 단일 스레드. MVTec capsule test 132장을 10회 반복 = **1320 프레임**.

| 지표 | 값 |
|------|-----|
| mean | **3.42 ms/frame** |
| p50 / p95 / p99 | 2.91 / 5.65 / 6.52 ms |
| min / max | 2.32 / 28.77 ms |
| stddev | 1.42 ms |
| **throughput** | **293 FPS** |
| frame drop | **0** |

![CPU baseline latency](docs/cpu_baseline.png)

*측정 환경: AMD EPYC 7B13 (단일 코어 사용), Ubuntu 24.04, g++ 13.3 `-O3 -march=native`, OpenCV 4.6, 이미지 1000×1000 그레이스케일. 이 수치가 Phase 2 이후 CUDA 가속의 비교 기준선이다.*

> 현재 검출 파라미터(threshold/min_area)는 미튜닝 상태로 정상 이미지 오검출이 있다. 처리 **속도** baseline 확보가 Phase 1 목적이며, 검출 품질 튜닝은 후속 작업이다.

## 측정 결과 — Phase 2 CUDA 포팅

absdiff·가우시안 블러(분리형)·이진화·모폴로지(open/close)를 직접 작성한 CUDA 커널로 처리. 동기 `cudaMemcpy`(H2D/D2H), Connected Component Labeling은 CPU 유지. 동일 입력(1320 프레임)에서 측정.

| 지표 | CPU baseline | **CUDA** | 향상 |
|------|------|------|------|
| mean | 3.42 ms | **1.90 ms** | **1.8×** |
| p95 | 5.65 ms | 2.29 ms | 2.5× |
| p99 | 6.52 ms | 2.50 ms | 2.6× |
| throughput | 293 FPS | **525 FPS** | |
| defect frames | 1250 | 1250 | 검출 결과 동일 |

![CPU vs CUDA](docs/phase2_compare.png)

**GPU 구간 분해** (프레임당 평균, CUDA event 측정):

| H2D | 커널 | D2H | 전송 비중 |
|-----|------|-----|-----------|
| 0.131 ms | 0.224 ms | 0.142 ms | **54.9%** |

→ 순수 GPU 구간은 ~0.50 ms뿐이고, detect() 1.90 ms의 나머지(~1.4 ms)는 **CPU 측 CCL·리사이즈**가 차지한다. 즉 병목이 픽셀 연산에서 *전송 + CPU 잔여 연산*으로 이동했다. 동기 전송이 GPU 구간의 절반 이상을 차지하므로 **Phase 3(pinned memory + CUDA stream 오버랩)**으로 직결되는 근거가 된다.

> 설계: CUDA 검출기는 pImpl로 디바이스 자원/커널을 `.cu` 안에 숨겨, `main.cpp`(g++)와 커널(nvcc)을 분리 컴파일한다. 가우시안 가중치·모폴로지 SE는 constant memory에 둔다(전 스레드 동일 읽기 → 브로드캐스트 캐시). CMake `-DUSE_CUDA=OFF`로 CPU 전용 빌드도 가능.

## 측정 결과 — Phase 3 pinned memory

입출력 호스트 버퍼를 `cudaHostAlloc`(page-locked)으로 바꾸고 `cudaMemcpyAsync`로 전송. **같은 실행 바이너리에서** `cuda`(pinned) / `cuda-pageable`(Phase 2 경로)를 선택해 동일 하드웨어 before/after 비교. GPU 구간은 CUDA event로 측정(1320 프레임 평균).

| GPU 구간 | pageable | **pinned** | |
|------|------|------|------|
| H2D | 0.132 ms | 0.131 ms | ≈ 동일 |
| kernels | 0.224 ms | 0.224 ms | 동일 |
| **D2H** | 0.144 ms | **0.082 ms** | **-43%** |
| GPU-side 합 | 0.500 ms | 0.437 ms | -13% |

![pinned transfer breakdown](docs/phase3_transfer.png)

**해석 (측정 기반):**
- pinned으로 **D2H DMA가 43% 단축**됐다. pageable 전송은 드라이버가 내부 pinned 버퍼로 staging한 뒤 복사하지만, page-locked 메모리는 DMA가 직접 접근하기 때문이다. (H2D는 1 MB 소형 전송이라 이 해상도에선 차이가 묻혔다.)
- 그러나 **end-to-end detect() 지연은 거의 변화 없음**(1.93 → 1.99 ms). 이유: (1) pageable 프레임을 pinned 버퍼로 옮기는 staging 복사가 추가되고, (2) detect() 시간의 대부분(~1.4 ms)은 여전히 **CPU 측 CCL**이 차지하기 때문이다.
- 즉 pinned 단독으로는 전송 *지연*만 줄 뿐, 동기 파이프라인에선 전체 처리량이 늘지 않는다. **진짜 이득은 이렇게 async 가능해진 전송을 다음 프레임의 커널 실행과 _오버랩_시킬 때(Phase 4) 발생한다.** Phase 3는 그 전제 조건(page-locked + 스트림 기반 async 경로)을 갖춘 단계다.

> "pinned = 무조건 빠름"이 아니라, *어디서 왜 이득이 생기는지*를 측정으로 분리한 것이 이 단계의 핵심이다.

## 측정 결과 — Phase 4 lock-free 파이프라인

Phase 3에서 GPU 구간(~0.44 ms)은 충분히 작고 **detect() 시간의 대부분이 CPU CCL**임을 확인했다. 따라서 단순 stream 오버랩(H2D∥compute∥D2H)만으론 end-to-end가 줄지 않는다(GPU가 병목이 아니므로). 대신 **GPU 스테이지와 CPU CCL 스테이지를 서로 다른 프레임에 대해 동시 실행**하면 처리량이 `sum`이 아니라 `max(GPU, CPU)`로 결정된다.

`grab → [SPSC] → GPU stage → [SPSC] → CPU CCL` 3-스테이지를 **직접 구현한 lock-free SPSC 큐**(단일 writer/reader, acquire/release, 캐시라인 분리)로 연결했다. 동일 스레드 예산(`CV_THREADS=10`)에서 측정:

| 구성 | per-frame | throughput | |
|------|------|------|------|
| CPU baseline | 4.0 ms | 251 FPS | |
| CUDA 직렬 | 1.33 ms | 751 FPS | 3.0× vs CPU |
| **CUDA 파이프라인** | **0.60 ms** | **1660 FPS** | **2.2× vs 직렬, 6.6× vs CPU** |

![Phase 4 throughput](docs/phase4_throughput.png)

- 파이프라인 per-frame(0.60 ms)이 직렬(1.33 ms = GPU+CCL 합)에서 **두 스테이지 중 큰 쪽(CCL ≈ 0.55 ms)으로 수렴** → 의도대로 `max(stage)` 동작.
- 바운디드 SPSC + 백프레셔로 **frame drop 0** 보장.

**문제해결 — oversubscription을 측정으로 규명:**
처음 파이프라인은 오히려 **117 FPS로 붕괴**했다. 원인 분석 결과, OpenCV가 컨테이너의 논리 코어(128)만큼 `parallel_for` 스레드를 띄우는데, 고프레임으로 CCL을 매 프레임 호출하니 **128-스레드 thread-pool 생성/동기 오버헤드**가 지배(CCL 0.55 ms → 8.5 ms)했다. busy-spin도 같은 코어를 뺏어 악화시켰다. → ① SPSC 대기를 **적응형 backoff(spin→yield→sleep)** 로, ② OpenCV 스레드를 **실제 가용 코어 수로 cap**하여 해결(117 → 1660 FPS, 14×). "라이브러리 기본값이 항상 최적은 아니다"를 측정으로 확인한 사례.

> 설계: `detect()`를 GPU 스테이지(`run_gpu_stage`)와 CPU 스테이지(`label_mask`)로 분리해 서로 다른 스레드가 다른 프레임을 처리한다. SPSC 큐는 실시간 트레이딩 시스템에서 쓰던 수신↔처리 분리 패턴을 그대로 이식한 것.

## 설계 핵심

- `IFrameSource` / `IDefectDetector` 인터페이스는 **프레임 경계에서만 호출**(프레임당 1회)되므로 가상 함수 비용이 무시 가능하다. 픽셀 단위 연산은 구현체 내부의 직선 루프에서 처리한다.
- 이 추상화 덕분에 MVTec 파일 / 실제 카메라 / CPU / CUDA 구현을 코드 변경 없이 교체할 수 있다.

## 알고리즘 (고전 비전)

정상 이미지 평균으로 기준 템플릿 생성 → 입력과 차분(absdiff) → 블러 → 이진화 → 모폴로지(open/close) → Connected Component Labeling → 결함 bbox.

## 데이터셋

[MVTec AD (MVTec Anomaly Detection)](https://www.mvtec.com/company/research/datasets/mvtec-ad)의 `capsule` 카테고리를 사용한다. 산업 표면 결함 검출용 공개 벤치마크 데이터셋이다.

- **라이선스**: CC BY-NC-SA 4.0 (연구·비상업 용도). 상업적 사용 불가.
- **데이터는 본 저장소에 포함되지 않는다** — 위 링크에서 직접 내려받아 `<category_root>` 아래에 다음 구조로 배치한다:
  ```
  capsule/
  ├── train/good/        정상 이미지 (기준 템플릿 학습용)
  ├── test/{good,crack,faulty_imprint,poke,scratch,squeeze}/
  └── ground_truth/      결함별 정답 마스크
  ```
- **인용 (citation)**:
  > P. Bergmann, M. Fauser, D. Sattlegger, C. Steger.
  > "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." *CVPR*, 2019.

## 빌드

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release           # CUDA 포함 (기본 USE_CUDA=ON)
# cmake .. -DUSE_CUDA=OFF                       # CPU 전용 (CUDA 없는 환경)
make -j$(nproc)
```

CUDA 빌드는 nvcc(아키텍처 sm_86 = RTX A5000/A4500 등 Ampere)를 사용한다.

## 실행

```bash
# ./defect_pipeline <category_root> [cpu|cuda|cuda-pageable|cuda-pipe] [loop_count] [out_csv]
#   cuda = pinned 직렬(Phase 3), cuda-pageable = pageable(Phase 2 비교용),
#   cuda-pipe = lock-free GPU∥CPU 파이프라인(Phase 4)
./defect_pipeline /root/mvtec/capsule cpu       10 ../bench/cpu_baseline.csv
./defect_pipeline /root/mvtec/capsule cuda      10 ../bench/cuda.csv
./defect_pipeline /root/mvtec/capsule cuda-pipe 10 ../bench/cuda_pipe.csv

# 스레드 예산 제어(코어 제한 컨테이너에서 oversubscription 방지):
CV_THREADS=10 ./defect_pipeline /root/mvtec/capsule cuda 10 ../bench/cuda.csv
```

`loop_count`로 스트림을 반복해 벤치마크용 프레임 수를 늘릴 수 있다.

## 측정 항목

- ms/frame: mean, min, max, stddev, p50/p95/p99
- throughput: FPS
- frame drop: 인덱스 연속성으로 검출 (목표 0)
- CSV: 프레임별 처리시간 로그 → matplotlib 시각화
