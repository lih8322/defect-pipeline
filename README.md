# GPU 가속 실시간 결함 검출 파이프라인

MVTec AD 데이터셋을 "가상 카메라 스트림"으로 공급하여, 산업 표면 결함을 실시간 검출하는 end-to-end 파이프라인. CPU baseline → CUDA 가속 → 멀티스레드 비동기화 단계로 발전시키며 각 단계의 처리속도(ms/frame, FPS)를 정량 측정한다.

## 현재 단계: Phase 1 — CPU baseline

| Phase | 내용 | 상태 |
|-------|------|------|
| 1 | CPU 단일스레드 baseline (OpenCV) + 처리시간 측정 | ✅ |
| 2 | CUDA 커널 포팅 (이진화·모폴로지·라벨링) | 예정 |
| 3 | pinned memory + CUDA stream 비동기화 | 예정 |
| 4 | grab/처리 스레드 분리 (lock-free 큐) | 예정 |

## 설계 핵심

- `IFrameSource` / `IDefectDetector` 인터페이스는 **프레임 경계에서만 호출**(프레임당 1회)되므로 가상 함수 비용이 무시 가능하다. 픽셀 단위 연산은 구현체 내부의 직선 루프에서 처리한다.
- 이 추상화 덕분에 MVTec 파일 / 실제 카메라 / CPU / CUDA 구현을 코드 변경 없이 교체할 수 있다.

## 알고리즘 (고전 비전)

정상 이미지 평균으로 기준 템플릿 생성 → 입력과 차분(absdiff) → 블러 → 이진화 → 모폴로지(open/close) → Connected Component Labeling → 결함 bbox.

## 빌드

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 실행

```bash
# ./defect_pipeline <category_root> [loop_count] [out_csv]
./defect_pipeline /root/mvtec/capsule 1 ../bench/cpu_baseline.csv
```

`loop_count`로 스트림을 반복해 벤치마크용 프레임 수를 늘릴 수 있다.

## 측정 항목

- ms/frame: mean, min, max, stddev, p50/p95/p99
- throughput: FPS
- frame drop: 인덱스 연속성으로 검출 (목표 0)
- CSV: 프레임별 처리시간 로그 → matplotlib 시각화
