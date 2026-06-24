#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

// 단일 생산자 / 단일 소비자(SPSC) 바운디드 lock-free 링 버퍼.
//
// 설계:
//  - 생산자는 head_만, 소비자는 tail_만 쓴다(서로의 인덱스는 읽기만) → CAS 불필요.
//  - head_/tail_를 acquire/release로 동기화해, 데이터 기록이 인덱스 공개보다
//    먼저 보이도록 보장(소비자가 본 인덱스 ⇒ 그 슬롯 데이터도 본다).
//  - 바운디드라 가득 차면 try_push 실패 → 호출자가 백프레셔 처리(프레임 누락 0).
//
// 트레이딩 시스템에서 쓰던 수신↔처리 스레드 분리 패턴을, 카메라 grab↔GPU↔CPU
// 스테이지 분리에 그대로 이식한 것.
template <typename T>
class SpscQueue {
public:
    explicit SpscQueue(std::size_t capacity)
        : buf_(capacity + 1), cap_(capacity + 1) {}  // 한 칸은 full/empty 구분용

    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    bool try_push(T v) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t nh = next(h);
        if (nh == tail_.load(std::memory_order_acquire)) return false;  // full
        buf_[h] = std::move(v);
        head_.store(nh, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;   // empty
        out = std::move(buf_[t]);
        tail_.store(next(t), std::memory_order_release);
        return true;
    }

    // 백프레셔 블로킹 헬퍼. 누락 없이 흘려보낸다.
    // 적응형 backoff: 짧게 스핀 → yield → 그래도 안 되면 짧은 sleep.
    // 스테이지 처리율이 다를 때(예: GPU≪CCL) 빠른 쪽이 순수 스핀으로 코어를
    // 점유해 느린 쪽을 굶기는 것을 막는다(코어 제한 환경에서 특히 중요).
    void push(T v) {
        int s = 0;
        while (!try_emplace(v)) backoff(s);
    }
    bool pop(T& out, const std::atomic<bool>& done) {
        int s = 0;
        while (!try_pop(out)) {
            if (done.load(std::memory_order_acquire)) {
                return try_pop(out);  // 종료 신호 후 잔여분까지 비운다
            }
            backoff(s);
        }
        return true;
    }

private:
    std::size_t next(std::size_t i) const { return (i + 1) % cap_; }

    // 실패 시 v를 소비하지 않는 push(성공 시에만 슬롯으로 move).
    bool try_emplace(T& v) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t nh = next(h);
        if (nh == tail_.load(std::memory_order_acquire)) return false;  // full
        buf_[h] = std::move(v);
        head_.store(nh, std::memory_order_release);
        return true;
    }

    static void backoff(int& spins) {
        if (spins < 64) {
            ++spins;  // 바쁜 스핀(짧게)
        } else if (spins < 256) {
            ++spins;
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    std::vector<T> buf_;
    const std::size_t cap_;
    // 캐시라인 분리: 생산자/소비자 인덱스의 false sharing 방지.
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};
