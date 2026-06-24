#pragma once
#include "Frame.hpp"
#include <optional>

// 프레임 공급원 추상화.
// 핵심 설계: grab()은 "프레임당 1회" 호출된다 → 가상 함수 비용은 무시 가능.
// 이 추상화 덕분에 MVTec 파일 / 실제 GenICam 카메라를 같은 코드로 처리한다.
class IFrameSource {
public:
    virtual ~IFrameSource() = default;

    // 다음 프레임을 가져온다. 더 없으면 nullopt.
    virtual std::optional<Frame> grab() = 0;

    // 전체 프레임 수 (알 수 있으면). 카메라면 0 반환 가능.
    virtual size_t size() const = 0;

    // 소스 식별용 이름
    virtual std::string name() const = 0;
};
