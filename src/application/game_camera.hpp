#pragma once

#include "input_state.hpp"

#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace danvulkan::application {

enum class CameraMode
{
    free,
    follow
};

class GameCamera
{
public:
    explicit GameCamera(CameraMode initialMode = CameraMode::free) noexcept;

    void frame(const glm::vec3& minimum, const glm::vec3& maximum,
        std::uint32_t viewportWidth, std::uint32_t viewportHeight);
    void setViewport(std::uint32_t width, std::uint32_t height);
    void setPose(const glm::vec3& position, const glm::vec3& target);
    void setFollowTarget(const glm::vec3& position, const glm::vec3& forwardDirection,
        float distance = 3.0f, float height = 1.6f);
    void update(const InputState& input, float deltaSeconds);

    [[nodiscard]] const glm::mat4& view() const noexcept { return view_; }
    [[nodiscard]] const glm::mat4& projection() const noexcept { return projection_; }
    [[nodiscard]] const glm::vec3& position() const noexcept { return position_; }
    [[nodiscard]] float movementSpeed() const noexcept { return movementSpeed_; }
    [[nodiscard]] CameraMode mode() const noexcept { return mode_; }

private:
    void rebuildView();
    void rebuildProjection();
    [[nodiscard]] glm::vec3 forward() const noexcept;

    glm::vec3 position_{0.0f, 0.0f, 3.0f};
    glm::mat4 view_{1.0f};
    glm::mat4 projection_{1.0f};
    float yawDegrees_ = 0.0f;
    float pitchDegrees_ = 0.0f;
    float movementSpeed_ = 1.0f;
    float fieldOfViewDegrees_ = 60.0f;
    float nearClip_ = 0.1f;
    float farClip_ = 100.0f;
    float aspect_ = 16.0f / 9.0f;
    CameraMode mode_;
    glm::vec3 followTarget_{0.0f};
    glm::vec3 followForward_{0.0f, 0.0f, 1.0f};
    float followDistance_ = 3.0f;
    float followHeight_ = 1.6f;
};

}
