#include "game_camera.hpp"

#include <algorithm>
#include <cmath>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

namespace danvulkan::application {
namespace {
constexpr float lookSensitivity = 0.05f;
constexpr float minimumPitch = -89.0f;
constexpr float maximumPitch = 89.0f;
constexpr float followNearPlaneRatio = 0.05f;
constexpr float minimumNearClip = 0.0001f;
constexpr float minimumDirectionLengthSquared = 0.000001f;
}

void GameCamera::frame(const glm::vec3& minimum, const glm::vec3& maximum,
    std::uint32_t viewportWidth, std::uint32_t viewportHeight)
{
    setViewport(viewportWidth, viewportHeight);

    const glm::vec3 center = (minimum + maximum) * 0.5f;
    const float radius = std::max(glm::length(maximum - minimum) * 0.5f, 0.1f);
    const float verticalHalfAngle = glm::radians(fieldOfViewDegrees_ * 0.5f);
    const float horizontalHalfAngle = std::atan(std::tan(verticalHalfAngle) * aspect_);
    const float limitingHalfAngle = std::min(verticalHalfAngle, horizontalHalfAngle);
    const float distance = radius / std::sin(limitingHalfAngle) * 1.15f;
    const glm::vec3 offsetDirection = glm::normalize(glm::vec3(0.0f, 0.35f, 1.0f));
    position_ = center + offsetDirection * distance;

    const glm::vec3 lookDirection = glm::normalize(center - position_);
    pitchDegrees_ = glm::degrees(std::asin(glm::clamp(lookDirection.y, -1.0f, 1.0f)));
    yawDegrees_ = glm::degrees(std::atan2(lookDirection.x, -lookDirection.z));
    movementSpeed_ = std::max(radius * 0.5f, 0.5f);
    nearClip_ = std::max(radius * 0.01f, 0.01f);
    farClip_ = std::max(distance + radius * 4.0f, 100.0f);
    rebuildProjection();
    rebuildView();
}

void GameCamera::setViewport(std::uint32_t width, std::uint32_t height)
{
    if (width == 0 || height == 0)
    {
        return;
    }
    aspect_ = static_cast<float>(width) / static_cast<float>(height);
    rebuildProjection();
}

void GameCamera::setPose(const glm::vec3& position, const glm::vec3& target)
{
    position_ = position;
    const glm::vec3 targetDirection = target - position;
    if (glm::dot(targetDirection, targetDirection) > minimumDirectionLengthSquared)
    {
        const glm::vec3 lookDirection = glm::normalize(targetDirection);
        pitchDegrees_ = glm::degrees(
            std::asin(glm::clamp(lookDirection.y, -1.0f, 1.0f)));
        yawDegrees_ = glm::degrees(std::atan2(lookDirection.x, -lookDirection.z));
    }
    rebuildView();
}

void GameCamera::setFollowTarget(const glm::vec3& position,
    const glm::vec3& forwardDirection, float distance, float height)
{
    followTarget_ = position;
    followDistance_ = std::max(std::abs(distance), minimumNearClip);
    followHeight_ = height;
    const glm::vec3 horizontalForward(forwardDirection.x, 0.0f, forwardDirection.z);
    if (glm::dot(horizontalForward, horizontalForward) > minimumDirectionLengthSquared)
    {
        followForward_ = glm::normalize(horizontalForward);
    }
}

void GameCamera::update(const InputState& input, float deltaSeconds)
{
    if (input.toggleCameraMode)
    {
        mode_ = mode_ == CameraMode::free ? CameraMode::follow : CameraMode::free;
        rebuildProjection();
    }

    if (mode_ == CameraMode::follow)
    {
        const glm::vec3 cameraPosition = followTarget_ -
            followForward_ * followDistance_ + glm::vec3(0.0f, followHeight_, 0.0f);
        setPose(cameraPosition, followTarget_);
        return;
    }

    yawDegrees_ += input.lookDeltaX * lookSensitivity;
    pitchDegrees_ = glm::clamp(pitchDegrees_ - input.lookDeltaY * lookSensitivity,
        minimumPitch, maximumPitch);

    const glm::vec3 cameraForward = forward();
    const glm::vec3 cameraRight = glm::normalize(
        glm::cross(cameraForward, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 movement(0.0f);
    if (input.moveForward)
    {
        movement += cameraForward;
    }
    if (input.moveBackward)
    {
        movement -= cameraForward;
    }
    if (input.moveLeft)
    {
        movement -= cameraRight;
    }
    if (input.moveRight)
    {
        movement += cameraRight;
    }
    if (glm::dot(movement, movement) > 0.0f)
    {
        position_ += glm::normalize(movement) * movementSpeed_ *
            std::max(deltaSeconds, 0.0f);
    }
    position_ += cameraForward * input.scrollDelta * movementSpeed_ * 0.25f;
    rebuildView();
}

glm::vec3 GameCamera::forward() const noexcept
{
    const float yaw = glm::radians(yawDegrees_);
    const float pitch = glm::radians(pitchDegrees_);
    return glm::normalize(glm::vec3(std::cos(pitch) * std::sin(yaw),
        std::sin(pitch), -std::cos(pitch) * std::cos(yaw)));
}

void GameCamera::rebuildView()
{
    view_ = glm::lookAt(position_, position_ + forward(), glm::vec3(0.0f, 1.0f, 0.0f));
}

void GameCamera::rebuildProjection()
{
    const float activeNearClip = mode_ == CameraMode::follow ?
        std::max(followDistance_ * followNearPlaneRatio, minimumNearClip) : nearClip_;
    projection_ = glm::perspective(glm::radians(fieldOfViewDegrees_), aspect_,
        activeNearClip, farClip_);
}

}
