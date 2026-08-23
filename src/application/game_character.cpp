#include "game_character.hpp"

#include "generated_terrain.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace danvulkan::application {
namespace {
constexpr float minimumDirectionLengthSquared = 0.000001f;

glm::vec3 moveTowards(const glm::vec3& current, const glm::vec3& target,
    float maximumDelta) noexcept
{
    const glm::vec3 offset = target - current;
    const float distance = glm::length(offset);
    if (distance <= maximumDelta || distance <= 0.0f)
    {
        return target;
    }
    return current + offset * (maximumDelta / distance);
}

bool finite(const CharacterMovementConfig& config) noexcept
{
    return std::isfinite(config.maximumSpeed) && std::isfinite(config.acceleration) &&
        std::isfinite(config.braking) && std::isfinite(config.turnSpeedRadians) &&
        std::isfinite(config.maximumSlopeRadians) && std::isfinite(config.groundOffset);
}

bool finite(const glm::vec3& value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
}

GameCharacter::GameCharacter(glm::vec3 initialPosition, glm::vec3 forwardDirection,
    float movementSpeed)
    : GameCharacter(initialPosition, forwardDirection,
          CharacterMovementConfig{movementSpeed, std::numeric_limits<float>::max(),
              std::numeric_limits<float>::max(), 10.0f, 0.959931f, 0.0f})
{
    instantResponse_ = true;
}

GameCharacter::GameCharacter(glm::vec3 initialPosition, glm::vec3 forwardDirection,
    CharacterMovementConfig config)
    : initialPosition_(initialPosition), position_(initialPosition), config_(config)
{
    const glm::vec3 horizontalForward(forwardDirection.x, 0.0f, forwardDirection.z);
    if (!finite(initialPosition_) || !finite(horizontalForward) ||
        glm::dot(horizontalForward, horizontalForward) <= minimumDirectionLengthSquared ||
        !finite(config_) || config_.maximumSpeed < 0.0f || config_.acceleration < 0.0f ||
        config_.braking < 0.0f || config_.turnSpeedRadians < 0.0f ||
        config_.maximumSlopeRadians < 0.0f ||
        config_.maximumSlopeRadians >= 1.57079632679f)
    {
        throw std::invalid_argument("game character movement configuration is invalid");
    }
    movementBasisForward_ = glm::normalize(horizontalForward);
    forward_ = movementBasisForward_;
}

void GameCharacter::updateMovement(const InputState& input, float deltaSeconds)
{
    updateMovement(input, deltaSeconds, nullptr, nullptr);
}

void GameCharacter::updateMovement(const InputState& input, float deltaSeconds,
    const GeneratedTerrain& terrain)
{
    updateMovement(input, deltaSeconds, &terrain, nullptr);
}

void GameCharacter::updateMovementOnSurface(const InputState& input, float deltaSeconds,
    const TerrainSampleQuery& sampleTerrain)
{
    updateMovement(input, deltaSeconds, nullptr, &sampleTerrain);
}

void GameCharacter::updateMovement(const InputState& input, float deltaSeconds,
    const GeneratedTerrain* terrain, const TerrainSampleQuery* sampleTerrain)
{
    // Match the follow camera's screen-space right axis. With the demo character facing
    // world +Z, the camera looks toward +Z and its right side is world -X.
    const glm::vec3 right = glm::normalize(
        glm::cross(movementBasisForward_, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 movement(0.0f);
    if (input.moveForward)
    {
        movement += movementBasisForward_;
    }
    if (input.moveBackward)
    {
        movement -= movementBasisForward_;
    }
    if (input.moveLeft)
    {
        movement -= right;
    }
    if (input.moveRight)
    {
        movement += right;
    }
    const float activeDelta = std::isfinite(deltaSeconds) ?
        std::max(deltaSeconds, 0.0f) : 0.0f;
    const bool hasInput = glm::dot(movement, movement) > 0.0f;
    const glm::vec3 desiredVelocity = hasInput ?
        glm::normalize(movement) * config_.maximumSpeed : glm::vec3(0.0f);
    const float response = hasInput ? config_.acceleration : config_.braking;
    const glm::vec3 previousVelocity = velocity_;
    velocity_ = moveTowards(velocity_, desiredVelocity, response * activeDelta);

    const glm::vec3 frameVelocity = instantResponse_ ? velocity_ :
        (previousVelocity + velocity_) * 0.5f;
    glm::vec3 candidate = position_ + frameVelocity * activeDelta;
    if (terrain != nullptr)
    {
        const glm::vec2 clamped = terrain->clampToBounds({candidate.x, candidate.z});
        candidate.x = clamped.x;
        candidate.z = clamped.y;
        const std::optional<TerrainSample> ground = terrain->sample(clamped);
        if (ground && ground->normal.y >= std::cos(config_.maximumSlopeRadians))
        {
            candidate.y = ground->height + config_.groundOffset;
        }
        else
        {
            candidate = position_;
            velocity_ = glm::vec3(0.0f);
        }
    }
    else if (sampleTerrain != nullptr)
    {
        const std::optional<TerrainSample> ground =
            (*sampleTerrain)({candidate.x, candidate.z});
        if (ground && ground->normal.y >= std::cos(config_.maximumSlopeRadians))
        {
            candidate.y = ground->height + config_.groundOffset;
        }
        else
        {
            candidate = position_;
            velocity_ = glm::vec3(0.0f);
        }
    }
    position_ = candidate;

    if (glm::dot(velocity_, velocity_) > minimumDirectionLengthSquared)
    {
        const glm::vec3 targetForward = glm::normalize(
            glm::vec3(velocity_.x, 0.0f, velocity_.z));
        const float currentYaw = std::atan2(forward_.x, forward_.z);
        const float targetYaw = std::atan2(targetForward.x, targetForward.z);
        const float deltaYaw = std::remainder(targetYaw - currentYaw, 6.28318530718f);
        const float turn = std::clamp(deltaYaw,
            -config_.turnSpeedRadians * activeDelta,
            config_.turnSpeedRadians * activeDelta);
        const float yaw = currentYaw + turn;
        forward_ = glm::vec3(std::sin(yaw), 0.0f, std::cos(yaw));
    }
}

void GameCharacter::placeOnGround(const GeneratedTerrain& terrain)
{
    const glm::vec2 clamped = terrain.clampToBounds({position_.x, position_.z});
    position_.x = clamped.x;
    position_.z = clamped.y;
    if (const std::optional<TerrainSample> ground = terrain.sample(clamped))
    {
        position_.y = ground->height + config_.groundOffset;
    }
}

bool GameCharacter::moving() const noexcept
{
    return glm::dot(velocity_, velocity_) > minimumDirectionLengthSquared;
}

glm::mat4 GameCharacter::worldOffset() const
{
    const float initialYaw = std::atan2(movementBasisForward_.x, movementBasisForward_.z);
    const float currentYaw = std::atan2(forward_.x, forward_.z);
    return glm::translate(glm::mat4(1.0f), position_) *
        glm::rotate(glm::mat4(1.0f), currentYaw - initialYaw,
            glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::translate(glm::mat4(1.0f), -initialPosition_);
}

}
