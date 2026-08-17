#include "game_character.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace danvulkan::application {
namespace {
constexpr float minimumDirectionLengthSquared = 0.000001f;
}

GameCharacter::GameCharacter(glm::vec3 initialPosition, glm::vec3 forwardDirection,
    float movementSpeed)
    : initialPosition_(initialPosition), position_(initialPosition),
      movementSpeed_(movementSpeed)
{
    const glm::vec3 horizontalForward(forwardDirection.x, 0.0f, forwardDirection.z);
    if (glm::dot(horizontalForward, horizontalForward) <= minimumDirectionLengthSquared ||
        !std::isfinite(movementSpeed) || movementSpeed < 0.0f)
    {
        throw std::invalid_argument("game character movement configuration is invalid");
    }
    forward_ = glm::normalize(horizontalForward);
}

void GameCharacter::updateMovement(const InputState& input, float deltaSeconds)
{
    // Match the follow camera's screen-space right axis. With the demo character facing
    // world +Z, the camera looks toward +Z and its right side is world -X.
    const glm::vec3 right = glm::normalize(
        glm::cross(forward_, glm::vec3(0.0f, 1.0f, 0.0f)));
    glm::vec3 movement(0.0f);
    if (input.moveForward)
    {
        movement += forward_;
    }
    if (input.moveBackward)
    {
        movement -= forward_;
    }
    if (input.moveLeft)
    {
        movement -= right;
    }
    if (input.moveRight)
    {
        movement += right;
    }
    if (glm::dot(movement, movement) > 0.0f)
    {
        position_ += glm::normalize(movement) * movementSpeed_ *
            std::max(deltaSeconds, 0.0f);
    }
}

glm::mat4 GameCharacter::worldOffset() const
{
    return glm::translate(glm::mat4(1.0f), position_ - initialPosition_);
}

}
