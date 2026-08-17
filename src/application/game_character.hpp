#pragma once

#include "input_state.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace danvulkan::application {

// Small gameplay-owned controller for a movable scene actor. Rendering remains unaware of
// controls; it receives only the resulting actor offset through SceneSubmission.
class GameCharacter
{
public:
    GameCharacter(glm::vec3 initialPosition, glm::vec3 forwardDirection,
        float movementSpeed);

    void updateMovement(const InputState& input, float deltaSeconds);

    [[nodiscard]] const glm::vec3& position() const noexcept { return position_; }
    [[nodiscard]] const glm::vec3& forward() const noexcept { return forward_; }
    [[nodiscard]] glm::mat4 worldOffset() const;

private:
    glm::vec3 initialPosition_{0.0f};
    glm::vec3 position_{0.0f};
    glm::vec3 forward_{0.0f, 0.0f, 1.0f};
    float movementSpeed_ = 1.0f;
};

}
