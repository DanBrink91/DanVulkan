#pragma once

#include "input_state.hpp"
#include "generated_terrain.hpp"

#include <functional>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace danvulkan::application {

using TerrainSampleQuery =
    std::function<std::optional<TerrainSample>(const glm::vec2&)>;

struct CharacterMovementConfig
{
    float maximumSpeed = 1.0f;
    float acceleration = 8.0f;
    float braking = 12.0f;
    float turnSpeedRadians = 10.0f;
    // Just under 55 degrees by default. Values must remain below a vertical slope.
    float maximumSlopeRadians = 0.959931f;
    float groundOffset = 0.0f;
};

// Small gameplay-owned controller for a movable scene actor. Rendering remains unaware of
// controls; it receives only the resulting actor offset through SceneSubmission.
class GameCharacter
{
public:
    GameCharacter(glm::vec3 initialPosition, glm::vec3 forwardDirection,
        float movementSpeed);
    GameCharacter(glm::vec3 initialPosition, glm::vec3 forwardDirection,
        CharacterMovementConfig config);

    void updateMovement(const InputState& input, float deltaSeconds);
    void updateMovement(const InputState& input, float deltaSeconds,
        const GeneratedTerrain& terrain);
    // Uses a composite or streamed surface without imposing the bounds of any one terrain chunk.
    void updateMovementOnSurface(const InputState& input, float deltaSeconds,
        const TerrainSampleQuery& sampleTerrain);
    void placeOnGround(const GeneratedTerrain& terrain);

    [[nodiscard]] const glm::vec3& position() const noexcept { return position_; }
    [[nodiscard]] const glm::vec3& forward() const noexcept { return forward_; }
    // Stable heading used to interpret WASD and orient the follow camera. The rendered actor's
    // forward direction is allowed to turn toward its current travel direction independently.
    [[nodiscard]] const glm::vec3& controlForward() const noexcept
    {
        return movementBasisForward_;
    }
    [[nodiscard]] const glm::vec3& velocity() const noexcept { return velocity_; }
    [[nodiscard]] bool moving() const noexcept;
    [[nodiscard]] glm::mat4 worldOffset() const;

private:
    void updateMovement(const InputState& input, float deltaSeconds,
        const GeneratedTerrain* terrain, const TerrainSampleQuery* sampleTerrain);

    glm::vec3 initialPosition_{0.0f};
    glm::vec3 position_{0.0f};
    glm::vec3 movementBasisForward_{0.0f, 0.0f, 1.0f};
    glm::vec3 forward_{0.0f, 0.0f, 1.0f};
    glm::vec3 velocity_{0.0f};
    CharacterMovementConfig config_;
    bool instantResponse_ = false;
};

}
