#pragma once

#include <danvulkan/renderer.hpp>

#include <glm/vec4.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace danvulkan
{
struct PlannedPointLight
{
    glm::vec4 positionRange;
    glm::vec4 colorIntensity;
};
static_assert(sizeof(PlannedPointLight) == 32);

struct LightingPlan
{
    std::vector<PlannedPointLight> pointLights;
    glm::vec4 environmentTintIntensity;
    // x rotation, y diffuse strength, z specular strength, w unused.
    glm::vec4 environmentControls;
};

// Valid input replaces output while retaining its point-light capacity for later frames.
// Invalid input leaves output unchanged.
[[nodiscard]] bool planLighting(
    std::span<const ScenePointLight> pointLights,
    const SceneEnvironment& environment,
    std::uint32_t capacity,
    LightingPlan& output);
}
