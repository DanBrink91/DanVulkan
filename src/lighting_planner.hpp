#pragma once

#include <danvulkan/renderer.hpp>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <array>
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

struct PlannedDirectionalLight
{
    glm::vec4 directionIntensity;
    glm::vec4 colorShadow;
};
static_assert(sizeof(PlannedDirectionalLight) == 32);

struct LightingPlan
{
    std::vector<PlannedPointLight> pointLights;
    std::array<PlannedDirectionalLight, MaxSceneDirectionalLights> directionalLights{};
    std::uint32_t directionalLightCount = 0;
    std::uint32_t shadowLightIndex = MaxSceneDirectionalLights;
    glm::vec4 environmentTintIntensity;
    // x rotation, y diffuse strength, z specular strength, w unused.
    glm::vec4 environmentControls;
    glm::vec4 atmosphereSkyZenithIntensity;
    glm::vec4 atmosphereSkyHorizonExponent;
    glm::vec4 atmosphereFogColorDensity;
    // x base height, y height falloff, z maximum opacity, w mist variation.
    glm::vec4 atmosphereFogParameters;
    // x god-ray strength, y maximum distance, z sun angular radius, w sun glow strength.
    glm::vec4 atmosphereScatteringParameters;
    // x coverage, y density, z world-space noise scale, w edge softness.
    glm::vec4 atmosphereCloudShapeParameters;
    // xy wind direction, z speed, w base altitude.
    glm::vec4 atmosphereCloudMovementParameters;
    // x ground-shadow strength, y silver lining, z brightness, w layer separation.
    glm::vec4 atmosphereCloudLightingParameters;
};

// Valid input replaces output while retaining its point-light capacity for later frames.
// Invalid input leaves output unchanged.
[[nodiscard]] bool planLighting(
    std::span<const ScenePointLight> pointLights,
    std::span<const SceneDirectionalLight> directionalLights,
    const SceneEnvironment& environment,
    const SceneAtmosphere& atmosphere,
    std::uint32_t capacity,
    LightingPlan& output);

[[nodiscard]] glm::mat4 directionalShadowViewProjection(
    const SceneDirectionalLight& light, const glm::vec3& focus,
    std::uint32_t shadowResolution);
}
