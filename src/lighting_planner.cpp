#include "lighting_planner.hpp"

#include <cmath>

namespace danvulkan
{
namespace
{
bool finite(glm::vec3 value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
}

bool planLighting(std::span<const ScenePointLight> pointLights,
    const SceneEnvironment& environment, std::uint32_t capacity, LightingPlan& output)
{
    if (capacity == 0 || capacity > MaxScenePointLights || pointLights.size() > capacity ||
        !finite(environment.tint) || !std::isfinite(environment.intensity) ||
        !std::isfinite(environment.rotation) || !std::isfinite(environment.diffuseStrength) ||
        !std::isfinite(environment.specularStrength) || environment.intensity < 0.0f ||
        environment.tint.x < 0.0f || environment.tint.y < 0.0f ||
        environment.tint.z < 0.0f || environment.diffuseStrength < 0.0f ||
        environment.specularStrength < 0.0f)
    {
        return false;
    }

    for (const ScenePointLight& light : pointLights)
    {
        if (!finite(light.position) || !finite(light.color) || !std::isfinite(light.range) ||
            !std::isfinite(light.intensity) || light.range < 0.0f || light.intensity < 0.0f ||
            light.color.x < 0.0f || light.color.y < 0.0f || light.color.z < 0.0f)
        {
            return false;
        }
    }

    output.pointLights.clear();
    output.pointLights.reserve(pointLights.size());
    for (const ScenePointLight& light : pointLights)
    {
        output.pointLights.push_back({glm::vec4(light.position, light.range),
            glm::vec4(light.color, light.intensity)});
    }
    output.environmentTintIntensity = glm::vec4(environment.tint, environment.intensity);
    output.environmentControls = {environment.rotation, environment.diffuseStrength,
        environment.specularStrength, 0.0f};
    return true;
}
}
