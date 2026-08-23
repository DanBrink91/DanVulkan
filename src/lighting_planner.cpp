#include "lighting_planner.hpp"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

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
    std::span<const SceneDirectionalLight> directionalLights,
    const SceneEnvironment& environment, const SceneAtmosphere& atmosphere,
    std::uint32_t capacity, LightingPlan& output)
{
    if (capacity == 0 || capacity > MaxScenePointLights || pointLights.size() > capacity ||
        directionalLights.size() > MaxSceneDirectionalLights ||
        !finite(environment.tint) || !std::isfinite(environment.intensity) ||
        !std::isfinite(environment.rotation) || !std::isfinite(environment.diffuseStrength) ||
        !std::isfinite(environment.specularStrength) || environment.intensity < 0.0f ||
        environment.tint.x < 0.0f || environment.tint.y < 0.0f ||
        environment.tint.z < 0.0f || environment.diffuseStrength < 0.0f ||
        environment.specularStrength < 0.0f ||
        !finite(atmosphere.skyZenithColor) || !finite(atmosphere.skyHorizonColor) ||
        !finite(atmosphere.fogColor) || !std::isfinite(atmosphere.skyIntensity) ||
        !std::isfinite(atmosphere.skyGradientExponent) ||
        !std::isfinite(atmosphere.fogDensity) ||
        !std::isfinite(atmosphere.fogBaseHeight) ||
        !std::isfinite(atmosphere.fogHeightFalloff) ||
        !std::isfinite(atmosphere.fogMaxOpacity) ||
        !std::isfinite(atmosphere.mistVariation) ||
        !std::isfinite(atmosphere.godRayStrength) ||
        !std::isfinite(atmosphere.godRayMaxDistance) ||
        !std::isfinite(atmosphere.sunAngularRadius) ||
        !std::isfinite(atmosphere.sunGlowStrength) ||
        !std::isfinite(atmosphere.cloudCoverage) ||
        !std::isfinite(atmosphere.cloudDensity) ||
        !std::isfinite(atmosphere.cloudScale) ||
        !std::isfinite(atmosphere.cloudSoftness) ||
        !std::isfinite(atmosphere.cloudWindDirection.x) ||
        !std::isfinite(atmosphere.cloudWindDirection.y) ||
        !std::isfinite(atmosphere.cloudWindSpeed) ||
        !std::isfinite(atmosphere.cloudAltitude) ||
        !std::isfinite(atmosphere.cloudShadowStrength) ||
        !std::isfinite(atmosphere.cloudSilverLiningStrength) ||
        !std::isfinite(atmosphere.cloudBrightness) ||
        !std::isfinite(atmosphere.cloudLayerSeparation) ||
        atmosphere.skyIntensity < 0.0f || atmosphere.skyGradientExponent <= 0.0f ||
        atmosphere.fogDensity < 0.0f || atmosphere.fogHeightFalloff < 0.0f ||
        atmosphere.fogMaxOpacity < 0.0f || atmosphere.fogMaxOpacity > 1.0f ||
        atmosphere.mistVariation < 0.0f || atmosphere.mistVariation > 1.0f ||
        atmosphere.godRayStrength < 0.0f || atmosphere.godRayMaxDistance <= 0.0f ||
        atmosphere.sunAngularRadius <= 0.0f || atmosphere.sunAngularRadius >= 0.25f ||
        atmosphere.sunGlowStrength < 0.0f ||
        atmosphere.cloudCoverage < 0.0f || atmosphere.cloudCoverage > 1.0f ||
        atmosphere.cloudDensity < 0.0f || atmosphere.cloudScale <= 0.0f ||
        atmosphere.cloudSoftness <= 0.0f || atmosphere.cloudSoftness > 0.5f ||
        glm::dot(atmosphere.cloudWindDirection, atmosphere.cloudWindDirection) < 0.000001f ||
        atmosphere.cloudWindSpeed < 0.0f || atmosphere.cloudAltitude <= 0.0f ||
        atmosphere.cloudShadowStrength < 0.0f || atmosphere.cloudShadowStrength > 1.0f ||
        atmosphere.cloudSilverLiningStrength < 0.0f || atmosphere.cloudBrightness < 0.0f ||
        atmosphere.cloudLayerSeparation <= 0.0f ||
        atmosphere.skyZenithColor.x < 0.0f || atmosphere.skyZenithColor.y < 0.0f ||
        atmosphere.skyZenithColor.z < 0.0f || atmosphere.skyHorizonColor.x < 0.0f ||
        atmosphere.skyHorizonColor.y < 0.0f || atmosphere.skyHorizonColor.z < 0.0f ||
        atmosphere.fogColor.x < 0.0f || atmosphere.fogColor.y < 0.0f ||
        atmosphere.fogColor.z < 0.0f)
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

    std::uint32_t shadowLightIndex = MaxSceneDirectionalLights;
    for (std::size_t index = 0; index < directionalLights.size(); ++index)
    {
        const SceneDirectionalLight& light = directionalLights[index];
        const float directionLengthSquared = glm::dot(light.direction, light.direction);
        if (!finite(light.direction) || directionLengthSquared < 0.000001f ||
            !finite(light.color) || !std::isfinite(light.intensity) ||
            !std::isfinite(light.shadowHalfExtent) || !std::isfinite(light.shadowDepth) ||
            light.intensity < 0.0f || light.color.x < 0.0f || light.color.y < 0.0f ||
            light.color.z < 0.0f || light.shadowHalfExtent <= 0.0f ||
            light.shadowDepth <= 0.0f ||
            (light.castsShadows && shadowLightIndex != MaxSceneDirectionalLights))
        {
            return false;
        }
        if (light.castsShadows)
        {
            shadowLightIndex = static_cast<std::uint32_t>(index);
        }
    }

    output.pointLights.clear();
    output.pointLights.reserve(pointLights.size());
    for (const ScenePointLight& light : pointLights)
    {
        output.pointLights.push_back({glm::vec4(light.position, light.range),
            glm::vec4(light.color, light.intensity)});
    }
    output.directionalLights = {};
    output.directionalLightCount = static_cast<std::uint32_t>(directionalLights.size());
    output.shadowLightIndex = shadowLightIndex;
    for (std::size_t index = 0; index < directionalLights.size(); ++index)
    {
        const SceneDirectionalLight& light = directionalLights[index];
        output.directionalLights[index] = {
            glm::vec4(glm::normalize(light.direction), light.intensity),
            glm::vec4(light.color, light.castsShadows ? 1.0f : 0.0f)};
    }
    output.environmentTintIntensity = glm::vec4(environment.tint, environment.intensity);
    output.environmentControls = {environment.rotation, environment.diffuseStrength,
        environment.specularStrength, 0.0f};
    output.atmosphereSkyZenithIntensity =
        glm::vec4(atmosphere.skyZenithColor, atmosphere.skyIntensity);
    output.atmosphereSkyHorizonExponent =
        glm::vec4(atmosphere.skyHorizonColor, atmosphere.skyGradientExponent);
    output.atmosphereFogColorDensity =
        glm::vec4(atmosphere.fogColor, atmosphere.fogDensity);
    output.atmosphereFogParameters = {atmosphere.fogBaseHeight,
        atmosphere.fogHeightFalloff, atmosphere.fogMaxOpacity, atmosphere.mistVariation};
    output.atmosphereScatteringParameters = {atmosphere.godRayStrength,
        atmosphere.godRayMaxDistance, atmosphere.sunAngularRadius,
        atmosphere.sunGlowStrength};
    output.atmosphereCloudShapeParameters = {atmosphere.cloudCoverage,
        atmosphere.cloudDensity, atmosphere.cloudScale, atmosphere.cloudSoftness};
    const glm::vec2 cloudWindDirection = glm::normalize(atmosphere.cloudWindDirection);
    output.atmosphereCloudMovementParameters = {cloudWindDirection.x,
        cloudWindDirection.y, atmosphere.cloudWindSpeed, atmosphere.cloudAltitude};
    output.atmosphereCloudLightingParameters = {atmosphere.cloudShadowStrength,
        atmosphere.cloudSilverLiningStrength, atmosphere.cloudBrightness,
        atmosphere.cloudLayerSeparation};
    return true;
}

glm::mat4 directionalShadowViewProjection(const SceneDirectionalLight& light,
    const glm::vec3& focus, std::uint32_t shadowResolution)
{
    const glm::vec3 direction = glm::normalize(light.direction);
    const glm::vec3 up = std::abs(direction.y) > 0.98f
        ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 snappedFocus = focus;
    if (shadowResolution > 0U)
    {
        const float texelWorldSize = (2.0f * light.shadowHalfExtent) /
            static_cast<float>(shadowResolution);
        const glm::vec3 right = glm::normalize(glm::cross(direction, up));
        const glm::vec3 lightUp = glm::normalize(glm::cross(right, direction));
        const float rightDistance = glm::dot(focus, right);
        const float upDistance = glm::dot(focus, lightUp);
        snappedFocus += right *
            (std::round(rightDistance / texelWorldSize) * texelWorldSize - rightDistance);
        snappedFocus += lightUp *
            (std::round(upDistance / texelWorldSize) * texelWorldSize - upDistance);
    }
    const glm::mat4 view = glm::lookAtRH(
        snappedFocus - direction * (light.shadowDepth * 0.5f), snappedFocus, up);

    glm::mat4 projection = glm::orthoRH_ZO(-light.shadowHalfExtent, light.shadowHalfExtent,
        -light.shadowHalfExtent, light.shadowHalfExtent, 0.01f, light.shadowDepth);
    projection[1][1] *= -1.0f;
    return projection * view;
}
}
