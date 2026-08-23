#include "src/lighting_planner.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
}

int main()
{
    std::vector<ScenePointLight> lights{
        {{1.0f, 2.0f, 3.0f}, 5.0f, {0.5f, 0.75f, 1.0f}, 20.0f},
        {{-1.0f, 0.0f, 2.0f}, 0.0f, {1.0f, 0.25f, 0.1f}, 4.0f}
    };
    SceneEnvironment environment;
    SceneAtmosphere atmosphere;
    std::vector<SceneDirectionalLight> directionalLights{
        {{-1.0f, -2.0f, 0.5f}, 3.0f, {1.0f, 0.9f, 0.8f}, true, 4.0f, 10.0f}
    };
    environment.intensity = 0.5f;
    environment.rotation = 1.0f;
    environment.diffuseStrength = 0.8f;
    environment.specularStrength = 1.2f;
    danvulkan::LightingPlan plan;
    require(danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan) &&
        plan.pointLights.size() == 2 && plan.directionalLightCount == 1 &&
        plan.shadowLightIndex == 0,
        "valid multiple-light input did not produce a plan");
    require(plan.pointLights[0].positionRange.w == 5.0f &&
        plan.pointLights[0].colorIntensity.w == 20.0f,
        "point-light range or intensity was not packed into the GPU ABI");
    require(plan.directionalLights[0].directionIntensity.w == 3.0f &&
        plan.directionalLights[0].colorShadow.w == 1.0f,
        "directional-light intensity or shadow ownership was not packed into the GPU ABI");
    const glm::mat4 shadowMatrix = danvulkan::directionalShadowViewProjection(
        directionalLights[0], {2.0f, 0.0f, -3.0f}, 2048);
    require(std::isfinite(shadowMatrix[0][0]) && std::isfinite(shadowMatrix[3][2]),
        "directional shadow planning produced a non-finite matrix");
    const glm::vec3 lightDirection = glm::normalize(directionalLights[0].direction);
    const glm::vec3 lightRight = glm::normalize(
        glm::cross(lightDirection, glm::vec3(0.0f, 1.0f, 0.0f)));
    const float shadowTexel = directionalLights[0].shadowHalfExtent * 2.0f / 2048.0f;
    const glm::mat4 stableA = danvulkan::directionalShadowViewProjection(
        directionalLights[0], glm::vec3(0.0f), 2048);
    const glm::mat4 stableB = danvulkan::directionalShadowViewProjection(
        directionalLights[0], lightRight * shadowTexel * 0.2f, 2048);
    bool stable = true;
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            stable = stable && std::abs(stableA[column][row] - stableB[column][row]) < 0.000001f;
        }
    }
    require(stable, "sub-texel focus motion changed the directional shadow projection");
    require(plan.environmentTintIntensity.w == 0.5f &&
        plan.environmentControls.x == 1.0f && plan.environmentControls.y == 0.8f &&
        plan.environmentControls.z == 1.2f,
        "environment controls were not preserved");
    require(plan.atmosphereFogColorDensity.w == atmosphere.fogDensity &&
        plan.atmosphereFogParameters.y == atmosphere.fogHeightFalloff &&
        plan.atmosphereScatteringParameters.x == atmosphere.godRayStrength &&
        plan.atmosphereCloudShapeParameters.x == atmosphere.cloudCoverage &&
        plan.atmosphereCloudShapeParameters.y == atmosphere.cloudDensity &&
        plan.atmosphereCloudMovementParameters.z == atmosphere.cloudWindSpeed &&
        plan.atmosphereCloudMovementParameters.w == atmosphere.cloudAltitude &&
        plan.atmosphereCloudLightingParameters.x == atmosphere.cloudShadowStrength,
        "atmosphere controls were not preserved");
    const auto* retainedStorage = plan.pointLights.data();

    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 1, plan),
        "a light list larger than capacity was accepted");
    require(plan.pointLights.size() == 2 && plan.pointLights.data() == retainedStorage,
        "invalid lighting input changed the previous valid plan");
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 0, plan) &&
        !danvulkan::planLighting(lights, directionalLights, environment, atmosphere,
            MaxScenePointLights + 1, plan),
        "invalid light-buffer capacities were accepted");
    lights[0].intensity = -1.0f;
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "a negative light intensity was accepted");
    lights[0].intensity = 1.0f;
    environment.rotation = std::numeric_limits<float>::infinity();
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "a non-finite environment rotation was accepted");
    environment.rotation = 0.0f;
    environment.tint.x = -0.1f;
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "a negative environment tint was accepted");
    environment.tint.x = 1.0f;
    atmosphere.fogMaxOpacity = 1.1f;
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "an invalid fog opacity was accepted");
    atmosphere.fogMaxOpacity = 0.8f;
    atmosphere.sunAngularRadius = 0.0f;
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "an invalid sun radius was accepted");
    atmosphere.sunAngularRadius = 0.0093f;
    atmosphere.cloudCoverage = 1.1f;
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "an invalid cloud coverage was accepted");
    atmosphere.cloudCoverage = 0.52f;
    atmosphere.cloudWindDirection = glm::vec2(0.0f);
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "a zero cloud wind direction was accepted");
    atmosphere.cloudWindDirection = {1.0f, 0.28f};
    atmosphere.cloudShadowStrength = -0.1f;
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "an invalid cloud shadow strength was accepted");
    atmosphere.cloudShadowStrength = 0.48f;
    lights.clear();
    require(danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan) &&
        plan.pointLights.empty() && plan.pointLights.data() == retainedStorage,
        "an empty light list did not retain planner storage");
    directionalLights.push_back(directionalLights.front());
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "multiple shadow-casting directional lights were accepted");
    directionalLights.resize(MaxSceneDirectionalLights + 1U);
    require(!danvulkan::planLighting(lights, directionalLights, environment, atmosphere, 8, plan),
        "directional light list larger than the shader ABI was accepted");
    return EXIT_SUCCESS;
}
