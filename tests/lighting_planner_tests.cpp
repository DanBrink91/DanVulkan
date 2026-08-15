#include "src/lighting_planner.hpp"

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
    environment.intensity = 0.5f;
    environment.rotation = 1.0f;
    environment.diffuseStrength = 0.8f;
    environment.specularStrength = 1.2f;
    danvulkan::LightingPlan plan;
    require(danvulkan::planLighting(lights, environment, 8, plan) &&
        plan.pointLights.size() == 2,
        "valid multiple-light input did not produce a plan");
    require(plan.pointLights[0].positionRange.w == 5.0f &&
        plan.pointLights[0].colorIntensity.w == 20.0f,
        "point-light range or intensity was not packed into the GPU ABI");
    require(plan.environmentTintIntensity.w == 0.5f &&
        plan.environmentControls.x == 1.0f && plan.environmentControls.y == 0.8f &&
        plan.environmentControls.z == 1.2f,
        "environment controls were not preserved");
    const auto* retainedStorage = plan.pointLights.data();

    require(!danvulkan::planLighting(lights, environment, 1, plan),
        "a light list larger than capacity was accepted");
    require(plan.pointLights.size() == 2 && plan.pointLights.data() == retainedStorage,
        "invalid lighting input changed the previous valid plan");
    require(!danvulkan::planLighting(lights, environment, 0, plan) &&
        !danvulkan::planLighting(lights, environment, MaxScenePointLights + 1, plan),
        "invalid light-buffer capacities were accepted");
    lights[0].intensity = -1.0f;
    require(!danvulkan::planLighting(lights, environment, 8, plan),
        "a negative light intensity was accepted");
    lights[0].intensity = 1.0f;
    environment.rotation = std::numeric_limits<float>::infinity();
    require(!danvulkan::planLighting(lights, environment, 8, plan),
        "a non-finite environment rotation was accepted");
    environment.rotation = 0.0f;
    environment.tint.x = -0.1f;
    require(!danvulkan::planLighting(lights, environment, 8, plan),
        "a negative environment tint was accepted");
    environment.tint.x = 1.0f;
    lights.clear();
    require(danvulkan::planLighting(lights, environment, 8, plan) &&
        plan.pointLights.empty() && plan.pointLights.data() == retainedStorage,
        "an empty light list did not retain planner storage");
    return EXIT_SUCCESS;
}
