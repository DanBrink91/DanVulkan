#include "src/scene_planner.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace
{
void require(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

danvulkan::ScenePlanningLimits generousLimits()
{
    return {
        .textureCapacity = 256,
        .materialCapacity = 2048,
        .minimumVertexCapacity = 16,
        .minimumIndexCapacity = 16,
        .transformCapacity = 2048,
        .drawCapacity = 2048,
        .jointMatrixCapacity = 16384
    };
}
}

int main()
{
    try
    {
        const danvulkan::assets::SceneAsset triangle =
            danvulkan::assets::loadScene("models/triangle.gltf");
        const danvulkan::ScenePlan trianglePlan =
            danvulkan::planScene(triangle, generousLimits());
        require(trianglePlan.meshes.size() == 1, "triangle mesh planning failed");
        require(trianglePlan.transforms.size() == 1, "triangle transform planning failed");
        require(trianglePlan.draws.size() == 1, "triangle draw planning failed");
        require(trianglePlan.usedVertexCount == 3 && trianglePlan.usedIndexCount == 3,
            "triangle geometry packing failed");
        require(trianglePlan.vertexCapacity == 16 && trianglePlan.indexCapacity == 16,
            "minimum geometry capacity was not retained");
        require(trianglePlan.vertices.size() == trianglePlan.usedVertexCount &&
            trianglePlan.indices.size() == trianglePlan.usedIndexCount,
            "the scene plan retained unused CPU geometry capacity");

        danvulkan::assets::SceneAsset composed =
            danvulkan::assets::loadScene("models/naruto_hiddenly_village.glb");
        static_cast<void>(composed.append(
            danvulkan::assets::loadScene("models/ninja_run_free_fire_emote.glb"),
            glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f))));
        const danvulkan::ScenePlan animatedPlan =
            danvulkan::planScene(composed, generousLimits());
        require(animatedPlan.draws.size() == 7, "composed scene lost planned draws");
        require(animatedPlan.jointMatrixCount == 375,
            "skinned node palette assignments were not planned");
        require(animatedPlan.transforms.size() < composed.nodes().size(),
            "non-renderable hierarchy nodes incorrectly consumed transform slots");

        danvulkan::ScenePlanningLimits constrained = generousLimits();
        constrained.drawCapacity = 1;
        bool capacityRejected = false;
        try
        {
            static_cast<void>(danvulkan::planScene(composed, constrained));
        }
        catch (const std::runtime_error&)
        {
            capacityRejected = true;
        }
        require(capacityRejected, "draw-capacity overflow was accepted");

        // SceneAsset is append-only, so build a cycle with the handles the two new nodes will
        // receive. The planner must reject it before animation or renderer traversal.
        danvulkan::assets::SceneAsset cycleScene;
        danvulkan::assets::TextureAsset texture;
        texture.width = 1;
        texture.height = 1;
        texture.rgba8.resize(4);
        const auto textureHandle = cycleScene.addTexture(std::move(texture));
        danvulkan::assets::MaterialAsset material;
        material.albedoTexture = textureHandle;
        const auto materialHandle = cycleScene.addMaterial(std::move(material));
        danvulkan::assets::MeshAsset mesh;
        mesh.material = materialHandle;
        mesh.vertices.resize(3);
        mesh.indices = {0, 1, 2};
        const auto meshHandle = cycleScene.addMesh(std::move(mesh));
        danvulkan::assets::NodeAsset first;
        first.meshes.push_back(meshHandle);
        first.children.push_back({1, 1});
        danvulkan::assets::NodeAsset second;
        second.children.push_back({0, 1});
        const auto firstHandle = cycleScene.addNode(std::move(first));
        static_cast<void>(cycleScene.addNode(std::move(second)));
        cycleScene.addRootNode(firstHandle);

        bool cycleRejected = false;
        try
        {
            static_cast<void>(danvulkan::planScene(cycleScene, generousLimits()));
        }
        catch (const std::runtime_error&)
        {
            cycleRejected = true;
        }
        require(cycleRejected, "cyclic scene hierarchy was accepted");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
