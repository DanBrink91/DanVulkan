#pragma once

#include <danvulkan/assets.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace danvulkan
{
struct ScenePlanningLimits
{
    std::uint32_t textureCapacity = 0;
    std::uint32_t materialCapacity = 0;
    std::uint32_t minimumVertexCapacity = 0;
    std::uint32_t minimumIndexCapacity = 0;
    std::uint32_t transformCapacity = 0;
    std::uint32_t drawCapacity = 0;
    std::uint32_t jointMatrixCapacity = 0;
    std::uint64_t maximumVertexBufferBytes = std::numeric_limits<std::uint64_t>::max();
};

struct PlannedSceneMesh
{
    std::uint32_t indexCount = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t vertexOffset = 0;
    std::uint32_t materialIndex = 0;
    assets::Bounds bounds{};
    std::string name;
};

struct PlannedSceneTransform
{
    assets::NodeHandle node;
    std::string name;
    glm::mat4 world{1.0f};
};

struct PlannedSceneDraw
{
    std::uint32_t meshIndex = 0;
    std::uint32_t transformIndex = 0;
    std::uint32_t jointOffset = 0;
    assets::NodeHandle node;
    assets::SkinHandle skin;
    assets::Bounds worldBounds{};
};

struct ScenePlan
{
    std::uint32_t vertexCapacity = 0;
    std::uint32_t indexCapacity = 0;
    std::uint32_t usedVertexCount = 0;
    std::uint32_t usedIndexCount = 0;
    std::uint32_t jointMatrixCount = 0;
    std::vector<assets::Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<PlannedSceneMesh> meshes;
    std::vector<PlannedSceneTransform> transforms;
    std::vector<PlannedSceneDraw> draws;
};

// Validates an authored scene and converts its hierarchy into deterministic packed geometry,
// transform, draw, bounds, and skin-palette assignments. This function creates no Vulkan objects.
[[nodiscard]] ScenePlan planScene(
    const assets::SceneAsset& scene, const ScenePlanningLimits& limits);
}
