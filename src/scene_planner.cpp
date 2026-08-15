#include "scene_planner.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace danvulkan
{
namespace
{
constexpr std::uint32_t initialGeneration = 1;

[[nodiscard]] bool finite(const glm::mat4& value)
{
    for (glm::length_t column = 0; column < 4; ++column)
    {
        for (glm::length_t row = 0; row < 4; ++row)
        {
            if (!std::isfinite(value[column][row]))
            {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] assets::Bounds transformedBounds(
    const assets::Bounds& bounds, const glm::mat4& transform)
{
    const float maximum = std::numeric_limits<float>::max();
    assets::Bounds result{glm::vec3(maximum), glm::vec3(-maximum)};
    for (std::uint32_t corner = 0; corner < 8; ++corner)
    {
        const glm::vec3 local{
            (corner & 1U) != 0 ? bounds.maxVertex.x : bounds.minVertex.x,
            (corner & 2U) != 0 ? bounds.maxVertex.y : bounds.minVertex.y,
            (corner & 4U) != 0 ? bounds.maxVertex.z : bounds.minVertex.z
        };
        const glm::vec3 world = glm::vec3(transform * glm::vec4(local, 1.0f));
        result.minVertex = glm::min(result.minVertex, world);
        result.maxVertex = glm::max(result.maxVertex, world);
    }
    return result;
}

void requireTexture(const assets::SceneAsset& scene, assets::TextureHandle handle,
    std::string_view role)
{
    if (handle && scene.find(handle) == nullptr)
    {
        throw std::runtime_error("scene material contains an invalid " + std::string(role) +
            " texture handle");
    }
}
}

ScenePlan planScene(const assets::SceneAsset& scene, const ScenePlanningLimits& limits)
{
    if (limits.textureCapacity == 0 || scene.textures().empty() ||
        scene.textures().size() > limits.textureCapacity)
    {
        throw std::runtime_error("scene texture count exceeds renderer capacity");
    }
    if (limits.materialCapacity == 0 || scene.materials().empty() ||
        scene.materials().size() > limits.materialCapacity)
    {
        throw std::runtime_error("scene material count exceeds renderer capacity");
    }
    if (limits.transformCapacity == 0 || limits.drawCapacity == 0 ||
        limits.jointMatrixCapacity == 0)
    {
        throw std::runtime_error("scene planning capacities must be greater than zero");
    }
    if (scene.meshes().empty() || scene.rootNodes().empty())
    {
        throw std::runtime_error("scene contains no renderable roots or meshes");
    }

    for (const assets::TextureAsset& texture : scene.textures())
    {
        const std::uint64_t rowBytes = static_cast<std::uint64_t>(texture.width) * 4U;
        const bool overflow = texture.height != 0 &&
            rowBytes > std::numeric_limits<std::uint64_t>::max() / texture.height;
        const std::uint64_t byteCount = overflow ? 0 :
            rowBytes * static_cast<std::uint64_t>(texture.height);
        if (texture.handle.generation != initialGeneration ||
            scene.find(texture.handle) != &texture || texture.width == 0 || texture.height == 0 ||
            overflow || byteCount != texture.rgba8.size())
        {
            throw std::runtime_error("scene contains invalid decoded texture data");
        }
    }

    for (const assets::MaterialAsset& material : scene.materials())
    {
        if (material.handle.generation != initialGeneration ||
            scene.find(material.handle) != &material)
        {
            throw std::runtime_error("scene contains an invalid material handle");
        }
        switch (material.alphaMode)
        {
        case assets::AlphaMode::opaque:
        case assets::AlphaMode::mask:
        case assets::AlphaMode::blend:
            break;
        default:
            throw std::runtime_error("scene contains an invalid material alpha mode");
        }
        requireTexture(scene, material.albedoTexture, "base-color");
        requireTexture(scene, material.normalTexture, "normal");
        requireTexture(scene, material.metallicRoughnessTexture, "metallic-roughness");
        requireTexture(scene, material.occlusionTexture, "occlusion");
        requireTexture(scene, material.emissiveTexture, "emissive");
    }

    ScenePlan plan;
    plan.meshes.resize(scene.meshes().size());
    std::uint64_t usedVertices = 0;
    std::uint64_t usedIndices = 0;
    for (const assets::MeshAsset& mesh : scene.meshes())
    {
        if (mesh.handle.generation != initialGeneration || scene.find(mesh.handle) != &mesh ||
            scene.find(mesh.material) == nullptr || mesh.vertices.empty() || mesh.indices.empty())
        {
            throw std::runtime_error("scene contains invalid mesh geometry or handles");
        }
        for (const std::uint32_t index : mesh.indices)
        {
            if (index >= mesh.vertices.size())
            {
                throw std::runtime_error("scene mesh contains an out-of-range vertex index");
            }
        }
        if (usedVertices + mesh.vertices.size() >
                static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            usedIndices + mesh.indices.size() > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("scene geometry exceeds renderer offset limits");
        }

        PlannedSceneMesh& destination = plan.meshes.at(mesh.handle.slot);
        destination.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
        destination.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
        destination.firstIndex = static_cast<std::uint32_t>(usedIndices);
        destination.vertexOffset = static_cast<std::uint32_t>(usedVertices);
        destination.materialIndex = mesh.material.slot;
        destination.bounds = mesh.bounds;
        destination.name = mesh.name;
        plan.vertices.insert(plan.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        plan.indices.insert(plan.indices.end(), mesh.indices.begin(), mesh.indices.end());
        usedVertices += mesh.vertices.size();
        usedIndices += mesh.indices.size();
    }

    for (const assets::SkinAsset& skin : scene.skins())
    {
        if (skin.handle.generation != initialGeneration || scene.find(skin.handle) != &skin ||
            skin.joints.empty() || skin.joints.size() != skin.inverseBindMatrices.size())
        {
            throw std::runtime_error("scene contains invalid skin data");
        }
        if (skin.skeleton && scene.find(skin.skeleton) == nullptr)
        {
            throw std::runtime_error("scene skin contains an invalid skeleton handle");
        }
        for (const assets::NodeHandle joint : skin.joints)
        {
            if (scene.find(joint) == nullptr)
            {
                throw std::runtime_error("scene skin contains an invalid joint handle");
            }
        }
    }

    for (const assets::AnimationClipAsset& clip : scene.animations())
    {
        if (clip.handle.generation != initialGeneration || scene.find(clip.handle) != &clip ||
            !std::isfinite(clip.startTime) || !std::isfinite(clip.endTime) ||
            clip.endTime < clip.startTime)
        {
            throw std::runtime_error("scene contains invalid animation clip metadata");
        }
        for (const assets::AnimationChannelAsset& channel : clip.channels)
        {
            const assets::NodeAsset* target = scene.find(channel.target);
            if (target == nullptr || !target->transformIsTrs || channel.times.empty() ||
                channel.times.size() != channel.values.size())
            {
                throw std::runtime_error("scene contains invalid animation channel data");
            }
            for (std::size_t index = 0; index < channel.times.size(); ++index)
            {
                if (!std::isfinite(channel.times[index]) ||
                    (index != 0 && channel.times[index] < channel.times[index - 1]))
                {
                    throw std::runtime_error("scene animation sample times are invalid");
                }
                for (glm::length_t component = 0; component < 4; ++component)
                {
                    if (!std::isfinite(channel.values[index][component]))
                    {
                        throw std::runtime_error("scene animation sample values are invalid");
                    }
                }
            }
        }
    }

    // 0 = unvisited, 1 = active recursion path, 2 = already owned by a root/parent.
    std::vector<std::uint8_t> nodeStates(scene.nodes().size(), 0);
    std::function<void(assets::NodeHandle, const glm::mat4&)> visit;
    visit = [&](assets::NodeHandle handle, const glm::mat4& parent)
    {
        const assets::NodeAsset* node = scene.find(handle);
        if (node == nullptr || handle.slot >= nodeStates.size())
        {
            throw std::runtime_error("scene hierarchy contains an invalid node handle");
        }
        if (nodeStates[handle.slot] == 1)
        {
            throw std::runtime_error("scene node hierarchy contains a cycle");
        }
        if (nodeStates[handle.slot] == 2)
        {
            throw std::runtime_error("scene node has multiple parents or duplicate roots");
        }
        if (!finite(node->localTransform))
        {
            throw std::runtime_error("scene node contains a non-finite transform");
        }
        nodeStates[handle.slot] = 1;
        const glm::mat4 world = parent * node->localTransform;

        std::uint32_t transformIndex = 0;
        if (!node->meshes.empty())
        {
            if (plan.transforms.size() >= limits.transformCapacity)
            {
                throw std::runtime_error("scene exceeds renderer transform capacity");
            }
            transformIndex = static_cast<std::uint32_t>(plan.transforms.size());
            plan.transforms.push_back({handle, node->name, world});
        }

        const assets::SkinAsset* skin = nullptr;
        if (node->skin)
        {
            skin = scene.find(node->skin);
            if (skin == nullptr)
            {
                throw std::runtime_error("scene node contains an invalid skin handle");
            }
        }
        std::uint32_t nodeJointOffset = 0;
        if (skin != nullptr)
        {
            if (skin->joints.size() > limits.jointMatrixCapacity - plan.jointMatrixCount)
            {
                throw std::runtime_error("scene exceeds renderer joint matrix capacity");
            }
            nodeJointOffset = plan.jointMatrixCount;
            plan.jointMatrixCount += static_cast<std::uint32_t>(skin->joints.size());
        }
        for (const assets::MeshHandle meshHandle : node->meshes)
        {
            const assets::MeshAsset* mesh = scene.find(meshHandle);
            if (mesh == nullptr || meshHandle.slot >= plan.meshes.size())
            {
                throw std::runtime_error("scene node contains an invalid mesh handle");
            }
            if (plan.draws.size() >= limits.drawCapacity)
            {
                throw std::runtime_error("scene exceeds renderer draw capacity");
            }

            if (skin != nullptr)
            {
                for (const assets::Vertex& vertex : mesh->vertices)
                {
                    for (glm::length_t component = 0; component < 4; ++component)
                    {
                        if (vertex.weights[component] > 0.0f &&
                            vertex.joints[component] >= skin->joints.size())
                        {
                            throw std::runtime_error(
                                "scene vertex references an out-of-range skin joint");
                        }
                    }
                }
            }
            plan.draws.push_back({meshHandle.slot, transformIndex, nodeJointOffset, handle,
                node->skin, transformedBounds(mesh->bounds, world)});
        }

        for (const assets::NodeHandle child : node->children)
        {
            visit(child, world);
        }
        nodeStates[handle.slot] = 2;
    };

    for (const assets::NodeHandle root : scene.rootNodes())
    {
        visit(root, glm::mat4(1.0f));
    }
    if (plan.draws.empty())
    {
        throw std::runtime_error("scene contains no mesh instances");
    }
    for (const assets::SkinAsset& skin : scene.skins())
    {
        for (const assets::NodeHandle joint : skin.joints)
        {
            if (nodeStates[joint.slot] != 2)
            {
                throw std::runtime_error("scene skin references a node outside the active scene");
            }
        }
    }
    for (const assets::AnimationClipAsset& clip : scene.animations())
    {
        for (const assets::AnimationChannelAsset& channel : clip.channels)
        {
            if (nodeStates[channel.target.slot] != 2)
            {
                throw std::runtime_error(
                    "scene animation targets a node outside the active scene");
            }
        }
    }

    plan.usedVertexCount = static_cast<std::uint32_t>(usedVertices);
    plan.usedIndexCount = static_cast<std::uint32_t>(usedIndices);
    plan.vertexCapacity = std::max(limits.minimumVertexCapacity, plan.usedVertexCount);
    plan.indexCapacity = std::max(limits.minimumIndexCapacity, plan.usedIndexCount);
    if (plan.vertexCapacity == 0 || plan.indexCapacity == 0 ||
        static_cast<std::uint64_t>(sizeof(assets::Vertex)) * plan.vertexCapacity >
            limits.maximumVertexBufferBytes)
    {
        throw std::runtime_error("scene geometry capacity exceeds renderer limits");
    }
    return plan;
}
}
