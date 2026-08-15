#include "src/scene_context.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cstdlib>
#include <iostream>

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

danvulkan::vk::MeshData mesh(std::uint32_t variant, std::int32_t transformIndex,
    danvulkan::assets::Bounds bounds)
{
    danvulkan::vk::MeshData result{};
    result.indexCount = 3;
    result.pipelineVariant = variant;
    result.drawData.transformIndex = transformIndex;
    result.localBounds = bounds;
    return result;
}
}

int main()
{
    using namespace danvulkan::vk;

    std::vector<GeometryRange> ranges{{0, 2}, {4, 2}};
    SceneContext::releaseGeometryRange(ranges, {2, 2});
    require(ranges.size() == 1 && ranges[0].offset == 0 && ranges[0].count == 6,
        "released adjacent geometry ranges must coalesce");
    require(SceneContext::hasGeometryRange(ranges, 6),
        "coalesced geometry storage must satisfy a contiguous allocation");
    const GeometryRange allocation = SceneContext::allocateGeometryRange(ranges, 3);
    require(allocation.offset == 0 && allocation.count == 3 && ranges.size() == 1 &&
        ranges[0].offset == 3 && ranges[0].count == 3,
        "geometry allocation must consume the front of the first fitting range");
    require(SceneContext::grownGeometryCapacity(16, 1, 128) == 32 &&
        SceneContext::grownGeometryCapacity(16, 40, 128) == 56,
        "geometry growth must choose the larger of doubling and the required capacity");

    SceneContext scene;
    const danvulkan::assets::Bounds nearBounds{{-0.1f, -0.1f, 0.1f}, {0.1f, 0.1f, 0.2f}};
    const danvulkan::assets::Bounds middleBounds{{-0.1f, -0.1f, 0.3f}, {0.1f, 0.1f, 0.4f}};
    const danvulkan::assets::Bounds farBounds{{-0.1f, -0.1f, 0.7f}, {0.1f, 0.1f, 0.8f}};
    const danvulkan::assets::Bounds culledBounds{{4.0f, -0.1f, 0.1f}, {4.2f, 0.1f, 0.2f}};
    scene.meshData = {
        mesh(static_cast<std::uint32_t>(PipelineVariant::opaque), 10, nearBounds),
        mesh(static_cast<std::uint32_t>(PipelineVariant::blend), 20, middleBounds),
        mesh(static_cast<std::uint32_t>(PipelineVariant::blend), 30, farBounds),
        mesh(static_cast<std::uint32_t>(PipelineVariant::opaque), 40, culledBounds)
    };
    scene.aabbs = {nearBounds, middleBounds, farBounds, culledBounds};
    scene.reserveFrameScratch();
    const SceneDrawCounts counts = scene.prepareDraws(
        glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0.0f));
    require(counts.active == 4 && counts.visible == 3,
        "draw preparation must report active and frustum-visible draws separately");
    require(scene.drawData.size() == 3 && scene.drawData[0].transformIndex == 10 &&
        scene.drawData[1].transformIndex == 30 && scene.drawData[2].transformIndex == 20,
        "opaque draws must retain their batch and transparent draws must sort back-to-front");
    require(scene.drawBatches[static_cast<std::size_t>(PipelineVariant::opaque)].commandCount == 1 &&
        scene.drawBatches[static_cast<std::size_t>(PipelineVariant::blend)].commandCount == 2,
        "draw preparation must emit per-pipeline indirect batches");
    require(scene.cpuScratchBytes() != 0,
        "draw preparation scratch must be retained for subsequent frames");

    const danvulkan::assets::Bounds translated = SceneContext::transformedBounds(nearBounds,
        glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f)));
    require(translated.minVertex.x > 1.8f && translated.minVertex.y > 2.8f &&
        translated.minVertex.z > 4.0f,
        "scene bounds must follow instance transforms");
    return EXIT_SUCCESS;
}
