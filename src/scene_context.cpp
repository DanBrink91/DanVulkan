#include "scene_context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace danvulkan::vk {
namespace {
std::array<glm::vec4, 6> frustumPlanes(
    const glm::mat4& view, const glm::mat4& projection)
{
    const glm::mat4 viewProjection = glm::transpose(projection * view);
    std::array<glm::vec4, 6> planes{
        viewProjection[3] + viewProjection[0], viewProjection[3] - viewProjection[0],
        viewProjection[3] - viewProjection[1], viewProjection[3] + viewProjection[1],
        viewProjection[3] + viewProjection[2], viewProjection[3] - viewProjection[2]
    };
    for (glm::vec4& plane : planes)
    {
        const float length = glm::length(glm::vec3(plane));
        if (length > 0.0f)
        {
            plane /= length;
        }
    }
    return planes;
}

bool boundsVisible(const assets::Bounds& bounds, std::span<const glm::vec4> planes)
{
    for (const glm::vec4& plane : planes)
    {
        const glm::vec3 normal = plane;
        const glm::vec3 axisVertex{
            normal.x < 0.0f ? bounds.minVertex.x : bounds.maxVertex.x,
            normal.y < 0.0f ? bounds.minVertex.y : bounds.maxVertex.y,
            normal.z < 0.0f ? bounds.minVertex.z : bounds.maxVertex.z
        };
        if (glm::dot(normal, axisVertex) + plane.w < 0.0f)
        {
            return false;
        }
    }
    return true;
}

float smoothPopulationTransition(float distance, float center, float halfWidth,
    float nearPopulation, float farPopulation)
{
    if (halfWidth <= 0.0f)
    {
        return distance <= center ? nearPopulation : farPopulation;
    }
    const float t = std::clamp(
        (distance - (center - halfWidth)) / (2.0f * halfWidth), 0.0f, 1.0f);
    const float smooth = t * t * (3.0f - 2.0f * t);
    return std::lerp(nearPopulation, farPopulation, smooth);
}
}

void SceneContext::reserveFrameScratch()
{
    std::size_t maximumDrawCount = 0;
    for (const MeshData& mesh : meshData)
    {
        maximumDrawCount += mesh.grass.enabled ? mesh.grass.tiles.size() : 1U;
    }
    indirectCommands.reserve(maximumDrawCount);
    drawData.reserve(maximumDrawCount);
    animationUpdatePolicies_.reserve(animationActors_.size());
    std::array<std::size_t, PipelineVariantCount> variantCounts{};
    std::size_t grassCount = 0;
    for (const MeshData& mesh : meshData)
    {
        if (mesh.pipelineVariant >= PipelineVariantCount)
        {
            throw std::runtime_error("scene contains an invalid pipeline variant");
        }
        if (mesh.grass.enabled)
        {
            ++grassCount;
        }
        else
        {
            ++variantCounts[mesh.pipelineVariant];
        }
    }
    for (std::size_t variant = 0; variant < PipelineVariantCount; ++variant)
    {
        visibleMeshScratch_[variant].reserve(variantCounts[variant]);
    }
    visibleGrassScratch_.reserve(grassCount);
}

assets::Bounds SceneContext::transformedBounds(const assets::Bounds& bounds,
    const glm::mat4& transform)
{
    const float maximum = std::numeric_limits<float>::max();
    assets::Bounds result{ glm::vec3(maximum), glm::vec3(-maximum) };
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

void SceneContext::setAnimationActorTransform(
    std::size_t actorIndex, const glm::mat4& worldOffset)
{
    AnimationActorState& actor = animationActors_.at(actorIndex);
    bool unchanged = true;
    for (glm::length_t column = 0; column < 4 && unchanged; ++column)
    {
        for (glm::length_t row = 0; row < 4; ++row)
        {
            unchanged = unchanged &&
                worldOffset[column][row] == actor.worldOffset[column][row];
        }
    }
    if (unchanged)
    {
        return;
    }
    const glm::mat4 delta = worldOffset * glm::inverse(actor.worldOffset);
    actor.worldOffset = worldOffset;
    actor.transformDirty = true;

    if (actor.hasBounds)
    {
        actor.conservativeBounds = transformedBounds(actor.conservativeBounds, delta);
    }
    for (const AnimatedDrawState& state : animatedDraws_)
    {
        if (state.animationInstance != actorIndex)
        {
            continue;
        }
        aabbs.at(state.meshDataIndex) = transformedBounds(aabbs.at(state.meshDataIndex), delta);
    }
}

AnimationSynchronizationTimings SceneContext::synchronizeAnimationPose(bool changedInstancesOnly)
{
    AnimationSynchronizationTimings timings;
    if (!animationPlayer_)
    {
        return timings;
    }
    const auto transformBegin = std::chrono::steady_clock::now();
    for (const AnimatedDrawState& state : animatedDraws_)
    {
        if (changedInstancesOnly &&
            (state.animationInstance == SkinPaletteState::noAnimationInstance ||
             (!animationPlayer_->instanceEvaluated(state.animationInstance) &&
              !animationActors_.at(state.animationInstance).transformDirty)))
        {
            continue;
        }
        if (state.transformIndex >= transformData.size() || state.meshDataIndex >= meshData.size() ||
            state.meshDataIndex >= aabbs.size())
        {
            throw std::runtime_error("animated renderer state is inconsistent");
        }
        const glm::mat4 world = animationActors_.at(state.animationInstance).worldOffset *
            animationPlayer_->worldTransform(state.node);
        transformData[state.transformIndex].model = world;
    }
    timings.transformUpdateMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - transformBegin).count();
    const auto boundsBegin = std::chrono::steady_clock::now();
    for (const AnimatedDrawState& state : animatedDraws_)
    {
        if (changedInstancesOnly &&
            !animationPlayer_->instanceEvaluated(state.animationInstance) &&
            !animationActors_.at(state.animationInstance).transformDirty)
        {
            continue;
        }
        const glm::mat4 world = animationActors_.at(state.animationInstance).worldOffset *
            animationPlayer_->worldTransform(state.node);
        aabbs[state.meshDataIndex] = transformedBounds(meshData[state.meshDataIndex].localBounds,
            world);
        AnimationActorState& actor = animationActors_.at(state.animationInstance);
        if (!actor.hasBounds)
        {
            actor.conservativeBounds = aabbs[state.meshDataIndex];
            actor.hasBounds = true;
        }
        else
        {
            actor.conservativeBounds.minVertex = glm::min(
                actor.conservativeBounds.minVertex, aabbs[state.meshDataIndex].minVertex);
            actor.conservativeBounds.maxVertex = glm::max(
                actor.conservativeBounds.maxVertex, aabbs[state.meshDataIndex].maxVertex);
        }
    }
    timings.boundsMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - boundsBegin).count();
    const auto paletteBegin = std::chrono::steady_clock::now();
    for (const SkinPaletteState& state : skinPalettes_)
    {
        if (changedInstancesOnly &&
            (state.animationInstance == SkinPaletteState::noAnimationInstance ||
             (!animationPlayer_->instanceEvaluated(state.animationInstance) &&
              !animationActors_.at(state.animationInstance).transformDirty)))
        {
            continue;
        }
        if (state.jointOffset > jointMatrices_.size() ||
            state.jointCount > jointMatrices_.size() - state.jointOffset)
        {
            throw std::runtime_error("animated renderer state is inconsistent");
        }
        animationPlayer_->writeSkinMatrices(state.skin,
            std::span<glm::mat4>(jointMatrices_).subspan(
                state.jointOffset, state.jointCount));
        if (state.animationInstance != SkinPaletteState::noAnimationInstance)
        {
            const glm::mat4& worldOffset =
                animationActors_.at(state.animationInstance).worldOffset;
            for (glm::mat4& joint : std::span<glm::mat4>(jointMatrices_).subspan(
                     state.jointOffset, state.jointCount))
            {
                joint = worldOffset * joint;
            }
        }
    }
    if (jointMatrices_.size() > JointMatrixCount)
    {
        throw std::runtime_error("animated scene exceeds the renderer joint matrix capacity");
    }
    timings.paletteGenerationMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - paletteBegin).count();
    for (AnimationActorState& actor : animationActors_)
    {
        actor.transformDirty = false;
    }
    return timings;
}

AnimationPlayer::InstanceUpdatePolicy SceneContext::planActorAnimationUpdate(
    const assets::Bounds& bounds, const glm::mat4& view, const glm::mat4& projection,
    const glm::vec3& cameraPosition, const AnimationUpdateSettings& settings)
{
    if (!std::isfinite(settings.fullRateDistance) || settings.fullRateDistance < 0.0f ||
        !std::isfinite(settings.reducedRateDistance) ||
        settings.reducedRateDistance < settings.fullRateDistance ||
        !std::isfinite(settings.mediumUpdatesPerSecond) ||
        settings.mediumUpdatesPerSecond <= 0.0f ||
        !std::isfinite(settings.farUpdatesPerSecond) || settings.farUpdatesPerSecond <= 0.0f)
    {
        throw std::invalid_argument("animation update settings are invalid");
    }
    if (settings.cullOffscreenActors &&
        !boundsVisible(bounds, frustumPlanes(view, projection)))
    {
        return {false, 0.0f};
    }
    const glm::vec3 center = (bounds.minVertex + bounds.maxVertex) * 0.5f;
    const float distance = glm::length(center - cameraPosition);
    if (distance <= settings.fullRateDistance)
    {
        return {true, 0.0f};
    }
    if (distance <= settings.reducedRateDistance)
    {
        return {true, 1.0f / settings.mediumUpdatesPerSecond};
    }
    return {true, 1.0f / settings.farUpdatesPerSecond};
}

AnimationUpdateCounts SceneContext::prepareAnimationUpdates(const glm::mat4& view,
    const glm::mat4& projection, const glm::vec3& cameraPosition,
    const AnimationUpdateSettings& settings)
{
    animationUpdatePolicies_.assign(animationActors_.size(), {});
    AnimationUpdateCounts counts;
    counts.actors = static_cast<std::uint32_t>(animationActors_.size());
    for (std::size_t index = 0; index < animationActors_.size(); ++index)
    {
        const AnimationActorState& actor = animationActors_[index];
        AnimationPlayer::InstanceUpdatePolicy policy;
        if (actor.hasBounds)
        {
            policy = planActorAnimationUpdate(
                actor.conservativeBounds, view, projection, cameraPosition, settings);
        }
        animationUpdatePolicies_[index] = policy;
        if (policy.evaluate)
        {
            ++counts.eligible;
        }
        else
        {
            ++counts.culled;
        }
    }
    return counts;
}

SceneDrawCounts SceneContext::prepareDraws(const glm::mat4& view,
    const glm::mat4& projection, const glm::vec3& cameraPosition)
{
    const std::array<glm::vec4, 6> planes = frustumPlanes(view, projection);

    drawData.clear();
    indirectCommands.clear();
    drawBatches.fill({});
    grassDrawBatch = {};
    visibleGrassScratch_.clear();
    for (std::vector<std::size_t>& visibleMeshes : visibleMeshScratch_)
    {
        visibleMeshes.clear();
    }
    if (aabbs.size() != meshData.size())
    {
        throw std::runtime_error("scene draw bounds are inconsistent");
    }
    for (std::size_t index = 0; index < aabbs.size(); ++index)
    {
        if (meshData[index].pipelineVariant >= PipelineVariantCount)
        {
            throw std::runtime_error("scene contains an invalid pipeline variant");
        }
        if (boundsVisible(aabbs[index], planes))
        {
            if (meshData[index].grass.enabled)
            {
                visibleGrassScratch_.push_back(index);
            }
            else
            {
                visibleMeshScratch_[meshData[index].pipelineVariant].push_back(index);
            }
        }
    }

    const auto distanceSquared = [&](std::size_t meshIndex)
    {
        const glm::vec3 center = (aabbs[meshIndex].minVertex +
            aabbs[meshIndex].maxVertex) * 0.5f;
        const glm::vec3 offset = center - cameraPosition;
        return glm::dot(offset, offset);
    };
    for (std::size_t variant = static_cast<std::size_t>(PipelineVariant::blend);
         variant < PipelineVariantCount; ++variant)
    {
        std::ranges::sort(visibleMeshScratch_[variant],
            [&](std::size_t left, std::size_t right)
            {
                return distanceSquared(left) > distanceSquared(right);
            });
    }

    for (std::size_t variant = 0; variant < PipelineVariantCount; ++variant)
    {
        DrawBatch& batch = drawBatches[variant];
        batch.firstCommand = static_cast<std::uint32_t>(indirectCommands.size());
        for (const std::size_t meshIndex : visibleMeshScratch_[variant])
        {
            const MeshData& mesh = meshData[meshIndex];
            VkDrawIndexedIndirectCommand command{};
            command.indexCount = mesh.indexCount;
            command.firstIndex = mesh.firstIndex;
            command.firstInstance = static_cast<std::uint32_t>(drawData.size());
            command.instanceCount = 1;
            command.vertexOffset = static_cast<std::int32_t>(mesh.vertexOffset);
            indirectCommands.push_back(command);
            drawData.push_back(mesh.drawData);
        }
        batch.commandCount = static_cast<std::uint32_t>(indirectCommands.size()) -
            batch.firstCommand;
    }

    grassDrawBatch.firstCommand = static_cast<std::uint32_t>(indirectCommands.size());
    for (const std::size_t meshIndex : visibleGrassScratch_)
    {
        MeshData& mesh = meshData[meshIndex];
        if (mesh.drawData.transformIndex < 0 ||
            static_cast<std::size_t>(mesh.drawData.transformIndex) >= transformData.size())
        {
            throw std::runtime_error("grass draw references an invalid transform");
        }
        const glm::mat4& transform = transformData[mesh.drawData.transformIndex].model;
        for (GrassTileDrawData& tile : mesh.grass.tiles)
        {
            const assets::Bounds bounds = transformedBounds(tile.localBounds, transform);
            if (!boundsVisible(bounds, planes))
            {
                continue;
            }
            const float nearestX = std::clamp(cameraPosition.x,
                bounds.minVertex.x, bounds.maxVertex.x);
            const float nearestZ = std::clamp(cameraPosition.z,
                bounds.minVertex.z, bounds.maxVertex.z);
            const glm::vec2 offset(cameraPosition.x - nearestX, cameraPosition.z - nearestZ);
            const float distance = glm::length(offset);

            std::uint8_t lod = tile.lodState;
            if (lod > 2U)
            {
                lod = distance <= mesh.grass.distances[0] ? 0U :
                    (distance <= mesh.grass.distances[1] ? 1U : 2U);
            }
            else if (lod == 0U &&
                distance > mesh.grass.distances[0] + mesh.grass.hysteresis)
            {
                lod = 1U;
            }
            else if (lod == 1U)
            {
                if (distance < mesh.grass.distances[0] - mesh.grass.hysteresis)
                {
                    lod = 0U;
                }
                else if (distance > mesh.grass.distances[1] + mesh.grass.hysteresis)
                {
                    lod = 2U;
                }
            }
            else if (lod == 2U &&
                distance < mesh.grass.distances[1] - mesh.grass.hysteresis)
            {
                lod = 1U;
            }
            tile.lodState = lod;

            float population = 0.0f;
            if (distance <= mesh.grass.distances[0] + mesh.grass.transitionBand)
            {
                population = smoothPopulationTransition(distance,
                    mesh.grass.distances[0], mesh.grass.transitionBand,
                    mesh.grass.populationRatios[0], mesh.grass.populationRatios[1]);
            }
            else if (distance <= mesh.grass.distances[1] + mesh.grass.transitionBand)
            {
                population = smoothPopulationTransition(distance,
                    mesh.grass.distances[1], mesh.grass.transitionBand,
                    mesh.grass.populationRatios[1], mesh.grass.populationRatios[2]);
            }
            else
            {
                population = smoothPopulationTransition(distance,
                    mesh.grass.distances[2], mesh.grass.transitionBand,
                    mesh.grass.populationRatios[2], 0.0f);
            }
            const std::uint32_t instanceCount = std::min(tile.bladeCount,
                static_cast<std::uint32_t>(std::floor(
                    static_cast<float>(tile.bladeCount) * population)));
            if (instanceCount == 0U)
            {
                continue;
            }

            VkDrawIndexedIndirectCommand command{};
            command.indexCount = mesh.grass.indexCounts[lod];
            command.firstIndex = mesh.grass.firstIndices[lod];
            command.firstInstance = static_cast<std::uint32_t>(drawData.size());
            command.instanceCount = instanceCount;
            command.vertexOffset = static_cast<std::int32_t>(mesh.vertexOffset);
            indirectCommands.push_back(command);
            DrawData tileDraw = mesh.drawData;
            constexpr std::uint32_t vertexWords = sizeof(Vertex) / sizeof(std::uint32_t);
            constexpr std::uint32_t grassRecordWords =
                sizeof(PackedRuntimeGrassBlade) / sizeof(std::uint32_t);
            tileDraw.vertexOffset = static_cast<std::int32_t>(
                mesh.vertexOffset * vertexWords + tile.firstBlade * grassRecordWords);
            tileDraw.jointOffset = static_cast<std::int32_t>(mesh.vertexOffset);
            drawData.push_back(tileDraw);
        }
    }
    grassDrawBatch.commandCount = static_cast<std::uint32_t>(indirectCommands.size()) -
        grassDrawBatch.firstCommand;
    return { static_cast<std::uint32_t>(meshData.size()),
        static_cast<std::uint32_t>(indirectCommands.size()),
        static_cast<std::uint32_t>(animatedDraws_.size()),
        static_cast<std::uint32_t>(jointMatrices_.size()) };
}

std::uint64_t SceneContext::cpuScratchBytes() const noexcept
{
    std::uint64_t result = jointMatrices_.capacity() * sizeof(glm::mat4) +
        indirectCommands.capacity() * sizeof(VkDrawIndexedIndirectCommand) +
        drawData.capacity() * sizeof(DrawData) +
        animationActors_.capacity() * sizeof(AnimationActorState) +
        animationUpdatePolicies_.capacity() * sizeof(AnimationPlayer::InstanceUpdatePolicy);
    for (const std::vector<std::size_t>& bucket : visibleMeshScratch_)
    {
        result += bucket.capacity() * sizeof(std::size_t);
    }
    result += visibleGrassScratch_.capacity() * sizeof(std::size_t);
    return result;
}

bool SceneContext::hasGeometryRange(std::span<const GeometryRange> ranges,
    std::uint32_t count) noexcept
{
    return std::ranges::any_of(ranges, [count](const GeometryRange& range)
    {
        return range.count >= count;
    });
}

GeometryRange SceneContext::allocateGeometryRange(std::vector<GeometryRange>& ranges,
    std::uint32_t count)
{
    const auto range = std::ranges::find_if(ranges, [count](const GeometryRange& candidate)
    {
        return candidate.count >= count;
    });
    if (range == ranges.end())
    {
        throw std::runtime_error("no contiguous geometry range satisfies the allocation");
    }
    const GeometryRange allocation{ range->offset, count };
    range->offset += count;
    range->count -= count;
    if (range->count == 0)
    {
        ranges.erase(range);
    }
    return allocation;
}

void SceneContext::releaseGeometryRange(std::vector<GeometryRange>& ranges,
    GeometryRange released)
{
    if (released.count == 0)
    {
        return;
    }
    const auto position = std::ranges::lower_bound(ranges, released.offset, {},
        &GeometryRange::offset);
    const auto inserted = ranges.insert(position, released);
    if (inserted != ranges.begin())
    {
        const auto previous = std::prev(inserted);
        if (previous->offset + previous->count == inserted->offset)
        {
            previous->count += inserted->count;
            ranges.erase(inserted);
        }
    }
    for (auto current = ranges.begin(); current != ranges.end();)
    {
        const auto next = std::next(current);
        if (next == ranges.end())
        {
            break;
        }
        if (current->offset + current->count == next->offset)
        {
            current->count += next->count;
            ranges.erase(next);
        }
        else
        {
            ++current;
        }
    }
}

std::uint32_t SceneContext::grownGeometryCapacity(std::uint32_t current,
    std::uint32_t requiredAdditional, std::uint32_t maximum)
{
    const std::uint64_t doubled = std::max<std::uint64_t>(1, current) * 2;
    const std::uint64_t required = static_cast<std::uint64_t>(current) + requiredAdditional;
    const std::uint64_t result = std::max(doubled, required);
    if (result > maximum)
    {
        throw std::runtime_error("geometry capacity exceeds renderer offset limits");
    }
    return static_cast<std::uint32_t>(result);
}

std::vector<VkDescriptorImageInfo> SceneContext::textureDescriptorInfos() const
{
    if (liveTextureCount_ == 0 || textureDescriptorCapacity_ < textures.size())
    {
        throw std::runtime_error("bindless texture descriptor state is inconsistent");
    }
    const Texture* fallback = nullptr;
    for (const std::optional<Texture>& texture : textures)
    {
        if (texture)
        {
            fallback = &*texture;
            break;
        }
    }
    if (fallback == nullptr)
    {
        throw std::runtime_error("bindless texture fallback is missing");
    }
    std::vector<VkDescriptorImageInfo> result(textureDescriptorCapacity_);
    for (std::uint32_t index = 0; index < textureDescriptorCapacity_; ++index)
    {
        const Texture& texture = index < textures.size() && textures[index]
            ? *textures[index] : *fallback;
        result[index] = { texture.sampler, texture.image.view(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    }
    return result;
}

void SceneContext::initializeImageGenerations(std::size_t imageCount)
{
    imageGeometryVersions_.assign(imageCount, geometryVersion_);
    imageGeometryRangeVersions_.assign(imageCount, geometryRangeVersion_);
    imageTextureVersions_.assign(imageCount, textureVersion_);
}

void SceneContext::prepareGeometryForImage(std::uint32_t imageIndex,
    DescriptorContext& descriptors)
{
    if (imageIndex >= descriptors.setCount() || imageIndex >= imageGeometryVersions_.size())
    {
        throw std::runtime_error("swapchain geometry descriptor state is inconsistent");
    }
    if (imageGeometryVersions_[imageIndex] == geometryVersion_)
    {
        return;
    }
    descriptors.updateVertex(imageIndex, { vertexBuffer, 0, vertexBuffer.size() });
    imageGeometryVersions_[imageIndex] = geometryVersion_;
    std::erase_if(retiredGeometry_, [&](const RetiredGeometry& retired)
    {
        return std::ranges::all_of(imageGeometryVersions_, [&](std::uint64_t version)
        {
            return version > retired.version;
        });
    });
}

void SceneContext::prepareGeometryRangesForImage(std::uint32_t imageIndex)
{
    if (imageIndex >= imageGeometryRangeVersions_.size())
    {
        throw std::runtime_error("swapchain geometry range state is inconsistent");
    }
    if (imageGeometryRangeVersions_[imageIndex] == geometryRangeVersion_)
    {
        return;
    }
    imageGeometryRangeVersions_[imageIndex] = geometryRangeVersion_;
    for (auto retired = retiredGeometryRanges_.begin(); retired != retiredGeometryRanges_.end();)
    {
        const bool complete = std::ranges::all_of(imageGeometryRangeVersions_,
            [&](std::uint64_t version) { return version > retired->version; });
        if (!complete)
        {
            ++retired;
            continue;
        }
        releaseGeometryRange(freeVertexRanges_, retired->vertices);
        releaseGeometryRange(freeIndexRanges_, retired->indices);
        retired = retiredGeometryRanges_.erase(retired);
    }
}

void SceneContext::prepareTexturesForImage(std::uint32_t imageIndex,
    DescriptorContext& descriptors, VkDevice device)
{
    if (imageIndex >= descriptors.setCount() || imageIndex >= imageTextureVersions_.size())
    {
        throw std::runtime_error("swapchain texture descriptor state is inconsistent");
    }
    if (imageTextureVersions_[imageIndex] == textureVersion_)
    {
        return;
    }
    descriptors.updateTextures(imageIndex, textureDescriptorInfos());
    imageTextureVersions_[imageIndex] = textureVersion_;
    for (auto retired = retiredTextures_.begin(); retired != retiredTextures_.end();)
    {
        const bool complete = std::ranges::all_of(imageTextureVersions_,
            [&](std::uint64_t version) { return version > retired->version; });
        if (!complete)
        {
            ++retired;
            continue;
        }
        vkDestroySampler(device, retired->texture.sampler, nullptr);
        retired->texture.sampler = VK_NULL_HANDLE;
        retired = retiredTextures_.erase(retired);
    }
}

void SceneContext::releaseSwapchainRetirements(VkDevice device)
{
    retiredGeometry_.clear();
    for (const RetiredGeometryRanges& retired : retiredGeometryRanges_)
    {
        releaseGeometryRange(freeVertexRanges_, retired.vertices);
        releaseGeometryRange(freeIndexRanges_, retired.indices);
    }
    retiredGeometryRanges_.clear();
    for (RetiredTexture& retired : retiredTextures_)
    {
        vkDestroySampler(device, retired.texture.sampler, nullptr);
        retired.texture.sampler = VK_NULL_HANDLE;
    }
    retiredTextures_.clear();
}

void SceneContext::resetScene(VkDevice device) noexcept
{
    for (std::optional<Texture>& texture : textures)
    {
        if (texture)
        {
            vkDestroySampler(device, texture->sampler, nullptr);
            texture->sampler = VK_NULL_HANDLE;
        }
    }
    for (RetiredTexture& retired : retiredTextures_)
    {
        vkDestroySampler(device, retired.texture.sampler, nullptr);
        retired.texture.sampler = VK_NULL_HANDLE;
    }
    textures.clear();
    retiredTextures_.clear();
    textureGenerations_.clear();
    freeTextureSlots_.clear();
    imageTextureVersions_.clear();
    liveTextureCount_ = 0;
    retiredGeometry_.clear();
    imageGeometryVersions_.clear();
    retiredGeometryRanges_.clear();
    imageGeometryRangeVersions_.clear();
    freeVertexRanges_.clear();
    freeIndexRanges_.clear();
    vertexBuffer.reset();
    indexBuffer.reset();
    matBuffers.clear();
    transformBuffers.clear();
    drawBuffers.clear();
    jointBuffers.clear();
    indirectCommandsBuffer.clear();
    matData.clear();
    transformData.clear();
    materialNames.clear();
    materialGenerations_.clear();
    materialAlive_.clear();
    freeMaterialSlots_.clear();
    instanceNames.clear();
    instanceGenerations.clear();
    instanceAlive.clear();
    freeInstanceSlots.clear();
    meshResources.clear();
    meshGenerations_.clear();
    meshAlive_.clear();
    freeMeshSlots_.clear();
    drawData.clear();
    meshData.clear();
    aabbs.clear();
    animationPlayer_.reset();
    animatedDraws_.clear();
    skinPalettes_.clear();
    animationActors_.clear();
    animationUpdatePolicies_.clear();
    jointMatrices_.clear();
    indirectCommands.clear();
    drawBatches.fill({});
    grassDrawBatch = {};
    visibleGrassScratch_.clear();
    for (std::vector<std::size_t>& bucket : visibleMeshScratch_)
    {
        bucket.clear();
    }
    vertexCapacity_ = 0;
    indexCapacity_ = 0;
    materialCapacity_ = 0;
    textureDescriptorCapacity_ = 0;
    sceneGeneration_ = 0;
    geometryVersion_ = 1;
    geometryRangeVersion_ = 1;
    textureVersion_ = 1;
}

} // namespace danvulkan::vk
