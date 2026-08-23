#define _CRT_SECURE_NO_WARNINGS
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <danvulkan/renderer.hpp>
#include <danvulkan/assets.hpp>

#include "glfw_platform.hpp"
#include "animation_player.hpp"
#include "attachment_context.hpp"
#include "descriptor_context.hpp"
#include "descriptor_planner.hpp"
#include "device_context.hpp"
#include "frame_context.hpp"
#include "lighting_context.hpp"
#include "lighting_planner.hpp"
#include "memory_planner.hpp"
#include "pipeline_context.hpp"
#include "pipeline_planner.hpp"
#include "presentation_context.hpp"
#include "scene_context.hpp"
#include "scene_planner.hpp"
#include "swapchain_context.hpp"
#include "upload_context.hpp"
#include "ui_context.hpp"
#include "vulkan_raii.hpp"
#include "vulkan_result.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>


#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <cstdint> // UINT32_MAX

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <vector>
#include <optional>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#include <filesystem> // hot reloading

#include "camera.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

using danvulkan::vk::AnimatedDrawState;
using danvulkan::vk::DrawBatch;
using danvulkan::vk::DrawData;
using danvulkan::vk::DrawDataCount;
using danvulkan::vk::GeometryRange;
using danvulkan::vk::JointMatrixCount;
using danvulkan::vk::MatDataCount;
using danvulkan::vk::MaterialData;
using danvulkan::vk::MeshData;
using danvulkan::vk::MeshResourceData;
using danvulkan::vk::PreparedSceneData;
using danvulkan::vk::RetiredGeometry;
using danvulkan::vk::RetiredGeometryRanges;
using danvulkan::vk::RetiredTexture;
using danvulkan::vk::Texture;
using danvulkan::vk::TransformData;
using danvulkan::vk::TransformDataCount;
using danvulkan::vk::Vertex;
using danvulkan::vk::makeVulkanStructure;

struct UniformBufferObject {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 inverseViewProjection;
    glm::mat4 directionalShadowViewProjection;
    glm::vec4 cameraPositionTime;
    glm::vec4 vegetationInteractorPositionRadius;
    glm::uvec4 lightingCounts;
    glm::vec4 environmentTintIntensity;
    glm::vec4 environmentControls;
    glm::vec4 atmosphereSkyZenithIntensity;
    glm::vec4 atmosphereSkyHorizonExponent;
    glm::vec4 atmosphereFogColorDensity;
    glm::vec4 atmosphereFogParameters;
    glm::vec4 atmosphereScatteringParameters;
    std::array<danvulkan::PlannedDirectionalLight,
        MaxSceneDirectionalLights> directionalLights;
    glm::vec4 atmosphereCloudShapeParameters;
    glm::vec4 atmosphereCloudMovementParameters;
    glm::vec4 atmosphereCloudLightingParameters;
};

static_assert(sizeof(UniformBufferObject) == 592,
    "UniformBufferObject must match the shader std140 layout");

using AABB = danvulkan::assets::Bounds;

using danvulkan::vk::PipelineVariant;
using danvulkan::vk::PipelineVariantCount;


static void checkVk(VkResult result, std::string_view operation)
{
    danvulkan::vk::check(result, operation);
}

// steady_clock measures frame latency, while this clock measures only CPU time consumed by the
// calling thread. Fence, swapchain-acquire, presentation, and scheduler waits do not advance it.
// Vulkan driver work performed by other threads is intentionally outside this measurement.
[[nodiscard]] static std::optional<double> currentThreadCpuMilliseconds() noexcept
{
#if defined(_WIN32)
    FILETIME creation{};
    FILETIME exit{};
    FILETIME kernel{};
    FILETIME user{};
    if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user) == 0)
    {
        return std::nullopt;
    }
    ULARGE_INTEGER kernelTicks{};
    kernelTicks.LowPart = kernel.dwLowDateTime;
    kernelTicks.HighPart = kernel.dwHighDateTime;
    ULARGE_INTEGER userTicks{};
    userTicks.LowPart = user.dwLowDateTime;
    userTicks.HighPart = user.dwHighDateTime;
    constexpr double millisecondsPerHundredNanoseconds = 1.0e-4;
    return static_cast<double>(kernelTicks.QuadPart + userTicks.QuadPart) *
        millisecondsPerHundredNanoseconds;
#else
    timespec timestamp{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &timestamp) != 0)
    {
        return std::nullopt;
    }
    return static_cast<double>(timestamp.tv_sec) * 1000.0 +
        static_cast<double>(timestamp.tv_nsec) * 1.0e-6;
#endif
}

class VulkanRenderer::Impl {
public:
    explicit Impl(RendererConfig config)
        : config_(std::move(config)),
          platform_(std::move(config_.platform)),
          MODEL_PATH(config_.modelPath.string())
    {
        if (!std::isfinite(config_.textureMipLodBias))
        {
            throw std::invalid_argument("texture mip LOD bias must be finite");
        }
        if (!std::isfinite(config_.animation.fullRateDistance) ||
            config_.animation.fullRateDistance < 0.0f ||
            !std::isfinite(config_.animation.reducedRateDistance) ||
            config_.animation.reducedRateDistance < config_.animation.fullRateDistance ||
            !std::isfinite(config_.animation.mediumUpdatesPerSecond) ||
            config_.animation.mediumUpdatesPerSecond <= 0.0f ||
            !std::isfinite(config_.animation.farUpdatesPerSecond) ||
            config_.animation.farUpdatesPerSecond <= 0.0f ||
            !std::isfinite(config_.animation.conservativeBoundsPadding) ||
            config_.animation.conservativeBoundsPadding < 0.0f ||
            !std::isfinite(config_.animation.adaptiveNlerpMaxAngleRadians) ||
            config_.animation.adaptiveNlerpMaxAngleRadians < 0.0f ||
            config_.animation.adaptiveNlerpMaxAngleRadians > std::numbers::pi_v<float>)
        {
            throw std::invalid_argument("renderer animation policy is invalid");
        }
    }

    ~Impl()
    {
        shutdownNoThrow();
    }

    void initialize()
    {
        if (initialized_)
        {
            return;
        }
        if (shutdown_)
        {
            throw std::logic_error("a shut down VulkanRenderer cannot be initialized again");
        }

        initPlatform();
        try
        {
            initVulkan();
            animationPreviousTime_ = std::chrono::steady_clock::now();
            initialized_ = true;
        }
        catch (...)
        {
            // Context resets are null-safe and preserve Vulkan dependency order, so partially
            // created scene resources cannot outlive their allocator after failed initialization.
            cleanup();
            throw;
        }
    }

    [[nodiscard]] bool isInitialized() const noexcept
    {
        return initialized_;
    }

    [[nodiscard]] bool shouldClose() const noexcept
    {
        return !initialized_ || platform_ == nullptr || platform_->shouldClose() ||
            (config_.maxFrames != 0 && renderedFrames_ >= config_.maxFrames);
    }

    [[nodiscard]] bool beginFrame()
    {
        requireInitialized("beginFrame");
        if (frameInProgress_)
        {
            throw std::logic_error("beginFrame called while a frame is already in progress");
        }
        if (shouldClose())
        {
            return false;
        }

        if (config_.resizeAtFrame != 0 && renderedFrames_ == config_.resizeAtFrame)
        {
            if (!platform_->requestWindowResize(config_.resizeWidth, config_.resizeHeight))
            {
                throw std::runtime_error(
                    "configured resize smoke test is unsupported by this platform adapter");
            }
        }

        frameBegin_ = std::chrono::steady_clock::now();
        frameThreadCpuBeginMilliseconds_ = currentThreadCpuMilliseconds();
        frameThread_ = std::this_thread::get_id();
        platform_->pollEvents();
        if (shouldClose())
        {
            return false;
        }

        const auto animationTime = std::chrono::steady_clock::now();
        animationDeltaSeconds_ = std::min(
            std::chrono::duration<float>(animationTime - animationPreviousTime_).count(), 0.1f);
        animationPreviousTime_ = animationTime;

        hasPendingSubmission_ = false;
        pendingUi_.vertices.clear();
        pendingUi_.commands.clear();
        hasPendingUi_ = false;
        frameInProgress_ = true;
        return true;
    }

    void submitScene(const SceneSubmission& submission)
    {
        requireFrameInProgress("submitScene");
        if (hasPendingSubmission_)
        {
            throw std::logic_error("submitScene may only be called once per frame");
        }
        pendingSubmission_ = submission;
        hasPendingSubmission_ = true;
    }

    void submitUi(const UiDrawData& drawData)
    {
        requireFrameInProgress("submitUi");
        if (hasPendingUi_)
        {
            throw std::logic_error("submitUi may only be called once per frame");
        }
        pendingUi_ = drawData;
        hasPendingUi_ = true;
    }

    void endFrame()
    {
        requireFrameInProgress("endFrame");
        if (!hasPendingSubmission_)
        {
            throw std::logic_error("endFrame requires one scene submission");
        }

        try
        {
            const auto animationBegin = std::chrono::steady_clock::now();
            applyAnimationActorTransforms(pendingSubmission_);
            updateAnimation(animationDeltaSeconds_, pendingSubmission_);
            animationCpuAvg_ = rollingAverage(
                animationCpuAvg_, millisecondsSince(animationBegin));
            drawFrame(pendingSubmission_);
            updateWindowTitle();
            ++renderedFrames_;
            hasPendingSubmission_ = false;
            hasPendingUi_ = false;
            frameInProgress_ = false;
        }
        catch (...)
        {
            hasPendingSubmission_ = false;
            hasPendingUi_ = false;
            frameInProgress_ = false;
            throw;
        }
    }

    [[nodiscard]] std::vector<SceneInstanceInfo> sceneInstances() const
    {
        requireInitialized("sceneInstances");
        std::vector<SceneInstanceInfo> result;
        result.reserve(scene_.transformData.size());
        for (std::size_t index = 0; index < scene_.transformData.size(); ++index)
        {
            if (!scene_.instanceAlive[index])
            {
                continue;
            }
            result.push_back({
                SceneInstanceHandle{ static_cast<std::uint32_t>(index),
                    scene_.instanceGenerations[index] },
                scene_.instanceNames[index],
                scene_.transformData[index].model
            });
        }
        return result;
    }

    [[nodiscard]] std::optional<SceneBounds> sceneBounds() const
    {
        requireInitialized("sceneBounds");
        if (scene_.aabbs.empty())
        {
            return std::nullopt;
        }

        SceneBounds result{scene_.aabbs.front().minVertex, scene_.aabbs.front().maxVertex};
        for (std::size_t index = 1; index < scene_.aabbs.size(); ++index)
        {
            result.minimum = glm::min(result.minimum, scene_.aabbs[index].minVertex);
            result.maximum = glm::max(result.maximum, scene_.aabbs[index].maxVertex);
        }
        return result;
    }

    [[nodiscard]] std::vector<SceneMaterialInfo> sceneMaterials() const
    {
        requireInitialized("sceneMaterials");
        std::vector<SceneMaterialInfo> result;
        result.reserve(scene_.matData.size());
        for (std::size_t index = 0; index < scene_.matData.size(); ++index)
        {
            if (!scene_.materialAlive_[index])
            {
                continue;
            }
            result.push_back({
                SceneMaterialHandle{ static_cast<std::uint32_t>(index),
                    scene_.materialGenerations_[index] },
                scene_.materialNames[index],
                runtimeProperties(scene_.matData[index]),
                runtimeTextures(scene_.matData[index]),
                static_cast<danvulkan::assets::AlphaMode>(scene_.matData[index].materialFlags.y),
                scene_.matData[index].materialFlags.z != 0,
                scene_.matData[index].materialFlags.w != 0
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<SceneTextureInfo> sceneTextures() const
    {
        requireInitialized("sceneTextures");
        std::vector<SceneTextureInfo> result;
        result.reserve(scene_.textures.size());
        for (std::size_t index = 0; index < scene_.textures.size(); ++index)
        {
            if (!scene_.textures[index])
            {
                continue;
            }
            const Texture& texture = *scene_.textures[index];
            result.push_back({
                SceneTextureHandle{ static_cast<std::uint32_t>(index),
                    scene_.textureGenerations_[index] },
                texture.name,
                texture.width,
                texture.height,
                texture.mipLevels,
                texture.colorSpace,
                texture.samplerConfig
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<SceneAnimationInfo> sceneAnimations() const
    {
        requireInitialized("sceneAnimations");
        std::vector<SceneAnimationInfo> result;
        if (!scene_.animationPlayer_)
        {
            return result;
        }
        result.reserve(scene_.animationPlayer_->clipCount());
        for (std::size_t index = 0; index < scene_.animationPlayer_->clipCount(); ++index)
        {
            result.push_back({
                SceneAnimationHandle{ static_cast<std::uint32_t>(index), scene_.sceneGeneration_ },
                std::string(scene_.animationPlayer_->clipName(index)),
                scene_.animationPlayer_->clipDuration(index)
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<SceneAnimationActorInfo> sceneAnimationActors() const
    {
        requireInitialized("sceneAnimationActors");
        std::vector<SceneAnimationActorInfo> result;
        if (!scene_.animationPlayer_)
        {
            return result;
        }
        result.reserve(scene_.animationActors_.size());
        for (std::size_t index = 0; index < scene_.animationActors_.size(); ++index)
        {
            result.push_back({
                SceneAnimationActorHandle{static_cast<std::uint32_t>(index),
                    scene_.sceneGeneration_},
                SceneAnimationHandle{
                    static_cast<std::uint32_t>(scene_.animationPlayer_->instanceClip(index)),
                    scene_.sceneGeneration_},
                scene_.animationActors_[index].worldOffset
            });
        }
        return result;
    }

    [[nodiscard]] AnimationPlaybackState animationPlaybackState() const
    {
        requireInitialized("animationPlaybackState");
        AnimationPlaybackState state;
        if (!scene_.animationPlayer_ || scene_.animationPlayer_->currentClip() ==
            danvulkan::AnimationPlayer::invalidClip)
        {
            return state;
        }
        const std::size_t selected = scene_.animationPlayer_->currentClip();
        state.clip = { static_cast<std::uint32_t>(selected), scene_.sceneGeneration_ };
        switch (scene_.animationPlayer_->status())
        {
        case danvulkan::AnimationPlayer::Status::stopped:
            state.status = AnimationPlaybackStatus::stopped;
            break;
        case danvulkan::AnimationPlayer::Status::playing:
            state.status = AnimationPlaybackStatus::playing;
            break;
        case danvulkan::AnimationPlayer::Status::paused:
            state.status = AnimationPlaybackStatus::paused;
            break;
        case danvulkan::AnimationPlayer::Status::finished:
            state.status = AnimationPlaybackStatus::finished;
            break;
        }
        state.positionSeconds = scene_.animationPlayer_->position();
        state.durationSeconds = scene_.animationPlayer_->clipDuration(selected);
        state.playbackSpeed = scene_.animationPlayer_->playbackSpeed();
        state.looping = scene_.animationPlayer_->looping();
        return state;
    }

    [[nodiscard]] RendererPerformanceStats performanceStats() const noexcept
    {
        return {renderedFrames_, frameAvg_, frameCpuAvg_, frameGpuAvg, animationCpuAvg_,
            animationEvaluationCpuAvg_, animationSynchronizationCpuAvg_,
            animationSamplingCpuAvg_, animationPoseResetCpuAvg_,
            animationTimelineResolutionCpuAvg_, animationVectorSamplingCpuAvg_,
            animationRotationSamplingCpuAvg_, animationTransformPropagationCpuAvg_,
            animationTransformUpdateCpuAvg_, animationBoundsCpuAvg_,
            animationPaletteGenerationCpuAvg_, animationPaletteUploadCpuAvg_, cullingCpuAvg_,
            bufferWriteCpuAvg_, commandRecordingCpuAvg_, lastActiveDrawCount_,
            lastVisibleDrawCount_, lastAnimatedDrawCount_, lastJointMatrixCount_,
            lastPointLightCount_, lastAnimationActorCount_,
            lastEvaluatedAnimationActorCount_, lastCulledAnimationActorCount_,
            lastSampledAnimationChannelCount_, lastSampledAnimationVectorChannelCount_,
            lastSampledAnimationRotationChannelCount_, lastNlerpAnimationRotationChannelCount_,
            lastAnimationClipChannelCount_, lastFoldedConstantAnimationChannelCount_,
            lastAnimationPropagatedNodeCount_, lastAnimationPoseComposedNodeCount_,
            lastAnimationCachedLocalNodeCount_};
    }

    [[nodiscard]] RendererMemoryStats memoryStats() const noexcept
    {
        if (!allocator || !device)
        {
            return lastMemoryStats_;
        }

        RendererMemoryStats result;
        VmaTotalStatistics statistics{};
        vmaCalculateStatistics(allocator, &statistics);
        result.blockBytes = statistics.total.statistics.blockBytes;
        result.allocationBytes = statistics.total.statistics.allocationBytes;
        result.blockCount = statistics.total.statistics.blockCount;
        result.allocationCount = statistics.total.statistics.allocationCount;

        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(allocator, budgets.data());
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(device.physicalDevice(), &memoryProperties);
        for (std::uint32_t index = 0; index < memoryProperties.memoryHeapCount; ++index)
        {
            result.heapUsageBytes += budgets[index].usage;
            result.heapBudgetBytes += budgets[index].budget;
        }
        result.attachmentSetCount = static_cast<std::uint32_t>(attachmentTargetCount_);
        result.msaaSamples = static_cast<std::uint32_t>(msaaSamples);
        result.prefersLazilyAllocatedAttachments =
            config_.memory.preferLazilyAllocatedAttachments;
        const danvulkan::vk::UploadArenaStats uploadStats = uploadContext_.arenaStats();
        result.stagingArenaBytes = uploadStats.capacityBytes;
        result.stagingArenaGrowthCount = uploadStats.growthCount;
        result.uploadSubmissionCount = uploadStats.uploadCount;
        result.retainedCpuGeometryBytes = 0;
        result.cpuScratchBytes = scene_.cpuScratchBytes();

        peakBlockBytes_ = std::max(peakBlockBytes_, result.blockBytes);
        peakAllocationBytes_ = std::max(peakAllocationBytes_, result.allocationBytes);
        peakHeapUsageBytes_ = std::max(peakHeapUsageBytes_, result.heapUsageBytes);
        peakBlockCount_ = std::max(peakBlockCount_, result.blockCount);
        peakAllocationCount_ = std::max(peakAllocationCount_, result.allocationCount);
        result.peakBlockBytes = peakBlockBytes_;
        result.peakAllocationBytes = peakAllocationBytes_;
        result.peakHeapUsageBytes = peakHeapUsageBytes_;
        result.peakBlockCount = peakBlockCount_;
        result.peakAllocationCount = peakAllocationCount_;
        return result;
    }

    void playAnimation(SceneAnimationHandle animation, bool restart)
    {
        requireSceneUpdateAllowed("playAnimation");
        requireValidAnimation(animation, "playAnimation");
        scene_.animationPlayer_->play(animation.slot, restart);
        static_cast<void>(scene_.synchronizeAnimationPose());
        animationPreviousTime_ = std::chrono::steady_clock::now();
    }

    void pauseAnimation()
    {
        requireSceneUpdateAllowed("pauseAnimation");
        requireAnimationPlayer("pauseAnimation").pause();
    }

    void resumeAnimation()
    {
        requireSceneUpdateAllowed("resumeAnimation");
        requireAnimationPlayer("resumeAnimation").resume();
        animationPreviousTime_ = std::chrono::steady_clock::now();
    }

    void stopAnimation()
    {
        requireSceneUpdateAllowed("stopAnimation");
        requireAnimationPlayer("stopAnimation").stop();
        static_cast<void>(scene_.synchronizeAnimationPose());
    }

    void seekAnimation(float positionSeconds)
    {
        requireSceneUpdateAllowed("seekAnimation");
        requireAnimationPlayer("seekAnimation").seek(positionSeconds);
        static_cast<void>(scene_.synchronizeAnimationPose());
        animationPreviousTime_ = std::chrono::steady_clock::now();
    }

    void setAnimationLooping(bool looping)
    {
        requireSceneUpdateAllowed("setAnimationLooping");
        requireAnimationPlayer("setAnimationLooping").setLooping(looping);
    }

    void setAnimationPlaybackSpeed(float speed)
    {
        requireSceneUpdateAllowed("setAnimationPlaybackSpeed");
        requireAnimationPlayer("setAnimationPlaybackSpeed").setPlaybackSpeed(speed);
    }

    void replaceScene(const danvulkan::assets::SceneAsset& scene)
    {
        requireSceneUpdateAllowed("replaceScene");
        if (scene_.textureVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("texture generation counter exhausted");
        }
        if (scene_.geometryVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("geometry buffer generation counter exhausted");
        }

        PreparedSceneData replacement = prepareSceneData(scene);
        scene_.retiredTextures_.reserve(scene_.retiredTextures_.size() + scene_.liveTextureCount_);
        scene_.retiredGeometry_.reserve(scene_.retiredGeometry_.size() + 1);

        std::vector<std::optional<Texture>> replacementTextures;
        replacementTextures.reserve(scene.textures().size());
        auto destroyReplacementSamplers = [&]() noexcept
        {
            for (std::optional<Texture>& texture : replacementTextures)
            {
                if (texture && texture->sampler != VK_NULL_HANDLE)
                {
                    vkDestroySampler(device, texture->sampler, nullptr);
                    texture->sampler = VK_NULL_HANDLE;
                }
            }
        };

        danvulkan::vk::Buffer replacementVertexBuffer;
        danvulkan::vk::Buffer replacementIndexBuffer;
        try
        {
            for (std::size_t index = 0; index < scene.textures().size(); ++index)
            {
                replacementTextures.emplace_back(createTextureImage(scene.textures()[index]));
                createTextureImageView(*replacementTextures.back(),
                    static_cast<std::uint32_t>(index));
                createTextureSampler(*replacementTextures.back(),
                    static_cast<std::uint32_t>(index));
            }
            replacementVertexBuffer = createDeviceLocalBuffer(
                sizeof(Vertex) * static_cast<VkDeviceSize>(replacement.vertexCapacity),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, replacement.uploadVertices.data(),
                sizeof(Vertex) * static_cast<VkDeviceSize>(replacement.uploadVertices.size()),
                "replacement scene vertex storage buffer");
            replacementIndexBuffer = createDeviceLocalBuffer(
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(replacement.indexCapacity),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, replacement.uploadIndices.data(),
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(replacement.uploadIndices.size()),
                "replacement scene index buffer");
        }
        catch (...)
        {
            destroyReplacementSamplers();
            throw;
        }

        for (std::optional<Texture>& texture : scene_.textures)
        {
            if (texture)
            {
                scene_.retiredTextures_.push_back({ scene_.textureVersion_, std::move(*texture) });
            }
        }
        scene_.textures = std::move(replacementTextures);
        ++scene_.textureVersion_;
        scene_.textureGenerations_.assign(scene_.textures.size(), replacement.generation);
        scene_.freeTextureSlots_.clear();
        scene_.liveTextureCount_ = static_cast<std::uint32_t>(scene_.textures.size());

        scene_.retiredGeometry_.push_back({ scene_.geometryVersion_, std::move(scene_.vertexBuffer),
            std::move(scene_.indexBuffer) });
        scene_.vertexBuffer = std::move(replacementVertexBuffer);
        scene_.indexBuffer = std::move(replacementIndexBuffer);
        ++scene_.geometryVersion_;

        scene_.sceneGeneration_ = replacement.generation;
        scene_.vertexCapacity_ = replacement.vertexCapacity;
        scene_.indexCapacity_ = replacement.indexCapacity;
        scene_.freeVertexRanges_ = std::move(replacement.freeVertexRanges);
        scene_.freeIndexRanges_ = std::move(replacement.freeIndexRanges);
        scene_.retiredGeometryRanges_.clear();
        scene_.geometryRangeVersion_ = 1;
        scene_.imageGeometryRangeVersions_.assign(descriptors.setCount(), scene_.geometryRangeVersion_);

        scene_.matData = std::move(replacement.materials);
        scene_.materialNames = std::move(replacement.materialNames);
        scene_.materialGenerations_.assign(scene_.matData.size(), scene_.sceneGeneration_);
        scene_.materialAlive_.assign(scene_.matData.size(), true);
        scene_.freeMaterialSlots_.clear();

        scene_.meshResources = std::move(replacement.meshResources);
        scene_.meshGenerations_.assign(scene_.meshResources.size(), scene_.sceneGeneration_);
        scene_.meshAlive_.assign(scene_.meshResources.size(), true);
        scene_.freeMeshSlots_.clear();

        scene_.transformData = std::move(replacement.transforms);
        scene_.instanceNames = std::move(replacement.instanceNames);
        scene_.instanceGenerations.assign(scene_.transformData.size(), scene_.sceneGeneration_);
        scene_.instanceAlive.assign(scene_.transformData.size(), true);
        scene_.freeInstanceSlots.clear();
        scene_.drawData = std::move(replacement.draws);
        scene_.meshData = std::move(replacement.meshes);
        scene_.aabbs = std::move(replacement.bounds);
        scene_.animationPlayer_.reset();
        scene_.animationPlayer_ = std::move(replacement.animationPlayer);
        scene_.animatedDraws_ = std::move(replacement.animatedDraws);
        scene_.skinPalettes_ = std::move(replacement.skinPalettes);
        scene_.animationActors_ = std::move(replacement.animationActors);
        scene_.jointMatrices_ = std::move(replacement.jointMatrices);
        scene_.reserveFrameScratch();
        captureMemoryPeak();
    }

    [[nodiscard]] SceneTextureHandle uploadTexture(
        const danvulkan::assets::TextureAsset& source)
    {
        requireSceneUpdateAllowed("uploadTexture");
        if (scene_.freeTextureSlots_.empty() && scene_.textures.size() >= scene_.textureDescriptorCapacity_)
        {
            throw std::runtime_error("renderer bindless texture capacity has been reached");
        }
        if (scene_.textureVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("texture generation counter exhausted");
        }

        const std::uint32_t textureIndex = scene_.freeTextureSlots_.empty()
            ? static_cast<std::uint32_t>(scene_.textures.size())
            : scene_.freeTextureSlots_.back();
        Texture texture = createTextureImage(source);
        createTextureImageView(texture, textureIndex);
        createTextureSampler(texture, textureIndex);

        if (textureIndex == scene_.textures.size())
        {
            scene_.textures.emplace_back(std::move(texture));
            scene_.textureGenerations_.push_back(scene_.sceneGeneration_);
        }
        else
        {
            scene_.textures[textureIndex].emplace(std::move(texture));
            scene_.freeTextureSlots_.pop_back();
        }
        ++scene_.liveTextureCount_;
        ++scene_.textureVersion_;
        captureMemoryPeak();
        return { textureIndex, scene_.textureGenerations_[textureIndex] };
    }

    void destroyTexture(SceneTextureHandle textureHandle)
    {
        requireSceneUpdateAllowed("destroyTexture");
        requireValidTexture(textureHandle, "destroyTexture");
        if (scene_.liveTextureCount_ <= 1)
        {
            throw std::runtime_error("cannot destroy the renderer's last fallback texture");
        }

        const std::int32_t textureSlot = static_cast<std::int32_t>(textureHandle.slot);
        bool referenced = false;
        for (std::size_t index = 0; index < scene_.matData.size(); ++index)
        {
            if (!scene_.materialAlive_[index])
            {
                continue;
            }
            const MaterialData& material = scene_.matData[index];
            referenced = material.textureIndices.x == textureSlot ||
                material.textureIndices.y == textureSlot ||
                material.textureIndices.z == textureSlot ||
                material.textureIndices.w == textureSlot ||
                material.materialFlags.x == textureSlot;
            if (referenced)
            {
                break;
            }
        }
        if (referenced)
        {
            throw std::runtime_error("cannot destroy a texture while a material references it");
        }
        if (scene_.textureVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("texture generation counter exhausted");
        }

        scene_.retiredTextures_.emplace_back();
        RetiredTexture& retired = scene_.retiredTextures_.back();
        retired.version = scene_.textureVersion_;
        retired.texture = std::move(*scene_.textures[textureHandle.slot]);
        scene_.textures[textureHandle.slot].reset();
        scene_.textureGenerations_[textureHandle.slot] = nextGeneration(
            scene_.textureGenerations_[textureHandle.slot]);
        scene_.freeTextureSlots_.push_back(textureHandle.slot);
        --scene_.liveTextureCount_;
        ++scene_.textureVersion_;
    }

    [[nodiscard]] SceneMaterialHandle createMaterial(
        const RuntimeMaterialDescription& source)
    {
        requireSceneUpdateAllowed("createMaterial");
        if (scene_.freeMaterialSlots_.empty() && scene_.matData.size() >= scene_.materialCapacity_)
        {
            throw std::runtime_error("renderer material capacity has been reached");
        }

        switch (source.alphaMode)
        {
        case danvulkan::assets::AlphaMode::opaque:
        case danvulkan::assets::AlphaMode::mask:
        case danvulkan::assets::AlphaMode::blend:
            break;
        default:
            throw std::invalid_argument("createMaterial received an invalid alpha mode");
        }

        MaterialData material{};
        material.baseColorFactor = source.properties.baseColorFactor;
        material.emissiveMetallic = glm::vec4(source.properties.emissiveFactor,
            source.properties.metallicFactor);
        material.roughnessNormalOcclusionAlpha = {
            source.properties.roughnessFactor,
            source.properties.normalScale,
            source.properties.occlusionStrength,
            source.properties.alphaCutoff
        };
        material.textureTiling = glm::vec4(source.properties.textureTiling, 0.0f, 0.0f);
        material.textureIndices = {
            textureIndex(source.textures.baseColor, "base-color"),
            textureIndex(source.textures.normal, "normal"),
            textureIndex(source.textures.metallicRoughness, "metallic-roughness"),
            textureIndex(source.textures.occlusion, "occlusion")
        };
        material.materialFlags = {
            textureIndex(source.textures.emissive, "emissive"),
            static_cast<std::int32_t>(source.alphaMode),
            source.doubleSided ? 1 : 0,
            source.unlit ? 1 : 0
        };

        const std::uint32_t materialIndex = scene_.freeMaterialSlots_.empty()
            ? static_cast<std::uint32_t>(scene_.matData.size())
            : scene_.freeMaterialSlots_.back();
        std::string materialName = source.name.empty()
            ? "runtime material " + std::to_string(materialIndex)
            : source.name;
        if (materialIndex == scene_.matData.size())
        {
            scene_.matData.push_back(material);
            scene_.materialNames.push_back(std::move(materialName));
            scene_.materialGenerations_.push_back(scene_.sceneGeneration_);
            scene_.materialAlive_.push_back(true);
        }
        else
        {
            scene_.matData[materialIndex] = material;
            scene_.materialNames[materialIndex] = std::move(materialName);
            scene_.materialAlive_[materialIndex] = true;
            scene_.freeMaterialSlots_.pop_back();
        }
        return { materialIndex, scene_.materialGenerations_[materialIndex] };
    }

    void destroyMaterial(SceneMaterialHandle materialHandle)
    {
        requireSceneUpdateAllowed("destroyMaterial");
        requireValidMaterial(materialHandle, "destroyMaterial");
        for (std::size_t index = 0; index < scene_.meshResources.size(); ++index)
        {
            if (scene_.meshAlive_[index] && scene_.meshResources[index].materialIndex ==
                    static_cast<std::int32_t>(materialHandle.slot))
            {
                throw std::runtime_error(
                    "cannot destroy a material while a mesh references it");
            }
        }

        scene_.matData[materialHandle.slot] = {};
        scene_.materialNames[materialHandle.slot].clear();
        scene_.materialAlive_[materialHandle.slot] = false;
        scene_.materialGenerations_[materialHandle.slot] = nextGeneration(
            scene_.materialGenerations_[materialHandle.slot]);
        scene_.freeMaterialSlots_.push_back(materialHandle.slot);
    }

    [[nodiscard]] std::vector<SceneMeshInfo> sceneMeshes() const
    {
        requireInitialized("sceneMeshes");
        std::vector<SceneMeshInfo> result;
        result.reserve(scene_.meshResources.size());
        for (std::size_t index = 0; index < scene_.meshResources.size(); ++index)
        {
            if (!scene_.meshAlive_[index])
            {
                continue;
            }
            const MeshResourceData& mesh = scene_.meshResources[index];
            result.push_back({
                SceneMeshHandle{ static_cast<std::uint32_t>(index), scene_.meshGenerations_[index] },
                mesh.name,
                SceneMaterialHandle{ static_cast<std::uint32_t>(mesh.materialIndex),
                    scene_.materialGenerations_[mesh.materialIndex] },
                mesh.bounds.minVertex,
                mesh.bounds.maxVertex
            });
        }
        return result;
    }

    [[nodiscard]] SceneMeshHandle uploadMesh(
        std::span<const danvulkan::assets::Vertex> sourceVertices,
        std::span<const std::uint32_t> sourceIndices,
        SceneMaterialHandle materialHandle,
        std::string name)
    {
        requireSceneUpdateAllowed("uploadMesh");
        requireValidMaterial(materialHandle, "uploadMesh");
        if (sourceVertices.empty() || sourceIndices.empty())
        {
            throw std::invalid_argument("uploadMesh requires non-empty vertex and index data");
        }
        if (sourceVertices.size() > static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()) ||
            sourceIndices.size() > static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::runtime_error("uploaded geometry exceeds renderer index limits");
        }
        for (const std::uint32_t index : sourceIndices)
        {
            if (index >= sourceVertices.size())
            {
                throw std::invalid_argument("uploadMesh contains an out-of-range vertex index");
            }
        }
        if (scene_.freeMeshSlots_.empty() && scene_.meshResources.size() >=
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::runtime_error("renderer mesh handle capacity has been reached");
        }

        const std::uint32_t meshSlot = scene_.freeMeshSlots_.empty()
            ? static_cast<std::uint32_t>(scene_.meshResources.size())
            : scene_.freeMeshSlots_.back();
        const std::uint32_t vertexCount = static_cast<std::uint32_t>(sourceVertices.size());
        const std::uint32_t indexCount = static_cast<std::uint32_t>(sourceIndices.size());

        ensureGeometryCapacity(vertexCount, indexCount);
        const GeometryRange vertexRange = scene_.allocateGeometryRange(scene_.freeVertexRanges_, vertexCount);
        const GeometryRange indexRange = scene_.allocateGeometryRange(scene_.freeIndexRanges_, indexCount);
        try
        {
            uploadDeviceLocalBufferRange(scene_.vertexBuffer,
                sizeof(Vertex) * static_cast<VkDeviceSize>(vertexRange.offset),
                sizeof(Vertex) * static_cast<VkDeviceSize>(vertexRange.count),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sourceVertices.data(),
                "runtime vertex range");
            uploadDeviceLocalBufferRange(scene_.indexBuffer,
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(indexRange.offset),
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(indexRange.count),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, sourceIndices.data(),
                "runtime index range");
        }
        catch (...)
        {
            scene_.releaseGeometryRange(scene_.freeVertexRanges_, vertexRange);
            scene_.releaseGeometryRange(scene_.freeIndexRanges_, indexRange);
            throw;
        }

        const float maximum = std::numeric_limits<float>::max();
        AABB bounds{ glm::vec3(maximum), glm::vec3(-maximum) };
        for (const Vertex& vertex : sourceVertices)
        {
            bounds.minVertex = glm::min(bounds.minVertex, vertex.pos);
            bounds.maxVertex = glm::max(bounds.maxVertex, vertex.pos);
        }

        MeshResourceData mesh;
        mesh.indexCount = indexCount;
        mesh.vertexCount = vertexCount;
        mesh.firstIndex = indexRange.offset;
        mesh.vertexOffset = vertexRange.offset;
        mesh.materialIndex = static_cast<std::int32_t>(materialHandle.slot);
        const MaterialData& material = scene_.matData[materialHandle.slot];
        mesh.pipelineVariant = static_cast<std::uint32_t>(material.materialFlags.y * 2 +
            material.materialFlags.z);
        mesh.bounds = bounds;
        mesh.name = name.empty()
            ? "runtime mesh " + std::to_string(meshSlot)
            : std::move(name);
        if (meshSlot == scene_.meshResources.size())
        {
            scene_.meshResources.push_back(std::move(mesh));
            scene_.meshGenerations_.push_back(scene_.sceneGeneration_);
            scene_.meshAlive_.push_back(true);
        }
        else
        {
            scene_.meshResources[meshSlot] = std::move(mesh);
            scene_.meshAlive_[meshSlot] = true;
            scene_.freeMeshSlots_.pop_back();
        }

        scene_.reserveFrameScratch();
        captureMemoryPeak();
        return { meshSlot, scene_.meshGenerations_[meshSlot] };
    }

    [[nodiscard]] SceneMeshHandle uploadGrass(
        std::span<const RuntimeGrassBlade> sourceBlades,
        SceneMaterialHandle materialHandle,
        const RuntimeGrassLodDescription& lod,
        std::string name)
    {
        requireSceneUpdateAllowed("uploadGrass");
        requireValidMaterial(materialHandle, "uploadGrass");
        PreparedRuntimeGrass prepared = prepareRuntimeGrass(
            std::vector<RuntimeGrassBlade>(sourceBlades.begin(), sourceBlades.end()), lod);
        RuntimeMeshUploadDescription upload{{}, prepared.indices, materialHandle,
            name.empty() ? "procedural Bezier grass" : std::move(name), &prepared};
        return uploadMeshBatch(std::span<const RuntimeMeshUploadDescription>(&upload, 1U)).front();
    }

    [[nodiscard]] std::vector<SceneMeshHandle> uploadMeshBatch(
        std::span<const RuntimeMeshUploadDescription> uploads)
    {
        requireSceneUpdateAllowed("uploadMeshBatch");
        if (uploads.empty())
        {
            throw std::invalid_argument("uploadMeshBatch requires at least one mesh");
        }
        if (!uploadContext_.asyncBufferUploadReady())
        {
            throw std::logic_error("uploadMeshBatch asynchronous staging slot is busy");
        }
        if (scene_.meshResources.size() - scene_.freeMeshSlots_.size() + uploads.size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::runtime_error("renderer mesh handle capacity has been reached");
        }

        std::uint64_t totalVertices64 = 0;
        std::uint64_t totalIndices64 = 0;
        for (const RuntimeMeshUploadDescription& upload : uploads)
        {
            requireValidMaterial(upload.material, "uploadMeshBatch");
            const bool grassUpload = upload.grass != nullptr;
            const std::size_t recordCount = grassUpload
                ? upload.grass->records.size() : upload.vertices.size();
            const std::size_t vertexUnits = grassUpload
                ? packedGrassGeometryUnits(recordCount)
                : recordCount;
            if (recordCount == 0U || upload.indices.empty())
            {
                throw std::invalid_argument("uploadMeshBatch contains an empty mesh");
            }
            if (vertexUnits >
                    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
                upload.indices.size() >
                    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
            {
                throw std::runtime_error("batched geometry exceeds renderer index limits");
            }
            if (std::ranges::any_of(upload.indices,
                    [&](std::uint32_t index)
                    {
                        return index >= (grassUpload ? grassTemplateVertexCount : recordCount);
                    }))
            {
                throw std::invalid_argument("uploadMeshBatch contains an out-of-range index");
            }
            if (grassUpload &&
                (!upload.vertices.empty() ||
                    upload.indices.data() != upload.grass->indices.data() ||
                    upload.indices.size() != upload.grass->indices.size() ||
                    upload.grass->tiles.empty()))
            {
                throw std::invalid_argument(
                    "uploadMeshBatch grass metadata does not describe its source spans");
            }
            totalVertices64 += vertexUnits;
            totalIndices64 += upload.indices.size();
        }
        if (totalVertices64 > std::numeric_limits<std::uint32_t>::max() ||
            totalIndices64 > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error("batched geometry capacity overflowed");
        }

        ensureGeometryCapacity(static_cast<std::uint32_t>(totalVertices64),
            static_cast<std::uint32_t>(totalIndices64));
        std::vector<GeometryRange> vertexRanges;
        std::vector<GeometryRange> indexRanges;
        vertexRanges.reserve(uploads.size());
        indexRanges.reserve(uploads.size());
        try
        {
            for (const RuntimeMeshUploadDescription& upload : uploads)
            {
                const std::size_t vertexUnits = upload.grass != nullptr
                    ? packedGrassGeometryUnits(upload.grass->records.size())
                    : upload.vertices.size();
                vertexRanges.push_back(scene_.allocateGeometryRange(scene_.freeVertexRanges_,
                    static_cast<std::uint32_t>(vertexUnits)));
                indexRanges.push_back(scene_.allocateGeometryRange(scene_.freeIndexRanges_,
                    static_cast<std::uint32_t>(upload.indices.size())));
            }
        }
        catch (...)
        {
            for (const GeometryRange range : vertexRanges)
            {
                scene_.releaseGeometryRange(scene_.freeVertexRanges_, range);
            }
            for (const GeometryRange range : indexRanges)
            {
                scene_.releaseGeometryRange(scene_.freeIndexRanges_, range);
            }
            throw;
        }

        std::vector<danvulkan::vk::BufferUploadRequest> requests;
        requests.reserve(uploads.size() * 2U);
        std::vector<MeshResourceData> resources(uploads.size());
        for (std::size_t index = 0; index < uploads.size(); ++index)
        {
            const RuntimeMeshUploadDescription& upload = uploads[index];
            const GeometryRange vertexRange = vertexRanges[index];
            const GeometryRange indexRange = indexRanges[index];
            const void* recordData = upload.grass != nullptr
                ? static_cast<const void*>(upload.grass->records.data())
                : static_cast<const void*>(upload.vertices.data());
            const VkDeviceSize recordBytes = upload.grass != nullptr
                ? sizeof(PackedRuntimeGrassBlade) *
                    static_cast<VkDeviceSize>(upload.grass->records.size())
                : sizeof(Vertex) * static_cast<VkDeviceSize>(upload.vertices.size());
            requests.push_back({scene_.vertexBuffer,
                sizeof(Vertex) * static_cast<VkDeviceSize>(vertexRange.offset),
                recordData, recordBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
            requests.push_back({scene_.indexBuffer,
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(indexRange.offset),
                upload.indices.data(),
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(indexRange.count),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT});

            const float maximum = std::numeric_limits<float>::max();
            AABB bounds{glm::vec3(maximum), glm::vec3(-maximum)};
            if (upload.grass != nullptr)
            {
                for (const PackedRuntimeGrassBlade& blade : upload.grass->records)
                {
                    bounds.minVertex = glm::min(bounds.minVertex, blade.base);
                    bounds.maxVertex = glm::max(bounds.maxVertex, blade.base);
                }
            }
            else
            {
                for (const Vertex& vertex : upload.vertices)
                {
                    bounds.minVertex = glm::min(bounds.minVertex, vertex.pos);
                    bounds.maxVertex = glm::max(bounds.maxVertex, vertex.pos);
                }
            }
            MeshResourceData& resource = resources[index];
            resource.indexCount = indexRange.count;
            resource.vertexCount = vertexRange.count;
            resource.firstIndex = indexRange.offset;
            resource.vertexOffset = vertexRange.offset;
            resource.materialIndex = static_cast<std::int32_t>(upload.material.slot);
            const MaterialData& material = scene_.matData[upload.material.slot];
            resource.pipelineVariant = static_cast<std::uint32_t>(
                material.materialFlags.y * 2 + material.materialFlags.z);
            resource.bounds = bounds;
            resource.name = upload.name;

            if (upload.grass != nullptr)
            {
                const PreparedRuntimeGrass& grass = *upload.grass;
                resource.grass.enabled = true;
                resource.grass.indexCounts = grassTemplateIndexCounts;
                resource.grass.firstIndices = {
                    resource.firstIndex + grassTemplateFirstIndexOffsets[0],
                    resource.firstIndex + grassTemplateFirstIndexOffsets[1],
                    resource.firstIndex + grassTemplateFirstIndexOffsets[2]};
                resource.grass.distances = {grass.lod.highDetailDistance,
                    grass.lod.mediumDetailDistance, grass.lod.maximumDistance};
                resource.grass.populationRatios = {
                    1.0f, grass.lod.mediumPopulation, grass.lod.lowPopulation};
                resource.grass.hysteresis = grass.lod.hysteresis;
                resource.grass.transitionBand = grass.lod.transitionBand;
                resource.grass.tiles.reserve(grass.tiles.size());
                for (const RuntimeGrassTileDescription& source : grass.tiles)
                {
                    resource.grass.tiles.push_back({
                        {source.boundsMinimum, source.boundsMaximum},
                        source.firstBlade, source.bladeCount});
                }
                const glm::vec3 padding(grass.maximumHeight * 0.5f);
                resource.bounds.minVertex -= padding;
                resource.bounds.maxVertex += padding;
                resource.bounds.maxVertex.y += grass.maximumHeight;
            }
        }

        try
        {
            uploadContext_.uploadBuffersAsync(requests, "runtime mesh batch");
        }
        catch (...)
        {
            for (const GeometryRange range : vertexRanges)
            {
                scene_.releaseGeometryRange(scene_.freeVertexRanges_, range);
            }
            for (const GeometryRange range : indexRanges)
            {
                scene_.releaseGeometryRange(scene_.freeIndexRanges_, range);
            }
            throw;
        }

        std::vector<SceneMeshHandle> handles;
        handles.reserve(resources.size());
        for (MeshResourceData& resource : resources)
        {
            std::uint32_t slot = 0;
            if (!scene_.freeMeshSlots_.empty())
            {
                slot = scene_.freeMeshSlots_.back();
                scene_.freeMeshSlots_.pop_back();
                scene_.meshResources[slot] = std::move(resource);
                scene_.meshAlive_[slot] = true;
            }
            else
            {
                slot = static_cast<std::uint32_t>(scene_.meshResources.size());
                if (resource.name.empty())
                {
                    resource.name = "runtime mesh " + std::to_string(slot);
                }
                scene_.meshResources.push_back(std::move(resource));
                scene_.meshGenerations_.push_back(scene_.sceneGeneration_);
                scene_.meshAlive_.push_back(true);
            }
            handles.push_back({slot, scene_.meshGenerations_[slot]});
        }
        scene_.reserveFrameScratch();
        captureMemoryPeak();
        return handles;
    }

    [[nodiscard]] bool runtimeUploadReady()
    {
        requireSceneUpdateAllowed("runtimeUploadReady");
        return uploadContext_.asyncBufferUploadReady();
    }

    void destroyMesh(SceneMeshHandle meshHandle)
    {
        requireSceneUpdateAllowed("destroyMesh");
        requireValidMesh(meshHandle, "destroyMesh");
        const bool referenced = std::ranges::any_of(scene_.meshData,
            [&](const MeshData& mesh)
            {
                return mesh.meshResourceSlot == meshHandle.slot;
            });
        if (referenced)
        {
            throw std::runtime_error("cannot destroy a mesh while an instance references it");
        }
        if (scene_.geometryRangeVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("geometry range retirement counter exhausted");
        }

        const MeshResourceData& resource = scene_.meshResources[meshHandle.slot];
        scene_.retiredGeometryRanges_.push_back({
            scene_.geometryRangeVersion_,
            { resource.vertexOffset, resource.vertexCount },
            { resource.firstIndex, resource.indexCount }
        });
        ++scene_.geometryRangeVersion_;
        scene_.meshResources[meshHandle.slot] = {};
        scene_.meshAlive_[meshHandle.slot] = false;
        scene_.meshGenerations_[meshHandle.slot] = nextGeneration(scene_.meshGenerations_[meshHandle.slot]);
        scene_.freeMeshSlots_.push_back(meshHandle.slot);
    }

    [[nodiscard]] SceneInstanceHandle createMeshInstance(SceneMeshHandle meshHandle,
        const glm::mat4& worldTransform, std::string name)
    {
        requireSceneUpdateAllowed("createMeshInstance");
        requireValidMesh(meshHandle, "createMeshInstance");
        if (scene_.meshData.size() >= DrawDataCount)
        {
            throw std::runtime_error("renderer instance draw capacity has been reached");
        }

        std::uint32_t instanceSlot = 0;
        if (!scene_.freeInstanceSlots.empty())
        {
            instanceSlot = scene_.freeInstanceSlots.back();
            scene_.freeInstanceSlots.pop_back();
        }
        else
        {
            if (scene_.transformData.size() >= TransformDataCount)
            {
                throw std::runtime_error("renderer instance transform capacity has been reached");
            }
            instanceSlot = static_cast<std::uint32_t>(scene_.transformData.size());
            scene_.transformData.push_back({});
            scene_.instanceNames.emplace_back();
            scene_.instanceGenerations.push_back(scene_.sceneGeneration_);
            scene_.instanceAlive.push_back(false);
        }

        scene_.transformData[instanceSlot].model = worldTransform;
        scene_.instanceNames[instanceSlot] = name.empty() ? scene_.meshResources[meshHandle.slot].name : std::move(name);
        scene_.instanceAlive[instanceSlot] = true;

        const MeshResourceData& resource = scene_.meshResources[meshHandle.slot];
        DrawData draw{};
        draw.materialIndex = resource.materialIndex;
        draw.transformIndex = static_cast<std::int32_t>(instanceSlot);
        draw.vertexOffset = static_cast<std::int32_t>(resource.vertexOffset);

        MeshData mesh{};
        mesh.indexCount = resource.indexCount;
        mesh.firstIndex = resource.firstIndex;
        mesh.vertexOffset = resource.vertexOffset;
        mesh.pipelineVariant = resource.pipelineVariant;
        mesh.meshResourceSlot = meshHandle.slot;
        mesh.drawData = draw;
        mesh.localBounds = resource.bounds;
        mesh.grass = resource.grass;
        scene_.meshData.push_back(mesh);
        scene_.aabbs.push_back(scene_.transformedBounds(resource.bounds, worldTransform));

        return { instanceSlot, scene_.instanceGenerations[instanceSlot] };
    }

    void destroyInstance(SceneInstanceHandle instanceHandle)
    {
        requireSceneUpdateAllowed("destroyInstance");
        requireValidInstance(instanceHandle, "destroyInstance");

        for (std::size_t index = scene_.meshData.size(); index-- > 0;)
        {
            if (scene_.meshData[index].drawData.transformIndex ==
                static_cast<std::int32_t>(instanceHandle.slot))
            {
                scene_.meshData.erase(scene_.meshData.begin() + static_cast<std::ptrdiff_t>(index));
                scene_.aabbs.erase(scene_.aabbs.begin() + static_cast<std::ptrdiff_t>(index));
            }
        }

        scene_.instanceAlive[instanceHandle.slot] = false;
        scene_.instanceNames[instanceHandle.slot].clear();
        scene_.instanceGenerations[instanceHandle.slot] = nextGeneration(
            scene_.instanceGenerations[instanceHandle.slot]);
        scene_.freeInstanceSlots.push_back(instanceHandle.slot);
    }

    void updateInstanceTransform(SceneInstanceHandle instanceHandle, const glm::mat4& worldTransform)
    {
        requireSceneUpdateAllowed("updateInstanceTransform");
        requireValidInstance(instanceHandle, "updateInstanceTransform");

        scene_.transformData[instanceHandle.slot].model = worldTransform;
        for (std::size_t index = 0; index < scene_.meshData.size(); ++index)
        {
            if (scene_.meshData[index].drawData.transformIndex ==
                static_cast<std::int32_t>(instanceHandle.slot))
            {
                scene_.aabbs[index] = scene_.transformedBounds(scene_.meshData[index].localBounds, worldTransform);
            }
        }
    }

    void updateMaterialProperties(SceneMaterialHandle material,
        const RuntimeMaterialProperties& properties)
    {
        requireSceneUpdateAllowed("updateMaterialProperties");
        requireValidMaterial(material, "updateMaterialProperties");

        MaterialData& destination = scene_.matData[material.slot];
        destination.baseColorFactor = properties.baseColorFactor;
        destination.emissiveMetallic = glm::vec4(properties.emissiveFactor,
            properties.metallicFactor);
        destination.roughnessNormalOcclusionAlpha = {
            properties.roughnessFactor,
            properties.normalScale,
            properties.occlusionStrength,
            properties.alphaCutoff
        };
        destination.textureTiling = glm::vec4(properties.textureTiling, 0.0f, 0.0f);
    }

    void updateMaterialTextures(SceneMaterialHandle materialHandle,
        const RuntimeMaterialTextures& source)
    {
        requireSceneUpdateAllowed("updateMaterialTextures");
        requireValidMaterial(materialHandle, "updateMaterialTextures");

        MaterialData& destination = scene_.matData[materialHandle.slot];
        destination.textureIndices = {
            textureIndex(source.baseColor, "base-color"),
            textureIndex(source.normal, "normal"),
            textureIndex(source.metallicRoughness, "metallic-roughness"),
            textureIndex(source.occlusion, "occlusion")
        };
        destination.materialFlags.x = textureIndex(source.emissive, "emissive");
    }

    void shutdown()
    {
        if (!initialized_)
        {
            if (shutdown_)
            {
                return;
            }
            shutdown_ = true;
            return;
        }
        if (frameInProgress_)
        {
            throw std::logic_error("shutdown called while a frame is in progress");
        }

        checkVk(vkDeviceWaitIdle(device), "vkDeviceWaitIdle(shutdown)");
        cleanup();
        initialized_ = false;
        shutdown_ = true;

        if (validationErrorSeen_.load())
        {
            throw std::runtime_error("Vulkan validation reported one or more errors");
        }
    }

    void run()
    {
        initialize();
        try
        {
            while (beginFrame())
            {
                update();
                submitScene(demoSceneSubmission());
                endFrame();
            }
            shutdown();
        }
        catch (...)
        {
            shutdownNoThrow();
            throw;
        }
    }

private:
    RendererConfig config_;
    std::shared_ptr<RendererPlatform> platform_;
    bool initialized_ = false;
    bool shutdown_ = false;
    bool frameInProgress_ = false;
    std::uint64_t renderedFrames_ = 0;
    std::chrono::steady_clock::time_point frameBegin_{};
    std::optional<double> frameThreadCpuBeginMilliseconds_;
    std::thread::id frameThread_{};
    std::chrono::steady_clock::time_point animationPreviousTime_{};
    float animationDeltaSeconds_ = 0.0f;
    SceneSubmission pendingSubmission_;
    UiDrawData pendingUi_;
    SceneSubmission demoSubmission_;
    std::vector<std::uint8_t> submittedAnimationActorScratch_;
    danvulkan::LightingPlan lightingPlan_;
    bool hasPendingSubmission_ = false;
    bool hasPendingUi_ = false;

    // Declared before all dependent Vulkan resources so they are destroyed last.
    danvulkan::vk::Instance instance;
    danvulkan::vk::DebugMessenger debugMessenger;
    danvulkan::vk::Surface surface;
    danvulkan::vk::DeviceContext device;
    danvulkan::vk::Allocator allocator;
    danvulkan::vk::SwapchainContext swapchain;
    danvulkan::vk::AttachmentContext attachments;
    danvulkan::vk::PresentationContext presentation;
    danvulkan::vk::DescriptorContext descriptors;
    danvulkan::vk::PipelineContext pipelines;
    danvulkan::vk::PipelineContext grassPipelines_;
    danvulkan::vk::PipelineContext skyPipelines_;
    danvulkan::vk::PipelineContext shadowPipelines_;
    danvulkan::vk::PipelineContext grassShadowPipelines_;
    danvulkan::vk::UiContext ui_;
    // Declared after the device and allocator so scene and lighting resources retire first.
    danvulkan::vk::SceneContext scene_;
    danvulkan::vk::LightingContext lighting_;

    const std::string MODEL_PATH;
    const std::string TEXTURE_PATH = "textures/viking_room.png";
    const std::string SHADER_PATH = "shaders/";
    const std::string COMPILED_SHADER_PATH = DANVULKAN_SHADER_DIR;
    std::unordered_map<std::string, std::filesystem::file_time_type> shaderPaths;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation",
        //"VK_LAYER_RENDERDOC_Capture"
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
    

    float timestampPeriod = 0.0f;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    std::size_t attachmentTargetCount_ = 0;
    RendererMemoryStats lastMemoryStats_{};
    mutable std::uint64_t peakBlockBytes_ = 0;
    mutable std::uint64_t peakAllocationBytes_ = 0;
    mutable std::uint64_t peakHeapUsageBytes_ = 0;
    mutable std::uint32_t peakBlockCount_ = 0;
    mutable std::uint32_t peakAllocationCount_ = 0;

    std::array<danvulkan::vk::FrameContext, MAX_FRAMES_IN_FLIGHT> frames;
    danvulkan::vk::UploadContext uploadContext_;

    size_t currentFrame = 0;

    // One per swap chain image
    std::vector<danvulkan::vk::Buffer> uniformBuffers;

    Camera camera;
    const std::chrono::steady_clock::time_point applicationStart_ =
        std::chrono::steady_clock::now();
    std::chrono::high_resolution_clock::time_point previousTime = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point lastTimeStamp = previousTime;
    glm::vec3 lightPos, lightSpeed;
    struct CameraDebugData
    {
        glm::vec3 forward;
        glm::vec3 right;
        glm::vec3 top;
    };

    std::array<float, 100> frameTimes;
    std::atomic_bool validationErrorSeen_ = false;
    double frameGpuAvg = 0.0;
    double frameAvg_ = 0.0;
    double frameCpuAvg_ = 0.0;
    double animationCpuAvg_ = 0.0;
    double animationEvaluationCpuAvg_ = 0.0;
    double animationSynchronizationCpuAvg_ = 0.0;
    double animationSamplingCpuAvg_ = 0.0;
    double animationPoseResetCpuAvg_ = 0.0;
    double animationTimelineResolutionCpuAvg_ = 0.0;
    double animationVectorSamplingCpuAvg_ = 0.0;
    double animationRotationSamplingCpuAvg_ = 0.0;
    double animationTransformPropagationCpuAvg_ = 0.0;
    double animationTransformUpdateCpuAvg_ = 0.0;
    double animationBoundsCpuAvg_ = 0.0;
    double animationPaletteGenerationCpuAvg_ = 0.0;
    double animationPaletteUploadCpuAvg_ = 0.0;
    double cullingCpuAvg_ = 0.0;
    double bufferWriteCpuAvg_ = 0.0;
    double commandRecordingCpuAvg_ = 0.0;
    std::uint32_t lastActiveDrawCount_ = 0;
    std::uint32_t lastVisibleDrawCount_ = 0;
    std::uint32_t lastAnimatedDrawCount_ = 0;
    std::uint32_t lastJointMatrixCount_ = 0;
    std::uint32_t lastPointLightCount_ = 0;
    std::uint32_t lastAnimationActorCount_ = 0;
    std::uint32_t lastEvaluatedAnimationActorCount_ = 0;
    std::uint32_t lastCulledAnimationActorCount_ = 0;
    std::uint32_t lastSampledAnimationChannelCount_ = 0;
    std::uint32_t lastSampledAnimationVectorChannelCount_ = 0;
    std::uint32_t lastSampledAnimationRotationChannelCount_ = 0;
    std::uint32_t lastNlerpAnimationRotationChannelCount_ = 0;
    std::uint32_t lastAnimationClipChannelCount_ = 0;
    std::uint32_t lastFoldedConstantAnimationChannelCount_ = 0;
    std::uint32_t lastAnimationPropagatedNodeCount_ = 0;
    std::uint32_t lastAnimationPoseComposedNodeCount_ = 0;
    std::uint32_t lastAnimationCachedLocalNodeCount_ = 0;

    [[nodiscard]] static double millisecondsSince(
        std::chrono::steady_clock::time_point begin) noexcept
    {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    }

    [[nodiscard]] static double rollingAverage(double current, double sample) noexcept
    {
        return current == 0.0 ? sample : current * 0.95 + sample * 0.05;
    }

    void captureMemoryPeak() const noexcept
    {
        static_cast<void>(memoryStats());
    }

    void initPlatform()
    {
        if (platform_ == nullptr)
        {
            platform_ = makeGlfwRendererPlatform(config_.applicationName, config_.width,
                config_.height);
        }
    }
    
    void initVulkan() 
    {
        createInstance();
        createDebugMessenger();
        createSurface();
        device.initialize(instance, surface, enableValidationLayers);
        timestampPeriod = device.properties().limits.timestampPeriod;
        const VkSampleCountFlags supportedSamples =
            device.properties().limits.framebufferColorSampleCounts &
            device.properties().limits.framebufferDepthSampleCounts;
        const std::optional<danvulkan::vk::MemoryPolicyPlan> memoryPlan =
            danvulkan::vk::planMemoryPolicy(supportedSamples,
                config_.memory.maxMsaaSamples, frames.size(),
                config_.memory.preferLazilyAllocatedAttachments);
        if (!memoryPlan)
        {
            throw std::invalid_argument("renderer memory policy is invalid for this device");
        }
        msaaSamples = memoryPlan->samples;
        attachmentTargetCount_ = memoryPlan->attachmentSetCount;
        createAllocator();
        const RendererFramebufferExtent framebuffer = platform_->framebufferExtent();
        swapchain.initialize(device, surface, { framebuffer.width, framebuffer.height },
            enableValidationLayers);
        createFrameContexts();
        createUploadContext();
        loadModel();
        lighting_.initialize(device, allocator, uploadContext_,
            config_.environmentMap ? &*config_.environmentMap : nullptr,
            config_.maxPointLights, swapchain.imageCount(),
            config_.directionalShadowResolution,
            enableValidationLayers);
        lightingPlan_.pointLights.reserve(config_.maxPointLights);
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createUiContext();
        createSwapchainAttachments();
        createTextureImageViews();
        createTextureSamplers();
        createUniformBuffers();
        createBindlessBuffers();
        updateIndirectBuffer();
        createDescriptorSets();
        presentation.initialize(device, swapchain.imageCount(), enableValidationLayers);
        //createIMGUI();
        initGame();
        captureMemoryPeak();
        // The decoded environment payload is only preparation data; LightingContext now owns
        // the uploaded image, view, sampler, and mip chain.
        config_.environmentMap.reset();
    }

    bool recreateSwapChain()
    {
        RendererFramebufferExtent extent = platform_->framebufferExtent();
        while (extent.width == 0 || extent.height == 0)
        {
            if (platform_->shouldClose())
            {
                return false;
            }
            platform_->waitEvents();
            extent = platform_->framebufferExtent();
        }

        waitForSwapchainRetirement();
        const std::size_t previousImageCount = swapchain.imageCount();
        swapchain.recreate(device, surface, { extent.width, extent.height },
            enableValidationLayers);
        presentation.recreate(swapchain.imageCount(), enableValidationLayers);
        if (swapchain.imageCount() != previousImageCount)
        {
            rebuildSwapchainIndexedResources();
        }
        camera.updateAspectRatio(
            swapchain.extent().width / static_cast<float>(swapchain.extent().height));
        createGraphicsPipeline();
        if (!ui_.compatible(swapchain.format()))
        {
            ui_.rebuild(swapchain.format(),
                std::filesystem::path(COMPILED_SHADER_PATH) / "ui_vert.spv",
                std::filesystem::path(COMPILED_SHADER_PATH) / "ui_frag.spv",
                enableValidationLayers);
        }
        recreateSwapchainAttachments();
        return true;
    }

    void waitForSwapchainRetirement()
    {
        for (danvulkan::vk::FrameContext& frame : frames)
        {
            if (const std::optional<double> gpuMilliseconds =
                    frame.waitForReuse(timestampPeriod))
            {
                frameGpuAvg = frameGpuAvg * 0.95 + *gpuMilliseconds * 0.05;
            }
        }
        checkVk(vkQueueWaitIdle(device.presentQueue()),
            "vkQueueWaitIdle(swapchain presentation retirement)");
    }

    void requireInitialized(std::string_view operation) const
    {
        if (!initialized_)
        {
            throw std::logic_error(std::string(operation) + " requires an initialized renderer");
        }
    }

    void requireFrameInProgress(std::string_view operation) const
    {
        requireInitialized(operation);
        if (!frameInProgress_)
        {
            throw std::logic_error(std::string(operation) + " requires beginFrame first");
        }
    }

    void requireSceneUpdateAllowed(std::string_view operation) const
    {
        requireInitialized(operation);
        if (frameInProgress_)
        {
            throw std::logic_error(std::string(operation) + " must be called between frames");
        }
    }

    [[nodiscard]] static std::uint32_t nextGeneration(std::uint32_t generation) noexcept
    {
        return generation == std::numeric_limits<std::uint32_t>::max() ? 1U : generation + 1U;
    }

    [[nodiscard]] std::uint32_t nextSceneGeneration() const
    {
        std::unordered_set<std::uint32_t> generations;
        generations.insert(scene_.instanceGenerations.begin(), scene_.instanceGenerations.end());
        generations.insert(scene_.materialGenerations_.begin(), scene_.materialGenerations_.end());
        generations.insert(scene_.meshGenerations_.begin(), scene_.meshGenerations_.end());
        generations.insert(scene_.textureGenerations_.begin(), scene_.textureGenerations_.end());

        std::uint32_t candidate = nextGeneration(scene_.sceneGeneration_);
        for (std::size_t attempt = 0; attempt <= generations.size(); ++attempt)
        {
            if (!generations.contains(candidate))
            {
                return candidate;
            }
            candidate = nextGeneration(candidate);
        }
        throw std::runtime_error("scene generation counter exhausted");
    }

    void requireValidInstance(SceneInstanceHandle instanceHandle,
        std::string_view operation) const
    {
        if (instanceHandle.slot >= scene_.transformData.size() ||
            !scene_.instanceAlive[instanceHandle.slot] ||
            instanceHandle.generation != scene_.instanceGenerations[instanceHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale instance handle");
        }
    }

    void requireValidTexture(SceneTextureHandle textureHandle,
        std::string_view operation) const
    {
        if (textureHandle.slot >= scene_.textures.size() ||
            !scene_.textures[textureHandle.slot] ||
            textureHandle.slot >= scene_.textureGenerations_.size() ||
            textureHandle.generation != scene_.textureGenerations_[textureHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale texture handle");
        }
    }

    void requireValidMaterial(SceneMaterialHandle materialHandle,
        std::string_view operation) const
    {
        if (materialHandle.slot >= scene_.matData.size() ||
            materialHandle.slot >= scene_.materialAlive_.size() ||
            !scene_.materialAlive_[materialHandle.slot] ||
            materialHandle.slot >= scene_.materialGenerations_.size() ||
            materialHandle.generation != scene_.materialGenerations_[materialHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale material handle");
        }
    }

    void requireValidMesh(SceneMeshHandle meshHandle,
        std::string_view operation) const
    {
        if (meshHandle.slot >= scene_.meshResources.size() ||
            meshHandle.slot >= scene_.meshAlive_.size() ||
            !scene_.meshAlive_[meshHandle.slot] ||
            meshHandle.slot >= scene_.meshGenerations_.size() ||
            meshHandle.generation != scene_.meshGenerations_[meshHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale mesh handle");
        }
    }

    [[nodiscard]] danvulkan::AnimationPlayer& requireAnimationPlayer(
        std::string_view operation)
    {
        if (!scene_.animationPlayer_ || scene_.animationPlayer_->clipCount() == 0)
        {
            throw std::logic_error(std::string(operation) +
                " requires a scene containing animation clips");
        }
        return *scene_.animationPlayer_;
    }

    void requireValidAnimation(SceneAnimationHandle animation,
        std::string_view operation) const
    {
        if (!scene_.animationPlayer_ || animation.generation != scene_.sceneGeneration_ ||
            animation.slot >= scene_.animationPlayer_->clipCount())
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale animation handle");
        }
    }

    [[nodiscard]] static RuntimeMaterialProperties runtimeProperties(const MaterialData& material)
    {
        RuntimeMaterialProperties properties;
        properties.baseColorFactor = material.baseColorFactor;
        properties.textureTiling = glm::vec2(material.textureTiling);
        properties.emissiveFactor = glm::vec3(material.emissiveMetallic);
        properties.metallicFactor = material.emissiveMetallic.w;
        properties.roughnessFactor = material.roughnessNormalOcclusionAlpha.x;
        properties.normalScale = material.roughnessNormalOcclusionAlpha.y;
        properties.occlusionStrength = material.roughnessNormalOcclusionAlpha.z;
        properties.alphaCutoff = material.roughnessNormalOcclusionAlpha.w;
        return properties;
    }

    [[nodiscard]] SceneTextureHandle textureHandle(std::int32_t textureIndex) const noexcept
    {
        if (textureIndex < 0 || static_cast<std::size_t>(textureIndex) >= scene_.textures.size() ||
            !scene_.textures[textureIndex] || static_cast<std::size_t>(textureIndex) >=
                scene_.textureGenerations_.size())
        {
            return {};
        }
        return { static_cast<std::uint32_t>(textureIndex), scene_.textureGenerations_[textureIndex] };
    }

    [[nodiscard]] RuntimeMaterialTextures runtimeTextures(const MaterialData& material) const noexcept
    {
        RuntimeMaterialTextures result;
        result.baseColor = textureHandle(material.textureIndices.x);
        result.normal = textureHandle(material.textureIndices.y);
        result.metallicRoughness = textureHandle(material.textureIndices.z);
        result.occlusion = textureHandle(material.textureIndices.w);
        result.emissive = textureHandle(material.materialFlags.x);
        return result;
    }

    [[nodiscard]] std::int32_t textureIndex(SceneTextureHandle texture,
        std::string_view role) const
    {
        if (!texture)
        {
            return -1;
        }
        if (texture.slot >= scene_.textures.size() || !scene_.textures[texture.slot] ||
            texture.slot >= scene_.textureGenerations_.size() ||
            texture.generation != scene_.textureGenerations_[texture.slot])
        {
            throw std::invalid_argument("invalid or stale " + std::string(role) +
                " texture handle");
        }
        return static_cast<std::int32_t>(texture.slot);
    }

    void updateWindowTitle()
    {
        const double frameMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frameBegin_).count();
        double frameCpuMilliseconds = frameMilliseconds;
        if (frameThread_ == std::this_thread::get_id() &&
            frameThreadCpuBeginMilliseconds_)
        {
            if (const std::optional<double> currentCpu = currentThreadCpuMilliseconds())
            {
                frameCpuMilliseconds = std::clamp(
                    *currentCpu - *frameThreadCpuBeginMilliseconds_, 0.0, frameMilliseconds);
            }
        }
        frameAvg_ = frameAvg_ * 0.95 + frameMilliseconds * 0.05;
        frameCpuAvg_ = frameCpuAvg_ * 0.95 + frameCpuMilliseconds * 0.05;
        const double currentFps = frameAvg_ > 0.0 ? 1000.0 / frameAvg_ : 0.0;
        char title[256];
        std::snprintf(title, sizeof(title),
            "DanVulkan frame: %.2f ms; cpu: %.2f ms; gpu: %.2f ms; FPS: %.2f",
            frameAvg_, frameCpuAvg_, frameGpuAvg, currentFps);
        platform_->setWindowTitle(title);
    }

    [[nodiscard]] const SceneSubmission& demoSceneSubmission()
    {
        demoSubmission_.view = camera.matrices.view;
        demoSubmission_.projection = camera.matrices.perspective;
        demoSubmission_.cameraPosition = glm::vec3(glm::inverse(camera.matrices.view)[3]);
        demoSubmission_.pointLights.front().position = lightPos;
        return demoSubmission_;
    }

    void shutdownNoThrow() noexcept
    {
        if (!initialized_)
        {
            return;
        }

        frameInProgress_ = false;
        hasPendingSubmission_ = false;
        hasPendingUi_ = false;
        vkDeviceWaitIdle(device);
        cleanup();
        initialized_ = false;
        shutdown_ = true;
    }

    void cleanupSwapChain()
    {
        attachments.reset();
        presentation.reset();
        swapchain.reset();
    }
    void cleanup() 
    {
        lastMemoryStats_ = memoryStats();
        cleanupSwapChain();
        ui_.reset();
        grassShadowPipelines_.reset();
        shadowPipelines_.reset();
        skyPipelines_.reset();
        grassPipelines_.reset();
        pipelines.reset();
        descriptors.reset();
        uniformBuffers.clear();
        lighting_.reset();
        scene_.resetScene(device);
        
        
        for (danvulkan::vk::FrameContext& frame : frames)
        {
            frame.reset();
        }
        uploadContext_.reset();
        allocator.reset();
        device.reset();
        surface.reset();
        debugMessenger.reset();
        instance.reset();
        platform_.reset();
    }
    
    void recreateGraphicsPipeline()
    {
        vkDeviceWaitIdle(device); // don't touch resources that may still be in use
        pipelines.rebuild(pipelineCreateInfo());
        danvulkan::vk::PipelineContextCreateInfo grassCreateInfo = pipelineCreateInfo();
        grassCreateInfo.vertexShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "grass_vert.spv";
        grassPipelines_.rebuild(grassCreateInfo);
        danvulkan::vk::PipelineContextCreateInfo skyCreateInfo = pipelineCreateInfo();
        skyCreateInfo.vertexShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "sky_vert.spv";
        skyCreateInfo.fragmentShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "sky_frag.spv";
        skyCreateInfo.background = true;
        skyPipelines_.rebuild(skyCreateInfo);
        ui_.rebuild(swapchain.format(),
            std::filesystem::path(COMPILED_SHADER_PATH) / "ui_vert.spv",
            std::filesystem::path(COMPILED_SHADER_PATH) / "ui_frag.spv",
            enableValidationLayers);
    }

    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested but not avaiable!");
        }
        uint32_t loaderVersion = VK_API_VERSION_1_0;
        if (vkEnumerateInstanceVersion(&loaderVersion) != VK_SUCCESS || loaderVersion < VK_API_VERSION_1_4)
        {
            throw std::runtime_error("Vulkan 1.4 loader required; install a current Vulkan SDK and GPU driver");
        }

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = config_.applicationName.c_str();
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "DanVulkan";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_4;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        const std::span<const char* const> platformExtensions =
            platform_->requiredVulkanInstanceExtensions();
        if (platformExtensions.empty())
        {
            throw std::runtime_error(
                "platform adapter did not provide Vulkan instance extensions");
        }
        std::vector<const char*> extensions(platformExtensions.begin(), platformExtensions.end());

        std::uint32_t availableExtensionCount = 0;
        checkVk(vkEnumerateInstanceExtensionProperties(
            nullptr, &availableExtensionCount, nullptr),
            "vkEnumerateInstanceExtensionProperties(count)");
        std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
        if (availableExtensionCount > 0)
        {
            checkVk(vkEnumerateInstanceExtensionProperties(
                nullptr, &availableExtensionCount, availableExtensions.data()),
                "vkEnumerateInstanceExtensionProperties");
        }
        const bool portabilityEnumerationAvailable = std::any_of(
            availableExtensions.begin(), availableExtensions.end(),
            [](const VkExtensionProperties& extension)
            {
                return std::string_view(extension.extensionName) ==
                    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
            });
        if (portabilityEnumerationAvailable)
        {
            const bool alreadyEnabled = std::any_of(
                extensions.begin(), extensions.end(), [](const char* extension)
                {
                    return std::string_view(extension) ==
                        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
                });
            if (!alreadyEnabled)
            {
                extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            }
            createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }

        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        checkVk(vkCreateInstance(&createInfo, nullptr, instance.put()), "vkCreateInstance");
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
        void* userData)
    {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        {
            static_cast<Impl*>(userData)->validationErrorSeen_.store(true);
        }
        const char* level = severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" :
            severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "warning" : "info";
        std::cerr << "[Vulkan " << level << "] " << callbackData->pMessage << '\n';
        return VK_FALSE;
    }

    void createDebugMessenger()
    {
        if (!enableValidationLayers)
        {
            return;
        }

        auto createInfo = makeVulkanStructure<VkDebugUtilsMessengerCreateInfoEXT>(
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT);
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = this;

        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create == nullptr)
        {
            throw std::runtime_error("VK_EXT_debug_utils was enabled but vkCreateDebugUtilsMessengerEXT is unavailable");
        }
        checkVk(create(instance, &createInfo, nullptr, debugMessenger.put(instance)),
                "vkCreateDebugUtilsMessengerEXT");
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound)
                return false;
        }
        return true;
    }

    void createAllocator()
    {
        VmaAllocatorCreateInfo createInfo{};
        createInfo.physicalDevice = device.physicalDevice();
        createInfo.device = device;
        createInfo.instance = instance;
        createInfo.vulkanApiVersion = VK_API_VERSION_1_4;
        checkVk(vmaCreateAllocator(&createInfo, allocator.put()), "vmaCreateAllocator");
    }

    void createSurface()
    {
        *surface.put(instance) = platform_->createVulkanSurface(instance);
        if (!surface)
        {
            throw std::runtime_error("platform adapter returned a null Vulkan surface");
        }
    }

#pragma region stuff
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags,
        std::string_view name, std::uint32_t mipLevels = 1)
    {
        auto createInfo = makeVulkanStructure<VkImageViewCreateInfo>(
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
        createInfo.image = image;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;

        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask = aspectFlags;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = mipLevels;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        VkImageView imageView = VK_NULL_HANDLE;
        checkVk(vkCreateImageView(device, &createInfo, nullptr, &imageView), "vkCreateImageView");
        setDebugName(VK_OBJECT_TYPE_IMAGE_VIEW, imageView, name);
        return imageView;
    }

    [[nodiscard]] danvulkan::vk::PipelineContextCreateInfo pipelineCreateInfo()
    {
        danvulkan::vk::PipelineContextCreateInfo createInfo;
        createInfo.descriptorLayout = descriptors.layout();
        createInfo.colorFormat = swapchain.format();
        createInfo.depthFormat = findDepthFormat();
        createInfo.samples = msaaSamples;
        createInfo.vertexShader = std::filesystem::path(COMPILED_SHADER_PATH) / "vert.spv";
        createInfo.fragmentShader = std::filesystem::path(COMPILED_SHADER_PATH) / "frag.spv";
        createInfo.enableDebugNames = enableValidationLayers;
        return createInfo;
    }

    void createGraphicsPipeline()
    {
        const danvulkan::vk::PipelineContextCreateInfo createInfo = pipelineCreateInfo();
        if (!pipelines)
        {
            pipelines.initialize(device, createInfo);
        }
        else if (!pipelines.compatible(createInfo.descriptorLayout, createInfo.colorFormat,
                     createInfo.depthFormat, createInfo.samples))
        {
            pipelines.rebuild(createInfo);
        }
        danvulkan::vk::PipelineContextCreateInfo grassCreateInfo = createInfo;
        grassCreateInfo.vertexShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "grass_vert.spv";
        if (!grassPipelines_)
        {
            grassPipelines_.initialize(device, grassCreateInfo);
        }
        else if (!grassPipelines_.compatible(grassCreateInfo.descriptorLayout,
                     grassCreateInfo.colorFormat, grassCreateInfo.depthFormat,
                     grassCreateInfo.samples))
        {
            grassPipelines_.rebuild(grassCreateInfo);
        }

        danvulkan::vk::PipelineContextCreateInfo skyCreateInfo = createInfo;
        skyCreateInfo.vertexShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "sky_vert.spv";
        skyCreateInfo.fragmentShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "sky_frag.spv";
        skyCreateInfo.background = true;
        if (!skyPipelines_)
        {
            skyPipelines_.initialize(device, skyCreateInfo);
        }
        else if (!skyPipelines_.compatible(skyCreateInfo.descriptorLayout,
                     skyCreateInfo.colorFormat, skyCreateInfo.depthFormat,
                     skyCreateInfo.samples))
        {
            skyPipelines_.rebuild(skyCreateInfo);
        }

        danvulkan::vk::PipelineContextCreateInfo shadowCreateInfo = createInfo;
        shadowCreateInfo.colorFormat = VK_FORMAT_UNDEFINED;
        shadowCreateInfo.depthFormat = lighting_.shadowFormat();
        shadowCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        shadowCreateInfo.vertexShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "vert_shadow.spv";
        shadowCreateInfo.fragmentShader.clear();
        shadowCreateInfo.depthOnly = true;
        if (!shadowPipelines_)
        {
            shadowPipelines_.initialize(device, shadowCreateInfo);
        }
        else if (!shadowPipelines_.compatible(shadowCreateInfo.descriptorLayout,
                     shadowCreateInfo.colorFormat, shadowCreateInfo.depthFormat,
                     shadowCreateInfo.samples))
        {
            shadowPipelines_.rebuild(shadowCreateInfo);
        }
        shadowCreateInfo.vertexShader =
            std::filesystem::path(COMPILED_SHADER_PATH) / "grass_vert_shadow.spv";
        if (!grassShadowPipelines_)
        {
            grassShadowPipelines_.initialize(device, shadowCreateInfo);
        }
        else if (!grassShadowPipelines_.compatible(shadowCreateInfo.descriptorLayout,
                     shadowCreateInfo.colorFormat, shadowCreateInfo.depthFormat,
                     shadowCreateInfo.samples))
        {
            grassShadowPipelines_.rebuild(shadowCreateInfo);
        }
    }

    void createUiContext()
    {
        danvulkan::vk::UiContextCreateInfo createInfo;
        createInfo.colorFormat = swapchain.format();
        createInfo.vertexShader = std::filesystem::path(COMPILED_SHADER_PATH) / "ui_vert.spv";
        createInfo.fragmentShader = std::filesystem::path(COMPILED_SHADER_PATH) / "ui_frag.spv";
        createInfo.frameCount = frames.size();
        createInfo.enableDebugNames = enableValidationLayers;
        ui_.initialize(device, allocator, createInfo);
    }

    void createFrameContexts()
    {
        const std::uint32_t graphicsQueueFamily =
            device.queueFamilies().graphicsFamily.value();
        for (std::size_t index = 0; index < frames.size(); ++index)
        {
            frames[index].initialize(device, graphicsQueueFamily,
                static_cast<std::uint32_t>(index), enableValidationLayers);
        }
    }
    
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(device.physicalDevice(), format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
            {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
            {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat()
    {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    [[nodiscard]] danvulkan::vk::AttachmentContextCreateInfo attachmentCreateInfo()
    {
        danvulkan::vk::AttachmentContextCreateInfo createInfo;
        createInfo.extent = swapchain.extent();
        createInfo.attachmentCount = attachmentTargetCount_;
        createInfo.colorFormat = swapchain.format();
        createInfo.depthFormat = findDepthFormat();
        createInfo.samples = msaaSamples;
        createInfo.preferLazilyAllocatedMemory =
            config_.memory.preferLazilyAllocatedAttachments;
        createInfo.enableDebugNames = enableValidationLayers;
        return createInfo;
    }

    void createUploadContext()
    {
        uploadContext_.initialize(device, allocator, device.graphicsQueue(),
            device.queueFamilies().graphicsFamily.value(), enableValidationLayers);
    }

    void createSwapchainAttachments()
    {
        attachments.initialize(device, allocator, attachmentCreateInfo());
    }

    void recreateSwapchainAttachments()
    {
        attachments.recreate(attachmentCreateInfo());
    }

    [[nodiscard]] Texture createTextureImage(const danvulkan::assets::TextureAsset& source)
    {
        if (source.width == 0 || source.height == 0 ||
            source.rgba8.size() != static_cast<std::size_t>(source.width) * source.height * 4U)
        {
            throw std::runtime_error("invalid RGBA8 texture payload: " + source.name);
        }

        Texture texture;
        texture.name = source.name;
        texture.width = source.width;
        texture.height = source.height;
        texture.colorSpace = source.colorSpace;
        texture.format = source.colorSpace == danvulkan::assets::ColorSpace::srgb
            ? VK_FORMAT_R8G8B8A8_SRGB
            : VK_FORMAT_R8G8B8A8_UNORM;
        VkFormatProperties formatProperties{};
        vkGetPhysicalDeviceFormatProperties(
            device.physicalDevice(), texture.format, &formatProperties);
        constexpr VkFormatFeatureFlags mipGenerationFeatures =
            VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        texture.mipLevels =
            (formatProperties.optimalTilingFeatures & mipGenerationFeatures) ==
                mipGenerationFeatures
            ? std::bit_width(std::max(source.width, source.height))
            : 1U;
        texture.samplerConfig = source.sampler;
        const VkDeviceSize imageSize = static_cast<VkDeviceSize>(source.rgba8.size());

        texture.image = createImage(source.width, source.height, VK_SAMPLE_COUNT_1_BIT, texture.format,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, source.name, texture.mipLevels);

        uploadContext_.uploadImage(texture.image, source.width, source.height, texture.mipLevels,
            source.rgba8.data(), imageSize, source.name);

        return texture;
    }
    
    danvulkan::vk::Image createImage(uint32_t width, uint32_t height, VkSampleCountFlagBits numSamples,
        VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties, std::string_view name, std::uint32_t mipLevels = 1)
    {
        auto imageInfo = makeVulkanStructure<VkImageCreateInfo>(
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = numSamples;
        imageInfo.flags = 0; // optional

        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationInfo.requiredFlags = properties;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        checkVk(vmaCreateImage(allocator, &imageInfo, &allocationInfo, &image, &allocation, nullptr),
                "vmaCreateImage");

        danvulkan::vk::Image ownedImage(device, allocator, image, allocation);
        const std::string terminatedName(name);
        vmaSetAllocationName(allocator, allocation, terminatedName.c_str());
        setDebugName(VK_OBJECT_TYPE_IMAGE, image, name);
        return ownedImage;
    }

    void createTextureImageViews()
    {
        for (std::uint32_t index = 0; index < scene_.textures.size(); ++index)
        {
            if (scene_.textures[index])
            {
                createTextureImageView(*scene_.textures[index], index);
            }
        }
    }

    void createTextureImageView(Texture& texture, std::uint32_t textureIndex)
    {
        texture.image.setView(createImageView(texture.image, texture.format,
            VK_IMAGE_ASPECT_COLOR_BIT, "texture image view " + std::to_string(textureIndex),
            texture.mipLevels));
    }
    
    static VkFilter textureFilter(danvulkan::assets::TextureFilter filter)
    {
        return filter == danvulkan::assets::TextureFilter::nearest ? VK_FILTER_NEAREST
                                                                   : VK_FILTER_LINEAR;
    }

    static VkSamplerAddressMode textureWrap(danvulkan::assets::TextureWrap wrap)
    {
        switch (wrap)
        {
        case danvulkan::assets::TextureWrap::mirroredRepeat:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case danvulkan::assets::TextureWrap::clampToEdge:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case danvulkan::assets::TextureWrap::repeat:
        default:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }

    void createTextureSamplers()
    {
        for (std::size_t index = 0; index < scene_.textures.size(); ++index)
        {
            if (scene_.textures[index])
            {
                createTextureSampler(*scene_.textures[index], static_cast<std::uint32_t>(index));
            }
        }
    }

    void createTextureSampler(Texture& texture, std::uint32_t textureIndex)
    {
        const VkPhysicalDeviceProperties& properties = device.properties();
        auto samplerInfo = makeVulkanStructure<VkSamplerCreateInfo>(
            VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
        samplerInfo.magFilter = textureFilter(texture.samplerConfig.magFilter);
        samplerInfo.minFilter = textureFilter(texture.samplerConfig.minFilter);
        samplerInfo.addressModeU = textureWrap(texture.samplerConfig.wrapU);
        samplerInfo.addressModeV = textureWrap(texture.samplerConfig.wrapV);
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = samplerInfo.minFilter == VK_FILTER_LINEAR &&
            samplerInfo.magFilter == VK_FILTER_LINEAR ? VK_TRUE : VK_FALSE;
        samplerInfo.maxAnisotropy = std::min(16.0f, properties.limits.maxSamplerAnisotropy);
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.mipmapMode = texture.samplerConfig.mipmapMode ==
                danvulkan::assets::TextureMipmapMode::nearest
            ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = device.supportsSamplerMipLodBias()
            ? std::clamp(config_.textureMipLodBias,
                -properties.limits.maxSamplerLodBias, properties.limits.maxSamplerLodBias)
            : 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(texture.mipLevels - 1);

        checkVk(vkCreateSampler(device, &samplerInfo, nullptr, &texture.sampler),
            "vkCreateSampler(material texture)");
        setDebugName(VK_OBJECT_TYPE_SAMPLER, texture.sampler,
            "material texture sampler " + std::to_string(textureIndex));
    }

    static void transitionImage(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageAspectFlags aspectMask,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkPipelineStageFlags2 sourceStage,
        VkAccessFlags2 sourceAccess,
        VkPipelineStageFlags2 destinationStage,
        VkAccessFlags2 destinationAccess)
    {
        auto barrier = makeVulkanStructure<VkImageMemoryBarrier2>(
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2);
        barrier.srcStageMask = sourceStage;
        barrier.srcAccessMask = sourceAccess;
        barrier.dstStageMask = destinationStage;
        barrier.dstAccessMask = destinationAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspectMask;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        auto dependencyInfo = makeVulkanStructure<VkDependencyInfo>(
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    void updateCommandBuffer(uint32_t currentFrameIndex, uint32_t imageIndex)
    {
        danvulkan::vk::FrameContext& frame = frames[currentFrameIndex];
        VkCommandBuffer commandBuffer = frame.beginCommands();

        const bool shadowWasInitialized = lighting_.shadowInitialized(imageIndex);
        transitionImage(commandBuffer, lighting_.shadowImage(imageIndex),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            shadowWasInitialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                 : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            shadowWasInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                 : VK_PIPELINE_STAGE_2_NONE,
            shadowWasInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        auto shadowAttachment = makeVulkanStructure<VkRenderingAttachmentInfo>(
            VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
        shadowAttachment.imageView = lighting_.shadowView(imageIndex);
        shadowAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        shadowAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        shadowAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        shadowAttachment.clearValue.depthStencil = {1.0f, 0};
        auto shadowRendering = makeVulkanStructure<VkRenderingInfo>(
            VK_STRUCTURE_TYPE_RENDERING_INFO);
        shadowRendering.renderArea = {{0, 0}, lighting_.shadowExtent()};
        shadowRendering.layerCount = 1;
        shadowRendering.pDepthAttachment = &shadowAttachment;
        vkCmdBeginRendering(commandBuffer, &shadowRendering);

        VkViewport shadowViewport{};
        shadowViewport.width = static_cast<float>(lighting_.shadowExtent().width);
        shadowViewport.height = static_cast<float>(lighting_.shadowExtent().height);
        shadowViewport.minDepth = 0.0f;
        shadowViewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
        const VkRect2D shadowScissor{{0, 0}, lighting_.shadowExtent()};
        vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);
        vkCmdBindIndexBuffer(commandBuffer, scene_.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        const VkDescriptorSet descriptorSet = descriptors.set(imageIndex);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            shadowPipelines_.layout(), 0, 1, &descriptorSet, 0, nullptr);
        for (std::size_t variant = 0; variant < PipelineVariantCount; ++variant)
        {
            const DrawBatch& batch = scene_.drawBatches[variant];
            if (batch.commandCount == 0)
            {
                continue;
            }
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                shadowPipelines_.pipeline(variant));
            const VkDeviceSize offset = static_cast<VkDeviceSize>(batch.firstCommand) *
                sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(commandBuffer,
                scene_.indirectCommandsBuffer[imageIndex], offset,
                batch.commandCount, sizeof(VkDrawIndexedIndirectCommand));
        }
        if (scene_.grassDrawBatch.commandCount != 0U)
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                grassShadowPipelines_.pipeline(static_cast<std::size_t>(
                    PipelineVariant::opaqueDoubleSided)));
            const VkDeviceSize offset =
                static_cast<VkDeviceSize>(scene_.grassDrawBatch.firstCommand) *
                sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(commandBuffer,
                scene_.indirectCommandsBuffer[imageIndex], offset,
                scene_.grassDrawBatch.commandCount,
                sizeof(VkDrawIndexedIndirectCommand));
        }
        vkCmdEndRendering(commandBuffer);
        transitionImage(commandBuffer, lighting_.shadowImage(imageIndex),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        lighting_.markShadowInitialized(imageIndex);

        transitionImage(commandBuffer, swapchain.images()[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        if (msaaSamples != VK_SAMPLE_COUNT_1_BIT)
        {
            transitionImage(commandBuffer, attachments.color(currentFrameIndex),
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        }

        const VkFormat depthFormat = attachments.depthFormat();
        VkImageAspectFlags depthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (danvulkan::vk::depthFormatHasStencil(depthFormat))
        {
            depthAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        transitionImage(commandBuffer, attachments.depth(currentFrameIndex), depthAspect,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        auto colorAttachment = makeVulkanStructure<VkRenderingAttachmentInfo>(
            VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
        colorAttachment.imageView = msaaSamples == VK_SAMPLE_COUNT_1_BIT
            ? swapchain.imageViews()[imageIndex]
            : attachments.colorView(currentFrameIndex);
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = msaaSamples == VK_SAMPLE_COUNT_1_BIT
            ? VK_ATTACHMENT_STORE_OP_STORE
            : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.clearValue.color = { { 0.012f, 0.022f, 0.038f, 1.0f } };
        if (msaaSamples != VK_SAMPLE_COUNT_1_BIT)
        {
            colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            colorAttachment.resolveImageView = swapchain.imageViews()[imageIndex];
            colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        auto depthAttachment = makeVulkanStructure<VkRenderingAttachmentInfo>(
            VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
        depthAttachment.imageView = attachments.depthView(currentFrameIndex);
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        auto renderingInfo = makeVulkanStructure<VkRenderingInfo>(
            VK_STRUCTURE_TYPE_RENDERING_INFO);
        renderingInfo.renderArea = { { 0, 0 }, swapchain.extent() };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;
        if (danvulkan::vk::depthFormatHasStencil(depthFormat))
        {
            renderingInfo.pStencilAttachment = &depthAttachment;
        }

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchain.extent().width);
        viewport.height = static_cast<float>(swapchain.extent().height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{ { 0, 0 }, swapchain.extent() };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            skyPipelines_.pipeline(static_cast<std::size_t>(
                PipelineVariant::opaqueDoubleSided)));
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            skyPipelines_.layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdBindIndexBuffer(commandBuffer, scene_.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelines.layout(), 0, 1, &descriptorSet, 0, nullptr);

        for (std::size_t variant = 0; variant < PipelineVariantCount; ++variant)
        {
            const DrawBatch& batch = scene_.drawBatches[variant];
            if (batch.commandCount == 0)
            {
                continue;
            }
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelines.pipeline(variant));
            const VkDeviceSize offset = static_cast<VkDeviceSize>(batch.firstCommand) *
                sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(commandBuffer, scene_.indirectCommandsBuffer[imageIndex], offset,
                batch.commandCount, sizeof(VkDrawIndexedIndirectCommand));
        }
        if (scene_.grassDrawBatch.commandCount != 0U)
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                grassPipelines_.pipeline(static_cast<std::size_t>(
                    PipelineVariant::opaqueDoubleSided)));
            const VkDeviceSize offset =
                static_cast<VkDeviceSize>(scene_.grassDrawBatch.firstCommand) *
                sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(commandBuffer,
                scene_.indirectCommandsBuffer[imageIndex], offset,
                scene_.grassDrawBatch.commandCount,
                sizeof(VkDrawIndexedIndirectCommand));
        }
        vkCmdEndRendering(commandBuffer);

        if (hasPendingUi_ && !pendingUi_.vertices.empty() && !pendingUi_.commands.empty())
        {
            ui_.prepare(currentFrameIndex, pendingUi_.vertices);

            auto memoryBarrier = makeVulkanStructure<VkMemoryBarrier2>(
                VK_STRUCTURE_TYPE_MEMORY_BARRIER_2);
            memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            memoryBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            memoryBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            auto dependency = makeVulkanStructure<VkDependencyInfo>(
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &memoryBarrier;
            vkCmdPipelineBarrier2(commandBuffer, &dependency);

            auto uiAttachment = makeVulkanStructure<VkRenderingAttachmentInfo>(
                VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO);
            uiAttachment.imageView = swapchain.imageViews()[imageIndex];
            uiAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            uiAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            uiAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            auto uiRendering = makeVulkanStructure<VkRenderingInfo>(
                VK_STRUCTURE_TYPE_RENDERING_INFO);
            uiRendering.renderArea = {{0, 0}, swapchain.extent()};
            uiRendering.layerCount = 1;
            uiRendering.colorAttachmentCount = 1;
            uiRendering.pColorAttachments = &uiAttachment;
            vkCmdBeginRendering(commandBuffer, &uiRendering);
            ui_.record(commandBuffer, currentFrameIndex, pendingUi_, swapchain.extent());
            vkCmdEndRendering(commandBuffer);
        }

        transitionImage(commandBuffer, swapchain.images()[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
        frame.endCommands();
    }

    void drawFrame(const SceneSubmission& submission)
    {
        danvulkan::vk::FrameContext& frame = frames[currentFrame];
        if (const std::optional<double> gpuMilliseconds = frame.waitForReuse(timestampPeriod))
        {
            frameGpuAvg = frameGpuAvg * 0.95 + *gpuMilliseconds * 0.05;
        }
        const bool framebufferResized = platform_->consumeFramebufferResize();

        uint32_t imageIndex;


        VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
            frame.imageAvailable(),
            VK_NULL_HANDLE, &imageIndex);
        // swapchain is incompoatible with surface and can't be used for rendering, usually after window resize
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        presentation.waitForImage(imageIndex);
        scene_.prepareGeometryForImage(imageIndex, descriptors);
        scene_.prepareGeometryRangesForImage(imageIndex);
        scene_.prepareTexturesForImage(imageIndex, descriptors, device);
        presentation.markImageInFlight(imageIndex, frame.inFlight());

        updateUniformBuffer(imageIndex, submission);
        const auto commandRecordingBegin = std::chrono::steady_clock::now();
        updateCommandBuffer(static_cast<uint32_t>(currentFrame), imageIndex);
        commandRecordingCpuAvg_ = rollingAverage(commandRecordingCpuAvg_,
            millisecondsSince(commandRecordingBegin));

        auto waitInfo = makeVulkanStructure<VkSemaphoreSubmitInfo>(
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO);
        waitInfo.semaphore = frame.imageAvailable();
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        auto commandBufferInfo = makeVulkanStructure<VkCommandBufferSubmitInfo>(
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO);
        commandBufferInfo.commandBuffer = frame.commandBuffer();

        auto signalInfo = makeVulkanStructure<VkSemaphoreSubmitInfo>(
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO);
        const VkSemaphore renderFinished = presentation.renderFinished(imageIndex);
        signalInfo.semaphore = renderFinished;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

        auto submitInfo = makeVulkanStructure<VkSubmitInfo2>(
            VK_STRUCTURE_TYPE_SUBMIT_INFO_2);
        submitInfo.waitSemaphoreInfoCount = 1;
        submitInfo.pWaitSemaphoreInfos = &waitInfo;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &signalInfo;
       
        frame.resetFenceForSubmit();
        checkVk(vkQueueSubmit2(device.graphicsQueue(), 1, &submitInfo, frame.inFlight()),
                "vkQueueSubmit2(frame)");
        frame.markSubmitted();

        auto presentInfo = makeVulkanStructure<VkPresentInfoKHR>(
            VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinished;

        VkSwapchainKHR swapChains[] = { swapchain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // optional

        result = vkQueuePresentKHR(device.presentQueue(), &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
            framebufferResized)
        {
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image");
        }
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    
    void update()
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        const float deltaSeconds = std::min(
            std::chrono::duration<float>(currentTime - previousTime).count(), 0.1f);
   
        /*
        ImGuiIO& io = ImGui::GetIO();



        // only Process keyboard stuff if imgui doesn't want it
        if (!io.WantCaptureKeyboard)
        {*/
            if (auto* glfwPlatform = dynamic_cast<GlfwRendererPlatform*>(platform_.get()))
            {
                const DemoInputState input = glfwPlatform->consumeDemoInput();
                camera.keys.down = input.moveDown;
                camera.keys.up = input.moveUp;
                camera.keys.left = input.moveLeft;
                camera.keys.right = input.moveRight;
                camera.rotate(glm::vec3(input.cursorDeltaY * 0.05f,
                    input.cursorDeltaX * 0.05f, 0.0f));
                camera.rotate(glm::vec3(0.0f, input.scrollDeltaY, 0.0f));
            }
            camera.update(deltaSeconds);
            /*
        }

        // Same with mouse capture
        if (!io.WantCaptureMouse)
        {*/
      //  }

        
        previousTime = currentTime;
        if (lightPos.x > 1100.0f || lightPos.x < -1900.0f)
        {
            lightSpeed.x *= -1.0f;
        }
        lightPos += lightSpeed;
        //checkFilesChanged();
    }

    void updateAnimation(float deltaSeconds, const SceneSubmission& submission)
    {
        if (!scene_.animationPlayer_)
        {
            lastAnimationActorCount_ = 0;
            lastEvaluatedAnimationActorCount_ = 0;
            lastCulledAnimationActorCount_ = 0;
            lastSampledAnimationChannelCount_ = 0;
            lastSampledAnimationVectorChannelCount_ = 0;
            lastSampledAnimationRotationChannelCount_ = 0;
            lastNlerpAnimationRotationChannelCount_ = 0;
            lastAnimationClipChannelCount_ = 0;
            lastFoldedConstantAnimationChannelCount_ = 0;
            lastAnimationPropagatedNodeCount_ = 0;
            lastAnimationPoseComposedNodeCount_ = 0;
            lastAnimationCachedLocalNodeCount_ = 0;
            return;
        }
        glm::mat4 projection = submission.projection;
        projection[1][1] *= -1.0f;
        const danvulkan::vk::AnimationUpdateCounts updateCounts =
            scene_.prepareAnimationUpdates(submission.view, projection,
                submission.cameraPosition, {
                    config_.animation.cullOffscreenActors,
                    config_.animation.fullRateDistance,
                    config_.animation.reducedRateDistance,
                    config_.animation.mediumUpdatesPerSecond,
                    config_.animation.farUpdatesPerSecond
                });
        lastAnimationActorCount_ = updateCounts.actors;
        lastCulledAnimationActorCount_ = updateCounts.culled;
        const auto evaluationBegin = std::chrono::steady_clock::now();
        scene_.animationPlayer_->update(deltaSeconds, scene_.animationUpdatePolicies());
        animationEvaluationCpuAvg_ = rollingAverage(animationEvaluationCpuAvg_,
            millisecondsSince(evaluationBegin));
        const danvulkan::AnimationPlayer::EvaluationTimings evaluation =
            scene_.animationPlayer_->evaluationTimings();
        lastEvaluatedAnimationActorCount_ = evaluation.evaluatedInstances;
        animationSamplingCpuAvg_ = rollingAverage(
            animationSamplingCpuAvg_, evaluation.samplingMilliseconds);
        animationPoseResetCpuAvg_ = rollingAverage(
            animationPoseResetCpuAvg_, evaluation.poseResetMilliseconds);
        animationTimelineResolutionCpuAvg_ = rollingAverage(
            animationTimelineResolutionCpuAvg_, evaluation.timelineResolutionMilliseconds);
        animationVectorSamplingCpuAvg_ = rollingAverage(
            animationVectorSamplingCpuAvg_, evaluation.vectorSamplingMilliseconds);
        animationRotationSamplingCpuAvg_ = rollingAverage(
            animationRotationSamplingCpuAvg_, evaluation.rotationSamplingMilliseconds);
        animationTransformPropagationCpuAvg_ = rollingAverage(
            animationTransformPropagationCpuAvg_, evaluation.transformPropagationMilliseconds);
        lastSampledAnimationChannelCount_ = evaluation.sampledChannels;
        lastSampledAnimationVectorChannelCount_ = evaluation.sampledVectorChannels;
        lastSampledAnimationRotationChannelCount_ = evaluation.sampledRotationChannels;
        lastNlerpAnimationRotationChannelCount_ = evaluation.nlerpRotationChannels;
        lastAnimationPropagatedNodeCount_ = evaluation.propagatedNodes;
        lastAnimationPoseComposedNodeCount_ = evaluation.poseComposedNodes;
        lastAnimationCachedLocalNodeCount_ = evaluation.cachedLocalNodes;
        const std::size_t selectedClip = scene_.animationPlayer_->currentClip();
        if (selectedClip != danvulkan::AnimationPlayer::invalidClip)
        {
            lastAnimationClipChannelCount_ = static_cast<std::uint32_t>(
                scene_.animationPlayer_->compiledChannelCount(selectedClip));
            lastFoldedConstantAnimationChannelCount_ = static_cast<std::uint32_t>(
                scene_.animationPlayer_->foldedConstantChannelCount(selectedClip));
        }
        else
        {
            lastAnimationClipChannelCount_ = 0;
            lastFoldedConstantAnimationChannelCount_ = 0;
        }
        const auto synchronizationBegin = std::chrono::steady_clock::now();
        const danvulkan::vk::AnimationSynchronizationTimings synchronization =
            scene_.synchronizeAnimationPose(true);
        animationSynchronizationCpuAvg_ = rollingAverage(animationSynchronizationCpuAvg_,
            millisecondsSince(synchronizationBegin));
        animationTransformUpdateCpuAvg_ = rollingAverage(
            animationTransformUpdateCpuAvg_, synchronization.transformUpdateMilliseconds);
        animationBoundsCpuAvg_ = rollingAverage(
            animationBoundsCpuAvg_, synchronization.boundsMilliseconds);
        animationPaletteGenerationCpuAvg_ = rollingAverage(animationPaletteGenerationCpuAvg_,
            synchronization.paletteGenerationMilliseconds);
    }

    void applyAnimationActorTransforms(const SceneSubmission& submission)
    {
        if (submission.animationActorTransforms.empty())
        {
            return;
        }
        submittedAnimationActorScratch_.assign(scene_.animationActors_.size(), 0U);
        for (const SceneAnimationActorTransform& submitted :
             submission.animationActorTransforms)
        {
            if (submitted.actor.generation != scene_.sceneGeneration_ ||
                submitted.actor.slot >= scene_.animationActors_.size())
            {
                throw std::invalid_argument(
                    "scene submission contains an invalid or stale animation actor handle");
            }
            if (submittedAnimationActorScratch_[submitted.actor.slot] != 0U)
            {
                throw std::invalid_argument(
                    "scene submission contains duplicate animation actor transforms");
            }
            submittedAnimationActorScratch_[submitted.actor.slot] = 1U;
            bool finite = true;
            for (glm::length_t column = 0; column < 4 && finite; ++column)
            {
                for (glm::length_t row = 0; row < 4; ++row)
                {
                    finite = finite && std::isfinite(submitted.worldOffset[column][row]);
                }
            }
            const float determinant = glm::determinant(submitted.worldOffset);
            if (!finite || !std::isfinite(determinant) || std::abs(determinant) < 0.000001f)
            {
                throw std::invalid_argument(
                    "scene submission contains a non-invertible animation actor transform");
            }
            scene_.setAnimationActorTransform(submitted.actor.slot, submitted.worldOffset);
        }
    }

    void rebuildSwapchainIndexedResources()
    {
        descriptors.resetSets();
        uniformBuffers.clear();
        scene_.matBuffers.clear();
        scene_.transformBuffers.clear();
        scene_.drawBuffers.clear();
        scene_.jointBuffers.clear();
        scene_.indirectCommandsBuffer.clear();

        // Recreation waits for every frame plus the presentation queue, so all per-image
        // generations are complete without idling unrelated device queues.
        scene_.releaseSwapchainRetirements(device);

        createUniformBuffers();
        createBindlessBuffers();
        lighting_.recreateBuffers(swapchain.imageCount());
        createDescriptorSets();
    }
#pragma endregion
    void updateUniformBuffer(uint32_t currentImage, const SceneSubmission& submission)
    {
        /*
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        */
        /*
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::6vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f),
            swapchain.extent().width / static_cast<float>(swapchain.extent().height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        */
        UniformBufferObject ubo = {};
        ubo.view = submission.view;
        ubo.projection = submission.projection;
        ubo.projection[1][1] *= -1.0f;
        ubo.inverseViewProjection = glm::inverse(ubo.projection * ubo.view);
        const float elapsedSeconds = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - applicationStart_).count();
        ubo.cameraPositionTime = glm::vec4(submission.cameraPosition, elapsedSeconds);
        ubo.vegetationInteractorPositionRadius =
            submission.vegetationInteractorPositionRadius;
        if (!danvulkan::planLighting(submission.pointLights, submission.directionalLights,
            submission.environment, submission.atmosphere,
            lighting_.pointLightCapacity(), lightingPlan_))
        {
            throw std::invalid_argument("scene submission contains invalid lighting");
        }
        ubo.lightingCounts.x = static_cast<std::uint32_t>(lightingPlan_.pointLights.size());
        ubo.lightingCounts.y = lightingPlan_.directionalLightCount;
        ubo.lightingCounts.z = lightingPlan_.shadowLightIndex < MaxSceneDirectionalLights
            ? lightingPlan_.shadowLightIndex + 1U : 0U;
        lastPointLightCount_ = ubo.lightingCounts.x;
        ubo.directionalLights = lightingPlan_.directionalLights;
        ubo.directionalShadowViewProjection = glm::mat4(1.0f);
        if (lightingPlan_.shadowLightIndex < submission.directionalLights.size())
        {
            const glm::vec3 shadowFocus =
                submission.vegetationInteractorPositionRadius.w > 0.0f
                ? glm::vec3(submission.vegetationInteractorPositionRadius)
                : submission.cameraPosition;
            ubo.directionalShadowViewProjection =
                danvulkan::directionalShadowViewProjection(
                    submission.directionalLights[lightingPlan_.shadowLightIndex],
                    shadowFocus, config_.directionalShadowResolution);
        }
        ubo.environmentTintIntensity = lightingPlan_.environmentTintIntensity;
        ubo.environmentControls = lightingPlan_.environmentControls;
        ubo.atmosphereSkyZenithIntensity = lightingPlan_.atmosphereSkyZenithIntensity;
        ubo.atmosphereSkyHorizonExponent = lightingPlan_.atmosphereSkyHorizonExponent;
        ubo.atmosphereFogColorDensity = lightingPlan_.atmosphereFogColorDensity;
        ubo.atmosphereFogParameters = lightingPlan_.atmosphereFogParameters;
        ubo.atmosphereScatteringParameters = lightingPlan_.atmosphereScatteringParameters;
        ubo.atmosphereCloudShapeParameters = lightingPlan_.atmosphereCloudShapeParameters;
        ubo.atmosphereCloudMovementParameters = lightingPlan_.atmosphereCloudMovementParameters;
        ubo.atmosphereCloudLightingParameters = lightingPlan_.atmosphereCloudLightingParameters;
        
        const auto cullingBegin = std::chrono::steady_clock::now();
        const danvulkan::vk::SceneDrawCounts drawCounts = scene_.prepareDraws(
            submission.view, ubo.projection, submission.cameraPosition);
        lastActiveDrawCount_ = drawCounts.active;
        lastVisibleDrawCount_ = drawCounts.visible;
        lastAnimatedDrawCount_ = drawCounts.animated;
        lastJointMatrixCount_ = drawCounts.joints;
        cullingCpuAvg_ = rollingAverage(cullingCpuAvg_, millisecondsSince(cullingBegin));

        // Update to GPU
        // NOTE: this is a hot area for code performance
        const auto bufferWriteBegin = std::chrono::steady_clock::now();
        lighting_.writePointLights(currentImage, lightingPlan_.pointLights);
        // UBO
        writeBuffer(uniformBuffers[currentImage], &ubo, sizeof(ubo));
        // Transform
        writeBuffer(scene_.transformBuffers[currentImage], scene_.transformData.data(), sizeof(TransformData) * scene_.transformData.size());
        // Material Data
        writeBuffer(scene_.matBuffers[currentImage], scene_.matData.data(), sizeof(MaterialData) * scene_.matData.size());
        // Draw Data
        writeBuffer(scene_.drawBuffers[currentImage], scene_.drawData.data(), sizeof(DrawData) * scene_.drawData.size());
        const auto paletteUploadBegin = std::chrono::steady_clock::now();
        if (!scene_.jointMatrices_.empty())
        {
            writeBuffer(scene_.jointBuffers[currentImage], scene_.jointMatrices_.data(),
                sizeof(glm::mat4) * scene_.jointMatrices_.size());
        }
        animationPaletteUploadCpuAvg_ = rollingAverage(animationPaletteUploadCpuAvg_,
            millisecondsSince(paletteUploadBegin));
        // indirect
        writeBuffer(scene_.indirectCommandsBuffer[currentImage], scene_.indirectCommands.data(),
            sizeof(VkDrawIndexedIndirectCommand) * scene_.indirectCommands.size());
        bufferWriteCpuAvg_ = rollingAverage(bufferWriteCpuAvg_,
            millisecondsSince(bufferWriteBegin));
    }

    template <typename Handle>
    static std::uint64_t debugHandleValue(Handle handle) noexcept
    {
        if constexpr (std::is_pointer_v<Handle>)
        {
            return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
        }
        else
        {
            return static_cast<std::uint64_t>(handle);
        }
    }

    template <typename Handle>
    void setDebugName(VkObjectType objectType, Handle handle, std::string_view name) const
    {
        if (!enableValidationLayers || handle == VK_NULL_HANDLE || !device)
        {
            return;
        }

        const auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
        if (setName == nullptr)
        {
            return;
        }

        const std::string terminatedName(name);
        auto nameInfo = makeVulkanStructure<VkDebugUtilsObjectNameInfoEXT>(
            VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT);
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = debugHandleValue(handle);
        nameInfo.pObjectName = terminatedName.c_str();
        checkVk(setName(device, &nameInfo), "vkSetDebugUtilsObjectNameEXT");
    }

    danvulkan::vk::Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, bool persistentlyMapped, std::string_view name)
    {
        auto bufferInfo = makeVulkanStructure<VkBufferCreateInfo>(
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocationInfo{};
        if (persistentlyMapped)
        {
            allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            allocationInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            allocationInfo.preferredFlags = properties;
        }
        else
        {
            allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocationInfo.requiredFlags = properties;
        }

        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo resultInfo{};
        checkVk(vmaCreateBuffer(allocator, &bufferInfo, &allocationInfo, &buffer, &allocation,
                    &resultInfo),
                "vmaCreateBuffer");

        danvulkan::vk::Buffer ownedBuffer(allocator, buffer, allocation, resultInfo.pMappedData, size);
        const std::string terminatedName(name);
        vmaSetAllocationName(allocator, allocation, terminatedName.c_str());
        setDebugName(VK_OBJECT_TYPE_BUFFER, buffer, name);
        return ownedBuffer;
    }

    void writeBuffer(danvulkan::vk::Buffer& buffer, const void* data, VkDeviceSize size)
    {
        if (size == 0)
        {
            return;
        }
        if (buffer.mapped() == nullptr || size > buffer.size())
        {
            throw std::runtime_error("invalid write to mapped Vulkan buffer");
        }
        memcpy(buffer.mapped(), data, static_cast<size_t>(size));
        checkVk(buffer.flush(0, size), "vmaFlushAllocation");
    }

    danvulkan::vk::Buffer createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        const void* bufferData, VkDeviceSize uploadSize, std::string_view name)
    {
        if (bufferData == nullptr || uploadSize == 0 || uploadSize > size)
        {
            throw std::invalid_argument("device-local buffer requires a valid initial payload");
        }
        auto destination = createBuffer(size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, name);
        uploadContext_.uploadBuffer(destination, 0, bufferData, uploadSize, usage, name);
        return destination;
    }

    danvulkan::vk::Buffer createEmptyDeviceLocalBuffer(
        VkDeviceSize size, VkBufferUsageFlags usage, std::string_view name)
    {
        return createBuffer(size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, name);
    }

    void uploadDeviceLocalBufferRange(danvulkan::vk::Buffer& destination,
        VkDeviceSize destinationOffset, VkDeviceSize size, VkBufferUsageFlags usage,
        const void* bufferData, std::string_view name)
    {
        if (size == 0 || destinationOffset > destination.size() ||
            size > destination.size() - destinationOffset)
        {
            throw std::runtime_error("invalid device-local geometry buffer upload range");
        }
        uploadContext_.uploadBuffer(destination, destinationOffset, bufferData, size, usage, name);
    }

    void ensureGeometryCapacity(std::uint32_t vertexCount, std::uint32_t indexCount)
    {
        const bool growVertices = !scene_.hasGeometryRange(scene_.freeVertexRanges_, vertexCount);
        const bool growIndices = !scene_.hasGeometryRange(scene_.freeIndexRanges_, indexCount);
        if (!growVertices && !growIndices)
        {
            return;
        }
        if (scene_.geometryVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("geometry buffer generation counter exhausted");
        }

        const std::uint32_t newVertexCapacity = growVertices
            ? scene_.grownGeometryCapacity(scene_.vertexCapacity_, vertexCount,
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
            : scene_.vertexCapacity_;
        const std::uint32_t newIndexCapacity = growIndices
            ? scene_.grownGeometryCapacity(scene_.indexCapacity_, indexCount,
                std::numeric_limits<std::uint32_t>::max())
            : scene_.indexCapacity_;

        const VkPhysicalDeviceProperties& physicalDeviceProperties = device.properties();
        if (sizeof(Vertex) * static_cast<VkDeviceSize>(newVertexCapacity) >
            physicalDeviceProperties.limits.maxStorageBufferRange)
        {
            throw std::runtime_error("grown vertex capacity exceeds maxStorageBufferRange");
        }

        auto newVertexBuffer = createEmptyDeviceLocalBuffer(
            sizeof(Vertex) * static_cast<VkDeviceSize>(newVertexCapacity),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            "grown scene vertex storage buffer");
        auto newIndexBuffer = createEmptyDeviceLocalBuffer(
            sizeof(std::uint32_t) * static_cast<VkDeviceSize>(newIndexCapacity),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            "grown scene index buffer");
        uploadContext_.copyBuffer(scene_.vertexBuffer, newVertexBuffer, scene_.vertexBuffer.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "preserve scene vertex storage");
        uploadContext_.copyBuffer(scene_.indexBuffer, newIndexBuffer, scene_.indexBuffer.size(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "preserve scene index storage");

        scene_.retiredGeometry_.emplace_back();
        RetiredGeometry& retired = scene_.retiredGeometry_.back();
        retired.version = scene_.geometryVersion_;
        retired.vertexBuffer = std::move(scene_.vertexBuffer);
        retired.indexBuffer = std::move(scene_.indexBuffer);
        scene_.vertexBuffer = std::move(newVertexBuffer);
        scene_.indexBuffer = std::move(newIndexBuffer);
        ++scene_.geometryVersion_;

        if (newVertexCapacity > scene_.vertexCapacity_)
        {
            scene_.releaseGeometryRange(scene_.freeVertexRanges_,
                { scene_.vertexCapacity_, newVertexCapacity - scene_.vertexCapacity_ });
        }
        if (newIndexCapacity > scene_.indexCapacity_)
        {
            scene_.releaseGeometryRange(scene_.freeIndexRanges_,
                { scene_.indexCapacity_, newIndexCapacity - scene_.indexCapacity_ });
        }
        scene_.vertexCapacity_ = newVertexCapacity;
        scene_.indexCapacity_ = newIndexCapacity;
        captureMemoryPeak();
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(swapchain.imageCount());

        for (size_t i = 0; i < swapchain.imageCount(); i++)
        {
            uniformBuffers[i] = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "uniform buffer " + std::to_string(i));
        }
    }

    void createBindlessBuffers()
    {
        scene_.matBuffers.resize(swapchain.imageCount());

        scene_.transformBuffers.resize(swapchain.imageCount());

        scene_.drawBuffers.resize(swapchain.imageCount());

        scene_.jointBuffers.resize(swapchain.imageCount());


        scene_.indirectCommandsBuffer.resize(swapchain.imageCount());

        VkDeviceSize matBufferSize = sizeof(MaterialData) * scene_.materialCapacity_;
        VkDeviceSize transformBufferSize = sizeof(TransformData) * TransformDataCount;
        VkDeviceSize drawBufferSize = sizeof(DrawData) * DrawDataCount;
        VkDeviceSize jointBufferSize = sizeof(glm::mat4) * JointMatrixCount;
        VkDeviceSize indirectBufferSize = sizeof(VkDrawIndexedIndirectCommand) * DrawDataCount;

        for (size_t i = 0; i < swapchain.imageCount(); i++)
        {
            // Material Data
            scene_.matBuffers[i] = createBuffer(matBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "material buffer " + std::to_string(i));

            // Transform
            scene_.transformBuffers[i] = createBuffer(transformBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "transform buffer " + std::to_string(i));

            // DrawData
            scene_.drawBuffers[i] = createBuffer(drawBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "draw buffer " + std::to_string(i));

            scene_.jointBuffers[i] = createBuffer(jointBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "joint matrix buffer " + std::to_string(i));

            // Indirect Draw
            scene_.indirectCommandsBuffer[i] = createBuffer(indirectBufferSize, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "indirect draw buffer " + std::to_string(i));

        }

    }

    void updateIndirectBuffer()
    {
       // VkDeviceSize bufferSize = sizeof(VkDrawIndexedIndirectCommand) * scene_.indirectCommands.size();
        //createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, scene_.indirectCommandsBuffer[0], indirectCommandsBufferMemory[0], scene_.indirectCommands.data());
    }

    void createDescriptorSetLayout()
    {
        const VkPhysicalDeviceProperties& properties = device.properties();
        const std::optional<std::uint32_t> capacity =
            danvulkan::vk::selectTextureDescriptorCapacity({
                config_.maxTextures,
                scene_.liveTextureCount_,
                properties.limits.maxPerStageDescriptorSamplers,
                properties.limits.maxDescriptorSetSamplers
            });
        if (!capacity)
        {
            throw std::runtime_error(
                "configured bindless texture capacity is invalid for this scene or device");
        }
        scene_.textureDescriptorCapacity_ = *capacity;
        scene_.textures.reserve(scene_.textureDescriptorCapacity_);
        scene_.textureGenerations_.reserve(scene_.textureDescriptorCapacity_);
        scene_.freeTextureSlots_.reserve(scene_.textureDescriptorCapacity_);
        descriptors.initialize(device, scene_.textureDescriptorCapacity_, enableValidationLayers);
    }

    void createDescriptorSets()
    {
        const std::size_t setCount = swapchain.imageCount();
        if (uniformBuffers.size() != setCount || scene_.matBuffers.size() != setCount ||
            scene_.drawBuffers.size() != setCount || scene_.transformBuffers.size() != setCount ||
            scene_.jointBuffers.size() != setCount)
        {
            throw std::runtime_error(
                "swapchain descriptor buffer counts are inconsistent");
        }
        std::vector<danvulkan::vk::DescriptorSetBindings> bindings(setCount);
        for (std::size_t index = 0; index < setCount; ++index)
        {
            bindings[index] = {
                { uniformBuffers[index], 0, sizeof(UniformBufferObject) },
                { scene_.matBuffers[index], 0, sizeof(MaterialData) * scene_.materialCapacity_ },
                { scene_.drawBuffers[index], 0, sizeof(DrawData) * DrawDataCount },
                { scene_.transformBuffers[index], 0,
                    sizeof(TransformData) * TransformDataCount },
                { scene_.vertexBuffer, 0, scene_.vertexBuffer.size() },
                { scene_.jointBuffers[index], 0, sizeof(glm::mat4) * JointMatrixCount },
                lighting_.lightBuffer(index)
            };
        }
        const std::vector<VkDescriptorImageInfo> textureInfos = scene_.textureDescriptorInfos();
        std::vector<std::array<VkDescriptorImageInfo, 4>> lightingInfos;
        lightingInfos.reserve(setCount);
        for (std::size_t index = 0; index < setCount; ++index)
        {
            lightingInfos.push_back(lighting_.lightingDescriptors(index));
        }
        descriptors.allocateSets(bindings, textureInfos, lightingInfos);
        scene_.initializeImageGenerations(descriptors.setCount());
    }

    void loadModel()
    {
        danvulkan::assets::SceneAsset scene;
        if (!MODEL_PATH.empty())
        {
            scene = danvulkan::assets::loadScene(MODEL_PATH);
        }
        for (const AdditionalSceneConfig& additional : config_.additionalScenes)
        {
            static_cast<void>(scene.append(
                danvulkan::assets::loadScene(additional.modelPath),
                additional.rootTransform));
        }
        uploadSceneAsset(scene);
    }

    [[nodiscard]] PreparedSceneData prepareSceneData(
        const danvulkan::assets::SceneAsset& scene) const
    {
        danvulkan::ScenePlan plan = danvulkan::planScene(scene, {
            .textureCapacity = scene_.textureDescriptorCapacity_,
            .materialCapacity = scene_.materialCapacity_,
            .minimumVertexCapacity = config_.initialVertexCapacity,
            .minimumIndexCapacity = config_.initialIndexCapacity,
            .transformCapacity = TransformDataCount,
            .drawCapacity = DrawDataCount,
            .jointMatrixCapacity = JointMatrixCount,
            .maximumVertexBufferBytes = device.properties().limits.maxStorageBufferRange
        });

        PreparedSceneData prepared;
        prepared.generation = nextSceneGeneration();
        prepared.vertexCapacity = plan.vertexCapacity;
        prepared.indexCapacity = plan.indexCapacity;
        prepared.uploadVertices = std::move(plan.vertices);
        prepared.uploadIndices = std::move(plan.indices);
        if (plan.usedVertexCount < plan.vertexCapacity)
        {
            prepared.freeVertexRanges.push_back(
                {plan.usedVertexCount, plan.vertexCapacity - plan.usedVertexCount});
        }
        if (plan.usedIndexCount < plan.indexCapacity)
        {
            prepared.freeIndexRanges.push_back(
                {plan.usedIndexCount, plan.indexCapacity - plan.usedIndexCount});
        }

        const auto textureIndex = [](danvulkan::assets::TextureHandle handle)
        {
            return handle ? static_cast<std::int32_t>(handle.slot) : -1;
        };
        prepared.materials.reserve(scene.materials().size());
        prepared.materialNames.reserve(scene.materials().size());
        for (const danvulkan::assets::MaterialAsset& source : scene.materials())
        {
            MaterialData material{};
            material.baseColorFactor = source.albedoTint;
            material.emissiveMetallic = glm::vec4(source.emissiveFactor,
                source.metallicFactor);
            material.roughnessNormalOcclusionAlpha = {source.roughnessFactor,
                source.normalScale, source.occlusionStrength, source.alphaCutoff};
            material.textureTiling = glm::vec4(source.textureTiling, 0.0f, 0.0f);
            material.textureIndices = {textureIndex(source.albedoTexture),
                textureIndex(source.normalTexture), textureIndex(source.metallicRoughnessTexture),
                textureIndex(source.occlusionTexture)};
            material.materialFlags = {textureIndex(source.emissiveTexture),
                static_cast<std::int32_t>(source.alphaMode), source.doubleSided ? 1 : 0,
                source.unlit ? 1 : 0};
            prepared.materials.push_back(material);
            prepared.materialNames.push_back(source.name);
        }

        prepared.meshResources.reserve(plan.meshes.size());
        for (const danvulkan::PlannedSceneMesh& source : plan.meshes)
        {
            const MaterialData& material = prepared.materials.at(source.materialIndex);
            prepared.meshResources.push_back({source.indexCount, source.vertexCount,
                source.firstIndex, source.vertexOffset,
                static_cast<std::int32_t>(source.materialIndex),
                static_cast<std::uint32_t>(material.materialFlags.y * 2 +
                    material.materialFlags.z),
                source.bounds, source.name});
        }
        prepared.transforms.reserve(plan.transforms.size());
        prepared.instanceNames.reserve(plan.transforms.size());
        for (const danvulkan::PlannedSceneTransform& transform : plan.transforms)
        {
            prepared.transforms.push_back({transform.world});
            prepared.instanceNames.push_back(transform.name);
        }

        if (!scene.skins().empty() || !scene.animations().empty())
        {
            prepared.animationPlayer.emplace(scene,
                danvulkan::AnimationPlayer::SamplingSettings{
                    config_.animation.adaptiveNlerpMaxAngleRadians});
            prepared.animationActors.resize(prepared.animationPlayer->instanceCount());
        }
        prepared.draws.reserve(plan.draws.size());
        prepared.meshes.reserve(plan.draws.size());
        prepared.bounds.reserve(plan.draws.size());
        prepared.jointMatrices.reserve(plan.jointMatrixCount);
        for (const danvulkan::PlannedSceneDraw& source : plan.draws)
        {
            const MeshResourceData& resource = prepared.meshResources.at(source.meshIndex);
            DrawData draw{};
            draw.materialIndex = resource.materialIndex;
            draw.transformIndex = static_cast<std::int32_t>(source.transformIndex);
            draw.vertexOffset = static_cast<std::int32_t>(resource.vertexOffset);
            if (source.skin)
            {
                draw.jointOffset = static_cast<std::int32_t>(source.jointOffset);
            }
            prepared.draws.push_back(draw);

            MeshData mesh{};
            mesh.indexCount = resource.indexCount;
            mesh.firstIndex = resource.firstIndex;
            mesh.vertexOffset = resource.vertexOffset;
            mesh.pipelineVariant = resource.pipelineVariant;
            mesh.meshResourceSlot = source.meshIndex;
            mesh.drawData = draw;
            mesh.localBounds = resource.bounds;
            const std::uint32_t meshDataIndex =
                static_cast<std::uint32_t>(prepared.meshes.size());
            prepared.meshes.push_back(mesh);
            prepared.bounds.push_back(source.worldBounds);
            std::optional<std::size_t> animationInstance;
            if (prepared.animationPlayer && prepared.animationPlayer->instanceCount() != 0)
            {
                animationInstance = source.skin ?
                    prepared.animationPlayer->instanceForSkin(source.skin) :
                    prepared.animationPlayer->instanceForNode(source.node);
            }
            if (animationInstance)
            {
                prepared.animatedDraws.push_back(
                    {source.node, source.transformIndex, meshDataIndex,
                        static_cast<std::uint32_t>(*animationInstance)});
                danvulkan::vk::AnimationActorState& actor =
                    prepared.animationActors.at(*animationInstance);
                if (!actor.hasBounds)
                {
                    actor.conservativeBounds = source.worldBounds;
                    actor.hasBounds = true;
                }
                else
                {
                    actor.conservativeBounds.minVertex = glm::min(
                        actor.conservativeBounds.minVertex, source.worldBounds.minVertex);
                    actor.conservativeBounds.maxVertex = glm::max(
                        actor.conservativeBounds.maxVertex, source.worldBounds.maxVertex);
                }
            }
            if (source.skin)
            {
                if (prepared.jointMatrices.size() == source.jointOffset)
                {
                    const danvulkan::assets::SkinAsset* skin = scene.find(source.skin);
                    if (skin == nullptr)
                    {
                        throw std::runtime_error("planned skin palette is invalid");
                    }
                    prepared.skinPalettes.push_back({source.skin, source.jointOffset,
                        static_cast<std::uint32_t>(skin->joints.size()), animationInstance ?
                            static_cast<std::uint32_t>(*animationInstance) :
                            danvulkan::vk::SkinPaletteState::noAnimationInstance});
                    prepared.animationPlayer->appendSkinMatrices(
                        source.skin, prepared.jointMatrices);
                }
            }
        }
        for (danvulkan::vk::AnimationActorState& actor : prepared.animationActors)
        {
            if (!actor.hasBounds)
            {
                continue;
            }
            const glm::vec3 extent = actor.conservativeBounds.maxVertex -
                actor.conservativeBounds.minVertex;
            const glm::vec3 padding = glm::max(
                extent * config_.animation.conservativeBoundsPadding, glm::vec3(0.01f));
            actor.conservativeBounds.minVertex -= padding;
            actor.conservativeBounds.maxVertex += padding;
        }
        if (prepared.jointMatrices.size() != plan.jointMatrixCount)
        {
            throw std::runtime_error("planned scene joint palette is inconsistent");
        }
        return prepared;
    }

    void uploadSceneAsset(const danvulkan::assets::SceneAsset& scene)
    {
        scene_.materialCapacity_ = std::min(config_.maxMaterials, MatDataCount);
        if (scene_.materialCapacity_ == 0)
        {
            throw std::runtime_error("configured material capacity must be greater than zero");
        }
        const VkPhysicalDeviceProperties& properties = device.properties();
        const std::optional<std::uint32_t> textureCapacity =
            danvulkan::vk::selectTextureDescriptorCapacity({config_.maxTextures,
                static_cast<std::uint32_t>(scene.textures().size()),
                properties.limits.maxPerStageDescriptorSamplers,
                properties.limits.maxDescriptorSetSamplers});
        if (!textureCapacity)
        {
            throw std::runtime_error(
                "configured bindless texture capacity is invalid for this scene or device");
        }
        scene_.textureDescriptorCapacity_ = *textureCapacity;
        PreparedSceneData prepared = prepareSceneData(scene);
        scene_.sceneGeneration_ = prepared.generation;
        scene_.vertexCapacity_ = 0;
        scene_.indexCapacity_ = 0;
        scene_.freeVertexRanges_.clear();
        scene_.freeIndexRanges_.clear();
        scene_.retiredGeometryRanges_.clear();
        scene_.imageGeometryRangeVersions_.clear();
        scene_.geometryRangeVersion_ = 1;
        scene_.matData.clear();
        scene_.transformData.clear();
        scene_.materialNames.clear();
        scene_.materialGenerations_.clear();
        scene_.materialAlive_.clear();
        scene_.freeMaterialSlots_.clear();
        scene_.instanceNames.clear();
        scene_.instanceGenerations.clear();
        scene_.instanceAlive.clear();
        scene_.freeInstanceSlots.clear();
        scene_.meshResources.clear();
        scene_.meshGenerations_.clear();
        scene_.meshAlive_.clear();
        scene_.freeMeshSlots_.clear();
        scene_.drawData.clear();
        scene_.meshData.clear();
        scene_.aabbs.clear();
        scene_.animationPlayer_.reset();
        scene_.animatedDraws_.clear();
        scene_.skinPalettes_.clear();
        scene_.animationActors_.clear();
        scene_.animationUpdatePolicies_.clear();
        scene_.jointMatrices_.clear();
        scene_.textures.clear();
        scene_.textureGenerations_.clear();
        scene_.freeTextureSlots_.clear();
        scene_.retiredTextures_.clear();
        scene_.liveTextureCount_ = 0;
        scene_.textureVersion_ = 1;

        scene_.animationPlayer_ = std::move(prepared.animationPlayer);
        scene_.animatedDraws_ = std::move(prepared.animatedDraws);
        scene_.skinPalettes_ = std::move(prepared.skinPalettes);
        scene_.animationActors_ = std::move(prepared.animationActors);
        scene_.jointMatrices_ = std::move(prepared.jointMatrices);

        scene_.vertexBuffer = createDeviceLocalBuffer(
            sizeof(Vertex) * static_cast<VkDeviceSize>(prepared.vertexCapacity),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, prepared.uploadVertices.data(),
            sizeof(Vertex) * static_cast<VkDeviceSize>(prepared.uploadVertices.size()),
            "scene vertex storage buffer");
        scene_.indexBuffer = createDeviceLocalBuffer(
            sizeof(std::uint32_t) * static_cast<VkDeviceSize>(prepared.indexCapacity),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, prepared.uploadIndices.data(),
            sizeof(std::uint32_t) * static_cast<VkDeviceSize>(prepared.uploadIndices.size()),
            "scene index buffer");

        for (const danvulkan::assets::TextureAsset& texture : scene.textures())
        {
            scene_.textures.emplace_back(createTextureImage(texture));
            scene_.textureGenerations_.push_back(scene_.sceneGeneration_);
            ++scene_.liveTextureCount_;
        }

        scene_.matData = std::move(prepared.materials);
        scene_.materialNames = std::move(prepared.materialNames);
        scene_.materialGenerations_.assign(scene_.matData.size(), scene_.sceneGeneration_);
        scene_.materialAlive_.assign(scene_.matData.size(), true);

        scene_.meshResources = std::move(prepared.meshResources);
        scene_.meshGenerations_.assign(scene_.meshResources.size(), scene_.sceneGeneration_);
        scene_.meshAlive_.assign(scene_.meshResources.size(), true);

        scene_.transformData = std::move(prepared.transforms);
        scene_.instanceNames = std::move(prepared.instanceNames);
        scene_.instanceGenerations.assign(scene_.transformData.size(), scene_.sceneGeneration_);
        scene_.instanceAlive.assign(scene_.transformData.size(), true);
        scene_.drawData = std::move(prepared.draws);
        scene_.meshData = std::move(prepared.meshes);
        scene_.aabbs = std::move(prepared.bounds);

        scene_.vertexCapacity_ = prepared.vertexCapacity;
        scene_.indexCapacity_ = prepared.indexCapacity;
        scene_.freeVertexRanges_ = std::move(prepared.freeVertexRanges);
        scene_.freeIndexRanges_ = std::move(prepared.freeIndexRanges);
        scene_.reserveFrameScratch();
    }

    void checkFilesChanged()
    {
        for (auto& file : std::filesystem::recursive_directory_iterator(SHADER_PATH)) 
        {
            if (file.path().extension().string() == ".spv") continue;
            auto last_write_time = std::filesystem::last_write_time(file);

            auto filePath = file.path().string();
            auto keyExists = shaderPaths.find(filePath) != shaderPaths.end();
            if (keyExists)
            {
                // File was written to since we last checked
                if (last_write_time != shaderPaths[filePath])
                {
                    std::cout << filePath << " was changed!" << std::endl;
                    shaderPaths[filePath] = last_write_time;
                    shaderFileChanged(file.path());
                }
            }
            else 
            {
                shaderPaths[filePath] = last_write_time;
            }
        }
    }

    void shaderFileChanged(std::filesystem::path shaderSourceFile)
    {
        std::string filePath = shaderSourceFile.string();
        const bool vertexStage = shaderSourceFile.stem().string().ends_with("vert");
        const std::string stage = vertexStage ? "vert" : "frag";
        const std::filesystem::path compiledPath = std::filesystem::path(COMPILED_SHADER_PATH) /
            (shaderSourceFile.stem().string() + ".spv");
        std::array<char, 128> buffer;
        std::string result;
#ifdef _WIN32
        const std::string cmd = "glslc.exe --target-env=vulkan1.4 -fshader-stage=" + stage + " \"" + filePath +
            "\" -o \"" + compiledPath.string() + "\" 2>&1";
        std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd.c_str(), "r"), _pclose);
#else
        const std::string cmd = "glslc --target-env=vulkan1.4 -fshader-stage=" + stage + " \"" + filePath +
            "\" -o \"" + compiledPath.string() + "\" 2>&1";
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
#endif
        if (!pipe) {
            throw std::runtime_error("failed to send shell command for shader reload");
        }
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        std::cout << result << std::endl;
        recreateGraphicsPipeline();
    }
    void initGame()
    {
        frameTimes.fill(0.0f);

        glm::vec3 sceneCenter(0.0f);
        float sceneRadius = 1.0f;
        if (!scene_.aabbs.empty())
        {
            const float maximum = std::numeric_limits<float>::max();
            glm::vec3 sceneMinimum(maximum);
            glm::vec3 sceneMaximum(-maximum);
            for (const AABB& bounds : scene_.aabbs)
            {
                sceneMinimum = glm::min(sceneMinimum, bounds.minVertex);
                sceneMaximum = glm::max(sceneMaximum, bounds.maxVertex);
            }
            sceneCenter = (sceneMinimum + sceneMaximum) * 0.5f;
            sceneRadius = std::max(glm::length(sceneMaximum - sceneMinimum) * 0.5f, 0.1f);
        }

        const float aspect = swapchain.extent().width /
            static_cast<float>(swapchain.extent().height);
        constexpr float verticalFieldOfView = 60.0f;
        const float verticalHalfAngle = glm::radians(verticalFieldOfView * 0.5f);
        const float horizontalHalfAngle = std::atan(std::tan(verticalHalfAngle) * aspect);
        const float limitingHalfAngle = std::min(verticalHalfAngle, horizontalHalfAngle);
        const float cameraDistance = sceneRadius / std::sin(limitingHalfAngle) * 1.15f;
        const glm::vec3 cameraOffsetDirection = glm::normalize(glm::vec3(0.0f, 0.35f, 1.0f));
        const glm::vec3 cameraWorldPosition =
            sceneCenter + cameraOffsetDirection * cameraDistance;
        const glm::vec3 lookDirection = glm::normalize(sceneCenter - cameraWorldPosition);
        const float pitch = glm::degrees(std::asin(glm::clamp(-lookDirection.y, -1.0f, 1.0f)));
        const float yaw = glm::degrees(std::atan2(lookDirection.x, -lookDirection.z));

        camera.flipY = false;
        camera.setPerspective(verticalFieldOfView, aspect,
            std::max(sceneRadius * 0.01f, 0.01f),
            std::max(cameraDistance + sceneRadius * 4.0f, 100.0f));
        // The legacy camera stores the inverse translation used by its view matrix.
        camera.setPosition(-cameraWorldPosition);
        camera.setRotation(glm::vec3(pitch, yaw, 0.0f));
        camera.setMovementSpeed(std::max(sceneRadius * 0.5f, 0.5f));

        lightPos = glm::vec3(0.0f, 0.0f, 2.0f);
        lightSpeed = glm::vec3(0.0f);
        /* lion head spots
        DrawData lion1 = scene_.drawData[375], lion2 = scene_.drawData[376];
        Vertex v1 = vertices[lion1.vertexOffset];
        Vertex v2 = vertices[lion2.vertexOffset];

        lightPos = v2.pos;
        lightSpeed = glm::normalize(v2.pos - v1.pos);
     
        cameraStop = v1.pos;*/
    }
};

VulkanRenderer::VulkanRenderer(RendererConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
}

VulkanRenderer::~VulkanRenderer() = default;
VulkanRenderer::VulkanRenderer(VulkanRenderer&&) noexcept = default;
VulkanRenderer& VulkanRenderer::operator=(VulkanRenderer&&) noexcept = default;

void VulkanRenderer::initialize()
{
    impl_->initialize();
}

bool VulkanRenderer::isInitialized() const noexcept
{
    return impl_ != nullptr && impl_->isInitialized();
}

bool VulkanRenderer::shouldClose() const noexcept
{
    return impl_ == nullptr || impl_->shouldClose();
}

bool VulkanRenderer::beginFrame()
{
    return impl_->beginFrame();
}

void VulkanRenderer::submitScene(const SceneSubmission& submission)
{
    impl_->submitScene(submission);
}

void VulkanRenderer::submitUi(const UiDrawData& drawData)
{
    impl_->submitUi(drawData);
}

void VulkanRenderer::endFrame()
{
    impl_->endFrame();
}

std::vector<SceneInstanceInfo> VulkanRenderer::sceneInstances() const
{
    return impl_->sceneInstances();
}

std::optional<SceneBounds> VulkanRenderer::sceneBounds() const
{
    return impl_->sceneBounds();
}

std::vector<SceneMaterialInfo> VulkanRenderer::sceneMaterials() const
{
    return impl_->sceneMaterials();
}

std::vector<SceneMeshInfo> VulkanRenderer::sceneMeshes() const
{
    return impl_->sceneMeshes();
}

std::vector<SceneTextureInfo> VulkanRenderer::sceneTextures() const
{
    return impl_->sceneTextures();
}

std::vector<SceneAnimationInfo> VulkanRenderer::sceneAnimations() const
{
    return impl_->sceneAnimations();
}

std::vector<SceneAnimationActorInfo> VulkanRenderer::sceneAnimationActors() const
{
    return impl_->sceneAnimationActors();
}

AnimationPlaybackState VulkanRenderer::animationPlaybackState() const
{
    return impl_->animationPlaybackState();
}

RendererPerformanceStats VulkanRenderer::performanceStats() const noexcept
{
    return impl_ != nullptr ? impl_->performanceStats() : RendererPerformanceStats{};
}

RendererMemoryStats VulkanRenderer::memoryStats() const noexcept
{
    return impl_ != nullptr ? impl_->memoryStats() : RendererMemoryStats{};
}

void VulkanRenderer::playAnimation(SceneAnimationHandle animation, bool restart)
{
    impl_->playAnimation(animation, restart);
}

void VulkanRenderer::pauseAnimation()
{
    impl_->pauseAnimation();
}

void VulkanRenderer::resumeAnimation()
{
    impl_->resumeAnimation();
}

void VulkanRenderer::stopAnimation()
{
    impl_->stopAnimation();
}

void VulkanRenderer::seekAnimation(float positionSeconds)
{
    impl_->seekAnimation(positionSeconds);
}

void VulkanRenderer::setAnimationLooping(bool looping)
{
    impl_->setAnimationLooping(looping);
}

void VulkanRenderer::setAnimationPlaybackSpeed(float speed)
{
    impl_->setAnimationPlaybackSpeed(speed);
}

void VulkanRenderer::replaceScene(const danvulkan::assets::SceneAsset& scene)
{
    impl_->replaceScene(scene);
}

SceneTextureHandle VulkanRenderer::uploadTexture(
    const danvulkan::assets::TextureAsset& texture)
{
    return impl_->uploadTexture(texture);
}

void VulkanRenderer::destroyTexture(SceneTextureHandle texture)
{
    impl_->destroyTexture(texture);
}

SceneMaterialHandle VulkanRenderer::createMaterial(
    const RuntimeMaterialDescription& material)
{
    return impl_->createMaterial(material);
}

void VulkanRenderer::destroyMaterial(SceneMaterialHandle material)
{
    impl_->destroyMaterial(material);
}

SceneMeshHandle VulkanRenderer::uploadMesh(
    std::span<const danvulkan::assets::Vertex> vertices,
    std::span<const std::uint32_t> indices,
    SceneMaterialHandle material,
    std::string name)
{
    return impl_->uploadMesh(vertices, indices, material, std::move(name));
}

SceneMeshHandle VulkanRenderer::uploadGrass(
    std::span<const RuntimeGrassBlade> blades,
    SceneMaterialHandle material,
    const RuntimeGrassLodDescription& lod,
    std::string name)
{
    return impl_->uploadGrass(blades, material, lod, std::move(name));
}

std::vector<SceneMeshHandle> VulkanRenderer::uploadMeshBatch(
    std::span<const RuntimeMeshUploadDescription> uploads)
{
    return impl_->uploadMeshBatch(uploads);
}

bool VulkanRenderer::runtimeUploadReady()
{
    return impl_->runtimeUploadReady();
}

void VulkanRenderer::destroyMesh(SceneMeshHandle mesh)
{
    impl_->destroyMesh(mesh);
}

SceneInstanceHandle VulkanRenderer::createMeshInstance(SceneMeshHandle mesh,
    const glm::mat4& worldTransform, std::string name)
{
    return impl_->createMeshInstance(mesh, worldTransform, std::move(name));
}

void VulkanRenderer::destroyInstance(SceneInstanceHandle instance)
{
    impl_->destroyInstance(instance);
}

void VulkanRenderer::updateInstanceTransform(SceneInstanceHandle instance,
    const glm::mat4& worldTransform)
{
    impl_->updateInstanceTransform(instance, worldTransform);
}

void VulkanRenderer::updateMaterialProperties(SceneMaterialHandle material,
    const RuntimeMaterialProperties& properties)
{
    impl_->updateMaterialProperties(material, properties);
}

void VulkanRenderer::updateMaterialTextures(SceneMaterialHandle material,
    const RuntimeMaterialTextures& textures)
{
    impl_->updateMaterialTextures(material, textures);
}

void VulkanRenderer::shutdown()
{
    impl_->shutdown();
}

void VulkanRenderer::run()
{
    impl_->run();
}
