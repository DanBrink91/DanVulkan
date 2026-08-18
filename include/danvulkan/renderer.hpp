#pragma once

#include <danvulkan/assets.hpp>
#include <danvulkan/platform.hpp>
#include <danvulkan/ui.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

struct AdditionalSceneConfig
{
    std::filesystem::path modelPath;
    glm::mat4 rootTransform{1.0f};
};

struct RendererMemoryPolicy
{
    // Zero selects the device's highest mutually supported color/depth sample count. Set to one
    // to disable MSAA, or to another power of two to cap the selected sample count.
    std::uint32_t maxMsaaSamples = 0;
    // Transient attachments remain fully functional when lazy memory is unavailable; this only
    // controls whether the allocator prefers a lazily allocated memory type when one is offered.
    bool preferLazilyAllocatedAttachments = true;
};

struct RendererAnimationPolicy
{
    // Actor bounds are tested before pose evaluation. Playback clocks continue advancing while
    // an off-screen pose retains its last evaluated transforms and palettes.
    bool cullOffscreenActors = true;
    // Actors within this distance evaluate every rendered frame.
    float fullRateDistance = 15.0f;
    // Actors between the full-rate and reduced-rate distances use mediumUpdatesPerSecond;
    // actors beyond this distance use farUpdatesPerSecond.
    float reducedRateDistance = 35.0f;
    float mediumUpdatesPerSecond = 30.0f;
    float farUpdatesPerSecond = 15.0f;
    // Initial grouped draw bounds retain this proportional margin and only expand afterward.
    float conservativeBoundsPadding = 0.2f;
    // Zero retains exact quaternion slerp. A positive value enables normalized linear
    // interpolation only for adjacent rotation keys within this angular distance in radians.
    float adaptiveNlerpMaxAngleRadians = 0.0f;
};

struct RendererConfig
{
    std::string applicationName = "DanVulkan";
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    std::filesystem::path modelPath = "models/naruto_hiddenly_village.glb";
    std::vector<AdditionalSceneConfig> additionalScenes;
    std::uint32_t maxTextures = 256;
    std::uint32_t maxMaterials = 2048;
    // Lighting uses a per-swapchain-image storage buffer. The shader ABI supports at most 256
    // lights; lower values reduce persistent host-visible memory.
    std::uint32_t maxPointLights = 64;
    // Optional decoded equirectangular image used for image-based lighting. RGBA8 and linear
    // RGBA32F payloads are accepted; assets::loadEnvironment preserves HDR radiance. Null selects
    // a small renderer-owned neutral sky, so the environment descriptors are always valid.
    std::optional<danvulkan::assets::TextureAsset> environmentMap;
    // Negative values retain a finer mip level for sharper minified textures. The default is
    // deliberately mild because the bundled scene packs broad terrain surfaces into an atlas.
    float textureMipLodBias = -0.5f;
    std::uint32_t initialVertexCapacity = 4096;
    std::uint32_t initialIndexCapacity = 8192;
    RendererMemoryPolicy memory;
    RendererAnimationPolicy animation;
    std::uint64_t maxFrames = 0;
    std::uint64_t resizeAtFrame = 0;
    std::uint32_t resizeWidth = 960;
    std::uint32_t resizeHeight = 540;
    // Null selects the renderer-owned GLFW backend. Supplying an adapter keeps native window
    // and input ownership in the host application.
    std::shared_ptr<RendererPlatform> platform;
};

inline constexpr std::uint32_t MaxScenePointLights = 256;

struct ScenePointLight
{
    glm::vec3 position{0.0f, 0.0f, 2.0f};
    // Zero disables range falloff beyond inverse-square attenuation.
    float range = 0.0f;
    glm::vec3 color{1.0f};
    float intensity = 25.0f;
};

struct SceneEnvironment
{
    glm::vec3 tint{1.0f};
    float intensity = 0.03f;
    // Rotation is around world up and is expressed in radians.
    float rotation = 0.0f;
    float diffuseStrength = 1.0f;
    float specularStrength = 1.0f;
};

struct SceneAnimationActorHandle
{
    static constexpr std::uint32_t invalidSlot = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t slot = invalidSlot;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != invalidSlot && generation != 0;
    }

    constexpr explicit operator bool() const noexcept { return valid(); }
    friend constexpr bool operator==(
        SceneAnimationActorHandle, SceneAnimationActorHandle) noexcept = default;
};

struct SceneAnimationActorTransform
{
    SceneAnimationActorHandle actor;
    // Applied above the actor's imported root transform. Identity retains its authored pose.
    glm::mat4 worldOffset{1.0f};
};

// Per-frame state for the active scene, initially loaded through RendererConfig::modelPath.
// Scene resources remain renderer-owned; applications own when and how this
// view is submitted.
struct SceneSubmission
{
    glm::mat4 view{1.0f};
    // Vulkan clip depth is [0, 1]. DanVulkan's CMake target exports the matching
    // GLM configuration and the renderer performs the framebuffer Y inversion.
    glm::mat4 projection{1.0f};
    glm::vec3 cameraPosition{0.0f};
    std::vector<ScenePointLight> pointLights{ScenePointLight{}};
    SceneEnvironment environment;
    std::vector<SceneAnimationActorTransform> animationActorTransforms;
};

struct SceneInstanceHandle
{
    static constexpr std::uint32_t invalidSlot = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t slot = invalidSlot;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != invalidSlot && generation != 0;
    }

    constexpr explicit operator bool() const noexcept { return valid(); }
    friend constexpr bool operator==(SceneInstanceHandle, SceneInstanceHandle) noexcept = default;
};

struct SceneMaterialHandle
{
    static constexpr std::uint32_t invalidSlot = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t slot = invalidSlot;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != invalidSlot && generation != 0;
    }

    constexpr explicit operator bool() const noexcept { return valid(); }
    friend constexpr bool operator==(SceneMaterialHandle, SceneMaterialHandle) noexcept = default;
};

struct SceneMeshHandle
{
    static constexpr std::uint32_t invalidSlot = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t slot = invalidSlot;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != invalidSlot && generation != 0;
    }

    constexpr explicit operator bool() const noexcept { return valid(); }
    friend constexpr bool operator==(SceneMeshHandle, SceneMeshHandle) noexcept = default;
};

struct SceneTextureHandle
{
    static constexpr std::uint32_t invalidSlot = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t slot = invalidSlot;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != invalidSlot && generation != 0;
    }

    constexpr explicit operator bool() const noexcept { return valid(); }
    friend constexpr bool operator==(SceneTextureHandle, SceneTextureHandle) noexcept = default;
};

struct SceneAnimationHandle
{
    static constexpr std::uint32_t invalidSlot = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t slot = invalidSlot;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != invalidSlot && generation != 0;
    }

    constexpr explicit operator bool() const noexcept { return valid(); }
    friend constexpr bool operator==(SceneAnimationHandle, SceneAnimationHandle) noexcept = default;
};

enum class AnimationPlaybackStatus
{
    stopped,
    playing,
    paused,
    finished
};

struct SceneAnimationInfo
{
    SceneAnimationHandle handle;
    std::string name;
    float durationSeconds = 0.0f;
};

struct SceneAnimationActorInfo
{
    SceneAnimationActorHandle handle;
    SceneAnimationHandle animation;
    glm::mat4 worldOffset{1.0f};
};

struct AnimationPlaybackState
{
    SceneAnimationHandle clip;
    AnimationPlaybackStatus status = AnimationPlaybackStatus::stopped;
    float positionSeconds = 0.0f;
    float durationSeconds = 0.0f;
    float playbackSpeed = 1.0f;
    bool looping = true;
};

// Mutable PBR factors for an uploaded or runtime-created material. Alpha mode and
// double-sided state select its pipeline variant and remain fixed after creation.
struct RuntimeMaterialProperties
{
    glm::vec4 baseColorFactor{1.0f};
    glm::vec2 textureTiling{1.0f};
    glm::vec3 emissiveFactor{0.0f};
    float metallicFactor = 0.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    float alphaCutoff = 0.5f;
};

struct RuntimeMaterialTextures
{
    SceneTextureHandle baseColor;
    SceneTextureHandle normal;
    SceneTextureHandle metallicRoughness;
    SceneTextureHandle occlusion;
    SceneTextureHandle emissive;
};

struct RuntimeMaterialDescription
{
    std::string name;
    RuntimeMaterialProperties properties;
    RuntimeMaterialTextures textures;
    danvulkan::assets::AlphaMode alphaMode = danvulkan::assets::AlphaMode::opaque;
    bool doubleSided = false;
    bool unlit = false;
};

struct SceneInstanceInfo
{
    SceneInstanceHandle handle;
    std::string name;
    // This is the model-to-world transform. Hierarchy evaluation happened during import.
    glm::mat4 worldTransform{1.0f};
};

struct SceneBounds
{
    glm::vec3 minimum{0.0f};
    glm::vec3 maximum{0.0f};
};

struct SceneMaterialInfo
{
    SceneMaterialHandle handle;
    std::string name;
    RuntimeMaterialProperties properties;
    RuntimeMaterialTextures textures;
    danvulkan::assets::AlphaMode alphaMode = danvulkan::assets::AlphaMode::opaque;
    bool doubleSided = false;
    bool unlit = false;
};

struct SceneMeshInfo
{
    SceneMeshHandle handle;
    std::string name;
    SceneMaterialHandle material;
    glm::vec3 localBoundsMin{0.0f};
    glm::vec3 localBoundsMax{0.0f};
};

struct SceneTextureInfo
{
    SceneTextureHandle handle;
    std::string name;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 1;
    danvulkan::assets::ColorSpace colorSpace = danvulkan::assets::ColorSpace::srgb;
    danvulkan::assets::TextureSampler sampler;
};

struct RendererPerformanceStats
{
    std::uint64_t renderedFrames = 0;
    // End-to-end wall time from beginFrame through presentation. This includes synchronization
    // and presentation pacing; frameCpuMilliseconds is CPU time consumed by the render thread.
    double frameMilliseconds = 0.0;
    double frameCpuMilliseconds = 0.0;
    double frameGpuMilliseconds = 0.0;
    double animationCpuMilliseconds = 0.0;
    double animationEvaluationCpuMilliseconds = 0.0;
    double animationSynchronizationCpuMilliseconds = 0.0;
    double animationSamplingCpuMilliseconds = 0.0;
    double animationPoseResetCpuMilliseconds = 0.0;
    double animationTimelineResolutionCpuMilliseconds = 0.0;
    double animationVectorSamplingCpuMilliseconds = 0.0;
    double animationRotationSamplingCpuMilliseconds = 0.0;
    double animationTransformPropagationCpuMilliseconds = 0.0;
    double animationTransformUpdateCpuMilliseconds = 0.0;
    double animationBoundsCpuMilliseconds = 0.0;
    double animationPaletteGenerationCpuMilliseconds = 0.0;
    double animationPaletteUploadCpuMilliseconds = 0.0;
    double cullingCpuMilliseconds = 0.0;
    double bufferWriteCpuMilliseconds = 0.0;
    double commandRecordingCpuMilliseconds = 0.0;
    std::uint32_t activeDraws = 0;
    std::uint32_t visibleDraws = 0;
    std::uint32_t animatedDraws = 0;
    std::uint32_t jointMatrices = 0;
    std::uint32_t pointLights = 0;
    std::uint32_t animationActors = 0;
    std::uint32_t evaluatedAnimationActors = 0;
    std::uint32_t culledAnimationActors = 0;
    std::uint32_t sampledAnimationChannels = 0;
    std::uint32_t sampledAnimationVectorChannels = 0;
    std::uint32_t sampledAnimationRotationChannels = 0;
    std::uint32_t nlerpAnimationRotationChannels = 0;
    std::uint32_t animationClipChannels = 0;
    std::uint32_t foldedConstantAnimationChannels = 0;
    std::uint32_t animationPropagatedNodes = 0;
    std::uint32_t animationPoseComposedNodes = 0;
    std::uint32_t animationCachedLocalNodes = 0;
};

struct RendererMemoryStats
{
    // Current VMA state.
    std::uint64_t blockBytes = 0;
    std::uint64_t allocationBytes = 0;
    std::uint64_t heapUsageBytes = 0;
    std::uint64_t heapBudgetBytes = 0;
    std::uint32_t blockCount = 0;
    std::uint32_t allocationCount = 0;
    // High-water marks sampled after initialization and resource mutations, including the point
    // where old and replacement scene resources coexist.
    std::uint64_t peakBlockBytes = 0;
    std::uint64_t peakAllocationBytes = 0;
    std::uint64_t peakHeapUsageBytes = 0;
    std::uint32_t peakBlockCount = 0;
    std::uint32_t peakAllocationCount = 0;
    // The upload arena is one persistently mapped allocation reused from offset zero after each
    // synchronous upload. CPU geometry is temporary and should be zero outside scene preparation.
    std::uint64_t stagingArenaBytes = 0;
    std::uint64_t stagingArenaGrowthCount = 0;
    std::uint64_t uploadSubmissionCount = 0;
    std::uint64_t retainedCpuGeometryBytes = 0;
    std::uint64_t cpuScratchBytes = 0;
    // Number of frame-indexed color/depth target sets, not the number of individual images.
    std::uint32_t attachmentSetCount = 0;
    std::uint32_t msaaSamples = 1;
    bool prefersLazilyAllocatedAttachments = false;
};

class VulkanRenderer
{
public:
    explicit VulkanRenderer(RendererConfig config = {});
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;
    VulkanRenderer(VulkanRenderer&&) noexcept;
    VulkanRenderer& operator=(VulkanRenderer&&) noexcept;

    void initialize();
    [[nodiscard]] bool isInitialized() const noexcept;
    [[nodiscard]] bool shouldClose() const noexcept;
    [[nodiscard]] bool beginFrame();
    void submitScene(const SceneSubmission& submission);
    // UI triangles are copied into renderer-owned frame scratch. Submission is optional and
    // renders as a single-sample overlay after scene resolve, independent of scene MSAA.
    void submitUi(const UiDrawData& drawData);
    void endFrame();

    // Resource handles stay valid for the current uploaded scene until their resource is
    // destroyed. Mutations happen between frames and reach a swapchain image only after its
    // in-flight fence waits.
    [[nodiscard]] std::vector<SceneInstanceInfo> sceneInstances() const;
    // The combined world-space bounds of all live scene draws. An empty scene has no bounds.
    // This is useful for application-owned camera framing and editor navigation.
    [[nodiscard]] std::optional<SceneBounds> sceneBounds() const;
    [[nodiscard]] std::vector<SceneMaterialInfo> sceneMaterials() const;
    [[nodiscard]] std::vector<SceneMeshInfo> sceneMeshes() const;
    [[nodiscard]] std::vector<SceneTextureInfo> sceneTextures() const;
    [[nodiscard]] std::vector<SceneAnimationInfo> sceneAnimations() const;
    // Actors are independently transformable animation instances. Their generation follows the
    // uploaded scene, so a replacement invalidates offsets retained by an application.
    [[nodiscard]] std::vector<SceneAnimationActorInfo> sceneAnimationActors() const;
    [[nodiscard]] AnimationPlaybackState animationPlaybackState() const;
    // Rolling measurements are diagnostic rather than benchmark guarantees. They remain
    // available after shutdown so automated stress modes can print their final sample.
    [[nodiscard]] RendererPerformanceStats performanceStats() const noexcept;
    // VMA totals describe current allocator blocks and logical allocations. Peak fields are
    // renderer-sampled high-water marks rather than driver guarantees. Heap usage/budget includes
    // memory reported by the driver for this process and remains a diagnostic estimate. The last
    // live snapshot remains available after shutdown.
    [[nodiscard]] RendererMemoryStats memoryStats() const noexcept;

    // Animation mutations happen between frames. Selecting a different clip always starts it
    // from the beginning; restart=false preserves an already-selected active or paused cursor.
    void playAnimation(SceneAnimationHandle animation, bool restart = true);
    void pauseAnimation();
    void resumeAnimation();
    // Stopping rewinds the selected clip and leaves its first pose applied.
    void stopAnimation();
    void seekAnimation(float positionSeconds);
    void setAnimationLooping(bool looping);
    void setAnimationPlaybackSpeed(float speed);

    // Prepares a complete replacement before committing it between frames. All handles from
    // the previous scene become stale when the replacement succeeds.
    void replaceScene(const danvulkan::assets::SceneAsset& scene);
    [[nodiscard]] SceneTextureHandle uploadTexture(
        const danvulkan::assets::TextureAsset& texture);
    // A texture must first be removed from every material. Its Vulkan resources are
    // retired only after all in-flight descriptor sets stop referencing its slot.
    void destroyTexture(SceneTextureHandle texture);
    [[nodiscard]] SceneMaterialHandle createMaterial(
        const RuntimeMaterialDescription& material);
    // Materials can be destroyed only after every mesh using them is removed.
    // The vacated GPU material index is retained as a reusable handle slot.
    void destroyMaterial(SceneMaterialHandle material);
    // Uploads indexed geometry into the renderer's combined device-local buffers. Indices are
    // local to the supplied vertex span; the returned mesh can be instantiated immediately.
    [[nodiscard]] SceneMeshHandle uploadMesh(
        std::span<const danvulkan::assets::Vertex> vertices,
        std::span<const std::uint32_t> indices,
        SceneMaterialHandle material,
        std::string name = {});
    // Mesh metadata can be destroyed only after all of its instances are removed.
    // Geometry storage reclamation is handled separately from handle invalidation.
    void destroyMesh(SceneMeshHandle mesh);
    [[nodiscard]] SceneInstanceHandle createMeshInstance(SceneMeshHandle mesh,
        const glm::mat4& worldTransform, std::string name = {});
    void destroyInstance(SceneInstanceHandle instance);
    void updateInstanceTransform(SceneInstanceHandle instance, const glm::mat4& worldTransform);
    void updateMaterialProperties(SceneMaterialHandle material,
        const RuntimeMaterialProperties& properties);
    void updateMaterialTextures(SceneMaterialHandle material,
        const RuntimeMaterialTextures& textures);

    void shutdown();

    // Convenience demo loop built on the step-driven API above.
    void run();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
