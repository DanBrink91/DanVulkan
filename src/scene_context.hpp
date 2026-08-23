#pragma once

#include <danvulkan/assets.hpp>
#include <danvulkan/grass.hpp>

#include "animation_player.hpp"
#include "descriptor_context.hpp"
#include "pipeline_planner.hpp"
#include "vulkan_raii.hpp"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace danvulkan::vk {

inline constexpr std::uint32_t DrawDataCount = 2048;
inline constexpr std::uint32_t MatDataCount = 2048;
inline constexpr std::uint32_t TransformDataCount = 2048;
inline constexpr std::uint32_t JointMatrixCount = 16384;

struct DrawData
{
    std::int32_t materialIndex = -1;
    std::int32_t transformIndex = -1;
    std::int32_t vertexOffset = 0;
    std::int32_t jointOffset = -1;
};

struct GrassTileDrawData
{
    assets::Bounds localBounds{};
    std::uint32_t firstBlade = 0;
    std::uint32_t bladeCount = 0;
    // 0..2 are active segment LODs; 255 means no previous frame has selected one.
    std::uint8_t lodState = 255U;
};

struct GrassDrawData
{
    bool enabled = false;
    std::array<std::uint32_t, 3> indexCounts{};
    std::array<std::uint32_t, 3> firstIndices{};
    std::array<float, 3> distances{};
    std::array<float, 3> populationRatios{1.0f, 0.5f, 0.18f};
    float hysteresis = 0.0f;
    float transitionBand = 0.0f;
    std::vector<GrassTileDrawData> tiles;
};

struct MeshData
{
    std::uint32_t indexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t vertexOffset = 0;
    std::uint32_t pipelineVariant = 0;
    std::uint32_t meshResourceSlot = 0;
    DrawData drawData;
    assets::Bounds localBounds;
    GrassDrawData grass;
};

struct MeshResourceData
{
    std::uint32_t indexCount = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t vertexOffset = 0;
    std::int32_t materialIndex = -1;
    std::uint32_t pipelineVariant = 0;
    assets::Bounds bounds{};
    std::string name;
    GrassDrawData grass;
};

struct GeometryRange
{
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
};

struct RetiredGeometry
{
    std::uint64_t version = 0;
    Buffer vertexBuffer;
    Buffer indexBuffer;
};

struct RetiredGeometryRanges
{
    std::uint64_t version = 0;
    GeometryRange vertices;
    GeometryRange indices;
};

struct MaterialData
{
    glm::vec4 baseColorFactor;
    glm::vec4 emissiveMetallic;
    glm::vec4 roughnessNormalOcclusionAlpha;
    glm::vec4 textureTiling;
    glm::ivec4 textureIndices;
    glm::ivec4 materialFlags;
};
static_assert(sizeof(MaterialData) == 96, "MaterialData must match shaders/frag.frag");

struct TransformData
{
    glm::mat4 model;
};

using Vertex = assets::Vertex;
static_assert(sizeof(Vertex) == 112, "CPU vertex layout must match shaders/vert.vert");
static_assert(sizeof(Vertex) % sizeof(std::uint32_t) == 0,
    "grass raw-word addressing requires a whole-word geometry stride");

struct SkinPaletteState
{
    static constexpr std::uint32_t noAnimationInstance =
        std::numeric_limits<std::uint32_t>::max();
    assets::SkinHandle skin;
    std::uint32_t jointOffset = 0;
    std::uint32_t jointCount = 0;
    std::uint32_t animationInstance = noAnimationInstance;
};

struct AnimationSynchronizationTimings
{
    double transformUpdateMilliseconds = 0.0;
    double boundsMilliseconds = 0.0;
    double paletteGenerationMilliseconds = 0.0;
};

struct AnimatedDrawState
{
    assets::NodeHandle node;
    std::uint32_t transformIndex = 0;
    std::uint32_t meshDataIndex = 0;
    std::uint32_t animationInstance = 0;
};

struct AnimationActorState
{
    assets::Bounds conservativeBounds{};
    glm::mat4 worldOffset{1.0f};
    bool hasBounds = false;
    bool transformDirty = false;
};

struct AnimationUpdateSettings
{
    bool cullOffscreenActors = true;
    float fullRateDistance = 15.0f;
    float reducedRateDistance = 35.0f;
    float mediumUpdatesPerSecond = 30.0f;
    float farUpdatesPerSecond = 15.0f;
};

struct AnimationUpdateCounts
{
    std::uint32_t actors = 0;
    std::uint32_t eligible = 0;
    std::uint32_t culled = 0;
};

struct Texture
{
    Image image;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkSampler sampler = VK_NULL_HANDLE;
    std::string name;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 1;
    assets::ColorSpace colorSpace = assets::ColorSpace::srgb;
    assets::TextureSampler samplerConfig;
};

struct RetiredTexture
{
    std::uint64_t version = 0;
    Texture texture;
};

struct PreparedSceneData
{
    std::uint32_t generation = 0;
    std::uint32_t vertexCapacity = 0;
    std::uint32_t indexCapacity = 0;
    std::vector<Vertex> uploadVertices;
    std::vector<std::uint32_t> uploadIndices;
    std::vector<GeometryRange> freeVertexRanges;
    std::vector<GeometryRange> freeIndexRanges;
    std::vector<MaterialData> materials;
    std::vector<std::string> materialNames;
    std::vector<MeshResourceData> meshResources;
    std::vector<TransformData> transforms;
    std::vector<std::string> instanceNames;
    std::vector<DrawData> draws;
    std::vector<MeshData> meshes;
    std::vector<assets::Bounds> bounds;
    std::optional<AnimationPlayer> animationPlayer;
    std::vector<AnimatedDrawState> animatedDraws;
    std::vector<SkinPaletteState> skinPalettes;
    std::vector<AnimationActorState> animationActors;
    std::vector<glm::mat4> jointMatrices;
};

struct DrawBatch
{
    std::uint32_t firstCommand = 0;
    std::uint32_t commandCount = 0;
};

struct SceneDrawCounts
{
    std::uint32_t active = 0;
    std::uint32_t visible = 0;
    std::uint32_t animated = 0;
    std::uint32_t joints = 0;
};

// Owns the complete uploaded-scene lifetime. Renderer orchestration borrows this state when
// recording a frame, but scene metadata, GPU storage, retirement queues, and draw scratch all
// live and reset together here.
class SceneContext
{
public:
    void reserveFrameScratch();
    [[nodiscard]] AnimationSynchronizationTimings synchronizeAnimationPose(
        bool changedInstancesOnly = false);
    void setAnimationActorTransform(std::size_t actorIndex, const glm::mat4& worldOffset);
    [[nodiscard]] AnimationUpdateCounts prepareAnimationUpdates(const glm::mat4& view,
        const glm::mat4& projection, const glm::vec3& cameraPosition,
        const AnimationUpdateSettings& settings);
    [[nodiscard]] std::span<const AnimationPlayer::InstanceUpdatePolicy>
        animationUpdatePolicies() const noexcept { return animationUpdatePolicies_; }
    [[nodiscard]] static AnimationPlayer::InstanceUpdatePolicy planActorAnimationUpdate(
        const assets::Bounds& bounds, const glm::mat4& view, const glm::mat4& projection,
        const glm::vec3& cameraPosition, const AnimationUpdateSettings& settings);
    [[nodiscard]] SceneDrawCounts prepareDraws(const glm::mat4& view,
        const glm::mat4& projection, const glm::vec3& cameraPosition);
    [[nodiscard]] std::uint64_t cpuScratchBytes() const noexcept;

    [[nodiscard]] std::vector<VkDescriptorImageInfo> textureDescriptorInfos() const;
    void initializeImageGenerations(std::size_t imageCount);
    void prepareGeometryForImage(std::uint32_t imageIndex, DescriptorContext& descriptors);
    void prepareGeometryRangesForImage(std::uint32_t imageIndex);
    void prepareTexturesForImage(std::uint32_t imageIndex, DescriptorContext& descriptors,
        VkDevice device);
    void releaseSwapchainRetirements(VkDevice device);
    void resetScene(VkDevice device) noexcept;

    [[nodiscard]] static bool hasGeometryRange(std::span<const GeometryRange> ranges,
        std::uint32_t count) noexcept;
    [[nodiscard]] static GeometryRange allocateGeometryRange(
        std::vector<GeometryRange>& ranges, std::uint32_t count);
    static void releaseGeometryRange(std::vector<GeometryRange>& ranges,
        GeometryRange released);
    [[nodiscard]] static std::uint32_t grownGeometryCapacity(std::uint32_t current,
        std::uint32_t requiredAdditional, std::uint32_t maximum);
    [[nodiscard]] static assets::Bounds transformedBounds(const assets::Bounds& bounds,
        const glm::mat4& transform);

    std::uint32_t vertexCapacity_ = 0;
    std::uint32_t indexCapacity_ = 0;
    std::vector<GeometryRange> freeVertexRanges_;
    std::vector<GeometryRange> freeIndexRanges_;
    std::uint64_t geometryRangeVersion_ = 1;
    std::vector<std::uint64_t> imageGeometryRangeVersions_;
    std::vector<RetiredGeometryRanges> retiredGeometryRanges_;

    std::vector<MaterialData> matData;
    std::vector<TransformData> transformData;
    std::vector<std::string> materialNames;
    std::vector<std::uint32_t> materialGenerations_;
    std::vector<bool> materialAlive_;
    std::vector<std::uint32_t> freeMaterialSlots_;
    std::vector<std::string> instanceNames;
    std::vector<std::uint32_t> instanceGenerations;
    std::vector<bool> instanceAlive;
    std::vector<std::uint32_t> freeInstanceSlots;
    std::vector<MeshResourceData> meshResources;
    std::vector<std::uint32_t> meshGenerations_;
    std::vector<bool> meshAlive_;
    std::vector<std::uint32_t> freeMeshSlots_;
    std::vector<DrawData> drawData;
    std::vector<MeshData> meshData;
    std::vector<assets::Bounds> aabbs;
    std::optional<AnimationPlayer> animationPlayer_;
    std::vector<AnimatedDrawState> animatedDraws_;
    std::vector<SkinPaletteState> skinPalettes_;
    std::vector<AnimationActorState> animationActors_;
    std::vector<AnimationPlayer::InstanceUpdatePolicy> animationUpdatePolicies_;
    std::vector<glm::mat4> jointMatrices_;

    std::vector<std::optional<Texture>> textures;
    std::vector<std::uint32_t> textureGenerations_;
    std::vector<std::uint32_t> freeTextureSlots_;
    std::vector<RetiredTexture> retiredTextures_;
    std::uint32_t liveTextureCount_ = 0;
    std::uint32_t materialCapacity_ = 0;
    std::uint32_t textureDescriptorCapacity_ = 0;
    std::uint64_t textureVersion_ = 1;
    std::vector<std::uint64_t> imageTextureVersions_;
    std::uint32_t sceneGeneration_ = 0;

    std::vector<VkDrawIndexedIndirectCommand> indirectCommands;
    std::array<DrawBatch, PipelineVariantCount> drawBatches{};
    std::array<std::vector<std::size_t>, PipelineVariantCount> visibleMeshScratch_;
    DrawBatch grassDrawBatch{};
    std::vector<std::size_t> visibleGrassScratch_;
    std::vector<Buffer> indirectCommandsBuffer;

    Buffer indexBuffer;
    Buffer vertexBuffer;
    std::uint64_t geometryVersion_ = 1;
    std::vector<std::uint64_t> imageGeometryVersions_;
    std::vector<RetiredGeometry> retiredGeometry_;

    std::vector<Buffer> matBuffers;
    std::vector<Buffer> transformBuffers;
    std::vector<Buffer> drawBuffers;
    std::vector<Buffer> jointBuffers;
};

} // namespace danvulkan::vk
