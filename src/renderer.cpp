#define _CRT_SECURE_NO_WARNINGS

#include <danvulkan/renderer.hpp>
#include <danvulkan/assets.hpp>

#include "glfw_platform.hpp"
#include "animation_player.hpp"
#include "descriptor_context.hpp"
#include "descriptor_planner.hpp"
#include "device_context.hpp"
#include "swapchain_context.hpp"
#include "upload_context.hpp"
#include "vulkan_raii.hpp"
#include "vulkan_result.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>


#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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
#include <tuple>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <filesystem> // hot reloading

#include "camera.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

struct DrawData
{
    int32_t materialIndex; // Index into material buffer
    int32_t transformIndex; // Index into transform buffer
    int32_t vertexOffset; // used to lookup attributes in vertex storage buffer
    int32_t jointOffset = -1; // -1 for rigid meshes, otherwise the first matrix in the joint palette

    // Gameplay data?
};

struct MeshData
{
    uint32_t indexCount;
    uint32_t firstIndex;
    uint32_t vertexOffset;
    uint32_t pipelineVariant;
    std::uint32_t meshResourceSlot;
    DrawData drawData;
    danvulkan::assets::Bounds localBounds;
};

struct MeshResourceData
{
    std::uint32_t indexCount = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t vertexOffset = 0;
    std::int32_t materialIndex = -1;
    std::uint32_t pipelineVariant = 0;
    danvulkan::assets::Bounds bounds{};
    std::string name;
};

struct RetiredGeometry
{
    std::uint64_t version = 0;
    danvulkan::vk::Buffer vertexBuffer;
    danvulkan::vk::Buffer indexBuffer;
};

struct GeometryRange
{
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
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

using Vertex = danvulkan::assets::Vertex;
static_assert(sizeof(Vertex) == 112, "CPU vertex layout must match shaders/vert.vert");

const uint32_t DrawDataCount = 2048, MatDataCount = 2048, TransformDataCount = 2048;
const uint32_t JointMatrixCount = 4096;

struct UniformBufferObject {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec4 cameraPositionTime;
    glm::vec4 lightPosition;
};

struct SkinnedDrawState
{
    danvulkan::assets::NodeHandle node;
    danvulkan::assets::SkinHandle skin;
    std::uint32_t transformIndex = 0;
    std::uint32_t jointOffset = 0;
    std::uint32_t meshDataIndex = 0;
};

struct AnimatedDrawState
{
    danvulkan::assets::NodeHandle node;
    std::uint32_t transformIndex = 0;
    std::uint32_t meshDataIndex = 0;
};
static_assert(sizeof(UniformBufferObject) == 160,
    "UniformBufferObject must match the shader std140 layout");

using AABB = danvulkan::assets::Bounds;

struct Texture
{
    danvulkan::vk::Image image;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkSampler sampler = VK_NULL_HANDLE;
    std::string name;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 1;
    danvulkan::assets::ColorSpace colorSpace = danvulkan::assets::ColorSpace::srgb;
    danvulkan::assets::TextureSampler samplerConfig;
};

struct PreparedSceneData
{
    std::uint32_t generation = 0;
    std::uint32_t vertexCapacity = 0;
    std::uint32_t indexCapacity = 0;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<GeometryRange> freeVertexRanges;
    std::vector<GeometryRange> freeIndexRanges;
    std::vector<MaterialData> materials;
    std::vector<std::string> materialNames;
    std::vector<MeshResourceData> meshResources;
    std::vector<TransformData> transforms;
    std::vector<std::string> instanceNames;
    std::vector<DrawData> draws;
    std::vector<MeshData> meshes;
    std::vector<AABB> bounds;
};

struct RetiredTexture
{
    std::uint64_t version = 0;
    Texture texture;
};

enum class PipelineVariant : std::uint32_t
{
    opaque = 0,
    opaqueDoubleSided,
    mask,
    maskDoubleSided,
    blend,
    blendDoubleSided,
    count
};

constexpr std::size_t PipelineVariantCount = static_cast<std::size_t>(PipelineVariant::count);

struct DrawBatch
{
    std::uint32_t firstCommand = 0;
    std::uint32_t commandCount = 0;
};

struct PointLight
{
    glm::vec3 position;
    float power;
    glm::vec3 color;
    float unused0;
};

struct FrameResources
{
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkFence inFlight = VK_NULL_HANDLE;
};

static std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

static void checkVk(VkResult result, std::string_view operation)
{
    danvulkan::vk::check(result, operation);
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
            platform_.reset();
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

        frameCpuBegin_ = std::chrono::steady_clock::now();
        platform_->pollEvents();
        if (shouldClose())
        {
            return false;
        }

        const auto animationTime = std::chrono::steady_clock::now();
        updateAnimation(std::min(
            std::chrono::duration<float>(animationTime - animationPreviousTime_).count(), 0.1f));
        animationPreviousTime_ = animationTime;

        pendingSubmission_.reset();
        frameInProgress_ = true;
        return true;
    }

    void submitScene(const SceneSubmission& submission)
    {
        requireFrameInProgress("submitScene");
        if (pendingSubmission_)
        {
            throw std::logic_error("submitScene may only be called once per frame");
        }
        pendingSubmission_ = submission;
    }

    void endFrame()
    {
        requireFrameInProgress("endFrame");
        if (!pendingSubmission_)
        {
            throw std::logic_error("endFrame requires one scene submission");
        }

        try
        {
            drawFrame(*pendingSubmission_);
            updateWindowTitle();
            ++renderedFrames_;
            pendingSubmission_.reset();
            frameInProgress_ = false;
        }
        catch (...)
        {
            pendingSubmission_.reset();
            frameInProgress_ = false;
            throw;
        }
    }

    [[nodiscard]] std::vector<SceneInstanceInfo> sceneInstances() const
    {
        requireInitialized("sceneInstances");
        std::vector<SceneInstanceInfo> result;
        result.reserve(transformData.size());
        for (std::size_t index = 0; index < transformData.size(); ++index)
        {
            if (!instanceAlive[index])
            {
                continue;
            }
            result.push_back({
                SceneInstanceHandle{ static_cast<std::uint32_t>(index),
                    instanceGenerations[index] },
                instanceNames[index],
                transformData[index].model
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<SceneMaterialInfo> sceneMaterials() const
    {
        requireInitialized("sceneMaterials");
        std::vector<SceneMaterialInfo> result;
        result.reserve(matData.size());
        for (std::size_t index = 0; index < matData.size(); ++index)
        {
            if (!materialAlive_[index])
            {
                continue;
            }
            result.push_back({
                SceneMaterialHandle{ static_cast<std::uint32_t>(index),
                    materialGenerations_[index] },
                materialNames[index],
                runtimeProperties(matData[index]),
                runtimeTextures(matData[index]),
                static_cast<danvulkan::assets::AlphaMode>(matData[index].materialFlags.y),
                matData[index].materialFlags.z != 0,
                matData[index].materialFlags.w != 0
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<SceneTextureInfo> sceneTextures() const
    {
        requireInitialized("sceneTextures");
        std::vector<SceneTextureInfo> result;
        result.reserve(textures.size());
        for (std::size_t index = 0; index < textures.size(); ++index)
        {
            if (!textures[index])
            {
                continue;
            }
            const Texture& texture = *textures[index];
            result.push_back({
                SceneTextureHandle{ static_cast<std::uint32_t>(index),
                    textureGenerations_[index] },
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
        if (!animationPlayer_)
        {
            return result;
        }
        result.reserve(animationPlayer_->clipCount());
        for (std::size_t index = 0; index < animationPlayer_->clipCount(); ++index)
        {
            result.push_back({
                SceneAnimationHandle{ static_cast<std::uint32_t>(index), sceneGeneration_ },
                std::string(animationPlayer_->clipName(index)),
                animationPlayer_->clipDuration(index)
            });
        }
        return result;
    }

    [[nodiscard]] AnimationPlaybackState animationPlaybackState() const
    {
        requireInitialized("animationPlaybackState");
        AnimationPlaybackState state;
        if (!animationPlayer_ || animationPlayer_->currentClip() ==
            danvulkan::AnimationPlayer::invalidClip)
        {
            return state;
        }
        const std::size_t selected = animationPlayer_->currentClip();
        state.clip = { static_cast<std::uint32_t>(selected), sceneGeneration_ };
        switch (animationPlayer_->status())
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
        state.positionSeconds = animationPlayer_->position();
        state.durationSeconds = animationPlayer_->clipDuration(selected);
        state.playbackSpeed = animationPlayer_->playbackSpeed();
        state.looping = animationPlayer_->looping();
        return state;
    }

    void playAnimation(SceneAnimationHandle animation, bool restart)
    {
        requireSceneUpdateAllowed("playAnimation");
        requireValidAnimation(animation, "playAnimation");
        animationPlayer_->play(animation.slot, restart);
        synchronizeAnimationPose();
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
        synchronizeAnimationPose();
    }

    void seekAnimation(float positionSeconds)
    {
        requireSceneUpdateAllowed("seekAnimation");
        requireAnimationPlayer("seekAnimation").seek(positionSeconds);
        synchronizeAnimationPose();
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
        if (textureVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("texture generation counter exhausted");
        }
        if (geometryVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("geometry buffer generation counter exhausted");
        }

        PreparedSceneData replacement = prepareReplacementScene(scene);
        retiredTextures_.reserve(retiredTextures_.size() + liveTextureCount_);
        retiredGeometry_.reserve(retiredGeometry_.size() + 1);

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
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, replacement.vertices.data(),
                "replacement scene vertex storage buffer");
            replacementIndexBuffer = createDeviceLocalBuffer(
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(replacement.indexCapacity),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, replacement.indices.data(),
                "replacement scene index buffer");
        }
        catch (...)
        {
            destroyReplacementSamplers();
            throw;
        }

        for (std::optional<Texture>& texture : textures)
        {
            if (texture)
            {
                retiredTextures_.push_back({ textureVersion_, std::move(*texture) });
            }
        }
        textures = std::move(replacementTextures);
        ++textureVersion_;
        textureGenerations_.assign(textures.size(), replacement.generation);
        freeTextureSlots_.clear();
        liveTextureCount_ = static_cast<std::uint32_t>(textures.size());

        retiredGeometry_.push_back({ geometryVersion_, std::move(vertexBuffer),
            std::move(indexBuffer) });
        vertexBuffer = std::move(replacementVertexBuffer);
        indexBuffer = std::move(replacementIndexBuffer);
        ++geometryVersion_;

        sceneGeneration_ = replacement.generation;
        vertexCapacity_ = replacement.vertexCapacity;
        indexCapacity_ = replacement.indexCapacity;
        vertices = std::move(replacement.vertices);
        indices = std::move(replacement.indices);
        freeVertexRanges_ = std::move(replacement.freeVertexRanges);
        freeIndexRanges_ = std::move(replacement.freeIndexRanges);
        retiredGeometryRanges_.clear();
        geometryRangeVersion_ = 1;
        imageGeometryRangeVersions_.assign(descriptors.setCount(), geometryRangeVersion_);

        matData = std::move(replacement.materials);
        materialNames = std::move(replacement.materialNames);
        materialGenerations_.assign(matData.size(), sceneGeneration_);
        materialAlive_.assign(matData.size(), true);
        freeMaterialSlots_.clear();

        meshResources = std::move(replacement.meshResources);
        meshGenerations_.assign(meshResources.size(), sceneGeneration_);
        meshAlive_.assign(meshResources.size(), true);
        freeMeshSlots_.clear();

        transformData = std::move(replacement.transforms);
        instanceNames = std::move(replacement.instanceNames);
        instanceGenerations.assign(transformData.size(), sceneGeneration_);
        instanceAlive.assign(transformData.size(), true);
        freeInstanceSlots.clear();
        drawData = std::move(replacement.draws);
        meshData = std::move(replacement.meshes);
        aabbs = std::move(replacement.bounds);
        animationPlayer_.reset();
        animatedDraws_.clear();
        skinnedDraws_.clear();
        jointMatrices_.clear();
    }

    [[nodiscard]] SceneTextureHandle uploadTexture(
        const danvulkan::assets::TextureAsset& source)
    {
        requireSceneUpdateAllowed("uploadTexture");
        if (freeTextureSlots_.empty() && textures.size() >= textureDescriptorCapacity_)
        {
            throw std::runtime_error("renderer bindless texture capacity has been reached");
        }
        if (textureVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("texture generation counter exhausted");
        }

        const std::uint32_t textureIndex = freeTextureSlots_.empty()
            ? static_cast<std::uint32_t>(textures.size())
            : freeTextureSlots_.back();
        Texture texture = createTextureImage(source);
        createTextureImageView(texture, textureIndex);
        createTextureSampler(texture, textureIndex);

        if (textureIndex == textures.size())
        {
            textures.emplace_back(std::move(texture));
            textureGenerations_.push_back(sceneGeneration_);
        }
        else
        {
            textures[textureIndex].emplace(std::move(texture));
            freeTextureSlots_.pop_back();
        }
        ++liveTextureCount_;
        ++textureVersion_;
        return { textureIndex, textureGenerations_[textureIndex] };
    }

    void destroyTexture(SceneTextureHandle textureHandle)
    {
        requireSceneUpdateAllowed("destroyTexture");
        requireValidTexture(textureHandle, "destroyTexture");
        if (liveTextureCount_ <= 1)
        {
            throw std::runtime_error("cannot destroy the renderer's last fallback texture");
        }

        const std::int32_t textureSlot = static_cast<std::int32_t>(textureHandle.slot);
        bool referenced = false;
        for (std::size_t index = 0; index < matData.size(); ++index)
        {
            if (!materialAlive_[index])
            {
                continue;
            }
            const MaterialData& material = matData[index];
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
        if (textureVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("texture generation counter exhausted");
        }

        retiredTextures_.emplace_back();
        RetiredTexture& retired = retiredTextures_.back();
        retired.version = textureVersion_;
        retired.texture = std::move(*textures[textureHandle.slot]);
        textures[textureHandle.slot].reset();
        textureGenerations_[textureHandle.slot] = nextGeneration(
            textureGenerations_[textureHandle.slot]);
        freeTextureSlots_.push_back(textureHandle.slot);
        --liveTextureCount_;
        ++textureVersion_;
    }

    [[nodiscard]] SceneMaterialHandle createMaterial(
        const RuntimeMaterialDescription& source)
    {
        requireSceneUpdateAllowed("createMaterial");
        if (freeMaterialSlots_.empty() && matData.size() >= materialCapacity_)
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

        const std::uint32_t materialIndex = freeMaterialSlots_.empty()
            ? static_cast<std::uint32_t>(matData.size())
            : freeMaterialSlots_.back();
        std::string materialName = source.name.empty()
            ? "runtime material " + std::to_string(materialIndex)
            : source.name;
        if (materialIndex == matData.size())
        {
            matData.push_back(material);
            materialNames.push_back(std::move(materialName));
            materialGenerations_.push_back(sceneGeneration_);
            materialAlive_.push_back(true);
        }
        else
        {
            matData[materialIndex] = material;
            materialNames[materialIndex] = std::move(materialName);
            materialAlive_[materialIndex] = true;
            freeMaterialSlots_.pop_back();
        }
        return { materialIndex, materialGenerations_[materialIndex] };
    }

    void destroyMaterial(SceneMaterialHandle materialHandle)
    {
        requireSceneUpdateAllowed("destroyMaterial");
        requireValidMaterial(materialHandle, "destroyMaterial");
        for (std::size_t index = 0; index < meshResources.size(); ++index)
        {
            if (meshAlive_[index] && meshResources[index].materialIndex ==
                    static_cast<std::int32_t>(materialHandle.slot))
            {
                throw std::runtime_error(
                    "cannot destroy a material while a mesh references it");
            }
        }

        matData[materialHandle.slot] = {};
        materialNames[materialHandle.slot].clear();
        materialAlive_[materialHandle.slot] = false;
        materialGenerations_[materialHandle.slot] = nextGeneration(
            materialGenerations_[materialHandle.slot]);
        freeMaterialSlots_.push_back(materialHandle.slot);
    }

    [[nodiscard]] std::vector<SceneMeshInfo> sceneMeshes() const
    {
        requireInitialized("sceneMeshes");
        std::vector<SceneMeshInfo> result;
        result.reserve(meshResources.size());
        for (std::size_t index = 0; index < meshResources.size(); ++index)
        {
            if (!meshAlive_[index])
            {
                continue;
            }
            const MeshResourceData& mesh = meshResources[index];
            result.push_back({
                SceneMeshHandle{ static_cast<std::uint32_t>(index), meshGenerations_[index] },
                mesh.name,
                SceneMaterialHandle{ static_cast<std::uint32_t>(mesh.materialIndex),
                    materialGenerations_[mesh.materialIndex] },
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
        if (freeMeshSlots_.empty() && meshResources.size() >=
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::runtime_error("renderer mesh handle capacity has been reached");
        }

        const std::uint32_t meshSlot = freeMeshSlots_.empty()
            ? static_cast<std::uint32_t>(meshResources.size())
            : freeMeshSlots_.back();
        const std::uint32_t vertexCount = static_cast<std::uint32_t>(sourceVertices.size());
        const std::uint32_t indexCount = static_cast<std::uint32_t>(sourceIndices.size());

        ensureGeometryCapacity(vertexCount, indexCount);
        const GeometryRange vertexRange = allocateGeometryRange(freeVertexRanges_, vertexCount);
        const GeometryRange indexRange = allocateGeometryRange(freeIndexRanges_, indexCount);
        std::ranges::copy(sourceVertices, vertices.begin() + vertexRange.offset);
        std::ranges::copy(sourceIndices, indices.begin() + indexRange.offset);
        try
        {
            uploadDeviceLocalBufferRange(vertexBuffer,
                sizeof(Vertex) * static_cast<VkDeviceSize>(vertexRange.offset),
                sizeof(Vertex) * static_cast<VkDeviceSize>(vertexRange.count),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sourceVertices.data(),
                "runtime vertex range");
            uploadDeviceLocalBufferRange(indexBuffer,
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(indexRange.offset),
                sizeof(std::uint32_t) * static_cast<VkDeviceSize>(indexRange.count),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, sourceIndices.data(),
                "runtime index range");
        }
        catch (...)
        {
            releaseGeometryRange(freeVertexRanges_, vertexRange);
            releaseGeometryRange(freeIndexRanges_, indexRange);
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
        const MaterialData& material = matData[materialHandle.slot];
        mesh.pipelineVariant = static_cast<std::uint32_t>(material.materialFlags.y * 2 +
            material.materialFlags.z);
        mesh.bounds = bounds;
        mesh.name = name.empty()
            ? "runtime mesh " + std::to_string(meshSlot)
            : std::move(name);
        if (meshSlot == meshResources.size())
        {
            meshResources.push_back(std::move(mesh));
            meshGenerations_.push_back(sceneGeneration_);
            meshAlive_.push_back(true);
        }
        else
        {
            meshResources[meshSlot] = std::move(mesh);
            meshAlive_[meshSlot] = true;
            freeMeshSlots_.pop_back();
        }

        return { meshSlot, meshGenerations_[meshSlot] };
    }

    void destroyMesh(SceneMeshHandle meshHandle)
    {
        requireSceneUpdateAllowed("destroyMesh");
        requireValidMesh(meshHandle, "destroyMesh");
        const bool referenced = std::ranges::any_of(meshData,
            [&](const MeshData& mesh)
            {
                return mesh.meshResourceSlot == meshHandle.slot;
            });
        if (referenced)
        {
            throw std::runtime_error("cannot destroy a mesh while an instance references it");
        }
        if (geometryRangeVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("geometry range retirement counter exhausted");
        }

        const MeshResourceData& resource = meshResources[meshHandle.slot];
        retiredGeometryRanges_.push_back({
            geometryRangeVersion_,
            { resource.vertexOffset, resource.vertexCount },
            { resource.firstIndex, resource.indexCount }
        });
        ++geometryRangeVersion_;
        meshResources[meshHandle.slot] = {};
        meshAlive_[meshHandle.slot] = false;
        meshGenerations_[meshHandle.slot] = nextGeneration(meshGenerations_[meshHandle.slot]);
        freeMeshSlots_.push_back(meshHandle.slot);
    }

    [[nodiscard]] SceneInstanceHandle createMeshInstance(SceneMeshHandle meshHandle,
        const glm::mat4& worldTransform, std::string name)
    {
        requireSceneUpdateAllowed("createMeshInstance");
        requireValidMesh(meshHandle, "createMeshInstance");
        if (meshData.size() >= DrawDataCount)
        {
            throw std::runtime_error("renderer instance draw capacity has been reached");
        }

        std::uint32_t instanceSlot = 0;
        if (!freeInstanceSlots.empty())
        {
            instanceSlot = freeInstanceSlots.back();
            freeInstanceSlots.pop_back();
        }
        else
        {
            if (transformData.size() >= TransformDataCount)
            {
                throw std::runtime_error("renderer instance transform capacity has been reached");
            }
            instanceSlot = static_cast<std::uint32_t>(transformData.size());
            transformData.push_back({});
            instanceNames.emplace_back();
            instanceGenerations.push_back(sceneGeneration_);
            instanceAlive.push_back(false);
        }

        transformData[instanceSlot].model = worldTransform;
        instanceNames[instanceSlot] = name.empty() ? meshResources[meshHandle.slot].name : std::move(name);
        instanceAlive[instanceSlot] = true;

        const MeshResourceData& resource = meshResources[meshHandle.slot];
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
        meshData.push_back(mesh);
        aabbs.push_back(transformedBounds(resource.bounds, worldTransform));

        return { instanceSlot, instanceGenerations[instanceSlot] };
    }

    void destroyInstance(SceneInstanceHandle instanceHandle)
    {
        requireSceneUpdateAllowed("destroyInstance");
        requireValidInstance(instanceHandle, "destroyInstance");

        for (std::size_t index = meshData.size(); index-- > 0;)
        {
            if (meshData[index].drawData.transformIndex ==
                static_cast<std::int32_t>(instanceHandle.slot))
            {
                meshData.erase(meshData.begin() + static_cast<std::ptrdiff_t>(index));
                aabbs.erase(aabbs.begin() + static_cast<std::ptrdiff_t>(index));
            }
        }

        instanceAlive[instanceHandle.slot] = false;
        instanceNames[instanceHandle.slot].clear();
        instanceGenerations[instanceHandle.slot] = nextGeneration(
            instanceGenerations[instanceHandle.slot]);
        freeInstanceSlots.push_back(instanceHandle.slot);
    }

    void updateInstanceTransform(SceneInstanceHandle instanceHandle, const glm::mat4& worldTransform)
    {
        requireSceneUpdateAllowed("updateInstanceTransform");
        requireValidInstance(instanceHandle, "updateInstanceTransform");

        transformData[instanceHandle.slot].model = worldTransform;
        for (std::size_t index = 0; index < meshData.size(); ++index)
        {
            if (meshData[index].drawData.transformIndex ==
                static_cast<std::int32_t>(instanceHandle.slot))
            {
                aabbs[index] = transformedBounds(meshData[index].localBounds, worldTransform);
            }
        }
    }

    void updateMaterialProperties(SceneMaterialHandle material,
        const RuntimeMaterialProperties& properties)
    {
        requireSceneUpdateAllowed("updateMaterialProperties");
        requireValidMaterial(material, "updateMaterialProperties");

        MaterialData& destination = matData[material.slot];
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

        MaterialData& destination = matData[materialHandle.slot];
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
    std::chrono::steady_clock::time_point frameCpuBegin_{};
    std::chrono::steady_clock::time_point animationPreviousTime_{};
    std::optional<SceneSubmission> pendingSubmission_;

    // Declared before all dependent Vulkan resources so they are destroyed last.
    danvulkan::vk::Instance instance;
    danvulkan::vk::DebugMessenger debugMessenger;
    danvulkan::vk::Surface surface;
    danvulkan::vk::DeviceContext device;
    danvulkan::vk::Allocator allocator;
    danvulkan::vk::SwapchainContext swapchain;
    danvulkan::vk::DescriptorContext descriptors;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
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
    std::vector<AABB> aabbs;
    std::optional<danvulkan::AnimationPlayer> animationPlayer_;
    std::vector<AnimatedDrawState> animatedDraws_;
    std::vector<SkinnedDrawState> skinnedDraws_;
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
    std::vector<PointLight> pointLights;
    std::uint32_t sceneGeneration_ = 0;

    std::vector<VkDrawIndexedIndirectCommand> indirectCommands;
    std::array<DrawBatch, PipelineVariantCount> drawBatches{};
    std::vector<danvulkan::vk::Buffer> indirectCommandsBuffer;

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
    

    VkQueryPool queryPoolTimestamp = VK_NULL_HANDLE;
    float timestampPeriod = 0.0f;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    
    std::array<VkPipeline, PipelineVariantCount> graphicsPipelines{};

    std::array<FrameResources, MAX_FRAMES_IN_FLIGHT> frames;
    danvulkan::vk::UploadContext uploadContext_;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> imagesInFlight;

    size_t currentFrame = 0;

    danvulkan::vk::Buffer indexBuffer;
    danvulkan::vk::Buffer vertexBuffer;
    std::uint64_t geometryVersion_ = 1;
    std::vector<std::uint64_t> imageGeometryVersions_;
    std::vector<RetiredGeometry> retiredGeometry_;

    std::vector<danvulkan::vk::Image> depthImages;
    std::vector<danvulkan::vk::Image> colorImages;

    // One per swap chain image
    std::vector<danvulkan::vk::Buffer> uniformBuffers;
    std::vector<danvulkan::vk::Buffer> matBuffers;
    std::vector<danvulkan::vk::Buffer> transformBuffers;
    std::vector<danvulkan::vk::Buffer> drawBuffers;
    std::vector<danvulkan::vk::Buffer> jointBuffers;

    Camera camera;
    glm::mat4 view;

    std::chrono::high_resolution_clock::time_point previousTime = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point lastTimeStamp = previousTime;
    glm::vec3 lightPos, lightSpeed, cameraStart, cameraStop;
    struct CameraDebugData
    {
        glm::vec3 forward;
        glm::vec3 right;
        glm::vec3 top;
    };

    std::array<float, 100> frameTimes;
    std::atomic_bool validationErrorSeen_ = false;
    float averageFrameTime = 0.0f;

    float culmativeDelta = 0.0f;

    double frameGpuAvg = 0.0;
    double frameCpuAvg = 0.0;

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
        msaaSamples = device.maxUsableSampleCount();
        createAllocator();
        const RendererFramebufferExtent framebuffer = platform_->framebufferExtent();
        swapchain.initialize(device, surface, { framebuffer.width, framebuffer.height },
            enableValidationLayers);
        createCommandPool();
        createUploadContext();
        createQueryPool();
        loadModel();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createColorResources();
        createDepthResources();
        createTextureImageViews();
        createTextureSamplers();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createBindlessBuffers();
        updateIndirectBuffer();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        //createIMGUI();
        initGame();
    }

    void recreateSwapChain()
    {
        // Handle minimized window, which has a framebuffer size of 0
        // we just pause until window in the foreground again
        RendererFramebufferExtent extent = platform_->framebufferExtent();
        while (extent.width == 0 || extent.height == 0)
        {
            platform_->waitEvents();
            extent = platform_->framebufferExtent();
        }
        vkDeviceWaitIdle(device); // don't touch resources that may still be in use
        const std::size_t previousImageCount = swapchain.imageCount();
        cleanupSwapChain(false);

        swapchain.recreate(device, surface, { extent.width, extent.height },
            enableValidationLayers);
        imagesInFlight.assign(swapchain.imageCount(), VK_NULL_HANDLE);
        if (swapchain.imageCount() != previousImageCount)
        {
            rebuildSwapchainIndexedResources();
        }
        camera.updateAspectRatio(
            swapchain.extent().width / static_cast<float>(swapchain.extent().height));
        createGraphicsPipeline();
        createDepthResources();
        createColorResources();
        createCommandBuffers();
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
        generations.insert(instanceGenerations.begin(), instanceGenerations.end());
        generations.insert(materialGenerations_.begin(), materialGenerations_.end());
        generations.insert(meshGenerations_.begin(), meshGenerations_.end());
        generations.insert(textureGenerations_.begin(), textureGenerations_.end());

        std::uint32_t candidate = nextGeneration(sceneGeneration_);
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

    [[nodiscard]] static bool hasGeometryRange(const std::vector<GeometryRange>& ranges,
        std::uint32_t count) noexcept
    {
        return std::ranges::any_of(ranges, [count](const GeometryRange& range)
        {
            return range.count >= count;
        });
    }

    [[nodiscard]] static GeometryRange allocateGeometryRange(
        std::vector<GeometryRange>& ranges, std::uint32_t count)
    {
        for (auto range = ranges.begin(); range != ranges.end(); ++range)
        {
            if (range->count < count)
            {
                continue;
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
        throw std::runtime_error("geometry range allocation failed after capacity planning");
    }

    static void releaseGeometryRange(std::vector<GeometryRange>& ranges,
        GeometryRange released)
    {
        if (released.count == 0)
        {
            return;
        }
        ranges.push_back(released);
        std::ranges::sort(ranges, {}, &GeometryRange::offset);

        std::vector<GeometryRange> merged;
        merged.reserve(ranges.size());
        for (const GeometryRange range : ranges)
        {
            if (merged.empty())
            {
                merged.push_back(range);
                continue;
            }
            GeometryRange& previous = merged.back();
            const std::uint64_t previousEnd = static_cast<std::uint64_t>(previous.offset) +
                previous.count;
            const std::uint64_t rangeEnd = static_cast<std::uint64_t>(range.offset) + range.count;
            if (range.offset > previousEnd)
            {
                merged.push_back(range);
                continue;
            }
            previous.count = static_cast<std::uint32_t>(
                std::max(previousEnd, rangeEnd) - previous.offset);
        }
        ranges = std::move(merged);
    }

    [[nodiscard]] static std::uint32_t grownGeometryCapacity(std::uint32_t current,
        std::uint32_t requiredContiguousCount, std::uint32_t maximum)
    {
        const std::uint64_t doubled = std::max<std::uint64_t>(1, current) * 2;
        const std::uint64_t required = static_cast<std::uint64_t>(current) +
            requiredContiguousCount;
        const std::uint64_t result = std::max(doubled, required);
        if (result > maximum)
        {
            throw std::runtime_error("geometry capacity exceeds renderer offset limits");
        }
        return static_cast<std::uint32_t>(result);
    }

    void requireValidInstance(SceneInstanceHandle instanceHandle,
        std::string_view operation) const
    {
        if (instanceHandle.slot >= transformData.size() ||
            !instanceAlive[instanceHandle.slot] ||
            instanceHandle.generation != instanceGenerations[instanceHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale instance handle");
        }
    }

    void requireValidTexture(SceneTextureHandle textureHandle,
        std::string_view operation) const
    {
        if (textureHandle.slot >= textures.size() ||
            !textures[textureHandle.slot] ||
            textureHandle.slot >= textureGenerations_.size() ||
            textureHandle.generation != textureGenerations_[textureHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale texture handle");
        }
    }

    void requireValidMaterial(SceneMaterialHandle materialHandle,
        std::string_view operation) const
    {
        if (materialHandle.slot >= matData.size() ||
            materialHandle.slot >= materialAlive_.size() ||
            !materialAlive_[materialHandle.slot] ||
            materialHandle.slot >= materialGenerations_.size() ||
            materialHandle.generation != materialGenerations_[materialHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale material handle");
        }
    }

    void requireValidMesh(SceneMeshHandle meshHandle,
        std::string_view operation) const
    {
        if (meshHandle.slot >= meshResources.size() ||
            meshHandle.slot >= meshAlive_.size() ||
            !meshAlive_[meshHandle.slot] ||
            meshHandle.slot >= meshGenerations_.size() ||
            meshHandle.generation != meshGenerations_[meshHandle.slot])
        {
            throw std::invalid_argument(std::string(operation) +
                " received an invalid or stale mesh handle");
        }
    }

    [[nodiscard]] danvulkan::AnimationPlayer& requireAnimationPlayer(
        std::string_view operation)
    {
        if (!animationPlayer_ || animationPlayer_->clipCount() == 0)
        {
            throw std::logic_error(std::string(operation) +
                " requires a scene containing animation clips");
        }
        return *animationPlayer_;
    }

    void requireValidAnimation(SceneAnimationHandle animation,
        std::string_view operation) const
    {
        if (!animationPlayer_ || animation.generation != sceneGeneration_ ||
            animation.slot >= animationPlayer_->clipCount())
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
        if (textureIndex < 0 || static_cast<std::size_t>(textureIndex) >= textures.size() ||
            !textures[textureIndex] || static_cast<std::size_t>(textureIndex) >=
                textureGenerations_.size())
        {
            return {};
        }
        return { static_cast<std::uint32_t>(textureIndex), textureGenerations_[textureIndex] };
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
        if (texture.slot >= textures.size() || !textures[texture.slot] ||
            texture.slot >= textureGenerations_.size() ||
            texture.generation != textureGenerations_[texture.slot])
        {
            throw std::invalid_argument("invalid or stale " + std::string(role) +
                " texture handle");
        }
        return static_cast<std::int32_t>(texture.slot);
    }

    [[nodiscard]] static AABB transformedBounds(const AABB& bounds, const glm::mat4& transform)
    {
        const float maximum = std::numeric_limits<float>::max();
        AABB result{ glm::vec3(maximum), glm::vec3(-maximum) };
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

    void updateWindowTitle()
    {
        const double frameCpuMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frameCpuBegin_).count();
        frameCpuAvg = frameCpuAvg * 0.95 + frameCpuMilliseconds * 0.05;
        const double currentFps = frameCpuAvg > 0.0 ? 1000.0 / frameCpuAvg : 0.0;
        char title[256];
        sprintf(title, "DanVulkan cpu %.2f ms; gpu: %.2f ms, FPS: %.2f", frameCpuAvg,
            frameGpuAvg, currentFps);
        platform_->setWindowTitle(title);
    }

    [[nodiscard]] SceneSubmission demoSceneSubmission() const
    {
        SceneSubmission submission;
        submission.view = camera.matrices.view;
        submission.projection = camera.matrices.perspective;
        submission.cameraPosition = glm::vec3(glm::inverse(camera.matrices.view)[3]);
        submission.lightPosition = lightPos;
        return submission;
    }

    void shutdownNoThrow() noexcept
    {
        if (!initialized_)
        {
            return;
        }

        frameInProgress_ = false;
        pendingSubmission_.reset();
        vkDeviceWaitIdle(device);
        cleanup();
        initialized_ = false;
        shutdown_ = true;
    }

    void cleanupSwapChain(bool releaseSwapchain = true)
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkFreeCommandBuffers(device, frames[i].commandPool, 1, &frames[i].commandBuffer);
        }
        for (VkPipeline& pipeline : graphicsPipelines)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        colorImages.clear();
        depthImages.clear();
        if (releaseSwapchain)
        {
            swapchain.reset();
        }
    }
    void cleanup() 
    {
        cleanupSwapChain();
        descriptors.reset();
        vkDestroyQueryPool(device, queryPoolTimestamp, nullptr);
        matBuffers.clear();
        drawBuffers.clear();
        transformBuffers.clear();
        jointBuffers.clear();
        uniformBuffers.clear();
        indirectCommandsBuffer.clear();
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
        textureGenerations_.clear();
        freeTextureSlots_.clear();
        retiredTextures_.clear();
        liveTextureCount_ = 0;
        imageTextureVersions_.clear();
        retiredGeometry_.clear();
        imageGeometryVersions_.clear();
        retiredGeometryRanges_.clear();
        imageGeometryRangeVersions_.clear();
        freeVertexRanges_.clear();
        freeIndexRanges_.clear();
        vertexBuffer.reset();
        indexBuffer.reset();
        
        
        destroySwapchainSyncObjects();
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(device, frames[i].imageAvailable, nullptr);
            vkDestroyFence(device, frames[i].inFlight, nullptr);
            vkDestroyCommandPool(device, frames[i].commandPool, nullptr);
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
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkFreeCommandBuffers(device, frames[i].commandPool, 1, &frames[i].commandBuffer);
        }        
        for (VkPipeline& pipeline : graphicsPipelines)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;

        createGraphicsPipeline();
        //createColorResources();
        //createDepthResources();
        createCommandBuffers();

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

        VkDebugUtilsMessengerCreateInfoEXT createInfo{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
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
        VkImageViewCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
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

    void createGraphicsPipeline()
    {
        auto [vertCreateInfo, vertShader] = loadShader(COMPILED_SHADER_PATH + "vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        auto  [fragCreateInfo, fragShader] = loadShader(COMPILED_SHADER_PATH + "frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertCreateInfo, fragCreateInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInputInfo.vertexBindingDescriptionCount = 0; // 1
        vertexInputInfo.pVertexBindingDescriptions = nullptr;// &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;//   static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology =  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.pViewports = nullptr;
        viewportState.scissorCount = 1;
        viewportState.pScissors = nullptr;

        constexpr std::array dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.sampleShadingEnable = VK_TRUE;
        multisampling.rasterizationSamples = msaaSamples;
        multisampling.minSampleShading = 0.2f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        // Dynamic State goes hereeeeeee

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        const VkDescriptorSetLayout descriptorLayout = descriptors.layout();
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        checkVk(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
                "vkCreatePipelineLayout");
        setDebugName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipelineLayout, "main pipeline layout");

        const VkFormat colorFormat = swapchain.format();
        const VkFormat depthFormat = findDepthFormat();
        VkPipelineRenderingCreateInfo renderingInfo{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;
        renderingInfo.depthAttachmentFormat = depthFormat;
        if (hasStencilComponent(depthFormat))
        {
            renderingInfo.stencilAttachmentFormat = depthFormat;
        }

        VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = VK_NULL_HANDLE;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // optional
        pipelineInfo.basePipelineIndex = -1; // optional
        pipelineInfo.pDepthStencilState = &depthStencil;

        for (std::size_t index = 0; index < graphicsPipelines.size(); ++index)
        {
            const bool doubleSided = (index % 2U) != 0;
            const bool blended = index >= static_cast<std::size_t>(PipelineVariant::blend);

            rasterizer.cullMode = doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
            colorBlendAttachment.blendEnable = blended ? VK_TRUE : VK_FALSE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            depthStencil.depthWriteEnable = blended ? VK_FALSE : VK_TRUE;

            checkVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                        &graphicsPipelines[index]),
                    "vkCreateGraphicsPipelines(material variant)");
            setDebugName(VK_OBJECT_TYPE_PIPELINE, graphicsPipelines[index],
                "material pipeline variant " + std::to_string(index));
        }

        vkDestroyShaderModule(device, fragShader, nullptr);
        vkDestroyShaderModule(device, vertShader, nullptr);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        // We don't delete shaderModule after creating pipeline, this could be bad?
        VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        checkVk(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule), "vkCreateShaderModule");

        return shaderModule;
    }

    std::tuple<VkPipelineShaderStageCreateInfo, VkShaderModule> loadShader(const std::string& filename, VkShaderStageFlagBits stage)
    {
        auto code = readFile(filename);
        VkShaderModule shaderModule = createShaderModule(code);
        setDebugName(VK_OBJECT_TYPE_SHADER_MODULE, shaderModule, filename);

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        shaderStageCreateInfo.stage = stage;
        shaderStageCreateInfo.module = shaderModule;
        shaderStageCreateInfo.pName = "main";

        return { shaderStageCreateInfo, shaderModule };
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.queueFamilyIndex = device.queueFamilies().graphicsFamily.value();
        poolInfo.flags = 0; // optional
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            checkVk(vkCreateCommandPool(device, &poolInfo, nullptr, &frames[i].commandPool),
                    "vkCreateCommandPool");
            setDebugName(VK_OBJECT_TYPE_COMMAND_POOL, frames[i].commandPool,
                "frame command pool " + std::to_string(i));
        }
    }

    void createQueryPool()
    {
        VkQueryPoolCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        createInfo.queryCount = 2;

        checkVk(vkCreateQueryPool(device, &createInfo, nullptr, &queryPoolTimestamp), "vkCreateQueryPool");
        setDebugName(VK_OBJECT_TYPE_QUERY_POOL, queryPoolTimestamp, "frame timestamp queries");
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

    bool hasStencilComponent(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createColorResources()
    {
        if (msaaSamples == VK_SAMPLE_COUNT_1_BIT)
        {
            colorImages.clear();
            return;
        }

        colorImages.resize(swapchain.imageCount());
        for (size_t i = 0; i < colorImages.size(); ++i)
        {
            colorImages[i] = createImage(swapchain.extent().width, swapchain.extent().height,
                msaaSamples, swapchain.format(), VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "MSAA color image " + std::to_string(i));
            colorImages[i].setView(createImageView(colorImages[i], swapchain.format(),
                VK_IMAGE_ASPECT_COLOR_BIT, "MSAA color image view " + std::to_string(i)));
        }
    }

    void createUploadContext()
    {
        uploadContext_.initialize(device, allocator, device.graphicsQueue(),
            device.queueFamilies().graphicsFamily.value(), enableValidationLayers);
    }

    void createDepthResources()
    {
        const VkFormat depthFormat = findDepthFormat();
        depthImages.resize(swapchain.imageCount());
        for (size_t i = 0; i < depthImages.size(); ++i)
        {
            depthImages[i] = createImage(swapchain.extent().width, swapchain.extent().height,
                msaaSamples,
                depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "depth image " + std::to_string(i));
            VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (hasStencilComponent(depthFormat))
            {
                aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            depthImages[i].setView(createImageView(depthImages[i], depthFormat, aspect,
                "depth image view " + std::to_string(i)));
        }
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
        VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
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
        for (std::uint32_t index = 0; index < textures.size(); ++index)
        {
            if (textures[index])
            {
                createTextureImageView(*textures[index], index);
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
        for (std::size_t index = 0; index < textures.size(); ++index)
        {
            if (textures[index])
            {
                createTextureSampler(*textures[index], static_cast<std::uint32_t>(index));
            }
        }
    }

    void createTextureSampler(Texture& texture, std::uint32_t textureIndex)
    {
        const VkPhysicalDeviceProperties& properties = device.properties();
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
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

    void createCommandBuffers()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocInfo.commandPool = frames[i].commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1; // TODO this is just a guess

            checkVk(vkAllocateCommandBuffers(device, &allocInfo, &frames[i].commandBuffer),
                    "vkAllocateCommandBuffers(frame)");
            setDebugName(VK_OBJECT_TYPE_COMMAND_BUFFER, frames[i].commandBuffer,
                "frame command buffer " + std::to_string(i));
        }

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
        VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
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

        VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    void updateCommandBuffer(uint32_t currentFrameIndex, uint32_t imageIndex)
    {
        VkCommandBuffer commandBuffer = frames[currentFrameIndex].commandBuffer;
        checkVk(vkResetCommandPool(device, frames[currentFrameIndex].commandPool, 0),
                "vkResetCommandPool");

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer(frame)");
        
        vkCmdResetQueryPool(commandBuffer, queryPoolTimestamp, 0, 2);
        vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 0);

        transitionImage(commandBuffer, swapchain.images()[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        if (msaaSamples != VK_SAMPLE_COUNT_1_BIT)
        {
            transitionImage(commandBuffer, colorImages[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        }

        const VkFormat depthFormat = findDepthFormat();
        VkImageAspectFlags depthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (hasStencilComponent(depthFormat))
        {
            depthAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        transitionImage(commandBuffer, depthImages[imageIndex], depthAspect,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = msaaSamples == VK_SAMPLE_COUNT_1_BIT
            ? swapchain.imageViews()[imageIndex]
            : colorImages[imageIndex].view();
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = msaaSamples == VK_SAMPLE_COUNT_1_BIT
            ? VK_ATTACHMENT_STORE_OP_STORE
            : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
        if (msaaSamples != VK_SAMPLE_COUNT_1_BIT)
        {
            colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            colorAttachment.resolveImageView = swapchain.imageViews()[imageIndex];
            colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthImages[imageIndex].view();
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, swapchain.extent() };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;
        if (hasStencilComponent(depthFormat))
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
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        const VkDescriptorSet descriptorSet = descriptors.set(imageIndex);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        for (std::size_t variant = 0; variant < PipelineVariantCount; ++variant)
        {
            const DrawBatch& batch = drawBatches[variant];
            if (batch.commandCount == 0)
            {
                continue;
            }
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                graphicsPipelines[variant]);
            const VkDeviceSize offset = static_cast<VkDeviceSize>(batch.firstCommand) *
                sizeof(VkDrawIndexedIndirectCommand);
            vkCmdDrawIndexedIndirect(commandBuffer, indirectCommandsBuffer[imageIndex], offset,
                batch.commandCount, sizeof(VkDrawIndexedIndirectCommand));
        }
        vkCmdEndRendering(commandBuffer);

        transitionImage(commandBuffer, swapchain.images()[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
        vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, queryPoolTimestamp, 1);

        checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer(frame)");
    }

    void createSyncObjects()
    {
        VkSemaphoreCreateInfo semaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };

        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            checkVk(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &frames[i].imageAvailable),
                    "vkCreateSemaphore(image available)");
            checkVk(vkCreateFence(device, &fenceInfo, nullptr, &frames[i].inFlight),
                    "vkCreateFence(frame in flight)");
            setDebugName(VK_OBJECT_TYPE_SEMAPHORE, frames[i].imageAvailable,
                "image available semaphore " + std::to_string(i));
            setDebugName(VK_OBJECT_TYPE_FENCE, frames[i].inFlight,
                "in-flight fence " + std::to_string(i));
        }
        createSwapchainSyncObjects();
    }

    void createSwapchainSyncObjects()
    {
        renderFinishedSemaphores.resize(swapchain.imageCount());
        imagesInFlight.assign(swapchain.imageCount(), VK_NULL_HANDLE);
        VkSemaphoreCreateInfo semaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i)
        {
            checkVk(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]),
                    "vkCreateSemaphore(render finished)");
            setDebugName(VK_OBJECT_TYPE_SEMAPHORE, renderFinishedSemaphores[i],
                "render finished semaphore " + std::to_string(i));
        }
    }

    void drawFrame(const SceneSubmission& submission)
    {
        FrameResources& frame = frames[currentFrame];
        checkVk(vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX),
                "vkWaitForFences(frame)");
        const bool framebufferResized = platform_->consumeFramebufferResize();

        uint32_t imageIndex;


        VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, frame.imageAvailable,
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

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            checkVk(vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX),
                    "vkWaitForFences(swapchain image)");
        }
        prepareGeometryForImage(imageIndex);
        prepareGeometryRangesForImage(imageIndex);
        prepareTexturesForImage(imageIndex);
        imagesInFlight[imageIndex] = frame.inFlight;

        updateUniformBuffer(imageIndex, submission);
        updateCommandBuffer(static_cast<uint32_t>(currentFrame), imageIndex);

        VkSemaphoreSubmitInfo waitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        waitInfo.semaphore = frame.imageAvailable;
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkCommandBufferSubmitInfo commandBufferInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        commandBufferInfo.commandBuffer = frame.commandBuffer;

        VkSemaphoreSubmitInfo signalInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        signalInfo.semaphore = renderFinishedSemaphores[imageIndex];
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.waitSemaphoreInfoCount = 1;
        submitInfo.pWaitSemaphoreInfos = &waitInfo;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &signalInfo;
       
        checkVk(vkResetFences(device, 1, &frame.inFlight), "vkResetFences");
        checkVk(vkQueueSubmit2(device.graphicsQueue(), 1, &submitInfo, frame.inFlight),
                "vkQueueSubmit2(frame)");

        VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];

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
        // GPU timing
        uint64_t timestampResults[2] = {};
        checkVk(vkGetQueryPoolResults(device, queryPoolTimestamp, 0, 2, sizeof(timestampResults),
            timestampResults, sizeof(timestampResults[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults");
        double frameGpuBegin = double(timestampResults[0]) * timestampPeriod * 1e-6;
        double frameGpuEnd = double(timestampResults[1]) * timestampPeriod * 1e-6;


        frameGpuAvg = frameGpuAvg * 0.95 + (frameGpuEnd - frameGpuBegin) * 0.05;
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

    void updateAnimation(float deltaSeconds)
    {
        if (!animationPlayer_)
        {
            return;
        }
        animationPlayer_->update(deltaSeconds);
        synchronizeAnimationPose();
    }

    void synchronizeAnimationPose()
    {
        if (!animationPlayer_)
        {
            return;
        }
        for (const AnimatedDrawState& state : animatedDraws_)
        {
            if (state.transformIndex >= transformData.size() ||
                state.meshDataIndex >= meshData.size() || state.meshDataIndex >= aabbs.size())
            {
                throw std::runtime_error("animated renderer state is inconsistent");
            }
            const glm::mat4& world = animationPlayer_->worldTransform(state.node);
            transformData[state.transformIndex].model = world;
            aabbs[state.meshDataIndex] = transformedBounds(
                meshData[state.meshDataIndex].localBounds, world);
        }
        jointMatrices_.clear();
        for (const SkinnedDrawState& state : skinnedDraws_)
        {
            if (jointMatrices_.size() != state.jointOffset)
            {
                throw std::runtime_error("animated renderer state is inconsistent");
            }
            animationPlayer_->appendSkinMatrices(state.skin, state.node, jointMatrices_);
        }
        if (jointMatrices_.size() > JointMatrixCount)
        {
            throw std::runtime_error("animated scene exceeds the renderer joint matrix capacity");
        }
    }

    void destroySwapchainSyncObjects() noexcept
    {
        for (VkSemaphore semaphore : renderFinishedSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        renderFinishedSemaphores.clear();
        imagesInFlight.clear();
    }

    void rebuildSwapchainIndexedResources()
    {
        destroySwapchainSyncObjects();
        descriptors.resetSets();
        uniformBuffers.clear();
        matBuffers.clear();
        transformBuffers.clear();
        drawBuffers.clear();
        jointBuffers.clear();
        indirectCommandsBuffer.clear();

        // Recreation waits for device idle, so every deferred generation is complete.
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

        createUniformBuffers();
        createBindlessBuffers();
        createDescriptorSets();
        createSwapchainSyncObjects();
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
        ubo.cameraPositionTime = glm::vec4(submission.cameraPosition, 1.0f);
        ubo.lightPosition = glm::vec4(submission.lightPosition, 1.0f);
        
        cullAABB(submission.view, ubo.projection, submission.cameraPosition);

        // Update to GPU
        // NOTE: this is a hot area for code performance
        // UBO
        writeBuffer(uniformBuffers[currentImage], &ubo, sizeof(ubo));
        // Transform
        writeBuffer(transformBuffers[currentImage], transformData.data(), sizeof(TransformData) * transformData.size());
        // Material Data
        writeBuffer(matBuffers[currentImage], matData.data(), sizeof(MaterialData) * matData.size());
        // Draw Data
        writeBuffer(drawBuffers[currentImage], drawData.data(), sizeof(DrawData) * drawData.size());
        if (!jointMatrices_.empty())
        {
            writeBuffer(jointBuffers[currentImage], jointMatrices_.data(),
                sizeof(glm::mat4) * jointMatrices_.size());
        }
        // indirect
        writeBuffer(indirectCommandsBuffer[currentImage], indirectCommands.data(),
            sizeof(VkDrawIndexedIndirectCommand) * indirectCommands.size());
    }

    void cullAABB(const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix,
        const glm::vec3& cameraPosition)
    {
        const glm::mat4 viewProjection = glm::transpose(projectionMatrix * viewMatrix);
        std::array<glm::vec4, 6> planes{
            viewProjection[3] + viewProjection[0],
            viewProjection[3] - viewProjection[0],
            viewProjection[3] - viewProjection[1],
            viewProjection[3] + viewProjection[1],
            viewProjection[3] + viewProjection[2],
            viewProjection[3] - viewProjection[2]
        };
        for (glm::vec4& plane : planes)
        {
            const float length = glm::length(glm::vec3(plane));
            if (length > 0.0f)
            {
                plane /= length;
            }
        }

        drawData.clear();
        indirectCommands.clear();
        drawBatches.fill({});
        std::array<std::vector<std::size_t>, PipelineVariantCount> visibleMeshes;
        for (size_t i = 0; i < aabbs.size(); i++)
        {
            bool cull = false;
            for (int planeID = 0; planeID < 6; ++planeID)
            {
                const glm::vec3 planeNormal = planes[planeID];
                const float planeConstant = planes[planeID].w;
                // check each axis to get the AABB vertex further away from the direction plane is facing (plane normal)
                glm::vec3 axisVert;

                // add position to the aabb here, we're all 0 here tho.
                if (planeNormal.x < 0.0f)
                    axisVert.x = aabbs[i].minVertex.x;
                else
                    axisVert.x = aabbs[i].maxVertex.x;

                if (planeNormal.y < 0.0)
                    axisVert.y = aabbs[i].minVertex.y;
                else
                    axisVert.y = aabbs[i].maxVertex.y;
                
                if (planeNormal.z < 0.0)
                    axisVert.z = aabbs[i].minVertex.z;
                else
                    axisVert.z = aabbs[i].maxVertex.z;

                if (glm::dot(planeNormal, axisVert) + planeConstant < 0.0f)
                {
                    cull = true;
                    break;
                }
            }
            if (!cull)
            {
                visibleMeshes[meshData[i].pipelineVariant].push_back(i);
            }
        }

        const auto distanceSquared = [&](std::size_t meshIndex)
        {
            const glm::vec3 center = (aabbs[meshIndex].minVertex + aabbs[meshIndex].maxVertex) * 0.5f;
            const glm::vec3 offset = center - cameraPosition;
            return glm::dot(offset, offset);
        };
        for (std::size_t variant = static_cast<std::size_t>(PipelineVariant::blend);
             variant < PipelineVariantCount; ++variant)
        {
            std::ranges::sort(visibleMeshes[variant], [&](std::size_t left, std::size_t right)
            {
                return distanceSquared(left) > distanceSquared(right);
            });
        }

        for (std::size_t variant = 0; variant < PipelineVariantCount; ++variant)
        {
            DrawBatch& batch = drawBatches[variant];
            batch.firstCommand = static_cast<std::uint32_t>(indirectCommands.size());
            for (const std::size_t meshIndex : visibleMeshes[variant])
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
            batch.commandCount = static_cast<std::uint32_t>(indirectCommands.size()) - batch.firstCommand;
        }
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
        VkDebugUtilsObjectNameInfoEXT nameInfo{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = debugHandleValue(handle);
        nameInfo.pObjectName = terminatedName.c_str();
        checkVk(setName(device, &nameInfo), "vkSetDebugUtilsObjectNameEXT");
    }

    danvulkan::vk::Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, bool persistentlyMapped, std::string_view name)
    {
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
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
        const void* bufferData, std::string_view name)
    {
        auto destination = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, name);
        uploadContext_.uploadBuffer(destination, 0, bufferData, size, usage, name);
        return destination;
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

    void createVertexBuffer()
    {
        const VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vertexBuffer = createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vertices.data(), "scene vertex storage buffer");
    }

    void createIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        indexBuffer = createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            indices.data(), "scene index buffer");
    }

    void ensureGeometryCapacity(std::uint32_t vertexCount, std::uint32_t indexCount)
    {
        const bool growVertices = !hasGeometryRange(freeVertexRanges_, vertexCount);
        const bool growIndices = !hasGeometryRange(freeIndexRanges_, indexCount);
        if (!growVertices && !growIndices)
        {
            return;
        }
        if (geometryVersion_ == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::runtime_error("geometry buffer generation counter exhausted");
        }

        const std::uint32_t newVertexCapacity = growVertices
            ? grownGeometryCapacity(vertexCapacity_, vertexCount,
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
            : vertexCapacity_;
        const std::uint32_t newIndexCapacity = growIndices
            ? grownGeometryCapacity(indexCapacity_, indexCount,
                std::numeric_limits<std::uint32_t>::max())
            : indexCapacity_;

        const VkPhysicalDeviceProperties& physicalDeviceProperties = device.properties();
        if (sizeof(Vertex) * static_cast<VkDeviceSize>(newVertexCapacity) >
            physicalDeviceProperties.limits.maxStorageBufferRange)
        {
            throw std::runtime_error("grown vertex capacity exceeds maxStorageBufferRange");
        }

        std::vector<Vertex> expandedVertices = vertices;
        expandedVertices.resize(newVertexCapacity);
        std::vector<std::uint32_t> expandedIndices = indices;
        expandedIndices.resize(newIndexCapacity);

        auto newVertexBuffer = createDeviceLocalBuffer(
            sizeof(Vertex) * static_cast<VkDeviceSize>(newVertexCapacity),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, expandedVertices.data(),
            "grown scene vertex storage buffer");
        auto newIndexBuffer = createDeviceLocalBuffer(
            sizeof(std::uint32_t) * static_cast<VkDeviceSize>(newIndexCapacity),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, expandedIndices.data(),
            "grown scene index buffer");

        retiredGeometry_.emplace_back();
        RetiredGeometry& retired = retiredGeometry_.back();
        retired.version = geometryVersion_;
        retired.vertexBuffer = std::move(vertexBuffer);
        retired.indexBuffer = std::move(indexBuffer);
        vertexBuffer = std::move(newVertexBuffer);
        indexBuffer = std::move(newIndexBuffer);
        ++geometryVersion_;

        vertices = std::move(expandedVertices);
        indices = std::move(expandedIndices);
        if (newVertexCapacity > vertexCapacity_)
        {
            releaseGeometryRange(freeVertexRanges_,
                { vertexCapacity_, newVertexCapacity - vertexCapacity_ });
        }
        if (newIndexCapacity > indexCapacity_)
        {
            releaseGeometryRange(freeIndexRanges_,
                { indexCapacity_, newIndexCapacity - indexCapacity_ });
        }
        vertexCapacity_ = newVertexCapacity;
        indexCapacity_ = newIndexCapacity;
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
        matBuffers.resize(swapchain.imageCount());

        transformBuffers.resize(swapchain.imageCount());

        drawBuffers.resize(swapchain.imageCount());

        jointBuffers.resize(swapchain.imageCount());


        indirectCommandsBuffer.resize(swapchain.imageCount());

        VkDeviceSize matBufferSize = sizeof(MaterialData) * materialCapacity_;
        VkDeviceSize transformBufferSize = sizeof(TransformData) * TransformDataCount;
        VkDeviceSize drawBufferSize = sizeof(DrawData) * DrawDataCount;
        VkDeviceSize jointBufferSize = sizeof(glm::mat4) * JointMatrixCount;
        VkDeviceSize indirectBufferSize = sizeof(VkDrawIndexedIndirectCommand) * DrawDataCount;

        for (size_t i = 0; i < swapchain.imageCount(); i++)
        {
            // Material Data
            matBuffers[i] = createBuffer(matBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "material buffer " + std::to_string(i));

            // Transform
            transformBuffers[i] = createBuffer(transformBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "transform buffer " + std::to_string(i));

            // DrawData
            drawBuffers[i] = createBuffer(drawBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "draw buffer " + std::to_string(i));

            jointBuffers[i] = createBuffer(jointBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "joint matrix buffer " + std::to_string(i));

            // Indirect Draw
            indirectCommandsBuffer[i] = createBuffer(indirectBufferSize, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, true,
                "indirect draw buffer " + std::to_string(i));

        }

    }

    void updateIndirectBuffer()
    {
       // VkDeviceSize bufferSize = sizeof(VkDrawIndexedIndirectCommand) * indirectCommands.size();
        //createDeviceLocalBuffer(bufferSize, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, indirectCommandsBuffer[0], indirectCommandsBufferMemory[0], indirectCommands.data());
    }

    void createDescriptorSetLayout()
    {
        const VkPhysicalDeviceProperties& properties = device.properties();
        const std::optional<std::uint32_t> capacity =
            danvulkan::vk::selectTextureDescriptorCapacity({
                config_.maxTextures,
                liveTextureCount_,
                properties.limits.maxPerStageDescriptorSamplers,
                properties.limits.maxDescriptorSetSamplers
            });
        if (!capacity)
        {
            throw std::runtime_error(
                "configured bindless texture capacity is invalid for this scene or device");
        }
        textureDescriptorCapacity_ = *capacity;
        textures.reserve(textureDescriptorCapacity_);
        textureGenerations_.reserve(textureDescriptorCapacity_);
        freeTextureSlots_.reserve(textureDescriptorCapacity_);
        descriptors.initialize(device, textureDescriptorCapacity_, enableValidationLayers);
    }

    [[nodiscard]] std::vector<VkDescriptorImageInfo> textureDescriptorInfos() const
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

        std::vector<VkDescriptorImageInfo> imageInfo(textureDescriptorCapacity_);
        for (std::uint32_t textureIndex = 0; textureIndex < textureDescriptorCapacity_; ++textureIndex)
        {
            const Texture& texture = textureIndex < textures.size() && textures[textureIndex]
                ? *textures[textureIndex]
                : *fallback;
            imageInfo[textureIndex].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo[textureIndex].imageView = texture.image.view();
            imageInfo[textureIndex].sampler = texture.sampler;
        }
        return imageInfo;
    }

    void createDescriptorSets()
    {
        const std::size_t setCount = swapchain.imageCount();
        if (uniformBuffers.size() != setCount || matBuffers.size() != setCount ||
            drawBuffers.size() != setCount || transformBuffers.size() != setCount ||
            jointBuffers.size() != setCount)
        {
            throw std::runtime_error(
                "swapchain descriptor buffer counts are inconsistent");
        }
        std::vector<danvulkan::vk::DescriptorSetBindings> bindings(setCount);
        for (std::size_t index = 0; index < setCount; ++index)
        {
            bindings[index] = {
                { uniformBuffers[index], 0, sizeof(UniformBufferObject) },
                { matBuffers[index], 0, sizeof(MaterialData) * materialCapacity_ },
                { drawBuffers[index], 0, sizeof(DrawData) * DrawDataCount },
                { transformBuffers[index], 0,
                    sizeof(TransformData) * TransformDataCount },
                { vertexBuffer, 0, sizeof(Vertex) * vertices.size() },
                { jointBuffers[index], 0, sizeof(glm::mat4) * JointMatrixCount }
            };
        }
        const std::vector<VkDescriptorImageInfo> textureInfos = textureDescriptorInfos();
        descriptors.allocateSets(bindings, textureInfos);
        imageGeometryVersions_.assign(descriptors.setCount(), geometryVersion_);
        imageGeometryRangeVersions_.assign(descriptors.setCount(), geometryRangeVersion_);
        imageTextureVersions_.assign(descriptors.setCount(), textureVersion_);
    }

    void prepareGeometryForImage(std::uint32_t imageIndex)
    {
        if (imageIndex >= descriptors.setCount() ||
            imageIndex >= imageGeometryVersions_.size())
        {
            throw std::runtime_error("swapchain geometry descriptor state is inconsistent");
        }
        if (imageGeometryVersions_[imageIndex] == geometryVersion_)
        {
            return;
        }

        descriptors.updateVertex(imageIndex,
            { vertexBuffer, 0, sizeof(Vertex) * vertices.size() });

        imageGeometryVersions_[imageIndex] = geometryVersion_;
        std::erase_if(retiredGeometry_, [&](const RetiredGeometry& retired)
        {
            return std::ranges::all_of(imageGeometryVersions_, [&](std::uint64_t imageVersion)
            {
                return imageVersion > retired.version;
            });
        });
    }

    void prepareGeometryRangesForImage(std::uint32_t imageIndex)
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
        for (auto retired = retiredGeometryRanges_.begin();
             retired != retiredGeometryRanges_.end();)
        {
            const bool noLongerReferenced = std::ranges::all_of(imageGeometryRangeVersions_,
                [&](std::uint64_t imageVersion)
                {
                    return imageVersion > retired->version;
                });
            if (!noLongerReferenced)
            {
                ++retired;
                continue;
            }
            releaseGeometryRange(freeVertexRanges_, retired->vertices);
            releaseGeometryRange(freeIndexRanges_, retired->indices);
            retired = retiredGeometryRanges_.erase(retired);
        }
    }

    void prepareTexturesForImage(std::uint32_t imageIndex)
    {
        if (imageIndex >= descriptors.setCount() ||
            imageIndex >= imageTextureVersions_.size())
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
            const bool noLongerReferenced = std::ranges::all_of(imageTextureVersions_,
                [&](std::uint64_t imageVersion)
                {
                    return imageVersion > retired->version;
                });
            if (!noLongerReferenced)
            {
                ++retired;
                continue;
            }
            vkDestroySampler(device, retired->texture.sampler, nullptr);
            retired->texture.sampler = VK_NULL_HANDLE;
            retired = retiredTextures_.erase(retired);
        }
    }

    void loadModel()
    {
        danvulkan::assets::SceneAsset scene = danvulkan::assets::loadScene(MODEL_PATH);
        for (const AdditionalSceneConfig& additional : config_.additionalScenes)
        {
            static_cast<void>(scene.append(
                danvulkan::assets::loadScene(additional.modelPath),
                additional.rootTransform));
        }
        uploadSceneAsset(scene);
    }

    [[nodiscard]] PreparedSceneData prepareReplacementScene(
        const danvulkan::assets::SceneAsset& scene) const
    {
        if (!scene.skins().empty() || !scene.animations().empty())
        {
            throw std::runtime_error(
                "animated whole-scene replacement is not yet supported");
        }
        if (scene.textures().empty() || scene.textures().size() > textureDescriptorCapacity_)
        {
            throw std::runtime_error(
                "replacement scene texture count exceeds renderer capacity");
        }
        if (scene.materials().empty() || scene.materials().size() > materialCapacity_)
        {
            throw std::runtime_error(
                "replacement scene material count exceeds renderer capacity");
        }
        if (scene.meshes().empty() || scene.meshes().size() >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::runtime_error("replacement scene contains an invalid mesh count");
        }

        for (const danvulkan::assets::TextureAsset& texture : scene.textures())
        {
            const std::uint64_t rowByteCount = static_cast<std::uint64_t>(texture.height) * 4;
            const bool byteCountOverflow = texture.width != 0 &&
                rowByteCount > std::numeric_limits<std::uint64_t>::max() / texture.width;
            const std::uint64_t byteCount = byteCountOverflow ? 0 :
                static_cast<std::uint64_t>(texture.width) * rowByteCount;
            if (texture.width == 0 || texture.height == 0 ||
                byteCountOverflow || byteCount != texture.rgba8.size())
            {
                throw std::runtime_error(
                    "replacement scene contains invalid decoded texture data");
            }
        }

        PreparedSceneData prepared;
        prepared.generation = nextSceneGeneration();
        prepared.materials.reserve(scene.materials().size());
        prepared.materialNames.reserve(scene.materials().size());

        const auto textureIndex = [&scene](danvulkan::assets::TextureHandle handle)
            -> std::int32_t
        {
            if (!handle)
            {
                return -1;
            }
            if (scene.find(handle) == nullptr ||
                handle.slot > static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()))
            {
                throw std::runtime_error(
                    "replacement scene contains an invalid texture handle");
            }
            return static_cast<std::int32_t>(handle.slot);
        };

        for (const danvulkan::assets::MaterialAsset& source : scene.materials())
        {
            switch (source.alphaMode)
            {
            case danvulkan::assets::AlphaMode::opaque:
            case danvulkan::assets::AlphaMode::mask:
            case danvulkan::assets::AlphaMode::blend:
                break;
            default:
                throw std::runtime_error(
                    "replacement scene contains an invalid material alpha mode");
            }

            MaterialData material{};
            material.baseColorFactor = source.albedoTint;
            material.emissiveMetallic = glm::vec4(source.emissiveFactor,
                source.metallicFactor);
            material.roughnessNormalOcclusionAlpha = {
                source.roughnessFactor,
                source.normalScale,
                source.occlusionStrength,
                source.alphaCutoff
            };
            material.textureTiling = glm::vec4(source.textureTiling, 0.0f, 0.0f);
            material.textureIndices = {
                textureIndex(source.albedoTexture),
                textureIndex(source.normalTexture),
                textureIndex(source.metallicRoughnessTexture),
                textureIndex(source.occlusionTexture)
            };
            material.materialFlags = {
                textureIndex(source.emissiveTexture),
                static_cast<std::int32_t>(source.alphaMode),
                source.doubleSided ? 1 : 0,
                source.unlit ? 1 : 0
            };
            prepared.materials.push_back(material);
            prepared.materialNames.push_back(source.name);
        }

        prepared.meshResources.resize(scene.meshes().size());
        std::uint64_t usedVertexCount = 0;
        std::uint64_t usedIndexCount = 0;
        for (const danvulkan::assets::MeshAsset& source : scene.meshes())
        {
            if (scene.find(source.material) == nullptr ||
                source.material.slot > static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()))
            {
                throw std::runtime_error(
                    "replacement scene contains an invalid material handle");
            }
            if (source.handle.slot >= prepared.meshResources.size() ||
                source.vertices.empty() || source.indices.empty())
            {
                throw std::runtime_error(
                    "replacement scene contains invalid mesh geometry");
            }
            for (const std::uint32_t index : source.indices)
            {
                if (index >= source.vertices.size())
                {
                    throw std::runtime_error(
                        "replacement scene contains an out-of-range vertex index");
                }
            }
            if (usedVertexCount + source.vertices.size() >
                    static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
                usedIndexCount + source.indices.size() >
                    std::numeric_limits<std::uint32_t>::max())
            {
                throw std::runtime_error(
                    "replacement scene geometry exceeds renderer offset limits");
            }

            MeshResourceData mesh;
            mesh.indexCount = static_cast<std::uint32_t>(source.indices.size());
            mesh.vertexCount = static_cast<std::uint32_t>(source.vertices.size());
            mesh.firstIndex = static_cast<std::uint32_t>(usedIndexCount);
            mesh.vertexOffset = static_cast<std::uint32_t>(usedVertexCount);
            mesh.materialIndex = static_cast<std::int32_t>(source.material.slot);
            const MaterialData& material = prepared.materials[source.material.slot];
            mesh.pipelineVariant = static_cast<std::uint32_t>(material.materialFlags.y * 2 +
                material.materialFlags.z);
            mesh.bounds = source.bounds;
            mesh.name = source.name;
            prepared.meshResources[source.handle.slot] = std::move(mesh);

            prepared.vertices.insert(prepared.vertices.end(), source.vertices.begin(),
                source.vertices.end());
            prepared.indices.insert(prepared.indices.end(), source.indices.begin(),
                source.indices.end());
            usedVertexCount += source.vertices.size();
            usedIndexCount += source.indices.size();
        }

        std::vector<bool> recursionStack(scene.nodes().size(), false);
        std::function<void(danvulkan::assets::NodeHandle, const glm::mat4&)> visitNode;
        visitNode = [&](danvulkan::assets::NodeHandle handle, const glm::mat4& parentTransform)
        {
            const danvulkan::assets::NodeAsset* node = scene.find(handle);
            if (node == nullptr || handle.slot >= recursionStack.size())
            {
                throw std::runtime_error(
                    "replacement scene contains an invalid node handle");
            }
            if (recursionStack[handle.slot])
            {
                throw std::runtime_error("replacement scene node hierarchy contains a cycle");
            }
            recursionStack[handle.slot] = true;

            const glm::mat4 worldTransform = parentTransform * node->localTransform;
            std::int32_t transformIndex = -1;
            if (!node->meshes.empty())
            {
                if (prepared.transforms.size() >= TransformDataCount)
                {
                    throw std::runtime_error(
                        "replacement scene exceeds renderer transform capacity");
                }
                transformIndex = static_cast<std::int32_t>(prepared.transforms.size());
                prepared.transforms.push_back({ worldTransform });
                prepared.instanceNames.push_back(node->name);
            }

            for (const danvulkan::assets::MeshHandle meshHandle : node->meshes)
            {
                if (scene.find(meshHandle) == nullptr ||
                    meshHandle.slot >= prepared.meshResources.size())
                {
                    throw std::runtime_error(
                        "replacement scene node contains an invalid mesh handle");
                }
                if (prepared.meshes.size() >= DrawDataCount)
                {
                    throw std::runtime_error(
                        "replacement scene exceeds renderer draw capacity");
                }

                const MeshResourceData& resource = prepared.meshResources[meshHandle.slot];
                DrawData draw{};
                draw.materialIndex = resource.materialIndex;
                draw.transformIndex = transformIndex;
                draw.vertexOffset = static_cast<std::int32_t>(resource.vertexOffset);
                prepared.draws.push_back(draw);

                MeshData mesh{};
                mesh.indexCount = resource.indexCount;
                mesh.firstIndex = resource.firstIndex;
                mesh.vertexOffset = resource.vertexOffset;
                mesh.pipelineVariant = resource.pipelineVariant;
                mesh.meshResourceSlot = meshHandle.slot;
                mesh.drawData = draw;
                mesh.localBounds = resource.bounds;
                prepared.meshes.push_back(mesh);
                prepared.bounds.push_back(transformedBounds(resource.bounds, worldTransform));
            }

            for (const danvulkan::assets::NodeHandle child : node->children)
            {
                visitNode(child, worldTransform);
            }
            recursionStack[handle.slot] = false;
        };

        for (const danvulkan::assets::NodeHandle root : scene.rootNodes())
        {
            visitNode(root, glm::mat4(1.0f));
        }
        if (prepared.draws.empty())
        {
            throw std::runtime_error("replacement scene contains no mesh instances");
        }

        prepared.vertexCapacity = std::max(config_.initialVertexCapacity,
            static_cast<std::uint32_t>(usedVertexCount));
        prepared.indexCapacity = std::max(config_.initialIndexCapacity,
            static_cast<std::uint32_t>(usedIndexCount));
        if (prepared.vertexCapacity == 0 || prepared.indexCapacity == 0)
        {
            throw std::runtime_error("replacement scene geometry capacity is invalid");
        }

        const VkPhysicalDeviceProperties& properties = device.properties();
        if (sizeof(Vertex) * static_cast<VkDeviceSize>(prepared.vertexCapacity) >
            properties.limits.maxStorageBufferRange)
        {
            throw std::runtime_error(
                "replacement scene vertex capacity exceeds maxStorageBufferRange");
        }

        prepared.vertices.resize(prepared.vertexCapacity);
        prepared.indices.resize(prepared.indexCapacity);
        if (usedVertexCount < prepared.vertexCapacity)
        {
            prepared.freeVertexRanges.push_back({ static_cast<std::uint32_t>(usedVertexCount),
                prepared.vertexCapacity - static_cast<std::uint32_t>(usedVertexCount) });
        }
        if (usedIndexCount < prepared.indexCapacity)
        {
            prepared.freeIndexRanges.push_back({ static_cast<std::uint32_t>(usedIndexCount),
                prepared.indexCapacity - static_cast<std::uint32_t>(usedIndexCount) });
        }
        return prepared;
    }

    void uploadSceneAsset(const danvulkan::assets::SceneAsset& scene)
    {
        sceneGeneration_ = nextGeneration(sceneGeneration_);
        materialCapacity_ = std::min(config_.maxMaterials, MatDataCount);
        if (materialCapacity_ == 0)
        {
            throw std::runtime_error("configured material capacity must be greater than zero");
        }
        vertices.clear();
        indices.clear();
        vertexCapacity_ = 0;
        indexCapacity_ = 0;
        freeVertexRanges_.clear();
        freeIndexRanges_.clear();
        retiredGeometryRanges_.clear();
        imageGeometryRangeVersions_.clear();
        geometryRangeVersion_ = 1;
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
        skinnedDraws_.clear();
        jointMatrices_.clear();
        textures.clear();
        textureGenerations_.clear();
        freeTextureSlots_.clear();
        retiredTextures_.clear();
        liveTextureCount_ = 0;
        textureVersion_ = 1;

        if (!scene.skins().empty() || !scene.animations().empty())
        {
            animationPlayer_.emplace(scene);
        }

        for (const danvulkan::assets::TextureAsset& texture : scene.textures())
        {
            textures.emplace_back(createTextureImage(texture));
            textureGenerations_.push_back(sceneGeneration_);
            ++liveTextureCount_;
        }

        const auto textureIndex = [&scene](danvulkan::assets::TextureHandle handle) -> std::int32_t
        {
            if (!handle)
            {
                return -1;
            }
            if (scene.find(handle) == nullptr ||
                handle.slot > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
            {
                throw std::runtime_error("asset contains an invalid texture handle");
            }
            return static_cast<std::int32_t>(handle.slot);
        };

        for (const danvulkan::assets::MaterialAsset& source : scene.materials())
        {
            MaterialData material{};
            material.baseColorFactor = source.albedoTint;
            material.emissiveMetallic = glm::vec4(source.emissiveFactor, source.metallicFactor);
            material.roughnessNormalOcclusionAlpha = {
                source.roughnessFactor, source.normalScale, source.occlusionStrength, source.alphaCutoff
            };
            material.textureTiling = glm::vec4(source.textureTiling, 0.0f, 0.0f);
            material.textureIndices = {
                textureIndex(source.albedoTexture),
                textureIndex(source.normalTexture),
                textureIndex(source.metallicRoughnessTexture),
                textureIndex(source.occlusionTexture)
            };
            material.materialFlags = {
                textureIndex(source.emissiveTexture),
                static_cast<std::int32_t>(source.alphaMode),
                source.doubleSided ? 1 : 0,
                source.unlit ? 1 : 0
            };
            matData.push_back(material);
            materialNames.push_back(source.name);
        }
        if (matData.empty() || matData.size() > materialCapacity_)
        {
            throw std::runtime_error("configured material capacity is invalid for this scene");
        }
        materialGenerations_.assign(matData.size(), sceneGeneration_);
        materialAlive_.assign(matData.size(), true);

        meshResources.resize(scene.meshes().size());
        meshGenerations_.assign(scene.meshes().size(), sceneGeneration_);
        meshAlive_.assign(scene.meshes().size(), false);

        for (const danvulkan::assets::MeshAsset& source : scene.meshes())
        {
            if (scene.find(source.material) == nullptr ||
                source.material.slot > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
            {
                throw std::runtime_error("asset contains an invalid material handle");
            }

            if (source.handle.slot >= meshResources.size() ||
                source.vertices.size() > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max()) - vertices.size() ||
                source.indices.size() > static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max()) - indices.size())
            {
                throw std::runtime_error("asset geometry exceeds renderer index limits");
            }

            const auto vertexOffset = static_cast<std::uint32_t>(vertices.size());
            const auto firstIndex = static_cast<std::uint32_t>(indices.size());
            vertices.insert(vertices.end(), source.vertices.begin(), source.vertices.end());
            indices.insert(indices.end(), source.indices.begin(), source.indices.end());

            MeshResourceData uploaded;
            uploaded.indexCount = static_cast<std::uint32_t>(source.indices.size());
            uploaded.vertexCount = static_cast<std::uint32_t>(source.vertices.size());
            uploaded.firstIndex = firstIndex;
            uploaded.vertexOffset = vertexOffset;
            uploaded.materialIndex = static_cast<std::int32_t>(source.material.slot);
            const MaterialData& material = matData[source.material.slot];
            uploaded.pipelineVariant = static_cast<std::uint32_t>(material.materialFlags.y * 2 +
                material.materialFlags.z);
            uploaded.bounds = source.bounds;
            uploaded.name = source.name;
            meshResources[source.handle.slot] = std::move(uploaded);
            meshAlive_[source.handle.slot] = true;
        }

        std::vector<bool> recursionStack(scene.nodes().size(), false);
        std::function<void(danvulkan::assets::NodeHandle, const glm::mat4&)> visitNode;
        visitNode = [&](danvulkan::assets::NodeHandle handle, const glm::mat4& parentTransform)
        {
            const danvulkan::assets::NodeAsset* node = scene.find(handle);
            if (node == nullptr || handle.slot >= recursionStack.size())
            {
                throw std::runtime_error("asset contains an invalid node handle");
            }
            if (recursionStack[handle.slot])
            {
                throw std::runtime_error("asset node hierarchy contains a cycle");
            }
            recursionStack[handle.slot] = true;

            const glm::mat4 worldTransform = parentTransform * node->localTransform;
            std::int32_t transformIndex = -1;
            if (!node->meshes.empty())
            {
                if (transformData.size() >= TransformDataCount)
                {
                    throw std::runtime_error("asset exceeds the renderer transform capacity");
                }
                transformIndex = static_cast<std::int32_t>(transformData.size());
                transformData.push_back(TransformData{ worldTransform });
                instanceNames.push_back(node->name);
                instanceGenerations.push_back(sceneGeneration_);
                instanceAlive.push_back(true);
            }

            for (const danvulkan::assets::MeshHandle meshHandle : node->meshes)
            {
                if (scene.find(meshHandle) == nullptr || meshHandle.slot >= meshResources.size())
                {
                    throw std::runtime_error("asset node contains an invalid mesh handle");
                }
                if (meshData.size() >= DrawDataCount)
                {
                    throw std::runtime_error("asset exceeds the renderer draw capacity");
                }

                const MeshResourceData& uploaded = meshResources[meshHandle.slot];
                DrawData draw{};
                draw.materialIndex = uploaded.materialIndex;
                draw.transformIndex = transformIndex;
                draw.vertexOffset = static_cast<std::int32_t>(uploaded.vertexOffset);
                std::uint32_t jointOffset = 0;
                if (node->skin)
                {
                    if (!animationPlayer_ || scene.find(node->skin) == nullptr)
                    {
                        throw std::runtime_error("asset node contains an invalid skin handle");
                    }
                    const danvulkan::assets::MeshAsset* sourceMesh = scene.find(meshHandle);
                    const danvulkan::assets::SkinAsset* skin = scene.find(node->skin);
                    for (const Vertex& vertex : sourceMesh->vertices)
                    {
                        for (glm::length_t component = 0; component < 4; ++component)
                        {
                            if (vertex.weights[component] > 0.0f &&
                                vertex.joints[component] >= skin->joints.size())
                            {
                                throw std::runtime_error(
                                    "asset vertex references an out-of-range skin joint");
                            }
                        }
                    }
                    jointOffset = static_cast<std::uint32_t>(jointMatrices_.size());
                    draw.jointOffset = static_cast<std::int32_t>(jointOffset);
                    animationPlayer_->appendSkinMatrices(
                        node->skin, handle, jointMatrices_);
                    if (jointMatrices_.size() > JointMatrixCount)
                    {
                        throw std::runtime_error(
                            "asset exceeds the renderer joint matrix capacity");
                    }
                }
                drawData.push_back(draw);

                MeshData mesh{};
                mesh.indexCount = uploaded.indexCount;
                mesh.firstIndex = uploaded.firstIndex;
                mesh.vertexOffset = uploaded.vertexOffset;
                mesh.pipelineVariant = uploaded.pipelineVariant;
                mesh.meshResourceSlot = meshHandle.slot;
                mesh.drawData = draw;
                mesh.localBounds = uploaded.bounds;
                const std::uint32_t meshDataIndex = static_cast<std::uint32_t>(meshData.size());
                meshData.push_back(mesh);
                aabbs.push_back(transformedBounds(uploaded.bounds, worldTransform));
                if (animationPlayer_)
                {
                    animatedDraws_.push_back({ handle,
                        static_cast<std::uint32_t>(transformIndex), meshDataIndex });
                }
                if (node->skin)
                {
                    skinnedDraws_.push_back({ handle, node->skin,
                        static_cast<std::uint32_t>(transformIndex), jointOffset, meshDataIndex });
                }
            }

            for (const danvulkan::assets::NodeHandle child : node->children)
            {
                visitNode(child, worldTransform);
            }
            recursionStack[handle.slot] = false;
        };

        for (const danvulkan::assets::NodeHandle root : scene.rootNodes())
        {
            visitNode(root, glm::mat4(1.0f));
        }
        if (drawData.empty())
        {
            throw std::runtime_error("asset scene contains no mesh instances");
        }

        const std::uint32_t usedVertexCount = static_cast<std::uint32_t>(vertices.size());
        const std::uint32_t usedIndexCount = static_cast<std::uint32_t>(indices.size());
        vertexCapacity_ = std::max(config_.initialVertexCapacity, usedVertexCount);
        indexCapacity_ = std::max(config_.initialIndexCapacity, usedIndexCount);
        if (vertexCapacity_ == 0 || indexCapacity_ == 0 ||
            vertexCapacity_ > static_cast<std::uint32_t>(
                std::numeric_limits<std::int32_t>::max()))
        {
            throw std::runtime_error("configured initial geometry capacity is invalid");
        }

        const VkPhysicalDeviceProperties& physicalDeviceProperties = device.properties();
        if (sizeof(Vertex) * static_cast<VkDeviceSize>(vertexCapacity_) >
            physicalDeviceProperties.limits.maxStorageBufferRange)
        {
            throw std::runtime_error("configured vertex capacity exceeds maxStorageBufferRange");
        }

        vertices.resize(vertexCapacity_);
        indices.resize(indexCapacity_);
        if (usedVertexCount < vertexCapacity_)
        {
            freeVertexRanges_.push_back({ usedVertexCount, vertexCapacity_ - usedVertexCount });
        }
        if (usedIndexCount < indexCapacity_)
        {
            freeIndexRanges_.push_back({ usedIndexCount, indexCapacity_ - usedIndexCount });
        }
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
        const std::filesystem::path compiledPath = std::filesystem::path(COMPILED_SHADER_PATH) /
            (shaderSourceFile.stem().string() + ".spv");
        std::array<char, 128> buffer;
        std::string result;
#ifdef _WIN32
        const std::string cmd = "glslc.exe --target-env=vulkan1.4 \"" + filePath +
            "\" -o \"" + compiledPath.string() + "\" 2>&1";
        std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd.c_str(), "r"), _pclose);
#else
        const std::string cmd = "glslc --target-env=vulkan1.4 \"" + filePath +
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
        if (!aabbs.empty())
        {
            const float maximum = std::numeric_limits<float>::max();
            glm::vec3 sceneMinimum(maximum);
            glm::vec3 sceneMaximum(-maximum);
            for (const AABB& bounds : aabbs)
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

        PointLight pl;
        pl.position = lightPos;
        pl.power = 1.0f;
        pl.color = glm::vec3(1.0f, 1.0f, 1.0f);
        pointLights.push_back(pl);
        /* lion head spots
        DrawData lion1 = drawData[375], lion2 = drawData[376];
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

void VulkanRenderer::endFrame()
{
    impl_->endFrame();
}

std::vector<SceneInstanceInfo> VulkanRenderer::sceneInstances() const
{
    return impl_->sceneInstances();
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

AnimationPlaybackState VulkanRenderer::animationPlaybackState() const
{
    return impl_->animationPlaybackState();
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
