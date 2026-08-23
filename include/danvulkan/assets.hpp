#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace danvulkan::assets
{
template <typename Tag>
struct Handle
{
    static constexpr std::uint32_t invalidSlot = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t slot = invalidSlot;
    std::uint32_t generation = 0;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return slot != invalidSlot && generation != 0;
    }

    constexpr explicit operator bool() const noexcept
    {
        return valid();
    }

    friend constexpr bool operator==(Handle, Handle) noexcept = default;
};

struct TextureTag;
struct MaterialTag;
struct MeshTag;
struct NodeTag;
struct SkinTag;
struct AnimationTag;

using TextureHandle = Handle<TextureTag>;
using MaterialHandle = Handle<MaterialTag>;
using MeshHandle = Handle<MeshTag>;
using NodeHandle = Handle<NodeTag>;
using SkinHandle = Handle<SkinTag>;
using AnimationHandle = Handle<AnimationTag>;

enum class ColorSpace
{
    linear,
    srgb
};

enum class TextureFilter
{
    nearest,
    linear
};

enum class TextureMipmapMode
{
    nearest,
    linear
};

enum class TextureWrap
{
    repeat,
    mirroredRepeat,
    clampToEdge
};

struct TextureSampler
{
    TextureFilter magFilter = TextureFilter::linear;
    TextureFilter minFilter = TextureFilter::linear;
    TextureMipmapMode mipmapMode = TextureMipmapMode::linear;
    TextureWrap wrapU = TextureWrap::repeat;
    TextureWrap wrapV = TextureWrap::repeat;
};

enum class AlphaMode
{
    opaque,
    mask,
    blend
};

struct Vertex
{
    glm::vec3 pos{};
    float unused0 = 0.0f;

    glm::vec3 normal{};
    float unused01 = 0.0f;

    glm::vec3 color{ 1.0f };
    float unused02 = 0.0f;

    glm::vec2 texCoord{};
    float unused03 = 0.0f;
    float unused04 = 0.0f;

    glm::vec3 tangent{};
    float tangentSign = 0.0f;

    glm::uvec4 joints{};
    glm::vec4 weights{};
};

struct Bounds
{
    glm::vec3 minVertex{};
    glm::vec3 maxVertex{};
};

struct TextureAsset
{
    TextureHandle handle;
    std::string name;
    std::filesystem::path sourcePath;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    ColorSpace colorSpace = ColorSpace::srgb;
    TextureSampler sampler;
    std::vector<std::byte> rgba8;
    // Linear floating-point texels used by HDR environments. Ordinary material textures keep
    // using rgba8; a decoded image must populate exactly one pixel payload.
    std::vector<float> rgba32f;
};

struct MaterialAsset
{
    MaterialHandle handle;
    std::string name;
    glm::vec4 albedoTint{ 1.0f };
    glm::vec2 textureTiling{ 1.0f };
    float reflectance = 1.0f;
    float metallicFactor = 0.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;
    float occlusionStrength = 1.0f;
    glm::vec3 emissiveFactor{};
    float alphaCutoff = 0.5f;
    AlphaMode alphaMode = AlphaMode::opaque;
    bool doubleSided = false;
    bool unlit = false;
    TextureHandle albedoTexture;
    TextureHandle normalTexture;
    TextureHandle metallicRoughnessTexture;
    TextureHandle occlusionTexture;
    TextureHandle emissiveTexture;
};

struct NodeAsset
{
    NodeHandle handle;
    std::string name;
    glm::mat4 localTransform{ 1.0f };
    glm::vec3 translation{};
    glm::vec4 rotation{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec3 scale{1.0f};
    bool transformIsTrs = false;
    std::vector<MeshHandle> meshes;
    std::vector<NodeHandle> children;
    SkinHandle skin;
};

struct MeshAsset
{
    MeshHandle handle;
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    MaterialHandle material;
    Bounds bounds;
};

struct SkinAsset
{
    SkinHandle handle;
    std::string name;
    NodeHandle skeleton;
    std::vector<NodeHandle> joints;
    std::vector<glm::mat4> inverseBindMatrices;
};

enum class AnimationTarget
{
    translation,
    rotation,
    scale
};

enum class AnimationInterpolation
{
    linear,
    step
};

struct AnimationChannelAsset
{
    NodeHandle target;
    AnimationTarget path = AnimationTarget::translation;
    AnimationInterpolation interpolation = AnimationInterpolation::linear;
    std::vector<float> times;
    // Translation and scale use xyz. Rotation uses a glTF quaternion in xyzw order.
    std::vector<glm::vec4> values;
};

struct AnimationClipAsset
{
    AnimationHandle handle;
    std::string name;
    float startTime = 0.0f;
    float endTime = 0.0f;
    std::vector<AnimationChannelAsset> channels;
};

struct AnimationNodeBindingAsset
{
    // The source is a node targeted by the shared clip. The target is the corresponding node in
    // this actor's hierarchy.
    NodeHandle source;
    NodeHandle target;
};

struct AnimationInstanceAsset
{
    std::string name;
    AnimationHandle clip;
    std::vector<AnimationNodeBindingAsset> nodeBindings;
    float initialPositionSeconds = 0.0f;
    float playbackSpeed = 1.0f;
    bool looping = true;
};

class SceneAsset
{
public:
    TextureHandle addTexture(TextureAsset texture);
    MaterialHandle addMaterial(MaterialAsset material);
    MeshHandle addMesh(MeshAsset mesh);
    NodeHandle addNode(NodeAsset node);
    SkinHandle addSkin(SkinAsset skin);
    AnimationHandle addAnimation(AnimationClipAsset animation);
    void addAnimationInstance(AnimationInstanceAsset instance);
    void addRootNode(NodeHandle node);

    // Moves another scene into this one, remapping every cross-resource handle. The optional
    // transform is applied above each appended root and is useful for composing authored assets.
    [[nodiscard]] std::vector<NodeHandle> append(
        SceneAsset scene, const glm::mat4& rootTransform = glm::mat4(1.0f));

    [[nodiscard]] const TextureAsset* find(TextureHandle handle) const noexcept;
    [[nodiscard]] const MaterialAsset* find(MaterialHandle handle) const noexcept;
    [[nodiscard]] const MeshAsset* find(MeshHandle handle) const noexcept;
    [[nodiscard]] const NodeAsset* find(NodeHandle handle) const noexcept;
    [[nodiscard]] const SkinAsset* find(SkinHandle handle) const noexcept;
    [[nodiscard]] const AnimationClipAsset* find(AnimationHandle handle) const noexcept;

    [[nodiscard]] std::span<const TextureAsset> textures() const noexcept { return textures_; }
    [[nodiscard]] std::span<const MaterialAsset> materials() const noexcept { return materials_; }
    [[nodiscard]] std::span<const MeshAsset> meshes() const noexcept { return meshes_; }
    [[nodiscard]] std::span<const NodeAsset> nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::span<const SkinAsset> skins() const noexcept { return skins_; }
    [[nodiscard]] std::span<const AnimationClipAsset> animations() const noexcept
    {
        return animations_;
    }
    [[nodiscard]] std::span<const AnimationInstanceAsset> animationInstances() const noexcept
    {
        return animationInstances_;
    }
    [[nodiscard]] std::span<const NodeHandle> rootNodes() const noexcept { return rootNodes_; }

private:
    std::vector<TextureAsset> textures_;
    std::vector<MaterialAsset> materials_;
    std::vector<MeshAsset> meshes_;
    std::vector<NodeAsset> nodes_;
    std::vector<SkinAsset> skins_;
    std::vector<AnimationClipAsset> animations_;
    std::vector<AnimationInstanceAsset> animationInstances_;
    std::vector<NodeHandle> rootNodes_;
};

// Compatibility importer for the existing demo assets. Importing and image decoding happen
// entirely on the CPU; no Vulkan objects are created by this function.
[[nodiscard]] SceneAsset loadObj(const std::filesystem::path& path);
[[nodiscard]] SceneAsset loadGltf(const std::filesystem::path& path);
[[nodiscard]] SceneAsset loadScene(const std::filesystem::path& path);
// Decodes an ordinary material texture into an RGBA8 payload. Color textures should use sRGB;
// data textures such as normal or roughness maps should request linear sampling.
[[nodiscard]] TextureAsset loadTexture(const std::filesystem::path& path,
    ColorSpace colorSpace = ColorSpace::srgb);
// Decodes an equirectangular environment into linear RGBA32F texels. HDR files retain values
// above one; conventional images are promoted and linearized by stb_image.
[[nodiscard]] TextureAsset loadEnvironment(const std::filesystem::path& path);
}
