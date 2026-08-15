#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <danvulkan/assets.hpp>

#include "assets_internal.hpp"

#include <tiny_obj_loader.h>

#include <glm/geometric.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace danvulkan::assets
{
namespace
{
constexpr std::uint32_t initialGeneration = 1;

template <typename Asset, typename TypedHandle>
const Asset* findAsset(std::span<const Asset> assets, TypedHandle handle) noexcept
{
    if (!handle || handle.slot >= assets.size())
    {
        return nullptr;
    }

    const Asset& asset = assets[handle.slot];
    return asset.handle == handle ? &asset : nullptr;
}

struct ObjIndex
{
    int vertex = -1;
    int normal = -1;
    int texCoord = -1;

    friend bool operator==(const ObjIndex&, const ObjIndex&) = default;
};

struct ObjIndexHash
{
    std::size_t operator()(const ObjIndex& index) const noexcept
    {
        std::size_t result = std::hash<int>{}(index.vertex);
        result ^= std::hash<int>{}(index.normal) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        result ^= std::hash<int>{}(index.texCoord) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        return result;
    }
};

struct MeshBuilder
{
    std::string name;
    int materialIndex = -1;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::unordered_map<ObjIndex, std::uint32_t, ObjIndexHash> vertexLookup;
    bool hasMissingNormals = false;
};

std::filesystem::path resolveTexturePath(const std::filesystem::path& modelPath,
                                         const std::string& textureName)
{
    const std::filesystem::path requested(textureName);
    if (requested.is_absolute() && std::filesystem::exists(requested))
    {
        return requested.lexically_normal();
    }

    const std::filesystem::path relativeToModel = modelPath.parent_path() / requested;
    if (std::filesystem::exists(relativeToModel))
    {
        return relativeToModel.lexically_normal();
    }

    // Preserve compatibility with legacy MTL files that used paths relative to the process.
    if (std::filesystem::exists(requested))
    {
        return requested.lexically_normal();
    }

    throw std::runtime_error("texture referenced by '" + modelPath.string() +
                             "' was not found: " + textureName);
}

TextureAsset decodeTexture(const std::filesystem::path& path, ColorSpace colorSpace)
{
    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &sourceChannels, STBI_rgb_alpha);
    if (pixels == nullptr)
    {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("failed to decode texture '" + path.string() + "': " +
                                 (reason != nullptr ? reason : "unknown stb_image error"));
    }
    if (width <= 0 || height <= 0)
    {
        stbi_image_free(pixels);
        throw std::runtime_error("decoded texture has invalid dimensions: " + path.string());
    }

    TextureAsset texture;
    texture.name = path.filename().string();
    texture.sourcePath = path;
    texture.width = static_cast<std::uint32_t>(width);
    texture.height = static_cast<std::uint32_t>(height);
    texture.colorSpace = colorSpace;

    const std::size_t byteCount = static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height) * 4U;
    texture.rgba8.resize(byteCount);
    std::memcpy(texture.rgba8.data(), pixels, byteCount);
    stbi_image_free(pixels);
    return texture;
}

TextureAsset decodeEnvironmentImpl(const std::filesystem::path& path)
{
    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    float* pixels = stbi_loadf(path.string().c_str(), &width, &height, &sourceChannels,
        STBI_rgb_alpha);
    if (pixels == nullptr)
    {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("failed to decode environment '" + path.string() + "': " +
            (reason != nullptr ? reason : "unknown stb_image error"));
    }
    if (width <= 0 || height <= 0)
    {
        stbi_image_free(pixels);
        throw std::runtime_error("decoded environment has invalid dimensions: " + path.string());
    }

    TextureAsset texture;
    texture.name = path.filename().string();
    texture.sourcePath = path;
    texture.width = static_cast<std::uint32_t>(width);
    texture.height = static_cast<std::uint32_t>(height);
    texture.colorSpace = ColorSpace::linear;
    texture.sampler.wrapU = TextureWrap::repeat;
    texture.sampler.wrapV = TextureWrap::clampToEdge;
    if (static_cast<std::size_t>(height) > std::numeric_limits<std::size_t>::max() /
        static_cast<std::size_t>(width) / 4U)
    {
        stbi_image_free(pixels);
        throw std::runtime_error("decoded environment dimensions overflow: " + path.string());
    }
    const std::size_t valueCount = static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height) * 4U;
    texture.rgba32f.assign(pixels, pixels + valueCount);
    stbi_image_free(pixels);
    return texture;
}

TextureAsset makeWhiteTextureImpl()
{
    TextureAsset texture;
    texture.name = "generated white texture";
    texture.width = 1;
    texture.height = 1;
    texture.colorSpace = ColorSpace::srgb;
    texture.rgba8 = {
        std::byte{ 0xff }, std::byte{ 0xff }, std::byte{ 0xff }, std::byte{ 0xff }
    };
    return texture;
}

glm::vec3 readVec3(const std::vector<tinyobj::real_t>& values, int index,
                   const glm::vec3& fallback)
{
    if (index < 0)
    {
        return fallback;
    }

    const std::size_t offset = static_cast<std::size_t>(index) * 3U;
    if (offset + 2U >= values.size())
    {
        throw std::runtime_error("OBJ attribute index is out of bounds");
    }
    return { values[offset], values[offset + 1U], values[offset + 2U] };
}

glm::vec2 readTexCoord(const std::vector<tinyobj::real_t>& values, int index)
{
    if (index < 0)
    {
        return {};
    }

    const std::size_t offset = static_cast<std::size_t>(index) * 2U;
    if (offset + 1U >= values.size())
    {
        throw std::runtime_error("OBJ texture-coordinate index is out of bounds");
    }
    return { values[offset], 1.0f - values[offset + 1U] };
}

std::uint32_t appendVertex(MeshBuilder& builder, const tinyobj::attrib_t& attributes,
                           const tinyobj::index_t& sourceIndex)
{
    const ObjIndex key{ sourceIndex.vertex_index, sourceIndex.normal_index,
                       sourceIndex.texcoord_index };
    if (const auto existing = builder.vertexLookup.find(key); existing != builder.vertexLookup.end())
    {
        return existing->second;
    }

    Vertex vertex;
    vertex.pos = readVec3(attributes.vertices, sourceIndex.vertex_index, {});
    vertex.normal = readVec3(attributes.normals, sourceIndex.normal_index, {});
    vertex.texCoord = readTexCoord(attributes.texcoords, sourceIndex.texcoord_index);
    builder.hasMissingNormals = builder.hasMissingNormals || sourceIndex.normal_index < 0;

    const auto index = static_cast<std::uint32_t>(builder.vertices.size());
    builder.vertices.push_back(vertex);
    builder.vertexLookup.emplace(key, index);
    return index;
}

void finishGeometryImpl(MeshAsset& mesh, bool generateNormals)
{
    const float maximum = std::numeric_limits<float>::max();
    mesh.bounds.minVertex = glm::vec3(maximum);
    mesh.bounds.maxVertex = glm::vec3(-maximum);

    std::vector<glm::vec3> accumulatedNormals(mesh.vertices.size());
    std::vector<glm::vec3> accumulatedTangents(mesh.vertices.size());
    std::vector<glm::vec3> accumulatedBitangents(mesh.vertices.size());

    for (std::size_t triangle = 0; triangle + 2U < mesh.indices.size(); triangle += 3U)
    {
        const std::array<std::uint32_t, 3> vertexIndices{
            mesh.indices[triangle], mesh.indices[triangle + 1U], mesh.indices[triangle + 2U]
        };
        Vertex& v0 = mesh.vertices[vertexIndices[0]];
        Vertex& v1 = mesh.vertices[vertexIndices[1]];
        Vertex& v2 = mesh.vertices[vertexIndices[2]];

        const glm::vec3 edge1 = v1.pos - v0.pos;
        const glm::vec3 edge2 = v2.pos - v0.pos;
        const glm::vec3 faceNormal = glm::cross(edge1, edge2);
        for (const std::uint32_t index : vertexIndices)
        {
            accumulatedNormals[index] += faceNormal;
        }

        const glm::vec2 deltaUv1 = v1.texCoord - v0.texCoord;
        const glm::vec2 deltaUv2 = v2.texCoord - v0.texCoord;
        const float determinant = deltaUv1.x * deltaUv2.y - deltaUv1.y * deltaUv2.x;
        if (std::abs(determinant) > std::numeric_limits<float>::epsilon())
        {
            const float inverse = 1.0f / determinant;
            const glm::vec3 tangent = inverse * (deltaUv2.y * edge1 - deltaUv1.y * edge2);
            const glm::vec3 bitangent = inverse * (-deltaUv2.x * edge1 + deltaUv1.x * edge2);
            for (const std::uint32_t index : vertexIndices)
            {
                accumulatedTangents[index] += tangent;
                accumulatedBitangents[index] += bitangent;
            }
        }
    }

    for (std::size_t index = 0; index < mesh.vertices.size(); ++index)
    {
        Vertex& vertex = mesh.vertices[index];
        mesh.bounds.minVertex = glm::min(mesh.bounds.minVertex, vertex.pos);
        mesh.bounds.maxVertex = glm::max(mesh.bounds.maxVertex, vertex.pos);

        if (generateNormals && glm::dot(accumulatedNormals[index], accumulatedNormals[index]) > 0.0f)
        {
            vertex.normal = glm::normalize(accumulatedNormals[index]);
        }

        if (glm::dot(vertex.tangent, vertex.tangent) > 0.0f)
        {
            vertex.tangent = glm::normalize(vertex.tangent -
                vertex.normal * glm::dot(vertex.normal, vertex.tangent));
            vertex.tangentSign = vertex.tangentSign < 0.0f ? -1.0f : 1.0f;
        }
        else
        {
            const glm::vec3 orthogonalTangent = accumulatedTangents[index] -
                vertex.normal * glm::dot(vertex.normal, accumulatedTangents[index]);
            if (glm::dot(orthogonalTangent, orthogonalTangent) > 0.0f)
            {
                vertex.tangent = glm::normalize(orthogonalTangent);
                vertex.tangentSign = glm::dot(glm::cross(vertex.normal, vertex.tangent),
                    accumulatedBitangents[index]) < 0.0f ? -1.0f : 1.0f;
            }
            else
            {
                const glm::vec3 reference = std::abs(vertex.normal.z) < 0.999f
                    ? glm::vec3(0.0f, 0.0f, 1.0f)
                    : glm::vec3(0.0f, 1.0f, 0.0f);
                vertex.tangent = glm::normalize(glm::cross(reference, vertex.normal));
                vertex.tangentSign = 1.0f;
            }
        }
    }
}
}

namespace detail
{
TextureAsset decodeTextureMemory(std::span<const std::byte> encodedBytes, std::string_view name,
                                 const std::filesystem::path& sourcePath, ColorSpace colorSpace)
{
    if (encodedBytes.empty() || encodedBytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("invalid encoded texture payload: " + std::string(name));
    }

    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    stbi_uc* pixels = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(encodedBytes.data()), static_cast<int>(encodedBytes.size()),
        &width, &height, &sourceChannels, STBI_rgb_alpha);
    if (pixels == nullptr || width <= 0 || height <= 0)
    {
        const char* reason = stbi_failure_reason();
        stbi_image_free(pixels);
        throw std::runtime_error("failed to decode texture '" + std::string(name) + "': " +
                                 (reason != nullptr ? reason : "unknown stb_image error"));
    }

    TextureAsset texture;
    texture.name = std::string(name);
    texture.sourcePath = sourcePath;
    texture.width = static_cast<std::uint32_t>(width);
    texture.height = static_cast<std::uint32_t>(height);
    texture.colorSpace = colorSpace;
    const std::size_t byteCount = static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height) * 4U;
    texture.rgba8.resize(byteCount);
    std::memcpy(texture.rgba8.data(), pixels, byteCount);
    stbi_image_free(pixels);
    return texture;
}

TextureAsset makeWhiteTexture()
{
    return makeWhiteTextureImpl();
}

void finishGeometry(MeshAsset& mesh, bool generateNormals)
{
    finishGeometryImpl(mesh, generateNormals);
}
}

TextureAsset loadEnvironment(const std::filesystem::path& path)
{
    return decodeEnvironmentImpl(path);
}

TextureHandle SceneAsset::addTexture(TextureAsset texture)
{
    const TextureHandle handle{ static_cast<std::uint32_t>(textures_.size()), initialGeneration };
    texture.handle = handle;
    textures_.push_back(std::move(texture));
    return handle;
}

MaterialHandle SceneAsset::addMaterial(MaterialAsset material)
{
    const MaterialHandle handle{ static_cast<std::uint32_t>(materials_.size()), initialGeneration };
    material.handle = handle;
    materials_.push_back(std::move(material));
    return handle;
}

MeshHandle SceneAsset::addMesh(MeshAsset mesh)
{
    const MeshHandle handle{ static_cast<std::uint32_t>(meshes_.size()), initialGeneration };
    mesh.handle = handle;
    meshes_.push_back(std::move(mesh));
    return handle;
}

NodeHandle SceneAsset::addNode(NodeAsset node)
{
    const NodeHandle handle{ static_cast<std::uint32_t>(nodes_.size()), initialGeneration };
    node.handle = handle;
    nodes_.push_back(std::move(node));
    return handle;
}

SkinHandle SceneAsset::addSkin(SkinAsset skin)
{
    const SkinHandle handle{ static_cast<std::uint32_t>(skins_.size()), initialGeneration };
    skin.handle = handle;
    skins_.push_back(std::move(skin));
    return handle;
}

AnimationHandle SceneAsset::addAnimation(AnimationClipAsset animation)
{
    const AnimationHandle handle{
        static_cast<std::uint32_t>(animations_.size()), initialGeneration
    };
    animation.handle = handle;
    animations_.push_back(std::move(animation));
    return handle;
}

void SceneAsset::addAnimationInstance(AnimationInstanceAsset instance)
{
    if (find(instance.clip) == nullptr)
    {
        throw std::invalid_argument("cannot add an animation instance with an invalid clip");
    }
    for (const AnimationNodeBindingAsset& binding : instance.nodeBindings)
    {
        if (find(binding.source) == nullptr || find(binding.target) == nullptr)
        {
            throw std::invalid_argument(
                "cannot add an animation instance with an invalid node binding");
        }
    }
    animationInstances_.push_back(std::move(instance));
}

void SceneAsset::addRootNode(NodeHandle node)
{
    if (find(node) == nullptr)
    {
        throw std::invalid_argument("cannot add an invalid scene root node");
    }
    rootNodes_.push_back(node);
}

std::vector<NodeHandle> SceneAsset::append(SceneAsset scene, const glm::mat4& rootTransform)
{
    const std::uint32_t textureOffset = static_cast<std::uint32_t>(textures_.size());
    const std::uint32_t materialOffset = static_cast<std::uint32_t>(materials_.size());
    const std::uint32_t meshOffset = static_cast<std::uint32_t>(meshes_.size());
    const std::uint32_t nodeOffset = static_cast<std::uint32_t>(nodes_.size());
    const std::uint32_t skinOffset = static_cast<std::uint32_t>(skins_.size());
    const std::uint32_t animationOffset = static_cast<std::uint32_t>(animations_.size());

    const auto remap = [](auto handle, std::uint32_t offset)
    {
        using HandleType = decltype(handle);
        if (!handle)
        {
            return HandleType{};
        }
        if (handle.generation != initialGeneration ||
            handle.slot > std::numeric_limits<std::uint32_t>::max() - offset)
        {
            throw std::invalid_argument("cannot append a scene with invalid resource handles");
        }
        return HandleType{ handle.slot + offset, initialGeneration };
    };

    for (TextureAsset& texture : scene.textures_)
    {
        addTexture(std::move(texture));
    }
    for (MaterialAsset& material : scene.materials_)
    {
        material.albedoTexture = remap(material.albedoTexture, textureOffset);
        material.normalTexture = remap(material.normalTexture, textureOffset);
        material.metallicRoughnessTexture = remap(
            material.metallicRoughnessTexture, textureOffset);
        material.occlusionTexture = remap(material.occlusionTexture, textureOffset);
        material.emissiveTexture = remap(material.emissiveTexture, textureOffset);
        addMaterial(std::move(material));
    }
    for (MeshAsset& mesh : scene.meshes_)
    {
        mesh.material = remap(mesh.material, materialOffset);
        addMesh(std::move(mesh));
    }
    for (SkinAsset& skin : scene.skins_)
    {
        skin.skeleton = remap(skin.skeleton, nodeOffset);
        for (NodeHandle& joint : skin.joints)
        {
            joint = remap(joint, nodeOffset);
        }
        addSkin(std::move(skin));
    }
    for (AnimationClipAsset& animation : scene.animations_)
    {
        for (AnimationChannelAsset& channel : animation.channels)
        {
            channel.target = remap(channel.target, nodeOffset);
        }
        addAnimation(std::move(animation));
    }

    std::vector<bool> isRoot(scene.nodes_.size(), false);
    for (const NodeHandle root : scene.rootNodes_)
    {
        if (!root || root.generation != initialGeneration || root.slot >= isRoot.size())
        {
            throw std::invalid_argument("cannot append a scene with an invalid root node");
        }
        isRoot[root.slot] = true;
    }
    for (std::size_t index = 0; index < scene.nodes_.size(); ++index)
    {
        NodeAsset& node = scene.nodes_[index];
        if (isRoot[index])
        {
            node.localTransform = rootTransform * node.localTransform;
            node.transformIsTrs = false;
        }
        node.skin = remap(node.skin, skinOffset);
        for (MeshHandle& mesh : node.meshes)
        {
            mesh = remap(mesh, meshOffset);
        }
        for (NodeHandle& child : node.children)
        {
            child = remap(child, nodeOffset);
        }
        addNode(std::move(node));
    }

    for (AnimationInstanceAsset& instance : scene.animationInstances_)
    {
        instance.clip = remap(instance.clip, animationOffset);
        for (AnimationNodeBindingAsset& binding : instance.nodeBindings)
        {
            binding.source = remap(binding.source, nodeOffset);
            binding.target = remap(binding.target, nodeOffset);
        }
        addAnimationInstance(std::move(instance));
    }

    std::vector<NodeHandle> appendedRoots;
    appendedRoots.reserve(scene.rootNodes_.size());
    for (const NodeHandle root : scene.rootNodes_)
    {
        const NodeHandle appended = remap(root, nodeOffset);
        addRootNode(appended);
        appendedRoots.push_back(appended);
    }
    return appendedRoots;
}

const TextureAsset* SceneAsset::find(TextureHandle handle) const noexcept
{
    return findAsset<TextureAsset>(textures_, handle);
}

const MaterialAsset* SceneAsset::find(MaterialHandle handle) const noexcept
{
    return findAsset<MaterialAsset>(materials_, handle);
}

const MeshAsset* SceneAsset::find(MeshHandle handle) const noexcept
{
    return findAsset<MeshAsset>(meshes_, handle);
}

const NodeAsset* SceneAsset::find(NodeHandle handle) const noexcept
{
    return findAsset<NodeAsset>(nodes_, handle);
}

const SkinAsset* SceneAsset::find(SkinHandle handle) const noexcept
{
    return findAsset<SkinAsset>(skins_, handle);
}

const AnimationClipAsset* SceneAsset::find(AnimationHandle handle) const noexcept
{
    return findAsset<AnimationClipAsset>(animations_, handle);
}

SceneAsset loadObj(const std::filesystem::path& path)
{
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> sourceMaterials;
    std::string warning;
    std::string error;
    const std::string baseDirectory = path.parent_path().string();

    if (!tinyobj::LoadObj(&attributes, &shapes, &sourceMaterials, &warning, &error,
                          path.string().c_str(), baseDirectory.c_str(), true))
    {
        throw std::runtime_error("failed to load OBJ '" + path.string() + "': " + warning + error);
    }

    SceneAsset scene;
    std::unordered_map<std::string, TextureHandle> loadedTextures;
    const TextureHandle fallbackAlbedo = scene.addTexture(makeWhiteTextureImpl());

    const auto loadTexture = [&](const std::string& name, ColorSpace colorSpace) -> TextureHandle
    {
        if (name.empty())
        {
            return {};
        }
        const std::filesystem::path texturePath = resolveTexturePath(path, name);
        const std::string key = texturePath.generic_string() +
                                (colorSpace == ColorSpace::srgb ? "#srgb" : "#linear");
        if (const auto existing = loadedTextures.find(key); existing != loadedTextures.end())
        {
            return existing->second;
        }

        const TextureHandle handle = scene.addTexture(decodeTexture(texturePath, colorSpace));
        loadedTextures.emplace(key, handle);
        return handle;
    };

    std::vector<MaterialHandle> materialHandles;
    materialHandles.reserve(std::max<std::size_t>(sourceMaterials.size(), 1U));
    for (const tinyobj::material_t& source : sourceMaterials)
    {
        MaterialAsset material;
        material.name = source.name;
        material.albedoTint = { source.diffuse[0], source.diffuse[1], source.diffuse[2], 1.0f };
        material.reflectance = std::max(source.shininess, 1.0f);
        material.roughnessFactor = std::clamp(
            std::sqrt(2.0f / (std::max(source.shininess, 0.0f) + 2.0f)), 0.045f, 1.0f);
        const std::string& albedoName = !source.diffuse_texname.empty()
            ? source.diffuse_texname
            : source.ambient_texname;
        material.albedoTexture = loadTexture(albedoName, ColorSpace::srgb);
        if (!material.albedoTexture)
        {
            material.albedoTexture = fallbackAlbedo;
        }
        material.normalTexture = loadTexture(source.bump_texname, ColorSpace::linear);
        materialHandles.push_back(scene.addMaterial(std::move(material)));
    }

    if (materialHandles.empty())
    {
        MaterialAsset material;
        material.name = "default material";
        material.albedoTexture = fallbackAlbedo;
        materialHandles.push_back(scene.addMaterial(std::move(material)));
    }

    std::vector<MeshHandle> importedMeshes;
    for (const tinyobj::shape_t& shape : shapes)
    {
        std::map<int, MeshBuilder> builders;
        std::size_t sourceIndexOffset = 0;
        for (std::size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face)
        {
            const int materialIndex = face < shape.mesh.material_ids.size()
                ? shape.mesh.material_ids[face]
                : -1;
            MeshBuilder& builder = builders[materialIndex];
            builder.name = shape.name;
            builder.materialIndex = materialIndex;

            const int vertexCount = shape.mesh.num_face_vertices[face];
            if (vertexCount != 3)
            {
                throw std::runtime_error("OBJ triangulation produced a non-triangle face");
            }
            for (int vertex = 0; vertex < vertexCount; ++vertex)
            {
                const tinyobj::index_t& sourceIndex = shape.mesh.indices[sourceIndexOffset +
                    static_cast<std::size_t>(vertex)];
                builder.indices.push_back(appendVertex(builder, attributes, sourceIndex));
            }
            sourceIndexOffset += static_cast<std::size_t>(vertexCount);
        }

        for (auto& [materialIndex, builder] : builders)
        {
            MeshAsset mesh;
            mesh.name = std::move(builder.name);
            if (builders.size() > 1U)
            {
                mesh.name += " material " + std::to_string(materialIndex);
            }
            mesh.vertices = std::move(builder.vertices);
            mesh.indices = std::move(builder.indices);
            mesh.material = materialIndex >= 0 &&
                            static_cast<std::size_t>(materialIndex) < materialHandles.size()
                ? materialHandles[static_cast<std::size_t>(materialIndex)]
                : materialHandles.front();
            finishGeometryImpl(mesh, builder.hasMissingNormals);
            importedMeshes.push_back(scene.addMesh(std::move(mesh)));
        }
    }

    if (scene.meshes().empty())
    {
        throw std::runtime_error("OBJ contains no renderable meshes: " + path.string());
    }

    NodeAsset root;
    root.name = path.stem().string();
    root.meshes = std::move(importedMeshes);
    scene.addRootNode(scene.addNode(std::move(root)));
    return scene;
}


SceneAsset loadScene(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension == ".obj")
    {
        return loadObj(path);
    }
    if (extension == ".gltf" || extension == ".glb")
    {
        return loadGltf(path);
    }
    throw std::invalid_argument("unsupported scene asset extension: " + extension);
}
}
