#include <danvulkan/assets.hpp>

#include "assets_internal.hpp"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <glm/glm.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace danvulkan::assets
{
namespace
{
std::vector<std::byte> readFileBytes(const std::filesystem::path& path, std::size_t offset = 0)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("failed to open glTF resource: " + path.string());
    }

    const auto end = file.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) < offset)
    {
        throw std::runtime_error("invalid glTF resource byte offset: " + path.string());
    }
    const std::size_t byteCount = static_cast<std::size_t>(end) - offset;
    std::vector<std::byte> bytes(byteCount);
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file)
    {
        throw std::runtime_error("failed to read glTF resource: " + path.string());
    }
    return bytes;
}

std::vector<std::byte> encodedImageBytes(const fastgltf::Asset& asset,
                                         const fastgltf::Image& image,
                                         const std::filesystem::path& assetPath)
{
    return std::visit(fastgltf::visitor{
        [&](const fastgltf::sources::Array& source)
        {
            return std::vector<std::byte>(source.bytes.data(), source.bytes.data() + source.bytes.size());
        },
        [&](const fastgltf::sources::Vector& source)
        {
            return std::vector<std::byte>(source.bytes.begin(), source.bytes.end());
        },
        [&](const fastgltf::sources::ByteView& source)
        {
            return std::vector<std::byte>(source.bytes.begin(), source.bytes.end());
        },
        [&](const fastgltf::sources::BufferView& source)
        {
            const auto bytes = fastgltf::DefaultBufferDataAdapter{}(asset, source.bufferViewIndex);
            return std::vector<std::byte>(bytes.begin(), bytes.end());
        },
        [&](const fastgltf::sources::URI& source)
        {
            if (!source.uri.isLocalPath())
            {
                throw std::runtime_error("remote glTF image URIs are not supported");
            }
            const auto pathView = source.uri.path();
            std::filesystem::path imagePath(std::string(pathView.begin(), pathView.end()));
            if (imagePath.is_relative())
            {
                imagePath = assetPath.parent_path() / imagePath;
            }
            return readFileBytes(imagePath.lexically_normal(), source.fileByteOffset);
        },
        [&](const auto&) -> std::vector<std::byte>
        {
            throw std::runtime_error("unsupported glTF image data source");
        }
    }, image.data);
}

glm::mat4 matrix(const fastgltf::math::fmat4x4& source)
{
    glm::mat4 result(1.0f);
    for (glm::length_t column = 0; column < 4; ++column)
    {
        for (glm::length_t row = 0; row < 4; ++row)
        {
            result[column][row] = source[static_cast<std::size_t>(column)]
                                        [static_cast<std::size_t>(row)];
        }
    }
    return result;
}

glm::mat4 localTransform(const fastgltf::Node& node)
{
    return matrix(fastgltf::getTransformMatrix(node));
}

AnimationTarget animationTarget(fastgltf::AnimationPath path)
{
    switch (path)
    {
    case fastgltf::AnimationPath::Translation: return AnimationTarget::translation;
    case fastgltf::AnimationPath::Rotation: return AnimationTarget::rotation;
    case fastgltf::AnimationPath::Scale: return AnimationTarget::scale;
    case fastgltf::AnimationPath::Weights:
    default:
        throw std::runtime_error("glTF morph-weight animation is not currently supported");
    }
}

AnimationInterpolation animationInterpolation(fastgltf::AnimationInterpolation interpolation)
{
    switch (interpolation)
    {
    case fastgltf::AnimationInterpolation::Linear: return AnimationInterpolation::linear;
    case fastgltf::AnimationInterpolation::Step: return AnimationInterpolation::step;
    case fastgltf::AnimationInterpolation::CubicSpline:
    default:
        throw std::runtime_error("glTF cubic-spline animation is not currently supported");
    }
}

AlphaMode convertAlphaMode(fastgltf::AlphaMode mode)
{
    switch (mode)
    {
    case fastgltf::AlphaMode::Mask: return AlphaMode::mask;
    case fastgltf::AlphaMode::Blend: return AlphaMode::blend;
    case fastgltf::AlphaMode::Opaque:
    default: return AlphaMode::opaque;
    }
}

TextureFilter convertBaseFilter(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return TextureFilter::nearest;
    default:
        return TextureFilter::linear;
    }
}

TextureMipmapMode convertMipmapMode(fastgltf::Filter filter)
{
    switch (filter)
    {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
        return TextureMipmapMode::nearest;
    default:
        return TextureMipmapMode::linear;
    }
}

TextureWrap convertWrap(fastgltf::Wrap wrap)
{
    switch (wrap)
    {
    case fastgltf::Wrap::ClampToEdge: return TextureWrap::clampToEdge;
    case fastgltf::Wrap::MirroredRepeat: return TextureWrap::mirroredRepeat;
    case fastgltf::Wrap::Repeat:
    default: return TextureWrap::repeat;
    }
}

TextureSampler textureSampler(const fastgltf::Asset& asset, const fastgltf::Texture& texture)
{
    TextureSampler result;
    if (!texture.samplerIndex.has_value())
    {
        return result;
    }
    if (*texture.samplerIndex >= asset.samplers.size())
    {
        throw std::runtime_error("glTF texture references an invalid sampler");
    }

    const fastgltf::Sampler& source = asset.samplers[*texture.samplerIndex];
    if (source.magFilter.has_value())
    {
        result.magFilter = convertBaseFilter(*source.magFilter);
    }
    if (source.minFilter.has_value())
    {
        result.minFilter = convertBaseFilter(*source.minFilter);
        result.mipmapMode = convertMipmapMode(*source.minFilter);
    }
    result.wrapU = convertWrap(source.wrapS);
    result.wrapV = convertWrap(source.wrapT);
    return result;
}

std::string imageName(const fastgltf::Image& image, std::size_t imageIndex)
{
    return image.name.empty() ? "glTF image " + std::to_string(imageIndex) : std::string(image.name);
}
}

SceneAsset loadGltf(const std::filesystem::path& path)
{
    auto gltfFile = fastgltf::MappedGltfFile::FromPath(path);
    if (!gltfFile)
    {
        throw std::runtime_error("failed to open glTF '" + path.string() + "': " +
                                 std::string(fastgltf::getErrorMessage(gltfFile.error())));
    }

    constexpr fastgltf::Extensions extensions = fastgltf::Extensions::KHR_mesh_quantization |
        fastgltf::Extensions::KHR_materials_unlit;
    constexpr fastgltf::Options options = fastgltf::Options::LoadExternalBuffers |
                                          fastgltf::Options::LoadExternalImages |
                                          fastgltf::Options::GenerateMeshIndices;
    fastgltf::Parser parser(extensions);
    auto parsed = parser.loadGltf(gltfFile.get(), path.parent_path(), options,
        fastgltf::Category::OnlyRenderable | fastgltf::Category::Scenes |
        fastgltf::Category::Skins | fastgltf::Category::Animations);
    if (parsed.error() != fastgltf::Error::None)
    {
        throw std::runtime_error("failed to parse glTF '" + path.string() + "': " +
                                 std::string(fastgltf::getErrorMessage(parsed.error())));
    }
    fastgltf::Asset asset = std::move(parsed.get());

    SceneAsset scene;
    const TextureHandle fallbackAlbedo = scene.addTexture(detail::makeWhiteTexture());
    std::unordered_map<std::uint64_t, TextureHandle> textureHandles;

    const auto loadTexture = [&](const auto& textureInfo, ColorSpace colorSpace) -> TextureHandle
    {
        if (!textureInfo.has_value())
        {
            return {};
        }
        const std::size_t textureIndex = textureInfo->textureIndex;
        if (textureIndex >= asset.textures.size() || !asset.textures[textureIndex].imageIndex.has_value())
        {
            throw std::runtime_error("glTF material references an invalid texture");
        }

        const std::uint64_t key = (static_cast<std::uint64_t>(textureIndex) << 1U) |
                                  (colorSpace == ColorSpace::srgb ? 1U : 0U);
        if (const auto existing = textureHandles.find(key); existing != textureHandles.end())
        {
            return existing->second;
        }

        const std::size_t imageIndex = *asset.textures[textureIndex].imageIndex;
        if (imageIndex >= asset.images.size())
        {
            throw std::runtime_error("glTF texture references an invalid image");
        }
        const fastgltf::Texture& sourceTexture = asset.textures[textureIndex];
        const fastgltf::Image& sourceImage = asset.images[imageIndex];
        const std::vector<std::byte> encoded = encodedImageBytes(asset, sourceImage, path);
        TextureAsset decoded = detail::decodeTextureMemory(
            encoded, imageName(sourceImage, imageIndex), path, colorSpace);
        decoded.sampler = textureSampler(asset, sourceTexture);
        const TextureHandle handle = scene.addTexture(std::move(decoded));
        textureHandles.emplace(key, handle);
        return handle;
    };

    std::vector<MaterialHandle> materialHandles;
    materialHandles.reserve(asset.materials.size());
    for (const fastgltf::Material& source : asset.materials)
    {
        MaterialAsset material;
        material.name = std::string(source.name);
        material.albedoTint = {
            source.pbrData.baseColorFactor[0], source.pbrData.baseColorFactor[1],
            source.pbrData.baseColorFactor[2], source.pbrData.baseColorFactor[3]
        };
        material.metallicFactor = source.pbrData.metallicFactor;
        material.roughnessFactor = source.pbrData.roughnessFactor;
        material.reflectance = std::max((1.0f - material.roughnessFactor) * 128.0f, 1.0f);
        material.emissiveFactor = {
            source.emissiveFactor[0], source.emissiveFactor[1], source.emissiveFactor[2]
        };
        material.alphaCutoff = source.alphaCutoff;
        material.alphaMode = convertAlphaMode(source.alphaMode);
        material.doubleSided = source.doubleSided;
        material.unlit = source.unlit;
        material.albedoTexture = loadTexture(source.pbrData.baseColorTexture, ColorSpace::srgb);
        if (!material.albedoTexture)
        {
            material.albedoTexture = fallbackAlbedo;
        }
        material.metallicRoughnessTexture =
            loadTexture(source.pbrData.metallicRoughnessTexture, ColorSpace::linear);
        material.normalTexture = loadTexture(source.normalTexture, ColorSpace::linear);
        if (source.normalTexture.has_value())
        {
            material.normalScale = source.normalTexture->scale;
        }
        material.occlusionTexture = loadTexture(source.occlusionTexture, ColorSpace::linear);
        if (source.occlusionTexture.has_value())
        {
            material.occlusionStrength = source.occlusionTexture->strength;
        }
        material.emissiveTexture = loadTexture(source.emissiveTexture, ColorSpace::srgb);
        materialHandles.push_back(scene.addMaterial(std::move(material)));
    }

    MaterialAsset defaultMaterial;
    defaultMaterial.name = "default glTF material";
    defaultMaterial.albedoTexture = fallbackAlbedo;
    const MaterialHandle defaultMaterialHandle = scene.addMaterial(std::move(defaultMaterial));

    std::vector<std::vector<MeshHandle>> meshHandles(asset.meshes.size());
    for (std::size_t meshIndex = 0; meshIndex < asset.meshes.size(); ++meshIndex)
    {
        const fastgltf::Mesh& sourceMesh = asset.meshes[meshIndex];
        for (std::size_t primitiveIndex = 0; primitiveIndex < sourceMesh.primitives.size(); ++primitiveIndex)
        {
            const fastgltf::Primitive& primitive = sourceMesh.primitives[primitiveIndex];
            if (primitive.type != fastgltf::PrimitiveType::Triangles)
            {
                throw std::runtime_error("only triangle glTF primitives are currently supported");
            }

            const auto* positionAttribute = primitive.findAttribute("POSITION");
            if (positionAttribute == primitive.attributes.end())
            {
                throw std::runtime_error("glTF primitive is missing POSITION data");
            }
            const fastgltf::Accessor& positionAccessor = asset.accessors[positionAttribute->accessorIndex];

            MeshAsset mesh;
            mesh.name = sourceMesh.name.empty() ? "glTF mesh " + std::to_string(meshIndex)
                                                : std::string(sourceMesh.name);
            if (sourceMesh.primitives.size() > 1U)
            {
                mesh.name += " primitive " + std::to_string(primitiveIndex);
            }
            mesh.vertices.resize(positionAccessor.count);
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                asset, positionAccessor, [&](const fastgltf::math::fvec3& value, std::size_t index)
                {
                    mesh.vertices[index].pos = { value[0], value[1], value[2] };
                });

            bool generateNormals = true;
            if (const auto* attribute = primitive.findAttribute("NORMAL");
                attribute != primitive.attributes.end())
            {
                generateNormals = false;
                const fastgltf::Accessor& accessor = asset.accessors[attribute->accessorIndex];
                if (accessor.count != mesh.vertices.size())
                {
                    throw std::runtime_error("glTF NORMAL count does not match POSITION count");
                }
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, accessor, [&](const fastgltf::math::fvec3& value, std::size_t index)
                    {
                        mesh.vertices[index].normal = { value[0], value[1], value[2] };
                    });
            }

            if (const auto* attribute = primitive.findAttribute("TEXCOORD_0");
                attribute != primitive.attributes.end())
            {
                const fastgltf::Accessor& accessor = asset.accessors[attribute->accessorIndex];
                if (accessor.count != mesh.vertices.size())
                {
                    throw std::runtime_error("glTF TEXCOORD_0 count does not match POSITION count");
                }
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
                    asset, accessor, [&](const fastgltf::math::fvec2& value, std::size_t index)
                    {
                        mesh.vertices[index].texCoord = { value[0], value[1] };
                    });
            }

            if (const auto* attribute = primitive.findAttribute("TANGENT");
                attribute != primitive.attributes.end())
            {
                const fastgltf::Accessor& accessor = asset.accessors[attribute->accessorIndex];
                if (accessor.count != mesh.vertices.size())
                {
                    throw std::runtime_error("glTF TANGENT count does not match POSITION count");
                }
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
                    asset, accessor, [&](const fastgltf::math::fvec4& value, std::size_t index)
                    {
                        mesh.vertices[index].tangent = { value[0], value[1], value[2] };
                        mesh.vertices[index].tangentSign = value[3];
                });
            }

            const auto* jointAttribute = primitive.findAttribute("JOINTS_0");
            const auto* weightAttribute = primitive.findAttribute("WEIGHTS_0");
            if ((jointAttribute == primitive.attributes.end()) !=
                (weightAttribute == primitive.attributes.end()))
            {
                throw std::runtime_error(
                    "glTF primitive must provide JOINTS_0 and WEIGHTS_0 together");
            }
            if (jointAttribute != primitive.attributes.end())
            {
                const fastgltf::Accessor& jointAccessor =
                    asset.accessors[jointAttribute->accessorIndex];
                const fastgltf::Accessor& weightAccessor =
                    asset.accessors[weightAttribute->accessorIndex];
                if (jointAccessor.count != mesh.vertices.size() ||
                    weightAccessor.count != mesh.vertices.size())
                {
                    throw std::runtime_error(
                        "glTF skin attribute count does not match POSITION count");
                }
                fastgltf::iterateAccessorWithIndex<fastgltf::math::uvec4>(
                    asset, jointAccessor,
                    [&](const fastgltf::math::uvec4& value, std::size_t index)
                    {
                        mesh.vertices[index].joints = {
                            value[0], value[1], value[2], value[3]
                        };
                    });
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
                    asset, weightAccessor,
                    [&](const fastgltf::math::fvec4& value, std::size_t index)
                    {
                        glm::vec4 weights(value[0], value[1], value[2], value[3]);
                        const float total = weights.x + weights.y + weights.z + weights.w;
                        mesh.vertices[index].weights = total > 0.0f ? weights / total
                                                                   : glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                    });
            }

            if (!primitive.indicesAccessor.has_value())
            {
                throw std::runtime_error("fastgltf failed to generate primitive indices");
            }
            const fastgltf::Accessor& indexAccessor = asset.accessors[*primitive.indicesAccessor];
            mesh.indices.resize(indexAccessor.count);
            fastgltf::copyFromAccessor<std::uint32_t>(asset, indexAccessor, mesh.indices.data());
            mesh.material = primitive.materialIndex.has_value()
                ? materialHandles.at(*primitive.materialIndex)
                : defaultMaterialHandle;
            detail::finishGeometry(mesh, generateNormals);
            meshHandles[meshIndex].push_back(scene.addMesh(std::move(mesh)));
        }
    }

    std::vector<NodeHandle> nodeHandles;
    nodeHandles.reserve(asset.nodes.size());
    for (std::size_t index = 0; index < asset.nodes.size(); ++index)
    {
        nodeHandles.push_back(NodeHandle{ static_cast<std::uint32_t>(index), 1U });
    }

    std::vector<SkinHandle> skinHandles;
    skinHandles.reserve(asset.skins.size());
    for (std::size_t skinIndex = 0; skinIndex < asset.skins.size(); ++skinIndex)
    {
        const fastgltf::Skin& source = asset.skins[skinIndex];
        SkinAsset skin;
        skin.name = source.name.empty() ? "glTF skin " + std::to_string(skinIndex)
                                       : std::string(source.name);
        if (source.skeleton.has_value())
        {
            skin.skeleton = nodeHandles.at(*source.skeleton);
        }
        skin.joints.reserve(source.joints.size());
        for (const std::size_t joint : source.joints)
        {
            skin.joints.push_back(nodeHandles.at(joint));
        }
        skin.inverseBindMatrices.assign(skin.joints.size(), glm::mat4(1.0f));
        if (source.inverseBindMatrices.has_value())
        {
            const fastgltf::Accessor& accessor =
                asset.accessors.at(*source.inverseBindMatrices);
            if (accessor.count != skin.joints.size())
            {
                throw std::runtime_error(
                    "glTF inverse-bind-matrix count does not match skin joint count");
            }
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fmat4x4>(
                asset, accessor,
                [&](const fastgltf::math::fmat4x4& value, std::size_t index)
                {
                    skin.inverseBindMatrices[index] = matrix(value);
                });
        }
        skinHandles.push_back(scene.addSkin(std::move(skin)));
    }

    for (std::size_t index = 0; index < asset.nodes.size(); ++index)
    {
        const fastgltf::Node& source = asset.nodes[index];
        NodeAsset node;
        node.name = source.name.empty() ? "glTF node " + std::to_string(index) : std::string(source.name);
        node.localTransform = localTransform(source);
        if (const auto* trs = std::get_if<fastgltf::TRS>(&source.transform))
        {
            node.translation = { trs->translation[0], trs->translation[1], trs->translation[2] };
            node.rotation = {
                trs->rotation[0], trs->rotation[1], trs->rotation[2], trs->rotation[3]
            };
            node.scale = { trs->scale[0], trs->scale[1], trs->scale[2] };
            node.transformIsTrs = true;
        }
        if (source.meshIndex.has_value())
        {
            node.meshes = meshHandles.at(*source.meshIndex);
        }
        if (source.skinIndex.has_value())
        {
            node.skin = skinHandles.at(*source.skinIndex);
        }
        for (const std::size_t child : source.children)
        {
            node.children.push_back(nodeHandles.at(child));
        }
        const NodeHandle actualHandle = scene.addNode(std::move(node));
        if (actualHandle != nodeHandles[index])
        {
            throw std::runtime_error("internal glTF node handle assignment mismatch");
        }
    }

    for (std::size_t animationIndex = 0; animationIndex < asset.animations.size();
         ++animationIndex)
    {
        const fastgltf::Animation& sourceAnimation = asset.animations[animationIndex];
        AnimationClipAsset animation;
        animation.name = sourceAnimation.name.empty()
            ? "glTF animation " + std::to_string(animationIndex)
            : std::string(sourceAnimation.name);
        animation.startTime = std::numeric_limits<float>::max();
        animation.endTime = std::numeric_limits<float>::lowest();
        animation.channels.reserve(sourceAnimation.channels.size());

        for (const fastgltf::AnimationChannel& sourceChannel : sourceAnimation.channels)
        {
            if (!sourceChannel.nodeIndex.has_value())
            {
                throw std::runtime_error("glTF animation channel has no target node");
            }
            if (sourceChannel.samplerIndex >= sourceAnimation.samplers.size())
            {
                throw std::runtime_error("glTF animation channel references an invalid sampler");
            }
            const fastgltf::AnimationSampler& sourceSampler =
                sourceAnimation.samplers[sourceChannel.samplerIndex];
            const fastgltf::Accessor& input = asset.accessors.at(sourceSampler.inputAccessor);
            const fastgltf::Accessor& output = asset.accessors.at(sourceSampler.outputAccessor);
            if (input.count == 0 || output.count != input.count)
            {
                throw std::runtime_error("glTF animation sampler has mismatched keyframe data");
            }

            AnimationChannelAsset channel;
            channel.target = nodeHandles.at(*sourceChannel.nodeIndex);
            channel.path = animationTarget(sourceChannel.path);
            channel.interpolation = animationInterpolation(sourceSampler.interpolation);
            channel.times.resize(input.count);
            fastgltf::copyFromAccessor<float>(asset, input, channel.times.data());
            if (!std::ranges::is_sorted(channel.times))
            {
                throw std::runtime_error("glTF animation keyframe times are not sorted");
            }
            channel.values.resize(output.count);
            if (channel.path == AnimationTarget::rotation)
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
                    asset, output,
                    [&](const fastgltf::math::fvec4& value, std::size_t index)
                    {
                        channel.values[index] = { value[0], value[1], value[2], value[3] };
                    });
            }
            else
            {
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                    asset, output,
                    [&](const fastgltf::math::fvec3& value, std::size_t index)
                    {
                        channel.values[index] = { value[0], value[1], value[2], 0.0f };
                    });
            }
            animation.startTime = std::min(animation.startTime, channel.times.front());
            animation.endTime = std::max(animation.endTime, channel.times.back());
            animation.channels.push_back(std::move(channel));
        }
        if (!animation.channels.empty())
        {
            scene.addAnimation(std::move(animation));
        }
    }

    if (!asset.scenes.empty())
    {
        const std::size_t sceneIndex = asset.defaultScene.value_or(0);
        if (sceneIndex >= asset.scenes.size())
        {
            throw std::runtime_error("glTF default scene index is invalid");
        }
        for (const std::size_t root : asset.scenes[sceneIndex].nodeIndices)
        {
            scene.addRootNode(nodeHandles.at(root));
        }
    }
    else
    {
        std::vector<bool> isChild(asset.nodes.size(), false);
        for (const fastgltf::Node& node : asset.nodes)
        {
            for (const std::size_t child : node.children)
            {
                isChild.at(child) = true;
            }
        }
        for (std::size_t index = 0; index < isChild.size(); ++index)
        {
            if (!isChild[index])
            {
                scene.addRootNode(nodeHandles[index]);
            }
        }
    }

    if (scene.meshes().empty() || scene.rootNodes().empty())
    {
        throw std::runtime_error("glTF contains no renderable scene: " + path.string());
    }
    return scene;
}
}
