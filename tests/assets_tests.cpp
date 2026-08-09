#include <danvulkan/assets.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace
{
void require(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

bool nearlyEqual(float left, float right)
{
    return std::abs(left - right) < 0.0001f;
}
}

int main()
{
    try
    {
        const danvulkan::assets::SceneAsset scene =
            danvulkan::assets::loadScene("models/triangle.obj");

        require(scene.textures().size() == 2, "expected fallback and material textures");
        require(scene.materials().size() == 1, "expected one material");
        require(scene.meshes().size() == 1, "expected one mesh");
        require(scene.nodes().size() == 1, "expected one OBJ root node");
        require(scene.rootNodes().size() == 1, "expected one OBJ scene root");

        const auto& material = scene.materials().front();
        require(scene.find(material.handle) == &material, "material handle did not resolve");
        require(scene.find(material.albedoTexture) != nullptr, "albedo texture handle did not resolve");
        auto staleTextureHandle = material.albedoTexture;
        ++staleTextureHandle.generation;
        require(scene.find(staleTextureHandle) == nullptr, "stale texture generation unexpectedly resolved");

        const auto& mesh = scene.meshes().front();
        require(scene.find(mesh.handle) == &mesh, "mesh handle did not resolve");
        require(scene.find(mesh.material) == &material, "mesh material handle did not resolve");
        require(mesh.vertices.size() == 3, "expected three unique vertices");
        require(mesh.indices.size() == 3, "expected one triangle");
        require(nearlyEqual(mesh.bounds.minVertex.x, -1.0f), "unexpected minimum x bound");
        require(nearlyEqual(mesh.bounds.maxVertex.y, 1.0f), "unexpected maximum y bound");

        const auto* albedo = scene.find(material.albedoTexture);
        require(albedo->width > 0 && albedo->height > 0, "decoded texture is empty");
        require(albedo->rgba8.size() == static_cast<std::size_t>(albedo->width) *
                                           albedo->height * 4U,
                "decoded texture byte count is invalid");

        const danvulkan::assets::SceneAsset gltf =
            danvulkan::assets::loadScene("models/triangle.gltf");
        require(gltf.textures().size() == 2, "expected fallback and glTF base-color textures");
        require(gltf.materials().size() == 2, "expected glTF and fallback materials");
        require(gltf.meshes().size() == 1, "expected one glTF primitive mesh");
        require(gltf.nodes().size() == 2, "expected two glTF nodes");
        require(gltf.rootNodes().size() == 1, "expected one glTF scene root");

        const auto& gltfMaterial = gltf.materials().front();
        require(nearlyEqual(gltfMaterial.metallicFactor, 0.2f), "metallic factor was not imported");
        require(nearlyEqual(gltfMaterial.roughnessFactor, 0.65f), "roughness factor was not imported");
        require(nearlyEqual(gltfMaterial.emissiveFactor.x, 0.02f),
                "emissive factor was not imported");
        require(nearlyEqual(gltfMaterial.alphaCutoff, 0.4f), "alpha cutoff was not imported");
        require(gltfMaterial.alphaMode == danvulkan::assets::AlphaMode::mask,
                "alpha mode was not imported");
        require(gltfMaterial.doubleSided, "double-sided state was not imported");
        require(gltf.find(gltfMaterial.albedoTexture) != nullptr,
                "glTF base-color texture handle did not resolve");
        const auto* gltfAlbedo = gltf.find(gltfMaterial.albedoTexture);
        require(gltfAlbedo->sampler.magFilter == danvulkan::assets::TextureFilter::linear,
                "glTF magnification filter was not imported");
        require(gltfAlbedo->sampler.minFilter == danvulkan::assets::TextureFilter::linear,
                "glTF minification filter was not imported");
        require(gltfAlbedo->sampler.mipmapMode == danvulkan::assets::TextureMipmapMode::linear,
                "glTF mipmap filter was not imported");
        require(gltfAlbedo->sampler.wrapU == danvulkan::assets::TextureWrap::repeat,
                "glTF horizontal wrap mode was not imported");
        require(std::abs(gltf.meshes().front().vertices.front().tangentSign) == 1.0f,
                "generated tangent handedness is invalid");

        const auto* root = gltf.find(gltf.rootNodes().front());
        require(root != nullptr && root->children.size() == 1, "glTF root hierarchy was not imported");
        require(nearlyEqual(root->localTransform[3].x, 0.25f), "glTF root transform was not imported");
        const auto* child = gltf.find(root->children.front());
        require(child != nullptr && child->meshes.size() == 1, "glTF mesh instance was not imported");
        require(nearlyEqual(child->localTransform[3].y, 0.25f), "glTF child transform was not imported");
        require(gltf.find(child->meshes.front()) == &gltf.meshes().front(),
                "glTF node mesh handle did not resolve");

        const danvulkan::assets::SceneAsset glb =
            danvulkan::assets::loadScene("models/naruto_hiddenly_village.glb");
        require(glb.materials().size() >= 1, "expected a GLB material");
        require(glb.meshes().size() == 2, "expected two GLB primitive meshes");
        require(glb.rootNodes().size() == 1, "expected one GLB scene root");
        const auto& glbMaterial = glb.materials().front();
        require(glbMaterial.unlit, "KHR_materials_unlit was not imported");
        const auto* glbAlbedo = glb.find(glbMaterial.albedoTexture);
        require(glbAlbedo != nullptr && glbAlbedo->width == 2048 && glbAlbedo->height == 2048,
                "embedded GLB atlas dimensions are incorrect");
        require(glbAlbedo->sampler.mipmapMode == danvulkan::assets::TextureMipmapMode::linear,
                "GLB trilinear sampler was not imported");

        danvulkan::assets::SceneAsset animated =
            danvulkan::assets::loadScene("models/ninja_run_free_fire_emote.glb");
        require(animated.meshes().size() == 5, "expected five animated character meshes");
        require(animated.skins().size() == 1, "expected one character skin");
        require(animated.animations().size() == 1, "expected one character animation");
        const auto& skin = animated.skins().front();
        require(skin.joints.size() == 75, "unexpected character joint count");
        require(skin.inverseBindMatrices.size() == skin.joints.size(),
                "character inverse bind matrices were not imported");
        require(animated.find(skin.joints.front()) != nullptr,
                "character skin joint handle did not resolve");
        const auto& clip = animated.animations().front();
        require(clip.channels.size() == 152, "unexpected character animation channel count");
        require(nearlyEqual(clip.endTime - clip.startTime, 0.52f),
                "unexpected character animation duration");
        require(animated.find(clip.channels.front().target) != nullptr,
                "animation target node handle did not resolve");
        bool foundSkinnedNode = false;
        for (const auto& node : animated.nodes())
        {
            if (!node.skin)
            {
                continue;
            }
            foundSkinnedNode = true;
            require(animated.find(node.skin) == &skin, "node skin handle did not resolve");
        }
        require(foundSkinnedNode, "character scene contains no skinned mesh node");
        const glm::vec4 weights = animated.meshes().front().vertices.front().weights;
        require(nearlyEqual(weights.x + weights.y + weights.z + weights.w, 1.0f),
                "character vertex skin weights were not normalized");

        danvulkan::assets::SceneAsset composed =
            danvulkan::assets::loadScene("models/naruto_hiddenly_village.glb");
        const std::size_t villageNodeCount = composed.nodes().size();
        const std::vector<danvulkan::assets::NodeHandle> appendedRoots =
            composed.append(std::move(animated));
        require(appendedRoots.size() == 1 && composed.rootNodes().size() == 2,
                "scene composition did not preserve both roots");
        require(composed.nodes().size() == villageNodeCount + 96,
                "scene composition lost character nodes");
        require(composed.find(composed.skins().front().joints.front()) != nullptr,
                "scene composition did not remap skin joints");
        require(composed.find(composed.animations().front().channels.front().target) != nullptr,
                "scene composition did not remap animation targets");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
