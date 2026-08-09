#include <danvulkan/renderer.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

int main(int argc, char** argv)
{
    try
    {
        RendererConfig config;
        config.additionalScenes.push_back({
            "models/ninja_run_free_fire_emote.glb",
            glm::translate(glm::mat4(1.0f), glm::vec3(2.18f, -0.335f, -2.89f)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(0.2f))
        });
        bool stepDriven = false;
        if (argc > 1 && std::string_view(argv[1]) == "--smoke-test")
        {
            config.maxFrames = 120;
        }
        else if (argc > 1 && std::string_view(argv[1]) == "--platform-smoke-test")
        {
            config.maxFrames = 120;
            config.platform = makeGlfwRendererPlatform(config.applicationName,
                config.width, config.height);
        }
        else if (argc > 1 && std::string_view(argv[1]) == "--resize-smoke-test")
        {
            config.maxFrames = 120;
            config.resizeAtFrame = 30;
        }
        else if (argc > 1 && std::string_view(argv[1]) == "--step-smoke-test")
        {
            config.maxFrames = 120;
            config.initialVertexCapacity = 6;
            config.initialIndexCapacity = 6;
            stepDriven = true;
        }

        VulkanRenderer renderer(std::move(config));
        if (!stepDriven)
        {
            renderer.run();
        }
        else
        {
            renderer.initialize();

            SceneSubmission submission;
            submission.view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -3.0f));
            submission.projection = glm::perspective(glm::radians(60.0f), 1280.0f / 720.0f,
                0.1f, 100.0f);
            submission.cameraPosition = glm::vec3(0.0f, 0.0f, 3.0f);

            const std::vector<SceneInstanceInfo> instances = renderer.sceneInstances();
            const std::vector<SceneMaterialInfo> materials = renderer.sceneMaterials();
            const std::vector<SceneMeshInfo> meshes = renderer.sceneMeshes();
            const std::vector<SceneTextureInfo> initialTextures = renderer.sceneTextures();
            const std::vector<SceneAnimationInfo> animations = renderer.sceneAnimations();
            if (instances.empty() || materials.empty() || meshes.empty() || initialTextures.empty())
            {
                throw std::runtime_error("step smoke scene did not expose runtime handles");
            }
            if (animations.size() != 1 || animations.front().durationSeconds <= 0.0f)
            {
                throw std::runtime_error("step smoke scene did not expose its animation clip");
            }
            renderer.setAnimationPlaybackSpeed(1.25f);
            renderer.setAnimationLooping(false);
            renderer.seekAnimation(animations.front().durationSeconds * 0.5f);
            renderer.pauseAnimation();
            AnimationPlaybackState playback = renderer.animationPlaybackState();
            if (playback.clip != animations.front().handle ||
                playback.status != AnimationPlaybackStatus::paused || playback.looping ||
                playback.playbackSpeed != 1.25f)
            {
                throw std::runtime_error("animation control state was not retained");
            }
            renderer.resumeAnimation();
            renderer.setAnimationLooping(true);

            std::uint64_t frame = 0;
            SceneInstanceHandle dynamicInstance;
            SceneInstanceHandle retiredInstance;
            SceneInstanceHandle uploadedInstance;
            SceneMaterialHandle runtimeMaterial;
            SceneMaterialHandle replacementMaterial;
            SceneTextureHandle runtimeTexture;
            SceneMeshHandle uploadedMesh;
            SceneMeshHandle replacementMesh;

            std::array<danvulkan::assets::Vertex, 3> runtimeVertices{};
            runtimeVertices[0].pos = glm::vec3(-0.2f, -0.2f, 0.0f);
            runtimeVertices[1].pos = glm::vec3(0.2f, -0.2f, 0.0f);
            runtimeVertices[2].pos = glm::vec3(0.0f, 0.2f, 0.0f);
            runtimeVertices[0].texCoord = glm::vec2(0.0f, 0.0f);
            runtimeVertices[1].texCoord = glm::vec2(1.0f, 0.0f);
            runtimeVertices[2].texCoord = glm::vec2(0.5f, 1.0f);
            for (danvulkan::assets::Vertex& vertex : runtimeVertices)
            {
                vertex.normal = glm::vec3(0.0f, 0.0f, 1.0f);
                vertex.tangent = glm::vec3(1.0f, 0.0f, 0.0f);
                vertex.tangentSign = 1.0f;
            }
            constexpr std::array<std::uint32_t, 3> runtimeIndices{ 0, 1, 2 };

            while (renderer.beginFrame())
            {
                renderer.submitScene(submission);
                renderer.endFrame();
                ++frame;

                if (frame == 10)
                {
                    danvulkan::assets::TextureAsset checker;
                    checker.name = "runtime checker";
                    checker.width = 2;
                    checker.height = 2;
                    checker.colorSpace = danvulkan::assets::ColorSpace::srgb;
                    checker.sampler.magFilter = danvulkan::assets::TextureFilter::nearest;
                    checker.sampler.minFilter = danvulkan::assets::TextureFilter::nearest;
                    checker.sampler.wrapU = danvulkan::assets::TextureWrap::clampToEdge;
                    checker.sampler.wrapV = danvulkan::assets::TextureWrap::clampToEdge;
                    checker.rgba8 = {
                        std::byte{255}, std::byte{32}, std::byte{32}, std::byte{255},
                        std::byte{32}, std::byte{255}, std::byte{32}, std::byte{255},
                        std::byte{32}, std::byte{32}, std::byte{255}, std::byte{255},
                        std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255}
                    };
                    runtimeTexture = renderer.uploadTexture(checker);
                    RuntimeMaterialTextures textureBindings = materials.front().textures;
                    textureBindings.baseColor = runtimeTexture;
                    renderer.updateMaterialTextures(materials.front().handle, textureBindings);

                    RuntimeMaterialDescription materialDescription;
                    materialDescription.name = "runtime checker material";
                    materialDescription.properties = materials.front().properties;
                    materialDescription.textures = textureBindings;
                    materialDescription.alphaMode = danvulkan::assets::AlphaMode::opaque;
                    materialDescription.doubleSided = true;
                    runtimeMaterial = renderer.createMaterial(materialDescription);

                    if (renderer.sceneTextures().size() != initialTextures.size() + 1)
                    {
                        throw std::runtime_error("runtime texture upload was not registered");
                    }
                    const std::vector<SceneMaterialInfo> updatedMaterials =
                        renderer.sceneMaterials();
                    if (updatedMaterials.size() != materials.size() + 1 ||
                        updatedMaterials.back().handle != runtimeMaterial ||
                        updatedMaterials.back().name != materialDescription.name ||
                        updatedMaterials.back().alphaMode != materialDescription.alphaMode ||
                        updatedMaterials.back().doubleSided != materialDescription.doubleSided ||
                        updatedMaterials.back().unlit != materialDescription.unlit)
                    {
                        throw std::runtime_error("runtime material creation was not registered");
                    }
                }
                else if (frame == 15)
                {
                    uploadedMesh = renderer.uploadMesh(runtimeVertices,
                        runtimeIndices, runtimeMaterial, "runtime triangle");
                    uploadedInstance = renderer.createMeshInstance(uploadedMesh,
                        glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.65f, 0.0f)),
                        "uploaded runtime triangle");
                }
                else if (frame == 30)
                {
                    const glm::mat4 rotated = glm::rotate(instances.front().worldTransform,
                        glm::radians(15.0f), glm::vec3(0.0f, 0.0f, 1.0f));
                    renderer.updateInstanceTransform(instances.front().handle, rotated);

                    RuntimeMaterialProperties properties = materials.front().properties;
                    properties.roughnessFactor = 0.75f;
                    properties.emissiveFactor += glm::vec3(0.01f, 0.0f, 0.0f);
                    renderer.updateMaterialProperties(materials.front().handle, properties);

                    const glm::mat4 duplicateTransform = glm::translate(
                        instances.front().worldTransform, glm::vec3(0.75f, 0.0f, 0.0f));
                    dynamicInstance = renderer.createMeshInstance(meshes.front().handle,
                        duplicateTransform, "runtime duplicate");
                }
                else if (frame == 45)
                {
                    bool referencedMaterialRejected = false;
                    try
                    {
                        renderer.destroyMaterial(runtimeMaterial);
                    }
                    catch (const std::runtime_error&)
                    {
                        referencedMaterialRejected = true;
                    }
                    if (!referencedMaterialRejected)
                    {
                        throw std::runtime_error("referenced runtime material was destroyed");
                    }

                    bool referencedMeshRejected = false;
                    try
                    {
                        renderer.destroyMesh(uploadedMesh);
                    }
                    catch (const std::runtime_error&)
                    {
                        referencedMeshRejected = true;
                    }
                    if (!referencedMeshRejected)
                    {
                        throw std::runtime_error("referenced runtime mesh was destroyed");
                    }

                    renderer.destroyInstance(uploadedInstance);
                    renderer.destroyMesh(uploadedMesh);
                    if (renderer.sceneMeshes().size() != meshes.size())
                    {
                        throw std::runtime_error("destroyed runtime mesh remained registered");
                    }
                    replacementMesh = renderer.uploadMesh(runtimeVertices,
                        runtimeIndices, runtimeMaterial, "runtime replacement triangle");
                    if (replacementMesh.slot != uploadedMesh.slot ||
                        replacementMesh.generation == uploadedMesh.generation)
                    {
                        throw std::runtime_error("runtime mesh slot was not safely reused");
                    }
                    try
                    {
                        const SceneInstanceHandle unexpectedInstance =
                            renderer.createMeshInstance(uploadedMesh, glm::mat4(1.0f));
                        (void)unexpectedInstance;
                        throw std::runtime_error("destroyed runtime mesh handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }

                    bool referencedTextureRejected = false;
                    try
                    {
                        renderer.destroyTexture(runtimeTexture);
                    }
                    catch (const std::runtime_error&)
                    {
                        referencedTextureRejected = true;
                    }
                    if (!referencedTextureRejected)
                    {
                        throw std::runtime_error("referenced runtime texture was destroyed");
                    }

                    renderer.updateMaterialTextures(materials.front().handle,
                        materials.front().textures);
                    renderer.updateMaterialTextures(runtimeMaterial, materials.front().textures);
                    renderer.destroyTexture(runtimeTexture);
                    if (renderer.sceneTextures().size() != initialTextures.size())
                    {
                        throw std::runtime_error("destroyed runtime texture remained registered");
                    }

                    danvulkan::assets::TextureAsset replacement;
                    replacement.name = "runtime replacement";
                    replacement.width = 1;
                    replacement.height = 1;
                    replacement.colorSpace = danvulkan::assets::ColorSpace::srgb;
                    replacement.rgba8 = {
                        std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255}
                    };
                    const SceneTextureHandle replacementHandle =
                        renderer.uploadTexture(replacement);
                    if (replacementHandle.slot != runtimeTexture.slot ||
                        replacementHandle.generation == runtimeTexture.generation)
                    {
                        throw std::runtime_error("runtime texture slot was not safely reused");
                    }

                    RuntimeMaterialTextures staleBindings = materials.front().textures;
                    staleBindings.baseColor = runtimeTexture;
                    try
                    {
                        renderer.updateMaterialTextures(runtimeMaterial, staleBindings);
                        throw std::runtime_error("destroyed runtime texture handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }
                }
                else if (frame == 60)
                {
                    renderer.destroyMesh(replacementMesh);
                    renderer.destroyMaterial(runtimeMaterial);
                    if (renderer.sceneMaterials().size() != materials.size())
                    {
                        throw std::runtime_error("destroyed runtime material remained registered");
                    }

                    RuntimeMaterialDescription replacementDescription;
                    replacementDescription.name = "runtime replacement material";
                    replacementDescription.properties = materials.front().properties;
                    replacementDescription.textures = materials.front().textures;
                    replacementDescription.alphaMode = danvulkan::assets::AlphaMode::opaque;
                    replacementDescription.doubleSided = true;
                    replacementMaterial = renderer.createMaterial(replacementDescription);
                    if (replacementMaterial.slot != runtimeMaterial.slot ||
                        replacementMaterial.generation == runtimeMaterial.generation)
                    {
                        throw std::runtime_error("runtime material slot was not safely reused");
                    }
                    try
                    {
                        renderer.updateMaterialProperties(runtimeMaterial,
                            materials.front().properties);
                        throw std::runtime_error(
                            "destroyed runtime material handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }

                    retiredInstance = dynamicInstance;
                    renderer.destroyInstance(retiredInstance);

                    try
                    {
                        renderer.updateInstanceTransform(retiredInstance, glm::mat4(1.0f));
                        throw std::runtime_error("destroyed runtime instance handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }
                }
                else if (frame == 90)
                {
                    const SceneMeshHandle reclaimedMesh = renderer.uploadMesh(runtimeVertices,
                        runtimeIndices, replacementMaterial, "reclaimed runtime triangle");
                    if (reclaimedMesh.slot != replacementMesh.slot ||
                        reclaimedMesh.generation == replacementMesh.generation)
                    {
                        throw std::runtime_error("reclaimed mesh slot was not safely reused");
                    }

                    const glm::mat4 replacementTransform = glm::translate(
                        instances.front().worldTransform, glm::vec3(-0.75f, 0.0f, 0.0f));
                    dynamicInstance = renderer.createMeshInstance(meshes.front().handle,
                        replacementTransform, "reused runtime slot");
                    if (dynamicInstance.slot != retiredInstance.slot ||
                        dynamicInstance.generation == retiredInstance.generation)
                    {
                        throw std::runtime_error("runtime instance slot was not safely reused");
                    }
                }
                else if (frame == 105)
                {
                    const std::vector<SceneInstanceInfo> previousInstances =
                        renderer.sceneInstances();
                    const std::vector<SceneMaterialInfo> previousMaterials =
                        renderer.sceneMaterials();
                    const std::vector<SceneMeshInfo> previousMeshes = renderer.sceneMeshes();
                    const std::vector<SceneTextureInfo> previousTextures =
                        renderer.sceneTextures();
                    bool invalidReplacementRejected = false;
                    try
                    {
                        renderer.replaceScene(danvulkan::assets::SceneAsset{});
                    }
                    catch (const std::runtime_error&)
                    {
                        invalidReplacementRejected = true;
                    }
                    if (!invalidReplacementRejected)
                    {
                        throw std::runtime_error("invalid replacement scene was accepted");
                    }
                    if (renderer.sceneInstances().size() != previousInstances.size() ||
                        renderer.sceneMaterials().size() != previousMaterials.size() ||
                        renderer.sceneMeshes().size() != previousMeshes.size() ||
                        renderer.sceneTextures().size() != previousTextures.size())
                    {
                        throw std::runtime_error(
                            "invalid replacement changed the active scene");
                    }

                    const danvulkan::assets::SceneAsset replacementScene =
                        danvulkan::assets::loadScene("models/triangle.gltf");
                    renderer.replaceScene(replacementScene);
                    const std::vector<SceneInstanceInfo> replacementInstances =
                        renderer.sceneInstances();
                    const std::vector<SceneMaterialInfo> replacementMaterials =
                        renderer.sceneMaterials();
                    const std::vector<SceneMeshInfo> replacementMeshes = renderer.sceneMeshes();
                    const std::vector<SceneTextureInfo> replacementTextures =
                        renderer.sceneTextures();
                    if (replacementInstances.empty() ||
                        replacementMaterials.size() != replacementScene.materials().size() ||
                        replacementMeshes.size() != replacementScene.meshes().size() ||
                        replacementTextures.size() != replacementScene.textures().size())
                    {
                        throw std::runtime_error(
                            "replacement scene resources were not registered");
                    }
                    if (!renderer.sceneAnimations().empty())
                    {
                        throw std::runtime_error("static replacement exposed stale animations");
                    }
                    try
                    {
                        renderer.playAnimation(animations.front().handle);
                        throw std::runtime_error("old scene animation handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }

                    try
                    {
                        renderer.updateInstanceTransform(previousInstances.back().handle,
                            glm::mat4(1.0f));
                        throw std::runtime_error("old scene instance handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }
                    try
                    {
                        renderer.updateMaterialProperties(previousMaterials.back().handle,
                            previousMaterials.back().properties);
                        throw std::runtime_error("old scene material handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }
                    try
                    {
                        const SceneInstanceHandle unexpectedInstance =
                            renderer.createMeshInstance(previousMeshes.back().handle,
                                glm::mat4(1.0f));
                        (void)unexpectedInstance;
                        throw std::runtime_error("old scene mesh handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }

                    RuntimeMaterialTextures staleTextureBinding =
                        replacementMaterials.front().textures;
                    staleTextureBinding.baseColor = previousTextures.back().handle;
                    try
                    {
                        renderer.updateMaterialTextures(replacementMaterials.front().handle,
                            staleTextureBinding);
                        throw std::runtime_error("old scene texture handle remained valid");
                    }
                    catch (const std::invalid_argument&)
                    {
                    }
                }
            }
            renderer.shutdown();
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
