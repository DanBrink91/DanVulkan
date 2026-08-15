#include <danvulkan/renderer.hpp>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

namespace
{
// Deterministically exercises the host-platform contract used during minimization. The native
// surface stays valid while the adapter reports a zero framebuffer until waitEvents restores it.
class MinimizeRestorePlatform final : public RendererPlatform
{
public:
    explicit MinimizeRestorePlatform(std::shared_ptr<RendererPlatform> delegate)
        : delegate_(std::move(delegate))
    {
    }

    bool shouldClose() const noexcept override { return delegate_->shouldClose(); }
    void pollEvents() override { delegate_->pollEvents(); }
    void waitEvents() override
    {
        if (minimized_)
        {
            delegate_->pollEvents();
            minimized_ = false;
            framebufferResized_ = true;
            return;
        }
        delegate_->waitEvents();
    }
    RendererFramebufferExtent framebufferExtent() const noexcept override
    {
        return minimized_ ? RendererFramebufferExtent{} : delegate_->framebufferExtent();
    }
    bool consumeFramebufferResize() noexcept override
    {
        const bool injectedResize = std::exchange(framebufferResized_, false);
        const bool nativeResize = delegate_->consumeFramebufferResize();
        return injectedResize || nativeResize;
    }
    std::span<const char* const> requiredVulkanInstanceExtensions() const noexcept override
    {
        return delegate_->requiredVulkanInstanceExtensions();
    }
    VkSurfaceKHR createVulkanSurface(VkInstance instance) override
    {
        return delegate_->createVulkanSurface(instance);
    }
    void setWindowTitle(std::string_view title) override { delegate_->setWindowTitle(title); }
    bool requestWindowResize(std::uint32_t width, std::uint32_t height) override
    {
        if (width == 0 || height == 0)
        {
            minimized_ = true;
            framebufferResized_ = true;
            return true;
        }
        return delegate_->requestWindowResize(width, height);
    }

private:
    std::shared_ptr<RendererPlatform> delegate_;
    bool minimized_ = false;
    bool framebufferResized_ = false;
};

constexpr std::uint32_t animatedStressModelCount = 24;

danvulkan::assets::SceneAsset makeAnimatedStressScene()
{
    danvulkan::assets::SceneAsset scene =
        danvulkan::assets::loadScene("models/naruto_hiddenly_village.glb");
    const danvulkan::assets::SceneAsset character =
        danvulkan::assets::loadScene("models/ninja_run_free_fire_emote.glb");
    if (character.animations().size() != 1)
    {
        throw std::runtime_error("animated stress fixture requires exactly one character clip");
    }

    danvulkan::assets::AnimationClipAsset aggregate;
    aggregate.name = "all stress characters";
    aggregate.startTime = character.animations().front().startTime;
    aggregate.endTime = character.animations().front().endTime;
    const std::uint32_t sharedMeshOffset = static_cast<std::uint32_t>(scene.meshes().size());
    for (std::uint32_t index = 0; index < animatedStressModelCount; ++index)
    {
        constexpr std::uint32_t columns = 6;
        const float x = (static_cast<float>(index % columns) - 2.5f) * 0.8f;
        const float z = (static_cast<float>(index / columns) - 1.5f) * 0.8f;
        const glm::mat4 placement =
            glm::translate(glm::mat4(1.0f), glm::vec3(x, -0.335f, z)) *
            glm::scale(glm::mat4(1.0f), glm::vec3(0.2f));
        if (index == 0)
        {
            const std::size_t firstAppendedClip = scene.animations().size();
            static_cast<void>(scene.append(character, placement));
            const danvulkan::assets::AnimationClipAsset& appended =
                scene.animations()[firstAppendedClip];
            aggregate.channels.insert(aggregate.channels.end(), appended.channels.begin(),
                appended.channels.end());
            continue;
        }

        const std::uint32_t nodeOffset = static_cast<std::uint32_t>(scene.nodes().size());
        const auto remapNode = [nodeOffset](danvulkan::assets::NodeHandle handle)
        {
            return handle ? danvulkan::assets::NodeHandle{handle.slot + nodeOffset, 1U} :
                danvulkan::assets::NodeHandle{};
        };
        std::vector<danvulkan::assets::SkinHandle> skinHandles;
        skinHandles.reserve(character.skins().size());
        for (const danvulkan::assets::SkinAsset& source : character.skins())
        {
            danvulkan::assets::SkinAsset skin = source;
            skin.skeleton = remapNode(skin.skeleton);
            for (danvulkan::assets::NodeHandle& joint : skin.joints)
            {
                joint = remapNode(joint);
            }
            skinHandles.push_back(scene.addSkin(std::move(skin)));
        }

        std::vector<bool> rootNodes(character.nodes().size(), false);
        for (const danvulkan::assets::NodeHandle root : character.rootNodes())
        {
            rootNodes.at(root.slot) = true;
        }
        for (std::size_t nodeIndex = 0; nodeIndex < character.nodes().size(); ++nodeIndex)
        {
            danvulkan::assets::NodeAsset node = character.nodes()[nodeIndex];
            if (rootNodes[nodeIndex])
            {
                node.localTransform = placement * node.localTransform;
                node.transformIsTrs = false;
            }
            for (danvulkan::assets::MeshHandle& mesh : node.meshes)
            {
                mesh = {mesh.slot + sharedMeshOffset, 1U};
            }
            if (node.skin)
            {
                node.skin = skinHandles.at(node.skin.slot);
            }
            for (danvulkan::assets::NodeHandle& child : node.children)
            {
                child = remapNode(child);
            }
            static_cast<void>(scene.addNode(std::move(node)));
        }
        for (const danvulkan::assets::NodeHandle root : character.rootNodes())
        {
            scene.addRootNode(remapNode(root));
        }
        for (const danvulkan::assets::AnimationChannelAsset& source :
            character.animations().front().channels)
        {
            danvulkan::assets::AnimationChannelAsset channel = source;
            channel.target = remapNode(channel.target);
            aggregate.channels.push_back(std::move(channel));
        }
    }
    static_cast<void>(scene.addAnimation(std::move(aggregate)));
    return scene;
}
}

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
        bool animatedStress = false;
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
        else if (argc > 1 && std::string_view(argv[1]) == "--minimize-smoke-test")
        {
            config.maxFrames = 120;
            config.resizeAtFrame = 30;
            config.resizeWidth = 0;
            config.resizeHeight = 0;
            config.platform = std::make_shared<MinimizeRestorePlatform>(
                makeGlfwRendererPlatform(config.applicationName, config.width, config.height));
        }
        else if (argc > 1 && std::string_view(argv[1]) == "--step-smoke-test")
        {
            config.maxFrames = 120;
            config.initialVertexCapacity = 6;
            config.initialIndexCapacity = 6;
            stepDriven = true;
        }
        else if (argc > 1 && std::string_view(argv[1]) == "--animated-stress-test")
        {
            config.maxFrames = 120;
            config.additionalScenes.clear();
            animatedStress = true;
        }

        VulkanRenderer renderer(std::move(config));
        if (animatedStress)
        {
            renderer.initialize();
            const danvulkan::assets::SceneAsset stressScene = makeAnimatedStressScene();
            renderer.replaceScene(stressScene);
            const std::vector<SceneAnimationInfo> animations = renderer.sceneAnimations();
            if (animations.empty())
            {
                throw std::runtime_error("animated stress scene exposed no clips");
            }
            renderer.playAnimation(animations.back().handle);

            SceneSubmission submission;
            submission.cameraPosition = glm::vec3(0.0f, 3.0f, 8.0f);
            submission.view = glm::lookAt(submission.cameraPosition, glm::vec3(0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f));
            submission.projection = glm::perspective(glm::radians(60.0f),
                1280.0f / 720.0f, 0.1f, 100.0f);
            while (renderer.beginFrame())
            {
                renderer.submitScene(submission);
                renderer.endFrame();
            }

            const RendererPerformanceStats stats = renderer.performanceStats();
            const RendererMemoryStats memory = renderer.memoryStats();
            std::cout << "animated stress: models=" << animatedStressModelCount
                      << " frames=" << stats.renderedFrames
                      << " active_draws=" << stats.activeDraws
                      << " visible_draws=" << stats.visibleDraws
                      << " joints=" << stats.jointMatrices << '\n'
                      << "animated stress timings: frame_cpu_ms=" << stats.frameCpuMilliseconds
                      << " frame_gpu_ms=" << stats.frameGpuMilliseconds
                      << " animation_ms=" << stats.animationCpuMilliseconds
                      << " animation_evaluation_ms="
                      << stats.animationEvaluationCpuMilliseconds
                      << " animation_sync_ms="
                      << stats.animationSynchronizationCpuMilliseconds
                      << " culling_ms=" << stats.cullingCpuMilliseconds
                      << " buffer_writes_ms=" << stats.bufferWriteCpuMilliseconds
                      << " command_recording_ms=" << stats.commandRecordingCpuMilliseconds
                      << '\n'
                      << "animated stress memory: blocks=" << memory.blockCount
                      << " allocations=" << memory.allocationCount
                      << " block_bytes=" << memory.blockBytes
                      << " allocation_bytes=" << memory.allocationBytes
                      << " heap_usage_bytes=" << memory.heapUsageBytes
                      << " heap_budget_bytes=" << memory.heapBudgetBytes
                      << " peak_block_bytes=" << memory.peakBlockBytes
                      << " peak_allocation_bytes=" << memory.peakAllocationBytes
                      << " staging_arena_bytes=" << memory.stagingArenaBytes
                      << " staging_growths=" << memory.stagingArenaGrowthCount
                      << " upload_submissions=" << memory.uploadSubmissionCount
                      << " cpu_geometry_bytes=" << memory.retainedCpuGeometryBytes
                      << " cpu_scratch_bytes=" << memory.cpuScratchBytes
                      << " attachment_sets=" << memory.attachmentSetCount
                      << " msaa_samples=" << memory.msaaSamples
                      << '\n';
            if (stats.renderedFrames != 120 ||
                stats.activeDraws < animatedStressModelCount * 5 ||
                stats.visibleDraws == 0 ||
                stats.jointMatrices != animatedStressModelCount * 375 ||
                stats.animationCpuMilliseconds <= 0.0 ||
                memory.attachmentSetCount != 2 || memory.blockCount == 0 ||
                memory.allocationCount == 0 || memory.stagingArenaBytes == 0 ||
                memory.stagingArenaGrowthCount == 0 || memory.uploadSubmissionCount == 0 ||
                memory.retainedCpuGeometryBytes != 0 || memory.cpuScratchBytes == 0 ||
                memory.peakAllocationBytes < memory.allocationBytes)
            {
                throw std::runtime_error("animated stress scene did not exercise its workload");
            }
            renderer.shutdown();
        }
        else if (!stepDriven)
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
