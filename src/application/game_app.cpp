#include "game_app.hpp"

#include "../glfw_platform.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace danvulkan::application {
namespace {
InputState translateInput(const DemoInputState& source, bool enabled) noexcept
{
    InputState result;
    if (!enabled)
    {
        return result;
    }
    result.moveForward = source.moveUp;
    result.moveBackward = source.moveDown;
    result.moveLeft = source.moveLeft;
    result.moveRight = source.moveRight;
    result.toggleCameraMode = source.toggleCameraMode;
    result.lookDeltaX = source.cursorDeltaX;
    result.lookDeltaY = source.cursorDeltaY;
    result.scrollDelta = source.scrollDeltaY;
    return result;
}
}

GameApp::GameApp(RendererConfig config, bool enableBackgroundMusic, bool showUiInitially,
    bool terrainTraversalCheck)
    : backgroundMusicEnabled_(enableBackgroundMusic),
      terrainTraversalCheck_(terrainTraversalCheck)
{
    GameWorld::configureStreamingCapacity(config);
    if (showUiInitially)
    {
        ui_.toggle();
    }
    platform_ = std::make_shared<GlfwRendererPlatform>(config.applicationName,
        config.width, config.height);
    config.platform = platform_;
    renderer_ = std::make_unique<VulkanRenderer>(std::move(config));
}

GameApp::~GameApp() = default;

void GameApp::run()
{
    renderer_->initialize();
    const RendererFramebufferExtent initialExtent = platform_->framebufferExtent();
    world_.initialize(*renderer_, initialExtent.width, initialExtent.height);
    if (backgroundMusicEnabled_)
    {
        try
        {
            audio_.playBackgroundMusic(std::filesystem::path("audio") / "bg.mp3");
        }
        catch (const std::exception& error)
        {
            // Rendering should remain usable on machines without an output device.
            std::cerr << "Background music disabled: " << error.what() << '\n';
        }
    }
    initializeUiControls();
    auto previousTime = std::chrono::steady_clock::now();
    const TerrainStreamingDiagnostics initialStreaming = world_.streamingDiagnostics();
    bool traversalComplete = false;
    bool traversalTargetReached = false;
    std::uint64_t traversalFrames = 0;
    std::uint64_t renderStallFrames = 0;
    std::uint64_t publishStallFrames = 0;
    double maximumFrameMilliseconds = 0.0;
    double maximumFrameCpuMilliseconds = 0.0;
    double maximumPublishMilliseconds = 0.0;
    constexpr double renderStallThresholdMilliseconds = 50.0;
    constexpr double publishStallThresholdMilliseconds = 8.0;
    constexpr double severeFrameStallMilliseconds = 1000.0;
    constexpr double severePublishStallMilliseconds = 500.0;

    try
    {
        while (renderer_->beginFrame())
        {
            const auto currentTime = std::chrono::steady_clock::now();
            const float measuredDeltaSeconds = std::clamp(
                std::chrono::duration<float>(currentTime - previousTime).count(), 0.0f, 0.1f);
            previousTime = currentTime;
            // Advance consistently on fast CI renderers and presentation-paced desktop drivers.
            // Generation still runs at real worker speed, so traversal stresses streaming without
            // making this check take the character's full real-time walking duration.
            const float deltaSeconds = terrainTraversalCheck_ ?
                0.1f : measuredDeltaSeconds;

            const DemoInputState platformInput = platform_->consumeDemoInput();
            if (platformInput.toggleUi)
            {
                ui_.toggle();
            }
            const RendererFramebufferExtent extent = platform_->framebufferExtent();
            const RendererPerformanceStats performance = renderer_->performanceStats();
            const RendererMemoryStats memory = renderer_->memoryStats();
            ui_.recordDiagnostics(performance, memory);
            const UiDrawData* uiDrawData = nullptr;
            if (ui_.visible())
            {
                UiInputState uiInput;
                uiInput.pointerX = platformInput.pointerX;
                uiInput.pointerY = platformInput.pointerY;
                uiInput.pointerDown = platformInput.pointerDown;
                uiInput.pointerPressed = platformInput.pointerPressed;
                uiInput.pointerReleased = platformInput.pointerReleased;
                uiInput.scrollDeltaY = platformInput.scrollDeltaY;
                uiInput.focusNext = platformInput.focusNext;
                uiInput.focusPrevious = platformInput.focusPrevious;
                uiInput.activateFocused = platformInput.activateFocused;
                uiInput.horizontalNavigation = platformInput.horizontalNavigation;
                uiInput.viewportWidth = extent.width;
                uiInput.viewportHeight = extent.height;
                uiDrawData = &ui_.build(uiInput, performance, memory, uiControls_);

                const UiInteractionResult interaction = ui_.interactionResult();
                if (platformInput.requestGameplayFocus ||
                    (ui_.interactionActive() && platformInput.pointerPressed &&
                        !interaction.wantsPointerInput))
                {
                    ui_.releaseInteraction();
                }
                else if (!ui_.interactionActive() && platformInput.pointerPressed &&
                    interaction.pointerOverUi)
                {
                    // The first click re-enters UI interaction without accidentally
                    // activating the widget beneath a captured cursor transition.
                    ui_.beginInteraction();
                }
            }

            const UiInteractionResult interaction = ui_.interactionResult();
            platform_->setCursorCaptured(
                !interaction.wantsPointerInput && !interaction.wantsKeyboardInput);
            const bool gameplayInputEnabled = !interaction.wantsKeyboardInput &&
                !platformInput.toggleUi;
            DemoInputState gameplayInput = platformInput;
            if (platformInput.pointerPressed || platformInput.toggleUi)
            {
                // Cursor capture changes reset the platform delta. Do not apply the delta
                // measured immediately before that transition to the camera.
                gameplayInput.cursorDeltaX = 0.0f;
                gameplayInput.cursorDeltaY = 0.0f;
            }
            InputState input = translateInput(gameplayInput, gameplayInputEnabled);
            if (terrainTraversalCheck_)
            {
                input = {};
                input.moveForward = !traversalTargetReached;
            }
            world_.update(input, deltaSeconds, extent.width, extent.height);
            renderer_->submitScene(world_.sceneSubmission());
            if (uiDrawData != nullptr)
            {
                renderer_->submitUi(*uiDrawData);
            }
            renderer_->endFrame();
            const auto publishStart = std::chrono::steady_clock::now();
            world_.streamTerrain(*renderer_);
            const double publishMilliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - publishStart).count();
            applyUiControls();

            if (terrainTraversalCheck_)
            {
                ++traversalFrames;
                const TerrainStreamingDiagnostics streaming = world_.streamingDiagnostics();
                const std::int32_t crossedChunks = std::abs(
                    streaming.playerChunk.z - initialStreaming.playerChunk.z);
                traversalTargetReached = crossedChunks >= 2;
                traversalComplete = traversalTargetReached &&
                    streaming.requestedWindowComplete && renderer_->runtimeUploadReady();

                if (traversalFrames % 500U == 0U)
                {
                    std::cout << "terrain traversal progress: frames=" << traversalFrames
                              << " player_chunk=[" << streaming.playerChunk.x << ','
                              << streaming.playerChunk.z << "] requested_center=["
                              << streaming.requestedCenter.x << ',' << streaming.requestedCenter.z
                              << "] complete_chunks=" << streaming.completeChunkCount
                              << std::endl;
                }

                const RendererPerformanceStats latest = renderer_->performanceStats();
                if (traversalFrames > 30U)
                {
                    maximumFrameMilliseconds = std::max(
                        maximumFrameMilliseconds, latest.frameMilliseconds);
                    maximumFrameCpuMilliseconds = std::max(
                        maximumFrameCpuMilliseconds, latest.frameCpuMilliseconds);
                    maximumPublishMilliseconds = std::max(
                        maximumPublishMilliseconds, publishMilliseconds);
                    renderStallFrames += latest.frameMilliseconds >
                        renderStallThresholdMilliseconds ? 1U : 0U;
                    publishStallFrames += publishMilliseconds >
                        publishStallThresholdMilliseconds ? 1U : 0U;
                }

                if (traversalComplete)
                {
                    std::cout << "terrain traversal check: frames=" << traversalFrames
                              << " crossed_chunks=" << crossedChunks
                              << " resident_chunks=" << streaming.residentChunkCount
                              << " complete_chunks=" << streaming.completeChunkCount
                              << " max_frame_ms=" << maximumFrameMilliseconds
                              << " max_frame_cpu_ms=" << maximumFrameCpuMilliseconds
                              << " max_publish_ms=" << maximumPublishMilliseconds
                              << " render_stalls_over_50ms=" << renderStallFrames
                              << " publish_stalls_over_8ms=" << publishStallFrames
                              << std::endl;
                    break;
                }
                // This mode can render far faster than an interactive application. Yield a small
                // amount of wall time so the check measures the persistent workers rather than
                // monopolizing their CPU cores with a synthetic foreground spin loop.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (terrainTraversalCheck_)
        {
            if (!traversalComplete)
            {
                const TerrainStreamingDiagnostics streaming = world_.streamingDiagnostics();
                std::cerr << "terrain traversal incomplete: frames=" << traversalFrames
                          << " player_chunk=[" << streaming.playerChunk.x << ','
                          << streaming.playerChunk.z << "] requested_center=["
                          << streaming.requestedCenter.x << ',' << streaming.requestedCenter.z
                          << "] resident_chunks=" << streaming.residentChunkCount
                          << " complete_chunks=" << streaming.completeChunkCount << '\n';
                throw std::runtime_error(
                    "terrain traversal check ended before crossing two chunks and completing "
                    "the requested streaming window");
            }
            if (maximumFrameMilliseconds > severeFrameStallMilliseconds ||
                maximumPublishMilliseconds > severePublishStallMilliseconds)
            {
                throw std::runtime_error(
                    "terrain traversal check detected a severe streaming stall");
            }
        }
        audio_.stopBackgroundMusic();
        renderer_->shutdown();
    }
    catch (...)
    {
        audio_.stopBackgroundMusic();
        if (renderer_->isInitialized())
        {
            try
            {
                renderer_->shutdown();
            }
            catch (...)
            {
            }
        }
        throw;
    }
}

void GameApp::initializeUiControls()
{
    uiControls_.audioAvailable = audio_.backgroundMusicActive();
    uiControls_.musicMuted = audio_.backgroundMusicMuted();
    uiControls_.musicVolume = audio_.backgroundMusicVolume();
    uiControls_.environment = world_.sceneSubmission().environment;
    uiControls_.atmosphere = world_.sceneSubmission().atmosphere;
    if (!world_.sceneSubmission().directionalLights.empty())
    {
        uiControls_.keyLightIntensity =
            world_.sceneSubmission().directionalLights.front().intensity;
    }

    const AnimationPlaybackState playback = renderer_->animationPlaybackState();
    uiControls_.animationAvailable = playback.clip.valid();
    uiControls_.animationPaused = playback.status == AnimationPlaybackStatus::paused;
    uiControls_.animationLooping = playback.looping;
    uiControls_.animationSpeed = playback.playbackSpeed;
    appliedUiControls_ = uiControls_;
}

void GameApp::applyUiControls()
{
    if (uiControls_.musicMuted != appliedUiControls_.musicMuted)
    {
        audio_.setBackgroundMusicMuted(uiControls_.musicMuted);
    }
    if (uiControls_.musicVolume != appliedUiControls_.musicVolume)
    {
        audio_.setBackgroundMusicVolume(uiControls_.musicVolume);
    }
    if (uiControls_.environment.intensity != appliedUiControls_.environment.intensity ||
        uiControls_.environment.rotation != appliedUiControls_.environment.rotation ||
        uiControls_.environment.diffuseStrength !=
            appliedUiControls_.environment.diffuseStrength ||
        uiControls_.environment.specularStrength !=
            appliedUiControls_.environment.specularStrength)
    {
        world_.setEnvironment(uiControls_.environment);
    }
    if (uiControls_.keyLightIntensity != appliedUiControls_.keyLightIntensity)
    {
        world_.setKeyLightIntensity(uiControls_.keyLightIntensity);
    }
    if (uiControls_.atmosphere.fogDensity != appliedUiControls_.atmosphere.fogDensity ||
        uiControls_.atmosphere.fogHeightFalloff !=
            appliedUiControls_.atmosphere.fogHeightFalloff ||
        uiControls_.atmosphere.mistVariation !=
            appliedUiControls_.atmosphere.mistVariation ||
        uiControls_.atmosphere.godRayStrength !=
            appliedUiControls_.atmosphere.godRayStrength ||
        uiControls_.atmosphere.cloudCoverage !=
            appliedUiControls_.atmosphere.cloudCoverage ||
        uiControls_.atmosphere.cloudDensity !=
            appliedUiControls_.atmosphere.cloudDensity ||
        uiControls_.atmosphere.cloudWindSpeed !=
            appliedUiControls_.atmosphere.cloudWindSpeed ||
        uiControls_.atmosphere.cloudShadowStrength !=
            appliedUiControls_.atmosphere.cloudShadowStrength)
    {
        world_.setAtmosphere(uiControls_.atmosphere);
    }

    if (!uiControls_.animationAvailable)
    {
        uiControls_.restartAnimationRequested = false;
        appliedUiControls_ = uiControls_;
        return;
    }

    AnimationPlaybackState playback = renderer_->animationPlaybackState();
    if (uiControls_.restartAnimationRequested)
    {
        renderer_->playAnimation(playback.clip, true);
        playback = renderer_->animationPlaybackState();
    }
    if (playback.looping != uiControls_.animationLooping)
    {
        renderer_->setAnimationLooping(uiControls_.animationLooping);
    }
    if (playback.playbackSpeed != uiControls_.animationSpeed)
    {
        renderer_->setAnimationPlaybackSpeed(uiControls_.animationSpeed);
    }
    if (uiControls_.animationPaused &&
        playback.status == AnimationPlaybackStatus::playing)
    {
        renderer_->pauseAnimation();
    }
    else if (!uiControls_.animationPaused &&
        playback.status == AnimationPlaybackStatus::paused)
    {
        renderer_->resumeAnimation();
    }
    uiControls_.restartAnimationRequested = false;
    appliedUiControls_ = uiControls_;
}

}
