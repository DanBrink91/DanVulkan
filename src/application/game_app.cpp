#include "game_app.hpp"

#include "../glfw_platform.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
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

GameApp::GameApp(RendererConfig config, bool enableBackgroundMusic, bool showUiInitially)
    : backgroundMusicEnabled_(enableBackgroundMusic)
{
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

    try
    {
        while (renderer_->beginFrame())
        {
            const auto currentTime = std::chrono::steady_clock::now();
            const float deltaSeconds = std::clamp(
                std::chrono::duration<float>(currentTime - previousTime).count(), 0.0f, 0.1f);
            previousTime = currentTime;

            const DemoInputState platformInput = platform_->consumeDemoInput();
            if (platformInput.toggleUi)
            {
                ui_.toggle();
            }
            const RendererFramebufferExtent extent = platform_->framebufferExtent();
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
                uiDrawData = &ui_.build(uiInput, renderer_->performanceStats(),
                    renderer_->memoryStats(), uiControls_);

                const bool clickedScene = platformInput.pointerPressed &&
                    !ui_.pointerOverUi();
                if (clickedScene || platformInput.requestGameplayFocus)
                {
                    ui_.dismiss();
                    uiDrawData = nullptr;
                }
            }

            platform_->setCursorCaptured(!ui_.visible());
            const bool gameplayInputEnabled = !ui_.visible() && !platformInput.toggleUi;
            DemoInputState gameplayInput = platformInput;
            if (platformInput.pointerPressed || platformInput.toggleUi)
            {
                // Cursor capture changes reset the platform delta. Do not apply the delta
                // measured immediately before that transition to the camera.
                gameplayInput.cursorDeltaX = 0.0f;
                gameplayInput.cursorDeltaY = 0.0f;
            }
            const InputState input = translateInput(gameplayInput, gameplayInputEnabled);
            world_.update(input, deltaSeconds, extent.width, extent.height);
            renderer_->submitScene(world_.sceneSubmission());
            if (uiDrawData != nullptr)
            {
                renderer_->submitUi(*uiDrawData);
            }
            renderer_->endFrame();
            applyUiControls();
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
    if (!world_.sceneSubmission().pointLights.empty())
    {
        uiControls_.keyLightIntensity =
            world_.sceneSubmission().pointLights.front().intensity;
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
