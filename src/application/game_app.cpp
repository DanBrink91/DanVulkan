#include "game_app.hpp"

#include "../glfw_platform.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <utility>

namespace danvulkan::application {
namespace {
InputState translateInput(const DemoInputState& source) noexcept
{
    InputState result;
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

GameApp::GameApp(RendererConfig config, bool enableBackgroundMusic)
    : backgroundMusicEnabled_(enableBackgroundMusic)
{
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
    auto previousTime = std::chrono::steady_clock::now();

    try
    {
        while (renderer_->beginFrame())
        {
            const auto currentTime = std::chrono::steady_clock::now();
            const float deltaSeconds = std::clamp(
                std::chrono::duration<float>(currentTime - previousTime).count(), 0.0f, 0.1f);
            previousTime = currentTime;

            const InputState input = translateInput(platform_->consumeDemoInput());
            const RendererFramebufferExtent extent = platform_->framebufferExtent();
            world_.update(input, deltaSeconds, extent.width, extent.height);
            renderer_->submitScene(world_.sceneSubmission());
            renderer_->endFrame();
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

}
