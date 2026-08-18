#pragma once

#include "game_audio.hpp"
#include "game_ui.hpp"
#include "game_world.hpp"

#include <memory>

class GlfwRendererPlatform;

namespace danvulkan::application {

class GameApp
{
public:
    explicit GameApp(RendererConfig config, bool enableBackgroundMusic = true,
        bool showUiInitially = false);
    ~GameApp();

    GameApp(const GameApp&) = delete;
    GameApp& operator=(const GameApp&) = delete;

    void run();

private:
    void initializeUiControls();
    void applyUiControls();

    std::shared_ptr<GlfwRendererPlatform> platform_;
    std::unique_ptr<VulkanRenderer> renderer_;
    GameWorld world_;
    GameAudio audio_;
    GameUi ui_;
    GameUiControls uiControls_;
    GameUiControls appliedUiControls_;
    bool backgroundMusicEnabled_ = true;
};

}
