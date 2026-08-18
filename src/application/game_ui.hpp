#pragma once

#include <danvulkan/renderer.hpp>
#include <danvulkan/ui.hpp>

namespace danvulkan::application {

struct GameUiControls
{
    bool audioAvailable = false;
    bool musicMuted = false;
    float musicVolume = 1.0f;
    bool animationAvailable = false;
    bool animationPaused = false;
    bool animationLooping = true;
    float animationSpeed = 1.0f;
    bool restartAnimationRequested = false;
    SceneEnvironment environment;
    float keyLightIntensity = 25.0f;
};

// Builds the demo's UI without exposing Vulkan or windowing types to gameplay code.
class GameUi
{
public:
    void toggle() noexcept;
    void dismiss() noexcept;
    [[nodiscard]] bool visible() const noexcept { return visible_; }
    [[nodiscard]] bool pointerOverUi() const noexcept { return ui_.pointerOverUi(); }
    [[nodiscard]] const UiDrawData& build(const UiInputState& input,
        const RendererPerformanceStats& performance,
        const RendererMemoryStats& memory, GameUiControls& controls);

private:
    ImmediateUi ui_;
    bool visible_ = false;
    bool showAnimationDetails_ = true;
};

}
