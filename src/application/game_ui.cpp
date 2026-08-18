#include "game_ui.hpp"

#include <cstdio>

namespace danvulkan::application {
namespace {
constexpr double bytesPerMegabyte = 1024.0 * 1024.0;

template <typename... Args>
void formattedText(ImmediateUi& ui, const char* format, Args... args)
{
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer), format, args...);
    ui.text(buffer);
}
}

void GameUi::toggle() noexcept
{
    visible_ = !visible_;
    if (!visible_)
    {
        ui_.clearInteraction();
    }
}

void GameUi::dismiss() noexcept
{
    if (visible_)
    {
        visible_ = false;
        ui_.clearInteraction();
    }
}

const UiDrawData& GameUi::build(const UiInputState& input,
    const RendererPerformanceStats& performance, const RendererMemoryStats& memory,
    GameUiControls& controls)
{
    ui_.beginFrame(input);
    if (!visible_)
    {
        return ui_.endFrame();
    }

    if (ui_.beginPanel("DANVULKAN  [F1] CLOSE", 390.0f))
    {
        ui_.text("TAB NAV  SHIFT TAB BACK");
        ui_.text("ENTER TOGGLE  ARROWS ADJUST");
        ui_.text("CLICK SCENE OR WASD TO RETURN");
        ui_.separator();
        ui_.text("AUDIO");
        if (controls.audioAvailable)
        {
            static_cast<void>(ui_.checkbox("MUSIC MUTED", controls.musicMuted));
            static_cast<void>(ui_.sliderFloat("MUSIC VOLUME", controls.musicVolume,
                0.0f, 1.0f));
        }
        else
        {
            ui_.text("BACKGROUND MUSIC UNAVAILABLE");
        }
        ui_.separator();
        ui_.text("ANIMATION");
        if (controls.animationAvailable)
        {
            static_cast<void>(ui_.checkbox("ANIMATION PAUSED", controls.animationPaused));
            static_cast<void>(ui_.checkbox("ANIMATION LOOPING", controls.animationLooping));
            static_cast<void>(ui_.sliderFloat("ANIMATION SPEED", controls.animationSpeed,
                0.1f, 2.0f));
            if (ui_.button("RESTART ANIMATION"))
            {
                controls.restartAnimationRequested = true;
            }
        }
        else
        {
            ui_.text("NO ANIMATION CLIP");
        }
        ui_.separator();
        ui_.text("LIGHTING AND ENVIRONMENT");
        static_cast<void>(ui_.sliderFloat("KEY LIGHT", controls.keyLightIntensity,
            0.0f, 100.0f));
        static_cast<void>(ui_.sliderFloat("ENV INTENSITY", controls.environment.intensity,
            0.0f, 2.0f));
        static_cast<void>(ui_.sliderFloat("ENV ROTATION", controls.environment.rotation,
            -3.14159f, 3.14159f));
        static_cast<void>(ui_.sliderFloat("ENV DIFFUSE", controls.environment.diffuseStrength,
            0.0f, 2.0f));
        static_cast<void>(ui_.sliderFloat("ENV SPECULAR", controls.environment.specularStrength,
            0.0f, 2.0f));
        ui_.separator();
        ui_.text("PERFORMANCE");
        formattedText(ui_, "FRAME  %6.2F MS   FPS  %5.1F", performance.frameMilliseconds,
            performance.frameMilliseconds > 0.0 ?
                1000.0 / performance.frameMilliseconds : 0.0);
        formattedText(ui_, "CPU    %6.2F MS   GPU  %5.2F MS",
            performance.frameCpuMilliseconds, performance.frameGpuMilliseconds);
        ui_.separator();
        formattedText(ui_, "DRAWS  %u / %u VISIBLE", performance.visibleDraws,
            performance.activeDraws);
        formattedText(ui_, "ANIMATED DRAWS  %u", performance.animatedDraws);
        formattedText(ui_, "ACTORS  %u   EVALUATED  %u", performance.animationActors,
            performance.evaluatedAnimationActors);
        static_cast<void>(ui_.checkbox("SHOW ANIMATION DETAILS", showAnimationDetails_));
        if (showAnimationDetails_)
        {
            formattedText(ui_, "ANIMATION  %6.2F MS", performance.animationCpuMilliseconds);
            formattedText(ui_, "SAMPLING   %6.2F MS  %u CHANNELS",
                performance.animationSamplingCpuMilliseconds,
                performance.sampledAnimationChannels);
            formattedText(ui_, "PROPAGATE  %6.2F MS  %u NODES",
                performance.animationTransformPropagationCpuMilliseconds,
                performance.animationPropagatedNodes);
            formattedText(ui_, "CULLED ACTORS  %u", performance.culledAnimationActors);
        }
        ui_.separator();
        formattedText(ui_, "GPU ALLOCATED  %6.1F MB",
            memory.allocationBytes / bytesPerMegabyte);
        formattedText(ui_, "GPU BLOCKS     %6.1F MB  %u",
            memory.blockBytes / bytesPerMegabyte, memory.blockCount);
        formattedText(ui_, "STAGING ARENA  %6.1F MB",
            memory.stagingArenaBytes / bytesPerMegabyte);
        ui_.endPanel();
    }
    return ui_.endFrame();
}

}
