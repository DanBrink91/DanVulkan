#include "game_ui.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

namespace danvulkan::application {
namespace {
constexpr double bytesPerMegabyte = 1024.0 * 1024.0;
constexpr glm::vec4 cyan(0.20f, 0.72f, 0.88f, 1.0f);
constexpr glm::vec4 green(0.35f, 0.82f, 0.45f, 1.0f);
constexpr glm::vec4 orange(0.95f, 0.58f, 0.22f, 1.0f);

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
    interactionActive_ = visible_;
    if (!interactionActive_)
    {
        ui_.clearInteraction();
    }
}

void GameUi::releaseInteraction() noexcept
{
    if (interactionActive_)
    {
        interactionActive_ = false;
        ui_.clearInteraction();
    }
}

void GameUi::beginInteraction() noexcept
{
    if (visible_)
    {
        interactionActive_ = true;
    }
}

void GameUi::recordDiagnostics(const RendererPerformanceStats& performance,
    const RendererMemoryStats& memory) noexcept
{
    if (performance.renderedFrames == lastRecordedFrame_)
    {
        return;
    }
    lastRecordedFrame_ = performance.renderedFrames;
    const std::array<float, static_cast<std::size_t>(HistoryMetric::count)> samples{{
        static_cast<float>(performance.frameMilliseconds),
        static_cast<float>(performance.frameCpuMilliseconds),
        static_cast<float>(performance.frameGpuMilliseconds),
        static_cast<float>(performance.animationCpuMilliseconds),
        static_cast<float>(performance.animationSamplingCpuMilliseconds),
        static_cast<float>(performance.animationTransformPropagationCpuMilliseconds),
        static_cast<float>(performance.visibleDraws),
        static_cast<float>(performance.activeDraws),
        static_cast<float>(performance.animatedDraws),
        static_cast<float>(memory.allocationBytes / bytesPerMegabyte),
        static_cast<float>(memory.blockBytes / bytesPerMegabyte),
        static_cast<float>(memory.stagingArenaBytes / bytesPerMegabyte)
    }};
    for (std::size_t metric = 0; metric < samples.size(); ++metric)
    {
        histories_[metric][historyWriteIndex_] = samples[metric];
    }
    historyWriteIndex_ = (historyWriteIndex_ + 1U) % historyCapacity;
    historySampleCount_ = std::min(historySampleCount_ + 1U, historyCapacity);
}

UiInteractionResult GameUi::interactionResult() const noexcept
{
    UiInteractionResult result = ui_.interactionResult();
    result.wantsPointerInput = visible_ && interactionActive_ &&
        (result.pointerOverUi || result.activeWidget != 0);
    result.wantsKeyboardInput = visible_ && interactionActive_;
    if (!interactionActive_)
    {
        result.activeWidget = 0;
        result.focusedWidget = 0;
    }
    return result;
}

std::span<const float> GameUi::history(HistoryMetric metric) const noexcept
{
    return std::span<const float>(histories_[static_cast<std::size_t>(metric)].data(),
        historySampleCount_);
}

std::size_t GameUi::historyFirstValue() const noexcept
{
    return historySampleCount_ == historyCapacity ? historyWriteIndex_ : 0U;
}

const UiDrawData& GameUi::build(const UiInputState& input,
    const RendererPerformanceStats& performance, const RendererMemoryStats& memory,
    GameUiControls& controls)
{
    UiInputState routedInput = input;
    const bool keyboardActivated = visible_ && !interactionActive_ &&
        (input.focusNext || input.focusPrevious);
    if (keyboardActivated)
    {
        beginInteraction();
        routedInput.focusNext = false;
        routedInput.focusPrevious = false;
    }
    routedInput.interactionEnabled = interactionActive_;
    ui_.beginFrame(routedInput);
    if (!visible_)
    {
        return ui_.endFrame();
    }

    if (ui_.beginPanel("DANVULKAN  [F1] HIDE", 390.0f))
    {
        ui_.text("TAB FOCUS/NAV  SHIFT TAB BACK");
        ui_.text("ENTER TOGGLE  ARROWS ADJUST");
        ui_.text("CLICK OUTSIDE OR WASD: GAME");
        if (ui_.collapsingHeader("AUDIO"))
        {
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
        }
        if (ui_.collapsingHeader("ANIMATION"))
        {
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
        }
        if (ui_.collapsingHeader("LIGHTING AND ENVIRONMENT"))
        {
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
        }
        if (ui_.collapsingHeader("FRAME TIMING"))
        {
            formattedText(ui_, "FRAME  %6.2F MS   FPS  %5.1F", performance.frameMilliseconds,
                performance.frameMilliseconds > 0.0 ?
                    1000.0 / performance.frameMilliseconds : 0.0);
            formattedText(ui_, "CPU    %6.2F MS   GPU  %5.2F MS",
                performance.frameCpuMilliseconds, performance.frameGpuMilliseconds);
            const std::array<UiPlotSeries, 3> timing{{
                {history(HistoryMetric::frame), cyan, historyFirstValue()},
                {history(HistoryMetric::cpu), green, historyFirstValue()},
                {history(HistoryMetric::gpu), orange, historyFirstValue()}
            }};
            ui_.plotLines("FRAME C  CPU G  GPU O", timing);
        }
        if (ui_.collapsingHeader("ANIMATION PROFILING", false))
        {
            formattedText(ui_, "ANIMATION  %6.2F MS", performance.animationCpuMilliseconds);
            formattedText(ui_, "SAMPLING   %6.2F MS  %u CHANNELS",
                performance.animationSamplingCpuMilliseconds,
                performance.sampledAnimationChannels);
            formattedText(ui_, "PROPAGATE  %6.2F MS  %u NODES",
                performance.animationTransformPropagationCpuMilliseconds,
                performance.animationPropagatedNodes);
            formattedText(ui_, "ACTORS  %u  EVAL %u  CULLED %u", performance.animationActors,
                performance.evaluatedAnimationActors, performance.culledAnimationActors);
            const std::array<UiPlotSeries, 3> animation{{
                {history(HistoryMetric::animation), cyan, historyFirstValue()},
                {history(HistoryMetric::sampling), green, historyFirstValue()},
                {history(HistoryMetric::propagation), orange, historyFirstValue()}
            }};
            ui_.plotLines("TOTAL C  SAMPLE G  PROP O", animation);
        }
        if (ui_.collapsingHeader("DRAW COUNTS", false))
        {
            formattedText(ui_, "DRAWS  %u / %u VISIBLE", performance.visibleDraws,
                performance.activeDraws);
            formattedText(ui_, "ANIMATED DRAWS  %u", performance.animatedDraws);
            const std::array<UiPlotSeries, 3> draws{{
                {history(HistoryMetric::activeDraws), cyan, historyFirstValue()},
                {history(HistoryMetric::visibleDraws), green, historyFirstValue()},
                {history(HistoryMetric::animatedDraws), orange, historyFirstValue()}
            }};
            ui_.plotLines("ACTIVE C  VISIBLE G  ANIM O", draws);
        }
        if (ui_.collapsingHeader("MEMORY", false))
        {
            formattedText(ui_, "GPU ALLOCATED  %6.1F MB",
                memory.allocationBytes / bytesPerMegabyte);
            formattedText(ui_, "GPU BLOCKS     %6.1F MB  %u",
                memory.blockBytes / bytesPerMegabyte, memory.blockCount);
            formattedText(ui_, "STAGING ARENA  %6.1F MB",
                memory.stagingArenaBytes / bytesPerMegabyte);
            const std::array<UiPlotSeries, 3> memoryLines{{
                {history(HistoryMetric::allocationMegabytes), cyan, historyFirstValue()},
                {history(HistoryMetric::blockMegabytes), green, historyFirstValue()},
                {history(HistoryMetric::stagingMegabytes), orange, historyFirstValue()}
            }};
            ui_.plotLines("ALLOC C  BLOCK G  STAGE O", memoryLines);
        }
        ui_.endPanel();
    }
    return ui_.endFrame();
}

}
