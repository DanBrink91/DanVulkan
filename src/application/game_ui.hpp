#pragma once

#include <danvulkan/renderer.hpp>
#include <danvulkan/ui.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

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
    void releaseInteraction() noexcept;
    void beginInteraction() noexcept;
    void recordDiagnostics(const RendererPerformanceStats& performance,
        const RendererMemoryStats& memory) noexcept;
    [[nodiscard]] bool visible() const noexcept { return visible_; }
    [[nodiscard]] bool interactionActive() const noexcept { return interactionActive_; }
    [[nodiscard]] UiInteractionResult interactionResult() const noexcept;
    [[nodiscard]] const UiDrawData& build(const UiInputState& input,
        const RendererPerformanceStats& performance,
        const RendererMemoryStats& memory, GameUiControls& controls);

private:
    static constexpr std::size_t historyCapacity = 120;
    enum class HistoryMetric : std::size_t
    {
        frame,
        cpu,
        gpu,
        animation,
        sampling,
        propagation,
        visibleDraws,
        activeDraws,
        animatedDraws,
        allocationMegabytes,
        blockMegabytes,
        stagingMegabytes,
        count
    };

    [[nodiscard]] std::span<const float> history(HistoryMetric metric) const noexcept;
    [[nodiscard]] std::size_t historyFirstValue() const noexcept;

    ImmediateUi ui_;
    std::array<std::array<float, historyCapacity>,
        static_cast<std::size_t>(HistoryMetric::count)> histories_{};
    std::size_t historyWriteIndex_ = 0;
    std::size_t historySampleCount_ = 0;
    std::uint64_t lastRecordedFrame_ = ~std::uint64_t{0};
    bool visible_ = false;
    bool interactionActive_ = false;
};

}
