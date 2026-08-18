#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

struct UiInputState
{
    float pointerX = 0.0f;
    float pointerY = 0.0f;
    bool pointerDown = false;
    bool pointerPressed = false;
    bool pointerReleased = false;
    float scrollDeltaY = 0.0f;
    bool focusNext = false;
    bool focusPrevious = false;
    bool activateFocused = false;
    int horizontalNavigation = 0;
    std::uint32_t viewportWidth = 0;
    std::uint32_t viewportHeight = 0;
};

struct UiVertex
{
    glm::vec2 position{0.0f};
    glm::vec2 glyphUv{0.0f};
    glm::vec4 color{1.0f};
    // A 5x7 monochrome glyph mask, packed row-major across two 32-bit words.
    glm::uvec2 glyphMask{0xffffffffU, 0x7U};
};

struct UiDrawCommand
{
    std::uint32_t firstVertex = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t clipX = 0;
    std::uint32_t clipY = 0;
    std::uint32_t clipWidth = 0;
    std::uint32_t clipHeight = 0;
};

struct UiDrawData
{
    std::vector<UiVertex> vertices;
    std::vector<UiDrawCommand> commands;
};

// A deliberately small immediate-mode UI. It owns interaction state and emits only
// renderer-neutral triangles; applications rebuild the widget tree every frame.
class ImmediateUi
{
public:
    void beginFrame(const UiInputState& input);
    // Height zero uses the available viewport height. Content that does not fit can be
    // scrolled while the pointer is over the panel.
    [[nodiscard]] bool beginPanel(std::string_view title, float width = 340.0f,
        float height = 0.0f);
    void endPanel();

    void text(std::string_view value);
    void separator();
    [[nodiscard]] bool button(std::string_view label);
    [[nodiscard]] bool checkbox(std::string_view label, bool& value);
    [[nodiscard]] bool sliderFloat(std::string_view label, float& value,
        float minimum, float maximum);

    [[nodiscard]] const UiDrawData& endFrame();
    [[nodiscard]] const UiDrawData& drawData() const noexcept { return drawData_; }
    // Reports whether the current frame's pointer lies over any emitted panel. Applications
    // can use this to dismiss a modal overlay when the user clicks back into the scene.
    [[nodiscard]] bool pointerOverUi() const noexcept { return pointerOverUi_; }
    void clearInteraction() noexcept;

private:
    [[nodiscard]] std::uint64_t widgetId(std::string_view label) const noexcept;
    [[nodiscard]] bool registerWidget(std::uint64_t id, float height);
    [[nodiscard]] bool hovered(float x, float y, float width, float height) const noexcept;
    void addRect(float x, float y, float width, float height, const glm::vec4& color);
    void addText(float x, float y, std::string_view value, const glm::vec4& color);
    void advance(float height);

    UiInputState input_;
    UiDrawData drawData_;
    float panelX_ = 16.0f;
    float panelY_ = 16.0f;
    float panelWidth_ = 340.0f;
    float panelHeightLimit_ = 0.0f;
    float panelInteractionBottom_ = 0.0f;
    float panelContentStartY_ = 0.0f;
    float panelScrollOffset_ = 0.0f;
    float cursorY_ = 16.0f;
    std::string panelTitle_;
    std::uint32_t panelFirstVertex_ = 0;
    std::uint32_t panelContentFirstVertex_ = 0;
    std::uint64_t panelId_ = 0;
    std::uint64_t activeId_ = 0;
    std::uint64_t focusedId_ = 0;
    std::vector<std::uint64_t> focusOrder_;
    struct PanelState
    {
        std::uint64_t id = 0;
        float scrollOffset = 0.0f;
        float contentHeight = 0.0f;
        float renderedHeight = 0.0f;
    };
    std::vector<PanelState> panelStates_;
    PanelState* panelState_ = nullptr;
    bool panelOpen_ = false;
    bool pointerOverUi_ = false;
};
