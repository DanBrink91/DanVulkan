#include <danvulkan/ui.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace {
constexpr float panelPadding = 12.0f;
constexpr float headerHeight = 30.0f;
constexpr float rowHeight = 20.0f;
constexpr float glyphWidth = 5.0f;
constexpr float glyphHeight = 7.0f;
constexpr float glyphScale = 2.0f;
constexpr float glyphAdvance = 12.0f;

constexpr glm::vec4 panelColor(0.035f, 0.045f, 0.065f, 0.94f);
constexpr glm::vec4 headerColor(0.09f, 0.18f, 0.28f, 0.98f);
constexpr glm::vec4 textColor(0.88f, 0.93f, 0.98f, 1.0f);
constexpr glm::vec4 mutedColor(0.42f, 0.50f, 0.58f, 1.0f);
constexpr glm::vec4 accentColor(0.20f, 0.72f, 0.88f, 1.0f);
constexpr glm::vec4 controlColor(0.10f, 0.13f, 0.17f, 1.0f);
constexpr glm::vec4 hoverColor(0.15f, 0.22f, 0.29f, 1.0f);
constexpr float scrollStep = 32.0f;

std::string_view glyphPattern(char value) noexcept
{
    if (value >= 'a' && value <= 'z')
    {
        value = static_cast<char>(value - 'a' + 'A');
    }
    switch (value)
    {
    case 'A': return "01110100011000111111100011000110001";
    case 'B': return "11110100011000111110100011000111110";
    case 'C': return "01111100001000010000100001000001111";
    case 'D': return "11110100011000110001100011000111110";
    case 'E': return "11111100001000011110100001000011111";
    case 'F': return "11111100001000011110100001000010000";
    case 'G': return "01111100001000010111100011000101111";
    case 'H': return "10001100011000111111100011000110001";
    case 'I': return "11111001000010000100001000010011111";
    case 'J': return "00111000100001000010000101001001100";
    case 'K': return "10001100101010011000101001001010001";
    case 'L': return "10000100001000010000100001000011111";
    case 'M': return "10001110111010110101100011000110001";
    case 'N': return "10001110011010110011100011000110001";
    case 'O': return "01110100011000110001100011000101110";
    case 'P': return "11110100011000111110100001000010000";
    case 'Q': return "01110100011000110001101011001001101";
    case 'R': return "11110100011000111110101001001010001";
    case 'S': return "01111100001000001110000010000111110";
    case 'T': return "11111001000010000100001000010000100";
    case 'U': return "10001100011000110001100011000101110";
    case 'V': return "10001100011000110001100010101000100";
    case 'W': return "10001100011000110101101011101110001";
    case 'X': return "10001100010101000100010101000110001";
    case 'Y': return "10001100010101000100001000010000100";
    case 'Z': return "11111000010001000100010001000011111";
    case '0': return "01110100011001110101110011000101110";
    case '1': return "00100011000010000100001000010001110";
    case '2': return "01110100010000100010001000100011111";
    case '3': return "11110000010000101110000010000111110";
    case '4': return "00010001100101010010111110001000010";
    case '5': return "11111100001000011110000010000111110";
    case '6': return "01110100001000011110100011000101110";
    case '7': return "11111000010001000100010000100001000";
    case '8': return "01110100011000101110100011000101110";
    case '9': return "01110100011000101111000010000101110";
    case '.': return "00000000000000000000000000011000110";
    case ':': return "00000001100011000000011000110000000";
    case '-': return "00000000000000011111000000000000000";
    case '/': return "00001000100010001000100001000000000";
    case '%': return "11001110100010001000100010101110011";
    case '(': return "00010001000100001000010000010000010";
    case ')': return "01000001000001000010000100010001000";
    case '[': return "01110010000100001000010000100001110";
    case ']': return "01110000100001000010000100001001110";
    case '+': return "00000001000010011111001000010000000";
    case '_': return "00000000000000000000000000000011111";
    case ' ': return "00000000000000000000000000000000000";
    default: return "01110100010000100010001000000000100";
    }
}

glm::uvec2 glyphMask(char value) noexcept
{
    const std::string_view pattern = glyphPattern(value);
    glm::uvec2 mask(0U);
    for (std::uint32_t index = 0; index < pattern.size(); ++index)
    {
        if (pattern[index] != '1')
        {
            continue;
        }
        if (index < 32)
        {
            mask.x |= 1U << index;
        }
        else
        {
            mask.y |= 1U << (index - 32U);
        }
    }
    return mask;
}

void addQuad(std::vector<UiVertex>& vertices, float x, float y, float width, float height,
    const glm::vec4& color, const glm::uvec2& mask)
{
    const std::array<glm::vec2, 6> positions{{
        {x, y}, {x + width, y}, {x + width, y + height},
        {x, y}, {x + width, y + height}, {x, y + height}
    }};
    const std::array<glm::vec2, 6> uvs{{
        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
        {0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}
    }};
    for (std::size_t index = 0; index < positions.size(); ++index)
    {
        vertices.push_back({positions[index], uvs[index], color, mask});
    }
}
}

void ImmediateUi::beginFrame(const UiInputState& input)
{
    if (panelOpen_)
    {
        throw std::logic_error("ImmediateUi::beginFrame called with an open panel");
    }
    input_ = input;
    drawData_.vertices.clear();
    drawData_.commands.clear();
    panelId_ = 0;
    pointerOverUi_ = false;

    if (!focusOrder_.empty() && (input_.focusNext || input_.focusPrevious))
    {
        const auto focused = std::find(focusOrder_.begin(), focusOrder_.end(), focusedId_);
        std::size_t index = focused == focusOrder_.end() ? 0U :
            static_cast<std::size_t>(focused - focusOrder_.begin());
        if (input_.focusPrevious)
        {
            index = index == 0 ? focusOrder_.size() - 1U : index - 1U;
        }
        else
        {
            index = (index + 1U) % focusOrder_.size();
        }
        focusedId_ = focusOrder_[index];
    }
    focusOrder_.clear();
}

bool ImmediateUi::beginPanel(std::string_view title, float width, float height)
{
    if (panelOpen_)
    {
        throw std::logic_error("ImmediateUi supports one open panel at a time");
    }
    panelOpen_ = true;
    panelWidth_ = std::max(width, 120.0f);
    panelFirstVertex_ = static_cast<std::uint32_t>(drawData_.vertices.size());
    panelId_ = widgetId(title);
    panelTitle_.assign(title);
    panelContentStartY_ = panelY_ + headerHeight + panelPadding;

    const float availableHeight = std::max(static_cast<float>(input_.viewportHeight) -
        panelY_ * 2.0f, headerHeight + panelPadding * 2.0f);
    panelHeightLimit_ = height > 0.0f ? std::clamp(height,
        headerHeight + panelPadding * 2.0f, availableHeight) : availableHeight;

    auto state = std::find_if(panelStates_.begin(), panelStates_.end(),
        [this](const PanelState& candidate) { return candidate.id == panelId_; });
    if (state == panelStates_.end())
    {
        panelStates_.push_back({panelId_});
        state = std::prev(panelStates_.end());
    }
    panelState_ = &*state;
    const float previousVisibleBody = std::max(panelState_->renderedHeight - headerHeight, 0.0f);
    const float previousMaxScroll = std::max(
        panelState_->contentHeight - previousVisibleBody + panelPadding, 0.0f);
    const float interactionHeight = panelState_->renderedHeight > 0.0f ?
        std::min(panelState_->renderedHeight, panelHeightLimit_) : panelHeightLimit_;
    if (input_.pointerX >= panelX_ && input_.pointerX < panelX_ + panelWidth_ &&
        input_.pointerY >= panelY_ && input_.pointerY < panelY_ + interactionHeight)
    {
        panelState_->scrollOffset = std::clamp(panelState_->scrollOffset -
            input_.scrollDeltaY * scrollStep, 0.0f, previousMaxScroll);
    }
    panelScrollOffset_ = panelState_->scrollOffset;
    panelInteractionBottom_ = panelY_ + interactionHeight;
    cursorY_ = panelContentStartY_ - panelScrollOffset_;

    // endPanel extends this placeholder after the final content height is known.
    addRect(panelX_, panelY_, panelWidth_, 1.0f, panelColor);
    panelContentFirstVertex_ = static_cast<std::uint32_t>(drawData_.vertices.size());
    return true;
}

void ImmediateUi::endPanel()
{
    if (!panelOpen_)
    {
        throw std::logic_error("ImmediateUi::endPanel called without beginPanel");
    }
    const float contentHeight = std::max(
        cursorY_ + panelScrollOffset_ - panelContentStartY_ + panelPadding, 0.0f);
    const float naturalHeight = headerHeight + panelPadding + contentHeight;
    const float panelHeight = std::clamp(naturalHeight,
        headerHeight + panelPadding * 2.0f, panelHeightLimit_);
    pointerOverUi_ = pointerOverUi_ ||
        (input_.pointerX >= panelX_ && input_.pointerX < panelX_ + panelWidth_ &&
            input_.pointerY >= panelY_ && input_.pointerY < panelY_ + panelHeight);
    const float visibleBodyHeight = std::max(panelHeight - headerHeight, 0.0f);
    const float maxScroll = std::max(contentHeight - visibleBodyHeight + panelPadding, 0.0f);
    const float clampedScroll = std::clamp(panelScrollOffset_, 0.0f, maxScroll);
    if (clampedScroll != panelScrollOffset_)
    {
        const float correction = panelScrollOffset_ - clampedScroll;
        for (std::size_t index = panelContentFirstVertex_;
             index < drawData_.vertices.size(); ++index)
        {
            drawData_.vertices[index].position.y += correction;
        }
        panelScrollOffset_ = clampedScroll;
    }
    panelState_->scrollOffset = panelScrollOffset_;
    panelState_->contentHeight = contentHeight;
    panelState_->renderedHeight = panelHeight;

    const std::array<glm::vec2, 6> backgroundPositions{{
        {panelX_, panelY_}, {panelX_ + panelWidth_, panelY_},
        {panelX_ + panelWidth_, panelY_ + panelHeight},
        {panelX_, panelY_}, {panelX_ + panelWidth_, panelY_ + panelHeight},
        {panelX_, panelY_ + panelHeight}
    }};
    for (std::size_t index = 0; index < backgroundPositions.size(); ++index)
    {
        drawData_.vertices[panelFirstVertex_ + index].position = backgroundPositions[index];
    }

    if (maxScroll > 0.0f)
    {
        const float trackX = panelX_ + panelWidth_ - 5.0f;
        const float trackY = panelY_ + headerHeight + 3.0f;
        const float trackHeight = std::max(panelHeight - headerHeight - 6.0f, 1.0f);
        const float thumbHeight = std::min(std::max(trackHeight * visibleBodyHeight /
            std::max(contentHeight, 1.0f), 20.0f), trackHeight);
        const float thumbTravel = std::max(trackHeight - thumbHeight, 0.0f);
        const float thumbY = trackY + thumbTravel * panelScrollOffset_ / maxScroll;
        addRect(trackX, trackY, 2.0f, trackHeight, controlColor);
        addRect(trackX - 1.0f, thumbY, 4.0f, thumbHeight, accentColor);
    }

    // Repaint the fixed header after the scrolling content so body geometry cannot overlap it.
    addRect(panelX_, panelY_, panelWidth_, headerHeight, headerColor);
    addText(panelX_ + panelPadding, panelY_ + 8.0f, panelTitle_, textColor);

    const float bottom = panelY_ + panelHeight;
    const float clipRight = std::min(panelX_ + panelWidth_,
        static_cast<float>(input_.viewportWidth));
    const float clipBottom = std::min(bottom,
        static_cast<float>(input_.viewportHeight));
    UiDrawCommand command;
    command.firstVertex = panelFirstVertex_;
    command.vertexCount = static_cast<std::uint32_t>(drawData_.vertices.size()) -
        panelFirstVertex_;
    command.clipX = static_cast<std::uint32_t>(std::max(panelX_, 0.0f));
    command.clipY = static_cast<std::uint32_t>(std::max(panelY_, 0.0f));
    command.clipWidth = static_cast<std::uint32_t>(std::max(clipRight - panelX_, 0.0f));
    command.clipHeight = static_cast<std::uint32_t>(std::max(clipBottom - panelY_, 0.0f));
    if (command.vertexCount != 0 && command.clipWidth != 0 && command.clipHeight != 0)
    {
        drawData_.commands.push_back(command);
    }
    panelState_ = nullptr;
    panelOpen_ = false;
}

void ImmediateUi::clearInteraction() noexcept
{
    activeId_ = 0;
    focusedId_ = 0;
    focusOrder_.clear();
}

void ImmediateUi::text(std::string_view value)
{
    addText(panelX_ + panelPadding, cursorY_ + 3.0f, value, textColor);
    advance(rowHeight);
}

void ImmediateUi::separator()
{
    addRect(panelX_ + panelPadding, cursorY_ + 6.0f,
        panelWidth_ - panelPadding * 2.0f, 1.0f, mutedColor);
    advance(13.0f);
}

bool ImmediateUi::button(std::string_view label)
{
    const float x = panelX_ + panelPadding;
    const float width = panelWidth_ - panelPadding * 2.0f;
    const std::uint64_t id = widgetId(label);
    const bool isFocused = registerWidget(id, rowHeight);
    const bool isHovered = hovered(x, cursorY_, width, rowHeight);
    addRect(x, cursorY_, width, rowHeight,
        isFocused ? accentColor : (isHovered ? hoverColor : controlColor));
    addText(x + 7.0f, cursorY_ + 3.0f, label, textColor);
    const bool pointerClicked = isHovered && input_.pointerPressed;
    const bool clicked = pointerClicked || (isFocused && input_.activateFocused);
    if (pointerClicked)
    {
        activeId_ = id;
    }
    if (pointerClicked)
    {
        focusedId_ = id;
    }
    if (input_.pointerReleased && activeId_ == id)
    {
        activeId_ = 0;
    }
    advance(rowHeight + 4.0f);
    return clicked;
}

bool ImmediateUi::checkbox(std::string_view label, bool& value)
{
    const float x = panelX_ + panelPadding;
    const float width = panelWidth_ - panelPadding * 2.0f;
    const std::uint64_t id = widgetId(label);
    const bool isFocused = registerWidget(id, rowHeight);
    const bool isHovered = hovered(x, cursorY_, width, rowHeight);
    addRect(x, cursorY_ + 2.0f, 16.0f, 16.0f,
        isFocused ? accentColor : (isHovered ? hoverColor : controlColor));
    if (value)
    {
        addRect(x + 4.0f, cursorY_ + 6.0f, 8.0f, 8.0f, accentColor);
    }
    addText(x + 24.0f, cursorY_ + 3.0f, label, textColor);
    const bool pointerChanged = isHovered && input_.pointerPressed;
    const bool changed = pointerChanged || (isFocused && input_.activateFocused);
    if (changed)
    {
        value = !value;
    }
    if (pointerChanged)
    {
        focusedId_ = id;
    }
    advance(rowHeight + 2.0f);
    return changed;
}

bool ImmediateUi::sliderFloat(std::string_view label, float& value,
    float minimum, float maximum)
{
    if (!(minimum < maximum))
    {
        throw std::invalid_argument("ImmediateUi slider requires minimum < maximum");
    }
    const float x = panelX_ + panelPadding;
    const float width = panelWidth_ - panelPadding * 2.0f;
    const std::uint64_t id = widgetId(label);
    const bool isFocused = registerWidget(id, rowHeight + 16.0f);
    char display[128];
    std::snprintf(display, sizeof(display), "%.*s  %.2f",
        static_cast<int>(label.size()), label.data(), static_cast<double>(value));
    addText(x, cursorY_ + 2.0f, display, textColor);
    const float trackY = cursorY_ + rowHeight;
    const bool isHovered = hovered(x, trackY - 5.0f, width, 14.0f);
    if (isHovered && input_.pointerPressed)
    {
        activeId_ = id;
        focusedId_ = id;
    }
    bool changed = false;
    if (activeId_ == id && input_.pointerDown)
    {
        const float normalized = std::clamp((input_.pointerX - x) / width, 0.0f, 1.0f);
        const float replacement = minimum + normalized * (maximum - minimum);
        changed = replacement != value;
        value = replacement;
    }
    if (isFocused && input_.horizontalNavigation != 0)
    {
        const float replacement = std::clamp(value +
            static_cast<float>(input_.horizontalNavigation) * (maximum - minimum) / 20.0f,
            minimum, maximum);
        changed = changed || replacement != value;
        value = replacement;
    }
    if (activeId_ == id && input_.pointerReleased)
    {
        activeId_ = 0;
    }
    const float normalized = std::clamp((value - minimum) / (maximum - minimum), 0.0f, 1.0f);
    addRect(x, trackY, width, 4.0f, controlColor);
    addRect(x, trackY, width * normalized, 4.0f, accentColor);
    addRect(x + width * normalized - 4.0f, trackY - 4.0f, 8.0f, 12.0f,
        isHovered || activeId_ == id || isFocused ? textColor : mutedColor);
    advance(rowHeight + 16.0f);
    return changed;
}

const UiDrawData& ImmediateUi::endFrame()
{
    if (panelOpen_)
    {
        throw std::logic_error("ImmediateUi::endFrame called with an open panel");
    }
    if (input_.pointerReleased)
    {
        activeId_ = 0;
    }
    if (focusedId_ != 0 &&
        std::find(focusOrder_.begin(), focusOrder_.end(), focusedId_) == focusOrder_.end())
    {
        focusedId_ = focusOrder_.empty() ? 0 : focusOrder_.front();
    }
    return drawData_;
}

std::uint64_t ImmediateUi::widgetId(std::string_view label) const noexcept
{
    std::uint64_t hash = 1469598103934665603ULL ^ panelId_;
    for (const char value : label)
    {
        hash ^= static_cast<unsigned char>(value);
        hash *= 1099511628211ULL;
    }
    return hash == 0 ? 1 : hash;
}

bool ImmediateUi::registerWidget(std::uint64_t id, float height)
{
    focusOrder_.push_back(id);
    if (focusedId_ == 0)
    {
        focusedId_ = id;
    }
    const bool focused = focusedId_ == id;
    if (focused && panelState_ != nullptr && (input_.focusNext || input_.focusPrevious))
    {
        const float visibleTop = panelY_ + headerHeight + panelPadding;
        const float visibleBottom = panelY_ + panelHeightLimit_ - panelPadding;
        float offsetAdjustment = 0.0f;
        if (cursorY_ < visibleTop)
        {
            offsetAdjustment = cursorY_ - visibleTop;
        }
        else if (cursorY_ + height > visibleBottom)
        {
            offsetAdjustment = cursorY_ + height - visibleBottom;
        }
        const float visibleBody = std::max(panelHeightLimit_ - headerHeight, 0.0f);
        const float maxScroll = std::max(
            panelState_->contentHeight - visibleBody + panelPadding, 0.0f);
        const float replacement = std::clamp(
            panelScrollOffset_ + offsetAdjustment, 0.0f, maxScroll);
        const float appliedAdjustment = replacement - panelScrollOffset_;
        if (appliedAdjustment != 0.0f)
        {
            for (std::size_t index = panelContentFirstVertex_;
                 index < drawData_.vertices.size(); ++index)
            {
                drawData_.vertices[index].position.y -= appliedAdjustment;
            }
            cursorY_ -= appliedAdjustment;
            panelScrollOffset_ = replacement;
            panelState_->scrollOffset = replacement;
        }
    }
    return focused;
}

bool ImmediateUi::hovered(float x, float y, float width, float height) const noexcept
{
    return input_.pointerX >= x && input_.pointerX < x + width &&
        input_.pointerY >= y && input_.pointerY < y + height &&
        (!panelOpen_ || (input_.pointerY >= panelContentStartY_ - panelPadding &&
            input_.pointerY < panelInteractionBottom_));
}

void ImmediateUi::addRect(float x, float y, float width, float height,
    const glm::vec4& color)
{
    if (width <= 0.0f || height <= 0.0f)
    {
        return;
    }
    addQuad(drawData_.vertices, x, y, width, height, color,
        glm::uvec2(0xffffffffU, 0x7U));
}

void ImmediateUi::addText(float x, float y, std::string_view value,
    const glm::vec4& color)
{
    for (const char character : value)
    {
        if (character != ' ')
        {
            addQuad(drawData_.vertices, x, y, glyphWidth * glyphScale,
                glyphHeight * glyphScale, color, glyphMask(character));
        }
        x += glyphAdvance;
    }
}

void ImmediateUi::advance(float height)
{
    cursorY_ += height;
}
