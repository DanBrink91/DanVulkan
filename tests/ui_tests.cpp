#include <danvulkan/ui.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void panelProducesClippedTriangles()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 1280;
    input.viewportHeight = 720;
    ui.beginFrame(input);
    require(ui.beginPanel("TEST"), "panel should open");
    ui.text("FRAME 16.67 MS");
    ui.separator();
    ui.endPanel();
    const UiDrawData& data = ui.endFrame();
    require(!data.vertices.empty(), "panel should emit vertices");
    require(data.vertices.size() % 6 == 0, "UI geometry should contain triangle quads");
    require(data.commands.size() == 1, "one panel should emit one command");
    require(data.commands.front().vertexCount == data.vertices.size(),
        "panel command should cover every emitted vertex");
    require(data.commands.front().clipWidth > 0 && data.commands.front().clipHeight > 0,
        "panel should emit a non-empty clip rectangle");
}

void checkboxChangesOnlyWhenPressedInside()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 1280;
    input.viewportHeight = 720;
    input.pointerX = 32.0f;
    input.pointerY = 62.0f;
    input.pointerDown = true;
    input.pointerPressed = true;
    bool value = false;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("TEST"));
    const bool changed = ui.checkbox("ENABLED", value);
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    require(changed && value, "checkbox should toggle on an inside press");

    input.pointerX = 900.0f;
    input.pointerPressed = true;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("TEST"));
    const bool changedOutside = ui.checkbox("ENABLED", value);
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    require(!changedOutside && value, "checkbox should ignore an outside press");
}

void sliderTracksPointerAndClamps()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 1280;
    input.viewportHeight = 720;
    input.pointerX = 184.0f;
    input.pointerY = 80.0f;
    input.pointerDown = true;
    input.pointerPressed = true;
    float value = 0.0f;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("TEST"));
    const bool changed = ui.sliderFloat("VALUE", value, 0.0f, 1.0f);
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    require(changed, "slider should react to a press on its track");
    require(std::abs(value - 0.5f) < 0.02f, "slider should map the pointer to its range");
}

void panelScrollsContentUnderFixedHeader()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 640;
    input.viewportHeight = 200;
    input.pointerX = 40.0f;
    input.pointerY = 100.0f;

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("SCROLL"));
    for (int row = 0; row < 20; ++row)
    {
        ui.text("ROW");
    }
    ui.endPanel();
    const float firstY = ui.endFrame().vertices.at(6).position.y;

    input.scrollDeltaY = -1.0f;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("SCROLL"));
    for (int row = 0; row < 20; ++row)
    {
        ui.text("ROW");
    }
    ui.endPanel();
    const UiDrawData& scrolled = ui.endFrame();
    require(std::abs(scrolled.vertices.at(6).position.y - (firstY - 32.0f)) < 0.01f,
        "mouse wheel should offset overflowing panel content");
    require(scrolled.commands.front().clipHeight == 168,
        "overflowing panel should remain clipped to the viewport");
}

void keyboardMovesFocusAndActivatesWidgets()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 640;
    input.viewportHeight = 480;
    bool first = false;
    bool second = false;

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("KEYBOARD"));
    static_cast<void>(ui.checkbox("FIRST", first));
    static_cast<void>(ui.checkbox("SECOND", second));
    ui.endPanel();
    static_cast<void>(ui.endFrame());

    input.focusNext = true;
    input.activateFocused = true;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("KEYBOARD"));
    static_cast<void>(ui.checkbox("FIRST", first));
    static_cast<void>(ui.checkbox("SECOND", second));
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    require(!first && second, "Tab then Enter should activate the next widget");
}

void keyboardAdjustsFocusedSlider()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 640;
    input.viewportHeight = 480;
    float value = 0.5f;

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("KEYBOARD SLIDER"));
    static_cast<void>(ui.sliderFloat("VALUE", value, 0.0f, 1.0f));
    ui.endPanel();
    static_cast<void>(ui.endFrame());

    input.horizontalNavigation = 1;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("KEYBOARD SLIDER"));
    const bool changed = ui.sliderFloat("VALUE", value, 0.0f, 1.0f);
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    require(changed && std::abs(value - 0.55f) < 0.001f,
        "Right should increment the focused slider by one step");
}

void panelReportsWhetherPointerIsOverUi()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 640;
    input.viewportHeight = 480;
    input.pointerX = 40.0f;
    input.pointerY = 40.0f;

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("HIT TEST", 200.0f, 120.0f));
    ui.text("CONTENT");
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    require(ui.pointerOverUi(), "pointer inside a panel should be captured by the UI");

    input.pointerX = 400.0f;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("HIT TEST", 200.0f, 120.0f));
    ui.text("CONTENT");
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    require(!ui.pointerOverUi(), "pointer outside every panel should not be captured");
}

void disabledInteractionStillDrawsWithoutChangingControls()
{
    ImmediateUi ui;
    UiInputState input;
    input.interactionEnabled = false;
    input.viewportWidth = 640;
    input.viewportHeight = 480;
    input.pointerX = 32.0f;
    input.pointerY = 62.0f;
    input.pointerDown = true;
    input.pointerPressed = true;
    input.activateFocused = true;
    bool value = false;

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("INACTIVE"));
    const bool changed = ui.checkbox("ENABLED", value);
    ui.endPanel();
    const UiDrawData& data = ui.endFrame();

    require(!changed && !value, "inactive UI should not consume pointer or keyboard input");
    require(!data.vertices.empty(), "inactive UI should remain visible");
    require(ui.pointerOverUi(), "inactive panels should still report their visible bounds");
    const UiInteractionResult& interaction = ui.interactionResult();
    require(interaction.pointerOverUi, "inactive UI should retain visual hit testing");
    require(!interaction.wantsPointerInput && !interaction.wantsKeyboardInput,
        "inactive UI should not request input ownership");
    require(interaction.activeWidget == 0 && interaction.focusedWidget == 0,
        "inactive UI should not expose active interaction IDs");
}

void interactionResultSeparatesPointerAndKeyboardOwnership()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 640;
    input.viewportHeight = 480;
    input.pointerX = 40.0f;
    input.pointerY = 64.0f;
    input.pointerDown = true;
    input.pointerPressed = true;

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("OWNERSHIP", 220.0f, 120.0f));
    static_cast<void>(ui.button("ACTIVE"));
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    const UiInteractionResult inside = ui.interactionResult();
    require(inside.pointerOverUi && inside.wantsPointerInput,
        "an interactive panel under the pointer should request pointer input");
    require(inside.wantsKeyboardInput,
        "an interactive panel with focusable widgets should request keyboard input");
    require(inside.activeWidget != 0 && inside.focusedWidget != 0,
        "a pressed button should report active and focused widget IDs");

    input.pointerX = 500.0f;
    input.pointerY = 300.0f;
    input.pointerDown = false;
    input.pointerPressed = false;
    input.pointerReleased = true;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("OWNERSHIP", 220.0f, 120.0f));
    static_cast<void>(ui.button("ACTIVE"));
    ui.endPanel();
    static_cast<void>(ui.endFrame());
    const UiInteractionResult outside = ui.interactionResult();
    require(!outside.pointerOverUi && !outside.wantsPointerInput,
        "a pointer outside an idle panel should remain available to the application");
    require(outside.wantsKeyboardInput,
        "keyboard focus should remain owned until the application releases UI interaction");
}

void collapsingHeadersRetainOpenState()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 640;
    input.viewportHeight = 480;

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("SECTIONS"));
    require(ui.collapsingHeader("DETAILS"), "header should honor its default open state");
    ui.endPanel();
    static_cast<void>(ui.endFrame());

    input.pointerX = 40.0f;
    input.pointerY = 64.0f;
    input.pointerPressed = true;
    input.pointerDown = true;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("SECTIONS"));
    require(!ui.collapsingHeader("DETAILS"), "clicking an open header should collapse it");
    ui.endPanel();
    static_cast<void>(ui.endFrame());

    input.pointerPressed = false;
    input.pointerDown = false;
    input.pointerReleased = true;
    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("SECTIONS"));
    require(!ui.collapsingHeader("DETAILS", true),
        "a collapsing header should retain its state across immediate-mode frames");
    ui.endPanel();
    static_cast<void>(ui.endFrame());
}

void plotLinesDrawsFixedHistorySeries()
{
    ImmediateUi ui;
    UiInputState input;
    input.viewportWidth = 640;
    input.viewportHeight = 480;
    const std::array<float, 4> wrappedValues{{3.0f, 4.0f, 1.0f, 2.0f}};
    const std::array<float, 4> comparison{{1.0f, 2.0f, 3.0f, 4.0f}};
    const std::array<UiPlotSeries, 2> series{{
        {wrappedValues, {0.2f, 0.7f, 0.9f, 1.0f}, 2U},
        {comparison, {0.4f, 0.8f, 0.4f, 1.0f}, 0U}
    }};

    ui.beginFrame(input);
    static_cast<void>(ui.beginPanel("PLOTS"));
    ui.plotLines("HISTORY", series, 0.0f, 4.0f, 60.0f);
    ui.endPanel();
    const UiDrawData& data = ui.endFrame();
    require(data.commands.size() == 1, "a plot should remain inside its panel draw command");
    require(data.vertices.size() >= 72,
        "multiple plot series should emit backgrounds, grid lines, and line segments");
}
}

int main()
{
    try
    {
        panelProducesClippedTriangles();
        checkboxChangesOnlyWhenPressedInside();
        sliderTracksPointerAndClamps();
        panelScrollsContentUnderFixedHeader();
        keyboardMovesFocusAndActivatesWidgets();
        keyboardAdjustsFocusedSlider();
        panelReportsWhetherPointerIsOverUi();
        disabledInteractionStillDrawsWithoutChangingControls();
        interactionResultSeparatesPointerAndKeyboardOwnership();
        collapsingHeadersRetainOpenState();
        plotLinesDrawsFixedHistorySeries();
        std::cout << "UI tests passed\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "UI test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
