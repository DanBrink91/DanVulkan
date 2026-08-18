#include "glfw_platform.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class GlfwRendererPlatform::Impl
{
public:
    Impl(std::string_view title, std::uint32_t width, std::uint32_t height)
    {
        if (glfwInit() != GLFW_TRUE)
        {
            throw std::runtime_error("failed to initialize GLFW");
        }
        glfwInitialized_ = true;
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height),
            std::string(title).c_str(), nullptr, nullptr);
        if (window_ == nullptr)
        {
            glfwTerminate();
            glfwInitialized_ = false;
            throw std::runtime_error("failed to create GLFW window");
        }

        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
        glfwSetKeyCallback(window_, keyCallback);
        glfwSetScrollCallback(window_, scrollCallback);
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwGetCursorPos(window_, &previousCursorX_, &previousCursorY_);

        std::uint32_t extensionCount = 0;
        const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        if (extensions == nullptr || extensionCount == 0)
        {
            glfwDestroyWindow(window_);
            window_ = nullptr;
            glfwTerminate();
            glfwInitialized_ = false;
            throw std::runtime_error(
                "GLFW did not provide the required Vulkan instance extensions");
        }
        instanceExtensions_.assign(extensions, extensions + extensionCount);
    }

    ~Impl()
    {
        if (window_ != nullptr)
        {
            glfwDestroyWindow(window_);
        }
        if (glfwInitialized_)
        {
            glfwTerminate();
        }
    }

    static void framebufferResizeCallback(GLFWwindow* window, int, int)
    {
        auto* platform = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        platform->framebufferResized_ = true;
    }

    static void scrollCallback(GLFWwindow* window, double, double yOffset)
    {
        auto* platform = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        platform->scrollDeltaY_ += static_cast<float>(yOffset);
    }

    static void keyCallback(GLFWwindow* window, int key, int, int action, int mods)
    {
        auto* platform = static_cast<Impl*>(glfwGetWindowUserPointer(window));
        if (key == GLFW_KEY_C && action == GLFW_RELEASE)
        {
            platform->cameraModeToggleRequested_ = true;
            platform->gameplayFocusRequested_ = true;
        }
        if (key == GLFW_KEY_F1 && action == GLFW_RELEASE)
        {
            platform->uiToggleRequested_ = true;
        }
        if (key == GLFW_KEY_TAB && action == GLFW_PRESS)
        {
            if ((mods & GLFW_MOD_SHIFT) != 0)
            {
                platform->focusPreviousRequested_ = true;
            }
            else
            {
                platform->focusNextRequested_ = true;
            }
        }
        if ((key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER || key == GLFW_KEY_SPACE) &&
            action == GLFW_PRESS)
        {
            platform->activateFocusedRequested_ = true;
        }
        if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) &&
            (action == GLFW_PRESS || action == GLFW_REPEAT))
        {
            platform->horizontalNavigation_ += key == GLFW_KEY_LEFT ? -1 : 1;
        }
        if ((key == GLFW_KEY_W || key == GLFW_KEY_A || key == GLFW_KEY_S ||
                key == GLFW_KEY_D) && action == GLFW_PRESS)
        {
            platform->gameplayFocusRequested_ = true;
        }
    }

    GLFWwindow* window_ = nullptr;
    bool glfwInitialized_ = false;
    bool framebufferResized_ = false;
    bool cameraModeToggleRequested_ = false;
    bool uiToggleRequested_ = false;
    bool gameplayFocusRequested_ = false;
    bool pointerDown_ = false;
    bool cursorCaptured_ = true;
    bool focusNextRequested_ = false;
    bool focusPreviousRequested_ = false;
    bool activateFocusedRequested_ = false;
    int horizontalNavigation_ = 0;
    double previousCursorX_ = 0.0;
    double previousCursorY_ = 0.0;
    float scrollDeltaY_ = 0.0f;
    std::vector<const char*> instanceExtensions_;
};

GlfwRendererPlatform::GlfwRendererPlatform(std::string_view title, std::uint32_t width,
    std::uint32_t height)
    : impl_(std::make_unique<Impl>(title, width, height))
{
}

GlfwRendererPlatform::~GlfwRendererPlatform() = default;

bool GlfwRendererPlatform::shouldClose() const noexcept
{
    return impl_ == nullptr || impl_->window_ == nullptr ||
        glfwWindowShouldClose(impl_->window_) == GLFW_TRUE;
}

void GlfwRendererPlatform::pollEvents()
{
    glfwPollEvents();
}

void GlfwRendererPlatform::waitEvents()
{
    glfwWaitEvents();
}

RendererFramebufferExtent GlfwRendererPlatform::framebufferExtent() const noexcept
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(impl_->window_, &width, &height);
    return { static_cast<std::uint32_t>(std::max(width, 0)),
        static_cast<std::uint32_t>(std::max(height, 0)) };
}

bool GlfwRendererPlatform::consumeFramebufferResize() noexcept
{
    return std::exchange(impl_->framebufferResized_, false);
}

std::span<const char* const>
GlfwRendererPlatform::requiredVulkanInstanceExtensions() const noexcept
{
    return impl_->instanceExtensions_;
}

VkSurfaceKHR GlfwRendererPlatform::createVulkanSurface(VkInstance instance)
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult result = glfwCreateWindowSurface(instance, impl_->window_, nullptr, &surface);
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("glfwCreateWindowSurface failed with VkResult " +
            std::to_string(result));
    }
    return surface;
}

void GlfwRendererPlatform::setWindowTitle(std::string_view title)
{
    glfwSetWindowTitle(impl_->window_, std::string(title).c_str());
}

bool GlfwRendererPlatform::requestWindowResize(std::uint32_t width, std::uint32_t height)
{
    glfwSetWindowSize(impl_->window_, static_cast<int>(width), static_cast<int>(height));
    return true;
}

DemoInputState GlfwRendererPlatform::consumeDemoInput()
{
    DemoInputState input;
    input.moveDown = glfwGetKey(impl_->window_, GLFW_KEY_S) == GLFW_PRESS;
    input.moveUp = glfwGetKey(impl_->window_, GLFW_KEY_W) == GLFW_PRESS;
    input.moveLeft = glfwGetKey(impl_->window_, GLFW_KEY_A) == GLFW_PRESS;
    input.moveRight = glfwGetKey(impl_->window_, GLFW_KEY_D) == GLFW_PRESS;
    input.toggleCameraMode = std::exchange(impl_->cameraModeToggleRequested_, false);
    input.toggleUi = std::exchange(impl_->uiToggleRequested_, false);
    input.requestGameplayFocus = std::exchange(impl_->gameplayFocusRequested_, false);
    if (glfwGetKey(impl_->window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(impl_->window_, GLFW_TRUE);
    }

    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(impl_->window_, &cursorX, &cursorY);
    int windowWidth = 0;
    int windowHeight = 0;
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetWindowSize(impl_->window_, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(impl_->window_, &framebufferWidth, &framebufferHeight);
    input.pointerX = static_cast<float>(cursorX) *
        (windowWidth > 0 ? static_cast<float>(framebufferWidth) / windowWidth : 1.0f);
    input.pointerY = static_cast<float>(cursorY) *
        (windowHeight > 0 ? static_cast<float>(framebufferHeight) / windowHeight : 1.0f);
    const bool pointerDown =
        glfwGetMouseButton(impl_->window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    input.pointerDown = pointerDown;
    input.pointerPressed = pointerDown && !impl_->pointerDown_;
    input.pointerReleased = !pointerDown && impl_->pointerDown_;
    impl_->pointerDown_ = pointerDown;
    input.cursorDeltaX = static_cast<float>(cursorX - impl_->previousCursorX_);
    // This camera stores a view-space pitch: moving the pointer upward must decrease it.
    input.cursorDeltaY = static_cast<float>(cursorY - impl_->previousCursorY_);
    impl_->previousCursorX_ = cursorX;
    impl_->previousCursorY_ = cursorY;
    input.scrollDeltaY = std::exchange(impl_->scrollDeltaY_, 0.0f);
    input.focusNext = std::exchange(impl_->focusNextRequested_, false);
    input.focusPrevious = std::exchange(impl_->focusPreviousRequested_, false);
    input.activateFocused = std::exchange(impl_->activateFocusedRequested_, false);
    input.horizontalNavigation = std::exchange(impl_->horizontalNavigation_, 0);
    return input;
}

void GlfwRendererPlatform::setCursorCaptured(bool captured)
{
    if (impl_->cursorCaptured_ == captured)
    {
        return;
    }
    impl_->cursorCaptured_ = captured;
    glfwSetInputMode(impl_->window_, GLFW_CURSOR,
        captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    glfwGetCursorPos(impl_->window_, &impl_->previousCursorX_, &impl_->previousCursorY_);
}

std::shared_ptr<RendererPlatform> makeGlfwRendererPlatform(std::string_view title,
    std::uint32_t width, std::uint32_t height)
{
    return std::make_shared<GlfwRendererPlatform>(title, width, height);
}
