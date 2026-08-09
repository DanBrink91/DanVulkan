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

    GLFWwindow* window_ = nullptr;
    bool glfwInitialized_ = false;
    bool framebufferResized_ = false;
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
    if (glfwGetKey(impl_->window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(impl_->window_, GLFW_TRUE);
    }

    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(impl_->window_, &cursorX, &cursorY);
    input.cursorDeltaX = static_cast<float>(cursorX - impl_->previousCursorX_);
    // This camera stores a view-space pitch: moving the pointer upward must decrease it.
    input.cursorDeltaY = static_cast<float>(cursorY - impl_->previousCursorY_);
    impl_->previousCursorX_ = cursorX;
    impl_->previousCursorY_ = cursorY;
    input.scrollDeltaY = std::exchange(impl_->scrollDeltaY_, 0.0f);
    return input;
}

std::shared_ptr<RendererPlatform> makeGlfwRendererPlatform(std::string_view title,
    std::uint32_t width, std::uint32_t height)
{
    return std::make_shared<GlfwRendererPlatform>(title, width, height);
}
