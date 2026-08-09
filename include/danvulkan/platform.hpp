#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include <vulkan/vulkan.h>

struct RendererFramebufferExtent
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// Host applications can implement this boundary around an existing native window. The renderer
// retains the shared adapter while initialized but never directly owns or interprets native input.
class RendererPlatform
{
public:
    virtual ~RendererPlatform() = default;

    [[nodiscard]] virtual bool shouldClose() const noexcept = 0;
    virtual void pollEvents() = 0;
    virtual void waitEvents() = 0;
    [[nodiscard]] virtual RendererFramebufferExtent framebufferExtent() const noexcept = 0;
    [[nodiscard]] virtual bool consumeFramebufferResize() noexcept = 0;

    // Returned extension names must remain valid for the lifetime of the adapter.
    [[nodiscard]] virtual std::span<const char* const>
        requiredVulkanInstanceExtensions() const noexcept = 0;
    // The renderer owns and destroys the returned surface before releasing this adapter.
    [[nodiscard]] virtual VkSurfaceKHR createVulkanSurface(VkInstance instance) = 0;

    // Optional conveniences used by the standalone demo and its resize smoke test.
    virtual void setWindowTitle(std::string_view) {}
    [[nodiscard]] virtual bool requestWindowResize(std::uint32_t, std::uint32_t)
    {
        return false;
    }
};

// Optional explicit construction of the same backend selected when RendererConfig::platform is
// null. The returned interface contains no GLFW types.
[[nodiscard]] std::shared_ptr<RendererPlatform> makeGlfwRendererPlatform(
    std::string_view title, std::uint32_t width, std::uint32_t height);
