#pragma once

#include "device_context.hpp"
#include "vulkan_raii.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <span>
#include <vector>

namespace danvulkan::vk
{
class SwapchainContext
{
public:
    SwapchainContext() = default;
    ~SwapchainContext() { reset(); }

    SwapchainContext(const SwapchainContext&) = delete;
    SwapchainContext& operator=(const SwapchainContext&) = delete;
    SwapchainContext(SwapchainContext&&) = delete;
    SwapchainContext& operator=(SwapchainContext&&) = delete;

    void initialize(const DeviceContext& device, VkSurfaceKHR surface,
        VkExtent2D framebufferExtent, bool enableDebugNames);
    void recreate(const DeviceContext& device, VkSurfaceKHR surface,
        VkExtent2D framebufferExtent, bool enableDebugNames);
    void reset() noexcept;

    [[nodiscard]] VkSwapchainKHR get() const noexcept { return swapchain_.get(); }
    [[nodiscard]] VkFormat format() const noexcept { return format_; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return extent_; }
    [[nodiscard]] std::span<const VkImage> images() const noexcept { return images_; }
    [[nodiscard]] std::span<const VkImageView> imageViews() const noexcept
    {
        return imageViews_;
    }
    [[nodiscard]] std::size_t imageCount() const noexcept { return images_.size(); }

    explicit operator bool() const noexcept { return static_cast<bool>(swapchain_); }
    operator VkSwapchainKHR() const noexcept { return swapchain_.get(); }

private:
    void create(const DeviceContext& device, VkSurfaceKHR surface,
        VkExtent2D framebufferExtent, bool enableDebugNames, bool requireExisting);
    void destroyImageViews() noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    danvulkan::vk::Swapchain swapchain_;
    std::vector<VkImage> images_;
    std::vector<VkImageView> imageViews_;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
};
}
