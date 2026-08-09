#include "swapchain_planner.hpp"

#include <algorithm>
#include <limits>

namespace danvulkan::vk
{
std::optional<VkSurfaceFormatKHR> selectSurfaceFormat(
    std::span<const VkSurfaceFormatKHR> formats) noexcept
{
    constexpr std::array preferredFormats{
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_R8G8B8A8_SRGB
    };
    if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED)
    {
        return VkSurfaceFormatKHR{
            preferredFormats.front(),
            formats.front().colorSpace
        };
    }
    for (VkFormat preferred : preferredFormats)
    {
        for (const VkSurfaceFormatKHR& available : formats)
        {
            if (available.format == preferred &&
                available.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return available;
            }
        }
    }
    if (formats.empty())
    {
        return std::nullopt;
    }
    return formats.front();
}

VkPresentModeKHR selectPresentMode(
    std::span<const VkPresentModeKHR> presentModes) noexcept
{
    for (VkPresentModeKHR available : presentModes)
    {
        if (available == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return available;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D selectSwapchainExtent(
    const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D framebufferExtent) noexcept
{
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    return {
        std::clamp(framebufferExtent.width,
            capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(framebufferExtent.height,
            capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

std::uint32_t selectSwapchainImageCount(
    const VkSurfaceCapabilitiesKHR& capabilities) noexcept
{
    std::uint32_t count = capabilities.minImageCount;
    if (count != std::numeric_limits<std::uint32_t>::max())
    {
        ++count;
    }
    if (capabilities.maxImageCount > 0)
    {
        count = std::min(count, capabilities.maxImageCount);
    }
    return count;
}

VkCompositeAlphaFlagBitsKHR selectCompositeAlpha(
    VkCompositeAlphaFlagsKHR supported) noexcept
{
    constexpr std::array candidates{
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
    };
    for (VkCompositeAlphaFlagBitsKHR candidate : candidates)
    {
        if ((supported & candidate) != 0)
        {
            return candidate;
        }
    }
    return VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

std::optional<SwapchainPlan> planSwapchain(
    const VkSurfaceCapabilitiesKHR& capabilities,
    std::span<const VkSurfaceFormatKHR> formats,
    std::span<const VkPresentModeKHR> presentModes,
    VkExtent2D framebufferExtent,
    const QueueFamilyIndices& queueFamilies) noexcept
{
    const std::optional<VkSurfaceFormatKHR> format = selectSurfaceFormat(formats);
    if (!format || presentModes.empty() || !queueFamilies.complete() ||
        (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0 ||
        capabilities.supportedCompositeAlpha == 0)
    {
        return std::nullopt;
    }

    SwapchainPlan plan;
    plan.surfaceFormat = *format;
    plan.presentMode = selectPresentMode(presentModes);
    plan.extent = selectSwapchainExtent(capabilities, framebufferExtent);
    if (plan.extent.width == 0 || plan.extent.height == 0)
    {
        return std::nullopt;
    }
    plan.imageCount = selectSwapchainImageCount(capabilities);
    plan.preTransform = capabilities.currentTransform;
    plan.compositeAlpha = selectCompositeAlpha(capabilities.supportedCompositeAlpha);
    plan.queueFamilyIndices = {
        *queueFamilies.graphicsFamily,
        *queueFamilies.presentFamily
    };
    if (queueFamilies.graphicsFamily != queueFamilies.presentFamily)
    {
        plan.sharingMode = VK_SHARING_MODE_CONCURRENT;
        plan.queueFamilyIndexCount = 2;
    }
    return plan;
}
}
