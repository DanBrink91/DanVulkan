#pragma once

#include "device_planner.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace danvulkan::vk
{
struct SwapchainPlan
{
    VkSurfaceFormatKHR surfaceFormat{};
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent{};
    std::uint32_t imageCount = 0;
    VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    std::array<std::uint32_t, 2> queueFamilyIndices{};
    std::uint32_t queueFamilyIndexCount = 0;
    VkSurfaceTransformFlagBitsKHR preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
};

[[nodiscard]] std::optional<VkSurfaceFormatKHR> selectSurfaceFormat(
    std::span<const VkSurfaceFormatKHR> formats) noexcept;
[[nodiscard]] VkPresentModeKHR selectPresentMode(
    std::span<const VkPresentModeKHR> presentModes) noexcept;
[[nodiscard]] VkExtent2D selectSwapchainExtent(
    const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D framebufferExtent) noexcept;
[[nodiscard]] std::uint32_t selectSwapchainImageCount(
    const VkSurfaceCapabilitiesKHR& capabilities) noexcept;
[[nodiscard]] VkCompositeAlphaFlagBitsKHR selectCompositeAlpha(
    VkCompositeAlphaFlagsKHR supported) noexcept;
[[nodiscard]] std::optional<SwapchainPlan> planSwapchain(
    const VkSurfaceCapabilitiesKHR& capabilities,
    std::span<const VkSurfaceFormatKHR> formats,
    std::span<const VkPresentModeKHR> presentModes,
    VkExtent2D framebufferExtent,
    const QueueFamilyIndices& queueFamilies) noexcept;
}
