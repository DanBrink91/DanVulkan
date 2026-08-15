#pragma once

#include "device_planner.hpp"
#include "vulkan_raii.hpp"

#include <vulkan/vulkan.h>

#include <vector>

namespace danvulkan::vk
{
struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class DeviceContext
{
public:
    DeviceContext() = default;
    ~DeviceContext() = default;

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&&) = delete;
    DeviceContext& operator=(DeviceContext&&) = delete;

    void initialize(VkInstance instance, VkSurfaceKHR surface, bool enableDebugNames);
    void reset() noexcept;

    [[nodiscard]] VkDevice get() const noexcept { return device_.get(); }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const noexcept { return physicalDevice_; }
    [[nodiscard]] VkQueue graphicsQueue() const noexcept { return graphicsQueue_; }
    [[nodiscard]] VkQueue presentQueue() const noexcept { return presentQueue_; }
    [[nodiscard]] const QueueFamilyIndices& queueFamilies() const noexcept
    {
        return queueFamilies_;
    }
    [[nodiscard]] const VkPhysicalDeviceProperties& properties() const noexcept
    {
        return properties_;
    }
    [[nodiscard]] bool supportsSamplerMipLodBias() const noexcept
    {
        return samplerMipLodBiasSupported_;
    }
    [[nodiscard]] VkSampleCountFlagBits maxUsableSampleCount() const noexcept;
    [[nodiscard]] SwapchainSupportDetails querySwapchainSupport(VkSurfaceKHR surface) const;

    explicit operator bool() const noexcept { return static_cast<bool>(device_); }
    operator VkDevice() const noexcept { return device_.get(); }

private:
    danvulkan::vk::Device device_;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    QueueFamilyIndices queueFamilies_;
    VkPhysicalDeviceProperties properties_{};
    bool samplerMipLodBiasSupported_ = true;
};
}
