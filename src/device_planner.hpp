#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace danvulkan::vk
{
struct QueueFamilyIndices
{
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    [[nodiscard]] bool complete() const noexcept
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct QueueFamilyCapabilities
{
    bool graphics = false;
    bool presentation = false;
};

[[nodiscard]] QueueFamilyIndices selectQueueFamilies(
    std::span<const QueueFamilyCapabilities> families) noexcept;

enum class DeviceClass
{
    other,
    integratedGpu,
    discreteGpu,
    virtualGpu,
    cpu
};

struct RequiredDeviceFeatures
{
    bool samplerAnisotropy = false;
    bool sampleRateShading = false;
    bool multiDrawIndirect = false;
    bool drawIndirectFirstInstance = false;
    bool shaderDrawParameters = false;
    bool shaderSampledImageArrayNonUniformIndexing = false;
    bool runtimeDescriptorArray = false;
    bool shaderDemoteToHelperInvocation = false;
    bool synchronization2 = false;
    bool dynamicRendering = false;

    [[nodiscard]] bool complete() const noexcept;
};

struct DeviceCandidateCapabilities
{
    QueueFamilyIndices queueFamilies;
    RequiredDeviceFeatures features;
    std::uint32_t apiVersion = 0;
    std::uint32_t maxImageDimension2D = 0;
    DeviceClass deviceClass = DeviceClass::other;
    bool requiredExtensions = false;
    bool swapchainAdequate = false;
};

[[nodiscard]] std::optional<std::uint64_t> scoreDeviceCandidate(
    const DeviceCandidateCapabilities& candidate, std::uint32_t minimumApiVersion) noexcept;
[[nodiscard]] std::optional<std::size_t> selectBestDeviceCandidate(
    std::span<const DeviceCandidateCapabilities> candidates,
    std::uint32_t minimumApiVersion) noexcept;
}
