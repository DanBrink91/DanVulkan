#include "device_planner.hpp"

namespace danvulkan::vk
{
QueueFamilyIndices selectQueueFamilies(
    std::span<const QueueFamilyCapabilities> families) noexcept
{
    QueueFamilyIndices result;
    for (std::size_t index = 0; index < families.size(); ++index)
    {
        if (families[index].graphics && families[index].presentation)
        {
            const auto selected = static_cast<std::uint32_t>(index);
            return { selected, selected };
        }
    }
    for (std::size_t index = 0; index < families.size(); ++index)
    {
        if (!result.graphicsFamily && families[index].graphics)
        {
            result.graphicsFamily = static_cast<std::uint32_t>(index);
        }
        if (!result.presentFamily && families[index].presentation)
        {
            result.presentFamily = static_cast<std::uint32_t>(index);
        }
    }
    return result;
}

bool RequiredDeviceFeatures::complete() const noexcept
{
    return samplerAnisotropy && sampleRateShading && multiDrawIndirect &&
        drawIndirectFirstInstance && shaderDrawParameters &&
        shaderSampledImageArrayNonUniformIndexing && runtimeDescriptorArray &&
        shaderDemoteToHelperInvocation && synchronization2 && dynamicRendering;
}

std::optional<std::uint64_t> scoreDeviceCandidate(
    const DeviceCandidateCapabilities& candidate, std::uint32_t minimumApiVersion) noexcept
{
    if (!candidate.queueFamilies.complete() || !candidate.features.complete() ||
        candidate.apiVersion < minimumApiVersion || !candidate.requiredExtensions ||
        !candidate.swapchainAdequate)
    {
        return std::nullopt;
    }

    std::uint64_t classScore = 0;
    switch (candidate.deviceClass)
    {
    case DeviceClass::discreteGpu: classScore = 4'000'000; break;
    case DeviceClass::integratedGpu: classScore = 3'000'000; break;
    case DeviceClass::virtualGpu: classScore = 2'000'000; break;
    case DeviceClass::cpu: classScore = 1'000'000; break;
    case DeviceClass::other: break;
    }
    return classScore + candidate.maxImageDimension2D;
}

std::optional<std::size_t> selectBestDeviceCandidate(
    std::span<const DeviceCandidateCapabilities> candidates,
    std::uint32_t minimumApiVersion) noexcept
{
    std::optional<std::size_t> bestIndex;
    std::uint64_t bestScore = 0;
    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        const std::optional<std::uint64_t> score =
            scoreDeviceCandidate(candidates[index], minimumApiVersion);
        if (score && (!bestIndex || *score > bestScore))
        {
            bestIndex = index;
            bestScore = *score;
        }
    }
    return bestIndex;
}
}
