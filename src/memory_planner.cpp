#include "memory_planner.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace danvulkan::vk
{
std::optional<MemoryPolicyPlan> planMemoryPolicy(
    VkSampleCountFlags supportedFramebufferSamples,
    std::uint32_t maximumSamples,
    std::size_t framesInFlight,
    bool preferLazilyAllocatedAttachments) noexcept
{
    constexpr std::array sampleCounts{
        VK_SAMPLE_COUNT_64_BIT,
        VK_SAMPLE_COUNT_32_BIT,
        VK_SAMPLE_COUNT_16_BIT,
        VK_SAMPLE_COUNT_8_BIT,
        VK_SAMPLE_COUNT_4_BIT,
        VK_SAMPLE_COUNT_2_BIT,
        VK_SAMPLE_COUNT_1_BIT
    };
    constexpr std::uint32_t allSupported = 0;

    if (framesInFlight == 0 || supportedFramebufferSamples == 0)
    {
        return std::nullopt;
    }
    if (maximumSamples != allSupported &&
        (maximumSamples > 64 || (maximumSamples & (maximumSamples - 1U)) != 0))
    {
        return std::nullopt;
    }

    for (const VkSampleCountFlagBits samples : sampleCounts)
    {
        const std::uint32_t count = static_cast<std::uint32_t>(samples);
        if ((supportedFramebufferSamples & samples) != 0 &&
            (maximumSamples == allSupported || count <= maximumSamples))
        {
            return MemoryPolicyPlan{
                samples, framesInFlight, preferLazilyAllocatedAttachments };
        }
    }
    return std::nullopt;
}

std::optional<std::uint64_t> planArenaCapacity(
    std::uint64_t currentBytes, std::uint64_t requiredBytes,
    std::uint64_t minimumBytes) noexcept
{
    if (requiredBytes == 0 || minimumBytes == 0)
    {
        return std::nullopt;
    }

    std::uint64_t capacity = std::max(currentBytes, minimumBytes);
    while (capacity < requiredBytes)
    {
        if (capacity > std::numeric_limits<std::uint64_t>::max() / 2U)
        {
            capacity = requiredBytes;
            break;
        }
        capacity *= 2U;
    }
    return capacity;
}
}
