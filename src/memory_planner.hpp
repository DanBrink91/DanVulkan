#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace danvulkan::vk
{
struct MemoryPolicyPlan
{
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    std::size_t attachmentSetCount = 0;
    bool preferLazilyAllocatedAttachments = true;
};

// A zero sample limit means the highest mutually supported color/depth sample count.
// Otherwise the selected count is the highest supported power of two at or below the limit.
[[nodiscard]] std::optional<MemoryPolicyPlan> planMemoryPolicy(
    VkSampleCountFlags supportedFramebufferSamples,
    std::uint32_t maximumSamples,
    std::size_t framesInFlight,
    bool preferLazilyAllocatedAttachments) noexcept;

// Returns a geometrically grown arena capacity that can hold requiredBytes. Existing capacity is
// retained, the first allocation is at least minimumBytes, and overflow is reported as nullopt.
[[nodiscard]] std::optional<std::uint64_t> planArenaCapacity(
    std::uint64_t currentBytes,
    std::uint64_t requiredBytes,
    std::uint64_t minimumBytes = 64U * 1024U) noexcept;
}
