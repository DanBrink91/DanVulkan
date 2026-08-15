#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>

namespace danvulkan::vk
{
class FrameContext
{
public:
    FrameContext() = default;
    ~FrameContext() { reset(); }

    FrameContext(const FrameContext&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;
    FrameContext(FrameContext&&) = delete;
    FrameContext& operator=(FrameContext&&) = delete;

    void initialize(VkDevice device, std::uint32_t graphicsQueueFamily,
        std::uint32_t frameIndex, bool enableDebugNames);
    void reset() noexcept;

    // Waiting for reuse also makes the previous timestamp pair available without a GPU-side
    // query wait. The first call has no previous submission and returns no duration.
    [[nodiscard]] std::optional<double> waitForReuse(float timestampPeriodNanoseconds);
    [[nodiscard]] VkCommandBuffer beginCommands();
    void endCommands();
    void resetFenceForSubmit();
    void markSubmitted() noexcept { submitted_ = true; }

    [[nodiscard]] VkCommandBuffer commandBuffer() const noexcept { return commandBuffer_; }
    [[nodiscard]] VkSemaphore imageAvailable() const noexcept { return imageAvailable_; }
    [[nodiscard]] VkFence inFlight() const noexcept { return inFlight_; }
    explicit operator bool() const noexcept { return commandPool_ != VK_NULL_HANDLE; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkSemaphore imageAvailable_ = VK_NULL_HANDLE;
    VkFence inFlight_ = VK_NULL_HANDLE;
    VkQueryPool timestampQueries_ = VK_NULL_HANDLE;
    bool recording_ = false;
    bool submitted_ = false;
};
}
