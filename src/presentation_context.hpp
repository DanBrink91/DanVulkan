#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <vector>

namespace danvulkan::vk
{
// Owns binary semaphores consumed by presentation and tracks which frame fence last used each
// swapchain image. Semaphores are retained when recreation preserves the image count.
class PresentationContext
{
public:
    PresentationContext() = default;
    ~PresentationContext() { reset(); }

    PresentationContext(const PresentationContext&) = delete;
    PresentationContext& operator=(const PresentationContext&) = delete;
    PresentationContext(PresentationContext&&) = delete;
    PresentationContext& operator=(PresentationContext&&) = delete;

    void initialize(VkDevice device, std::size_t imageCount, bool enableDebugNames);
    void recreate(std::size_t imageCount, bool enableDebugNames);
    void reset() noexcept;

    void waitForImage(std::size_t imageIndex) const;
    void markImageInFlight(std::size_t imageIndex, VkFence fence);
    [[nodiscard]] VkSemaphore renderFinished(std::size_t imageIndex) const;
    [[nodiscard]] std::size_t imageCount() const noexcept { return imageFences_.size(); }
    explicit operator bool() const noexcept { return device_ != VK_NULL_HANDLE; }

private:
    [[nodiscard]] std::vector<VkSemaphore> createSemaphores(
        std::size_t imageCount, bool enableDebugNames) const;
    void destroySemaphores() noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<VkSemaphore> renderFinished_;
    std::vector<VkFence> imageFences_;
};
}
