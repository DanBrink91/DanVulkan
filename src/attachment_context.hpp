#pragma once

#include "vulkan_raii.hpp"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <vector>

namespace danvulkan::vk
{
struct AttachmentContextCreateInfo
{
    VkExtent2D extent{};
    std::size_t attachmentCount = 0;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    bool preferLazilyAllocatedMemory = true;
    bool enableDebugNames = false;
};

// Owns the renderer-created color/depth targets indexed by frame in flight. Replacement is
// transactional: all new images and views are ready before the active set is retired.
class AttachmentContext
{
public:
    AttachmentContext() = default;
    ~AttachmentContext() { reset(); }

    AttachmentContext(const AttachmentContext&) = delete;
    AttachmentContext& operator=(const AttachmentContext&) = delete;
    AttachmentContext(AttachmentContext&&) = delete;
    AttachmentContext& operator=(AttachmentContext&&) = delete;

    void initialize(VkDevice device, VmaAllocator allocator,
        const AttachmentContextCreateInfo& createInfo);
    void recreate(const AttachmentContextCreateInfo& createInfo);
    void reset() noexcept;

    [[nodiscard]] const Image& color(std::size_t attachmentIndex) const;
    [[nodiscard]] const Image& depth(std::size_t attachmentIndex) const;
    [[nodiscard]] VkImageView colorView(std::size_t attachmentIndex) const;
    [[nodiscard]] VkImageView depthView(std::size_t attachmentIndex) const;
    [[nodiscard]] VkFormat depthFormat() const noexcept { return createInfo_.depthFormat; }
    [[nodiscard]] bool hasMultisampleColor() const noexcept
    {
        return createInfo_.samples != VK_SAMPLE_COUNT_1_BIT;
    }
    explicit operator bool() const noexcept { return device_ != VK_NULL_HANDLE; }

private:
    void create(const AttachmentContextCreateInfo& createInfo, bool requireExisting);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    AttachmentContextCreateInfo createInfo_{};
    std::vector<Image> colorImages_;
    std::vector<Image> depthImages_;
};
}
