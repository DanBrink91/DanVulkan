#pragma once

#include "vulkan_raii.hpp"

#include <cstdint>
#include <string_view>

namespace danvulkan::vk
{
class UploadContext
{
public:
    UploadContext() = default;
    ~UploadContext();

    UploadContext(const UploadContext&) = delete;
    UploadContext& operator=(const UploadContext&) = delete;
    UploadContext(UploadContext&&) = delete;
    UploadContext& operator=(UploadContext&&) = delete;

    void initialize(VkDevice device, VmaAllocator allocator, VkQueue queue,
        std::uint32_t queueFamilyIndex, bool enableDebugNames);
    void reset() noexcept;

    void uploadBuffer(VkBuffer destination, VkDeviceSize destinationOffset,
        const void* data, VkDeviceSize size, VkBufferUsageFlags destinationUsage,
        std::string_view name);
    void uploadImage(VkImage destination, std::uint32_t width, std::uint32_t height,
        std::uint32_t mipLevels, const void* rgba8, VkDeviceSize size, std::string_view name);

private:
    [[nodiscard]] Buffer createStagingBuffer(
        const void* data, VkDeviceSize size, std::string_view name) const;
    [[nodiscard]] VkCommandBuffer beginCommands();
    void submitAndWait(VkCommandBuffer commandBuffer);
    void setDebugName(VkObjectType objectType, std::uint64_t handle,
        std::string_view name) const;
    static void transitionImage(VkCommandBuffer commandBuffer, VkImage image,
        VkImageLayout oldLayout, VkImageLayout newLayout,
        VkPipelineStageFlags2 sourceStage, VkAccessFlags2 sourceAccess,
        VkPipelineStageFlags2 destinationStage, VkAccessFlags2 destinationAccess,
        std::uint32_t baseMipLevel, std::uint32_t levelCount);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool enableDebugNames_ = false;
};
} // namespace danvulkan::vk
