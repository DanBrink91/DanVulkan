#pragma once

#include "vulkan_raii.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace danvulkan::vk
{
struct UploadArenaStats
{
    VkDeviceSize capacityBytes = 0;
    std::uint64_t growthCount = 0;
    std::uint64_t uploadCount = 0;
};

struct ImageUploadLevel
{
    VkDeviceSize byteOffset = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct BufferUploadRequest
{
    VkBuffer destination = VK_NULL_HANDLE;
    VkDeviceSize destinationOffset = 0;
    const void* data = nullptr;
    VkDeviceSize size = 0;
    VkBufferUsageFlags destinationUsage = 0;
};

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
    void uploadImageMipChain(VkImage destination, std::span<const ImageUploadLevel> levels,
        const void* data, VkDeviceSize size, std::string_view name);
    void copyBuffer(VkBuffer source, VkBuffer destination, VkDeviceSize size,
        VkBufferUsageFlags destinationUsage, std::string_view name);
    // Records many disjoint range copies into one ordered graphics-queue submission. The source
    // bytes are copied into a retained staging arena before this returns; readiness only governs
    // when that arena and command buffer can be reused for the next batch.
    void uploadBuffersAsync(
        std::span<const BufferUploadRequest> requests, std::string_view name);
    [[nodiscard]] bool asyncBufferUploadReady();

    [[nodiscard]] UploadArenaStats arenaStats() const noexcept
    {
        return {staging_.size() + asyncStaging_.size(), arenaGrowthCount_, uploadCount_};
    }

private:
    void writeStaging(const void* data, VkDeviceSize size, std::string_view name);
    void ensureStagingCapacity(VkDeviceSize requiredSize, std::string_view name);
    void ensureAsyncStagingCapacity(VkDeviceSize requiredSize, std::string_view name);
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
    Buffer staging_;
    VkCommandPool asyncCommandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer asyncCommandBuffer_ = VK_NULL_HANDLE;
    VkFence asyncFence_ = VK_NULL_HANDLE;
    Buffer asyncStaging_;
    bool asyncPending_ = false;
    std::uint64_t arenaGrowthCount_ = 0;
    std::uint64_t uploadCount_ = 0;
    bool enableDebugNames_ = false;
};
} // namespace danvulkan::vk
