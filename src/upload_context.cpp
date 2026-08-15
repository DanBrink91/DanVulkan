#include "upload_context.hpp"

#include "memory_planner.hpp"
#include "vulkan_result.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace danvulkan::vk
{
namespace
{
template <typename Handle>
std::uint64_t handleValue(Handle handle) noexcept
{
    if constexpr (std::is_pointer_v<Handle>)
    {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
    }
    else
    {
        return static_cast<std::uint64_t>(handle);
    }
}
} // namespace

UploadContext::~UploadContext()
{
    reset();
}

void UploadContext::initialize(VkDevice device, VmaAllocator allocator, VkQueue queue,
    std::uint32_t queueFamilyIndex, bool enableDebugNames)
{
    if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE || queue == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("upload context requires a device, allocator, and queue");
    }

    reset();
    device_ = device;
    allocator_ = allocator;
    queue_ = queue;
    enableDebugNames_ = enableDebugNames;

    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_),
        "vkCreateCommandPool(upload)");

    VkCommandBufferAllocateInfo allocateInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocateInfo.commandPool = commandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device_, &allocateInfo, &commandBuffer_),
        "vkAllocateCommandBuffers(upload)");

    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    check(vkCreateFence(device_, &fenceInfo, nullptr, &fence_), "vkCreateFence(upload)");

    setDebugName(VK_OBJECT_TYPE_COMMAND_POOL, handleValue(commandPool_), "upload command pool");
    setDebugName(VK_OBJECT_TYPE_COMMAND_BUFFER, handleValue(commandBuffer_),
        "upload command buffer");
    setDebugName(VK_OBJECT_TYPE_FENCE, handleValue(fence_), "upload fence");
}

void UploadContext::reset() noexcept
{
    staging_.reset();
    if (device_ != VK_NULL_HANDLE)
    {
        if (fence_ != VK_NULL_HANDLE)
        {
            vkDestroyFence(device_, fence_, nullptr);
        }
        if (commandPool_ != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
    }

    fence_ = VK_NULL_HANDLE;
    commandBuffer_ = VK_NULL_HANDLE;
    commandPool_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    allocator_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    arenaGrowthCount_ = 0;
    uploadCount_ = 0;
    enableDebugNames_ = false;
}

void UploadContext::uploadBuffer(VkBuffer destination, VkDeviceSize destinationOffset,
    const void* data, VkDeviceSize size, VkBufferUsageFlags destinationUsage,
    std::string_view name)
{
    if (destination == VK_NULL_HANDLE || data == nullptr || size == 0)
    {
        throw std::invalid_argument("buffer upload requires a destination and non-empty data");
    }

    writeStaging(data, size, name);
    VkCommandBuffer commandBuffer = beginCommands();

    VkBufferCopy copyRegion{};
    copyRegion.dstOffset = destinationOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, staging_, destination, 1, &copyRegion);

    VkBufferMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    if ((destinationUsage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0)
    {
        barrier.dstStageMask |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
        barrier.dstAccessMask |= VK_ACCESS_2_INDEX_READ_BIT;
    }
    if ((destinationUsage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0)
    {
        barrier.dstStageMask |= VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        barrier.dstAccessMask |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    }
    if ((destinationUsage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
    {
        barrier.dstStageMask |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        barrier.dstAccessMask |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    }
    if ((destinationUsage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0)
    {
        barrier.dstStageMask |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        barrier.dstAccessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    }
    if (barrier.dstStageMask == VK_PIPELINE_STAGE_2_NONE)
    {
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    }
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = destination;
    barrier.offset = destinationOffset;
    barrier.size = size;

    VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

    submitAndWait(commandBuffer);
    ++uploadCount_;
}

void UploadContext::uploadImage(VkImage destination, std::uint32_t width,
    std::uint32_t height, std::uint32_t mipLevels, const void* rgba8, VkDeviceSize size,
    std::string_view name)
{
    const std::uint32_t maximumMipLevels =
        std::bit_width(std::max(width, height));
    if (destination == VK_NULL_HANDLE || rgba8 == nullptr || width == 0 || height == 0 ||
        mipLevels == 0 || mipLevels > maximumMipLevels || size == 0)
    {
        throw std::invalid_argument("image upload requires a destination and non-empty pixels");
    }

    writeStaging(rgba8, size, name);
    VkCommandBuffer commandBuffer = beginCommands();

    transitionImage(commandBuffer, destination, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, mipLevels);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { width, height, 1 };
    vkCmdCopyBufferToImage(commandBuffer, staging_, destination,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    std::int32_t mipWidth = static_cast<std::int32_t>(width);
    std::int32_t mipHeight = static_cast<std::int32_t>(height);
    for (std::uint32_t mipLevel = 1; mipLevel < mipLevels; ++mipLevel)
    {
        transitionImage(commandBuffer, destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT, mipLevel - 1, 1);

        const std::int32_t nextWidth = std::max(mipWidth / 2, 1);
        const std::int32_t nextHeight = std::max(mipHeight / 2, 1);
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = mipLevel - 1;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = mipLevel;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[1] = { nextWidth, nextHeight, 1 };
        vkCmdBlitImage(commandBuffer, destination, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        transitionImage(commandBuffer, destination, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, mipLevel - 1, 1);
        mipWidth = nextWidth;
        mipHeight = nextHeight;
    }

    transitionImage(commandBuffer, destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, mipLevels - 1, 1);

    submitAndWait(commandBuffer);
    ++uploadCount_;
}

void UploadContext::uploadImageMipChain(VkImage destination,
    std::span<const ImageUploadLevel> levels, const void* data, VkDeviceSize size,
    std::string_view name)
{
    if (destination == VK_NULL_HANDLE || levels.empty() || data == nullptr || size == 0 ||
        levels.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument("mip-chain upload requires a destination and pixel levels");
    }
    for (std::size_t index = 0; index < levels.size(); ++index)
    {
        const ImageUploadLevel& level = levels[index];
        if (level.width == 0 || level.height == 0 || level.byteOffset >= size ||
            (index > 0 && (level.width > levels[index - 1].width ||
                level.height > levels[index - 1].height)))
        {
            throw std::invalid_argument("mip-chain upload contains an invalid level");
        }
    }

    writeStaging(data, size, name);
    VkCommandBuffer commandBuffer = beginCommands();
    const std::uint32_t levelCount = static_cast<std::uint32_t>(levels.size());
    transitionImage(commandBuffer, destination, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, 0, levelCount);

    std::vector<VkBufferImageCopy> regions(levels.size());
    for (std::size_t index = 0; index < levels.size(); ++index)
    {
        regions[index].bufferOffset = levels[index].byteOffset;
        regions[index].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[index].imageSubresource.mipLevel = static_cast<std::uint32_t>(index);
        regions[index].imageSubresource.layerCount = 1;
        regions[index].imageExtent = {levels[index].width, levels[index].height, 1};
    }
    vkCmdCopyBufferToImage(commandBuffer, staging_, destination,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()),
        regions.data());
    transitionImage(commandBuffer, destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, levelCount);
    submitAndWait(commandBuffer);
    ++uploadCount_;
}

void UploadContext::copyBuffer(VkBuffer source, VkBuffer destination, VkDeviceSize size,
    VkBufferUsageFlags destinationUsage, std::string_view name)
{
    if (source == VK_NULL_HANDLE || destination == VK_NULL_HANDLE || size == 0)
    {
        throw std::invalid_argument("buffer copy requires two buffers and a non-zero size");
    }

    VkCommandBuffer commandBuffer = beginCommands();
    const VkBufferCopy copyRegion{0, 0, size};
    vkCmdCopyBuffer(commandBuffer, source, destination, 1, &copyRegion);

    VkBufferMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    barrier.dstAccessMask = VK_ACCESS_2_NONE;
    if ((destinationUsage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0)
    {
        barrier.dstStageMask |= VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
        barrier.dstAccessMask |= VK_ACCESS_2_INDEX_READ_BIT;
    }
    if ((destinationUsage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0)
    {
        barrier.dstStageMask |= VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT;
        barrier.dstAccessMask |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    }
    if ((destinationUsage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
    {
        barrier.dstStageMask |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        barrier.dstAccessMask |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    }
    if (barrier.dstStageMask == VK_PIPELINE_STAGE_2_NONE)
    {
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    }
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = destination;
    barrier.size = size;

    VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    submitAndWait(commandBuffer);
    ++uploadCount_;
    static_cast<void>(name);
}

void UploadContext::writeStaging(
    const void* data, VkDeviceSize size, std::string_view name)
{
    ensureStagingCapacity(size, name);
    if (staging_.mapped() == nullptr)
    {
        throw std::runtime_error("upload staging arena was not mapped");
    }
    std::memcpy(staging_.mapped(), data, static_cast<std::size_t>(size));
    check(staging_.flush(0, size), "vmaFlushAllocation(upload staging arena)");
}

void UploadContext::ensureStagingCapacity(VkDeviceSize requiredSize, std::string_view name)
{
    if (device_ == VK_NULL_HANDLE || allocator_ == VK_NULL_HANDLE || queue_ == VK_NULL_HANDLE)
    {
        throw std::logic_error("upload context is not initialized");
    }

    if (requiredSize <= staging_.size())
    {
        return;
    }
    const std::optional<std::uint64_t> capacity = planArenaCapacity(
        staging_.size(), requiredSize);
    if (!capacity)
    {
        throw std::overflow_error("upload staging arena capacity overflowed");
    }

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = static_cast<VkDeviceSize>(*capacity);
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    allocationInfo.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo resultInfo{};
    check(vmaCreateBuffer(allocator_, &bufferInfo, &allocationInfo, &buffer, &allocation,
              &resultInfo),
        "vmaCreateBuffer(upload staging)");

    Buffer replacement(allocator_, buffer, allocation, resultInfo.pMappedData, bufferInfo.size);
    if (replacement.mapped() == nullptr)
    {
        throw std::runtime_error("upload staging arena was not mapped");
    }

    const std::string terminatedName = std::string(name) + " staging arena";
    vmaSetAllocationName(allocator_, allocation, terminatedName.c_str());
    setDebugName(VK_OBJECT_TYPE_BUFFER, handleValue(buffer), terminatedName);
    staging_ = std::move(replacement);
    ++arenaGrowthCount_;
}

VkCommandBuffer UploadContext::beginCommands()
{
    if (device_ == VK_NULL_HANDLE || commandPool_ == VK_NULL_HANDLE ||
        commandBuffer_ == VK_NULL_HANDLE || fence_ == VK_NULL_HANDLE)
    {
        throw std::logic_error("upload context is not initialized");
    }

    check(vkResetFences(device_, 1, &fence_), "vkResetFences(upload)");
    check(vkResetCommandPool(device_, commandPool_, 0), "vkResetCommandPool(upload)");

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commandBuffer_, &beginInfo), "vkBeginCommandBuffer(upload)");
    return commandBuffer_;
}

void UploadContext::submitAndWait(VkCommandBuffer commandBuffer)
{
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer(upload)");

    VkCommandBufferSubmitInfo commandBufferInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO
    };
    commandBufferInfo.commandBuffer = commandBuffer;
    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;

    check(vkQueueSubmit2(queue_, 1, &submitInfo, fence_), "vkQueueSubmit2(upload)");
    check(vkWaitForFences(device_, 1, &fence_, VK_TRUE,
              std::numeric_limits<std::uint64_t>::max()),
        "vkWaitForFences(upload)");
}

void UploadContext::setDebugName(VkObjectType objectType, std::uint64_t handle,
    std::string_view name) const
{
    if (!enableDebugNames_ || handle == 0 || device_ == VK_NULL_HANDLE)
    {
        return;
    }

    const auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));
    if (setName == nullptr)
    {
        return;
    }

    const std::string terminatedName(name);
    VkDebugUtilsObjectNameInfoEXT nameInfo{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    nameInfo.objectType = objectType;
    nameInfo.objectHandle = handle;
    nameInfo.pObjectName = terminatedName.c_str();
    check(setName(device_, &nameInfo), "vkSetDebugUtilsObjectNameEXT(upload)");
}

void UploadContext::transitionImage(VkCommandBuffer commandBuffer, VkImage image,
    VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlags2 sourceStage,
    VkAccessFlags2 sourceAccess, VkPipelineStageFlags2 destinationStage,
    VkAccessFlags2 destinationAccess, std::uint32_t baseMipLevel, std::uint32_t levelCount)
{
    VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = sourceStage;
    barrier.srcAccessMask = sourceAccess;
    barrier.dstStageMask = destinationStage;
    barrier.dstAccessMask = destinationAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = baseMipLevel;
    barrier.subresourceRange.levelCount = levelCount;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}
} // namespace danvulkan::vk
