#include "attachment_context.hpp"

#include "pipeline_planner.hpp"
#include "vulkan_result.hpp"

#include <vk_mem_alloc.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

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

template <typename Handle>
void setDebugName(VkDevice device, VkObjectType type, Handle handle, std::string_view name)
{
    const auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (setName == nullptr)
    {
        return;
    }
    const std::string terminatedName(name);
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handleValue(handle);
    info.pObjectName = terminatedName.c_str();
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(attachment context)");
}

Image createAttachment(VkDevice device, VmaAllocator allocator, VkExtent2D extent,
    VkSampleCountFlagBits samples, VkFormat format, VkImageUsageFlags usage,
    VkImageAspectFlags aspect, std::string_view name, bool preferLazilyAllocatedMemory,
    bool enableDebugNames)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = { extent.width, extent.height, 1 };
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = samples;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    allocationInfo.preferredFlags = preferLazilyAllocatedMemory
        ? VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT : 0;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    check(vmaCreateImage(allocator, &imageInfo, &allocationInfo, &image, &allocation, nullptr),
        "vmaCreateImage(swapchain attachment)");

    Image ownedImage(device, allocator, image, allocation);
    const std::string terminatedName(name);
    vmaSetAllocationName(allocator, allocation, terminatedName.c_str());
    if (enableDebugNames)
    {
        setDebugName(device, VK_OBJECT_TYPE_IMAGE, image, name);
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspect;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    check(vkCreateImageView(device, &viewInfo, nullptr, &view),
        "vkCreateImageView(swapchain attachment)");
    ownedImage.setView(view);
    if (enableDebugNames)
    {
        setDebugName(device, VK_OBJECT_TYPE_IMAGE_VIEW, view, terminatedName + " view");
    }
    return ownedImage;
}

bool sameConfiguration(const AttachmentContextCreateInfo& left,
    const AttachmentContextCreateInfo& right) noexcept
{
    return left.extent.width == right.extent.width &&
        left.extent.height == right.extent.height &&
        left.attachmentCount == right.attachmentCount &&
        left.colorFormat == right.colorFormat && left.depthFormat == right.depthFormat &&
        left.samples == right.samples &&
        left.preferLazilyAllocatedMemory == right.preferLazilyAllocatedMemory;
}
}

void AttachmentContext::initialize(VkDevice device, VmaAllocator allocator,
    const AttachmentContextCreateInfo& createInfo)
{
    if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("attachment context requires a device and allocator");
    }
    device_ = device;
    allocator_ = allocator;
    try
    {
        create(createInfo, false);
    }
    catch (...)
    {
        reset();
        throw;
    }
}

void AttachmentContext::recreate(const AttachmentContextCreateInfo& createInfo)
{
    create(createInfo, true);
}

void AttachmentContext::create(const AttachmentContextCreateInfo& createInfo,
    bool requireExisting)
{
    const bool hasAttachments = createInfo_.attachmentCount != 0;
    if ((requireExisting && (device_ == VK_NULL_HANDLE || !hasAttachments)) ||
        (!requireExisting && hasAttachments))
    {
        throw std::logic_error(requireExisting
            ? "attachment context must be initialized before recreation"
            : "attachment context is already initialized");
    }
    if (createInfo.extent.width == 0 || createInfo.extent.height == 0 ||
        createInfo.attachmentCount == 0 || createInfo.colorFormat == VK_FORMAT_UNDEFINED ||
        createInfo.depthFormat == VK_FORMAT_UNDEFINED || createInfo.samples == 0)
    {
        throw std::invalid_argument("attachment context requires complete image configuration");
    }
    if (requireExisting && sameConfiguration(createInfo_, createInfo))
    {
        return;
    }

    std::vector<Image> replacementColors;
    std::vector<Image> replacementDepths;
    replacementColors.reserve(createInfo.samples == VK_SAMPLE_COUNT_1_BIT
        ? 0 : createInfo.attachmentCount);
    replacementDepths.reserve(createInfo.attachmentCount);
    for (std::size_t index = 0; index < createInfo.attachmentCount; ++index)
    {
        if (createInfo.samples != VK_SAMPLE_COUNT_1_BIT)
        {
            replacementColors.push_back(createAttachment(device_, allocator_, createInfo.extent,
                createInfo.samples, createInfo.colorFormat,
                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT, "MSAA color image " + std::to_string(index),
                createInfo.preferLazilyAllocatedMemory,
                createInfo.enableDebugNames));
        }

        VkImageAspectFlags depthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (depthFormatHasStencil(createInfo.depthFormat))
        {
            depthAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        replacementDepths.push_back(createAttachment(device_, allocator_, createInfo.extent,
            createInfo.samples, createInfo.depthFormat,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            depthAspect, "depth image " + std::to_string(index),
            createInfo.preferLazilyAllocatedMemory, createInfo.enableDebugNames));
    }

    colorImages_ = std::move(replacementColors);
    depthImages_ = std::move(replacementDepths);
    createInfo_ = createInfo;
}

const Image& AttachmentContext::color(std::size_t attachmentIndex) const
{
    if (!hasMultisampleColor() || attachmentIndex >= colorImages_.size())
    {
        throw std::out_of_range("frame color attachment index is out of range");
    }
    return colorImages_[attachmentIndex];
}

const Image& AttachmentContext::depth(std::size_t attachmentIndex) const
{
    if (attachmentIndex >= depthImages_.size())
    {
        throw std::out_of_range("frame depth attachment index is out of range");
    }
    return depthImages_[attachmentIndex];
}

VkImageView AttachmentContext::colorView(std::size_t attachmentIndex) const
{
    return color(attachmentIndex).view();
}

VkImageView AttachmentContext::depthView(std::size_t attachmentIndex) const
{
    return depth(attachmentIndex).view();
}

void AttachmentContext::reset() noexcept
{
    colorImages_.clear();
    depthImages_.clear();
    createInfo_ = {};
    allocator_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}
}
