#include "swapchain_context.hpp"

#include "swapchain_planner.hpp"
#include "vulkan_result.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
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
void setDebugName(VkDevice device, VkObjectType type, Handle handle, const std::string& name)
{
    const auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (setName == nullptr)
    {
        return;
    }
    auto info = makeVulkanStructure<VkDebugUtilsObjectNameInfoEXT>(
        VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT);
    info.objectType = type;
    info.objectHandle = handleValue(handle);
    info.pObjectName = name.c_str();
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(swapchain context)");
}
}

void SwapchainContext::initialize(const DeviceContext& device, VkSurfaceKHR surface,
    VkExtent2D framebufferExtent, bool enableDebugNames)
{
    create(device, surface, framebufferExtent, enableDebugNames, false);
}

void SwapchainContext::recreate(const DeviceContext& device, VkSurfaceKHR surface,
    VkExtent2D framebufferExtent, bool enableDebugNames)
{
    create(device, surface, framebufferExtent, enableDebugNames, true);
}

void SwapchainContext::create(const DeviceContext& device, VkSurfaceKHR surface,
    VkExtent2D framebufferExtent, bool enableDebugNames, bool requireExisting)
{
    if (!device || surface == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("swapchain context requires a device and surface");
    }
    if (requireExisting != static_cast<bool>(swapchain_))
    {
        throw std::logic_error(requireExisting
            ? "swapchain context must be initialized before recreation"
            : "swapchain context is already initialized");
    }

    const SwapchainSupportDetails support = device.querySwapchainSupport(surface);
    const std::optional<SwapchainPlan> plan = planSwapchain(support.capabilities,
        support.formats, support.presentModes, framebufferExtent, device.queueFamilies());
    if (!plan)
    {
        throw std::runtime_error("surface does not provide a usable swapchain configuration");
    }

    auto createInfo = makeVulkanStructure<VkSwapchainCreateInfoKHR>(
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
    createInfo.surface = surface;
    createInfo.minImageCount = plan->imageCount;
    createInfo.imageFormat = plan->surfaceFormat.format;
    createInfo.imageColorSpace = plan->surfaceFormat.colorSpace;
    createInfo.imageExtent = plan->extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = plan->sharingMode;
    createInfo.queueFamilyIndexCount = plan->queueFamilyIndexCount;
    createInfo.pQueueFamilyIndices = plan->queueFamilyIndexCount > 0
        ? plan->queueFamilyIndices.data() : nullptr;
    createInfo.preTransform = plan->preTransform;
    createInfo.compositeAlpha = plan->compositeAlpha;
    createInfo.presentMode = plan->presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = swapchain_.get();

    danvulkan::vk::Swapchain replacement;
    check(vkCreateSwapchainKHR(device, &createInfo, nullptr, replacement.put(device)),
        "vkCreateSwapchainKHR");

    std::uint32_t imageCount = 0;
    check(vkGetSwapchainImagesKHR(device, replacement, &imageCount, nullptr),
        "vkGetSwapchainImagesKHR(count)");
    std::vector<VkImage> replacementImages(imageCount);
    check(vkGetSwapchainImagesKHR(
        device, replacement, &imageCount, replacementImages.data()),
        "vkGetSwapchainImagesKHR");
    replacementImages.resize(imageCount);

    std::vector<VkImageView> replacementViews;
    replacementViews.reserve(replacementImages.size());
    try
    {
        for (std::size_t index = 0; index < replacementImages.size(); ++index)
        {
            auto viewInfo = makeVulkanStructure<VkImageViewCreateInfo>(
                VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
            viewInfo.image = replacementImages[index];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = plan->surfaceFormat.format;
            viewInfo.components = {
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY
            };
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            VkImageView view = VK_NULL_HANDLE;
            check(vkCreateImageView(device, &viewInfo, nullptr, &view),
                "vkCreateImageView(swapchain)");
            replacementViews.push_back(view);
            if (enableDebugNames)
            {
                setDebugName(device, VK_OBJECT_TYPE_IMAGE_VIEW, view,
                    "swapchain image view " + std::to_string(index));
            }
        }
        if (enableDebugNames)
        {
            setDebugName(device, VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                replacement.get(), "main swapchain");
        }
    }
    catch (...)
    {
        for (VkImageView view : replacementViews)
        {
            vkDestroyImageView(device, view, nullptr);
        }
        throw;
    }

    destroyImageViews();
    swapchain_ = std::move(replacement);
    device_ = device;
    images_ = std::move(replacementImages);
    imageViews_ = std::move(replacementViews);
    format_ = plan->surfaceFormat.format;
    extent_ = plan->extent;
}

void SwapchainContext::destroyImageViews() noexcept
{
    for (VkImageView view : imageViews_)
    {
        vkDestroyImageView(device_, view, nullptr);
    }
    imageViews_.clear();
}

void SwapchainContext::reset() noexcept
{
    destroyImageViews();
    images_.clear();
    swapchain_.reset();
    device_ = VK_NULL_HANDLE;
    format_ = VK_FORMAT_UNDEFINED;
    extent_ = {};
}
}
