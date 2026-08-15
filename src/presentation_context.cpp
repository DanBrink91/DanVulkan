#include "presentation_context.hpp"

#include "vulkan_result.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

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

void setDebugName(VkDevice device, VkSemaphore semaphore, const std::string& name)
{
    const auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (setName == nullptr)
    {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = VK_OBJECT_TYPE_SEMAPHORE;
    info.objectHandle = handleValue(semaphore);
    info.pObjectName = name.c_str();
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(presentation context)");
}
}

void PresentationContext::initialize(VkDevice device, std::size_t imageCount,
    bool enableDebugNames)
{
    if (device_ != VK_NULL_HANDLE)
    {
        throw std::logic_error("presentation context is already initialized");
    }
    if (device == VK_NULL_HANDLE || imageCount == 0)
    {
        throw std::invalid_argument("presentation context requires a device and swapchain images");
    }
    device_ = device;
    try
    {
        renderFinished_ = createSemaphores(imageCount, enableDebugNames);
        imageFences_.assign(imageCount, VK_NULL_HANDLE);
    }
    catch (...)
    {
        reset();
        throw;
    }
}

void PresentationContext::recreate(std::size_t imageCount, bool enableDebugNames)
{
    if (device_ == VK_NULL_HANDLE)
    {
        throw std::logic_error("presentation context must be initialized before recreation");
    }
    if (imageCount == 0)
    {
        throw std::invalid_argument("presentation context requires swapchain images");
    }
    if (imageCount != renderFinished_.size())
    {
        std::vector<VkSemaphore> replacement = createSemaphores(imageCount, enableDebugNames);
        destroySemaphores();
        renderFinished_ = std::move(replacement);
    }
    imageFences_.assign(imageCount, VK_NULL_HANDLE);
}

std::vector<VkSemaphore> PresentationContext::createSemaphores(
    std::size_t imageCount, bool enableDebugNames) const
{
    std::vector<VkSemaphore> semaphores;
    semaphores.reserve(imageCount);
    try
    {
        VkSemaphoreCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (std::size_t index = 0; index < imageCount; ++index)
        {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            check(vkCreateSemaphore(device_, &createInfo, nullptr, &semaphore),
                "vkCreateSemaphore(render finished)");
            semaphores.push_back(semaphore);
            if (enableDebugNames)
            {
                setDebugName(device_, semaphore,
                    "render finished semaphore " + std::to_string(index));
            }
        }
    }
    catch (...)
    {
        for (VkSemaphore semaphore : semaphores)
        {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
        throw;
    }
    return semaphores;
}

void PresentationContext::waitForImage(std::size_t imageIndex) const
{
    if (imageIndex >= imageFences_.size())
    {
        throw std::out_of_range("swapchain presentation image index is out of range");
    }
    const VkFence fence = imageFences_[imageIndex];
    if (fence != VK_NULL_HANDLE)
    {
        check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences(swapchain image)");
    }
}

void PresentationContext::markImageInFlight(std::size_t imageIndex, VkFence fence)
{
    if (imageIndex >= imageFences_.size() || fence == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("invalid swapchain image fence assignment");
    }
    imageFences_[imageIndex] = fence;
}

VkSemaphore PresentationContext::renderFinished(std::size_t imageIndex) const
{
    if (imageIndex >= renderFinished_.size())
    {
        throw std::out_of_range("swapchain presentation semaphore index is out of range");
    }
    return renderFinished_[imageIndex];
}

void PresentationContext::destroySemaphores() noexcept
{
    for (VkSemaphore semaphore : renderFinished_)
    {
        vkDestroySemaphore(device_, semaphore, nullptr);
    }
    renderFinished_.clear();
}

void PresentationContext::reset() noexcept
{
    destroySemaphores();
    imageFences_.clear();
    device_ = VK_NULL_HANDLE;
}
}
