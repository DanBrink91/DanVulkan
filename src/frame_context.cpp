#include "frame_context.hpp"

#include "vulkan_result.hpp"

#include <array>
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

template <typename Handle>
void setDebugName(VkDevice device, VkObjectType type, Handle handle, const std::string& name)
{
    const auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (setName == nullptr)
    {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handleValue(handle);
    info.pObjectName = name.c_str();
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(frame context)");
}
}

void FrameContext::initialize(VkDevice device, std::uint32_t graphicsQueueFamily,
    std::uint32_t frameIndex, bool enableDebugNames)
{
    if (device_ != VK_NULL_HANDLE)
    {
        throw std::logic_error("frame context is already initialized");
    }
    if (device == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("frame context requires a device");
    }

    device_ = device;
    try
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = graphicsQueueFamily;
        check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_),
            "vkCreateCommandPool(frame)");

        VkCommandBufferAllocateInfo commandInfo{};
        commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandInfo.commandPool = commandPool_;
        commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandInfo.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &commandInfo, &commandBuffer_),
            "vkAllocateCommandBuffers(frame)");

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailable_),
            "vkCreateSemaphore(image available)");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateFence(device_, &fenceInfo, nullptr, &inFlight_),
            "vkCreateFence(frame in flight)");

        VkQueryPoolCreateInfo queryInfo{};
        queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = 2;
        check(vkCreateQueryPool(device_, &queryInfo, nullptr, &timestampQueries_),
            "vkCreateQueryPool(frame timestamps)");

        if (enableDebugNames)
        {
            const std::string suffix = " " + std::to_string(frameIndex);
            setDebugName(device_, VK_OBJECT_TYPE_COMMAND_POOL, commandPool_,
                "frame command pool" + suffix);
            setDebugName(device_, VK_OBJECT_TYPE_COMMAND_BUFFER, commandBuffer_,
                "frame command buffer" + suffix);
            setDebugName(device_, VK_OBJECT_TYPE_SEMAPHORE, imageAvailable_,
                "image available semaphore" + suffix);
            setDebugName(device_, VK_OBJECT_TYPE_FENCE, inFlight_,
                "in-flight fence" + suffix);
            setDebugName(device_, VK_OBJECT_TYPE_QUERY_POOL, timestampQueries_,
                "frame timestamp queries" + suffix);
        }
    }
    catch (...)
    {
        reset();
        throw;
    }
}

std::optional<double> FrameContext::waitForReuse(float timestampPeriodNanoseconds)
{
    if (device_ == VK_NULL_HANDLE)
    {
        throw std::logic_error("frame context is not initialized");
    }
    check(vkWaitForFences(device_, 1, &inFlight_, VK_TRUE, UINT64_MAX),
        "vkWaitForFences(frame)");
    if (!submitted_)
    {
        return std::nullopt;
    }

    std::array<std::uint64_t, 2> timestamps{};
    check(vkGetQueryPoolResults(device_, timestampQueries_, 0,
        static_cast<std::uint32_t>(timestamps.size()), sizeof(timestamps), timestamps.data(),
        sizeof(timestamps[0]), VK_QUERY_RESULT_64_BIT),
        "vkGetQueryPoolResults(completed frame)");
    submitted_ = false;
    return static_cast<double>(timestamps[1] - timestamps[0]) *
        static_cast<double>(timestampPeriodNanoseconds) * 1e-6;
}

VkCommandBuffer FrameContext::beginCommands()
{
    if (device_ == VK_NULL_HANDLE || recording_)
    {
        throw std::logic_error("frame command recording cannot begin in the current state");
    }
    check(vkResetCommandPool(device_, commandPool_, 0), "vkResetCommandPool(frame)");
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commandBuffer_, &beginInfo), "vkBeginCommandBuffer(frame)");
    vkCmdResetQueryPool(commandBuffer_, timestampQueries_, 0, 2);
    vkCmdWriteTimestamp2(commandBuffer_, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        timestampQueries_, 0);
    recording_ = true;
    return commandBuffer_;
}

void FrameContext::endCommands()
{
    if (!recording_)
    {
        throw std::logic_error("frame command recording has not begun");
    }
    vkCmdWriteTimestamp2(commandBuffer_, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        timestampQueries_, 1);
    check(vkEndCommandBuffer(commandBuffer_), "vkEndCommandBuffer(frame)");
    recording_ = false;
}

void FrameContext::resetFenceForSubmit()
{
    if (recording_)
    {
        throw std::logic_error("cannot submit while frame commands are recording");
    }
    check(vkResetFences(device_, 1, &inFlight_), "vkResetFences(frame)");
}

void FrameContext::reset() noexcept
{
    recording_ = false;
    submitted_ = false;
    if (timestampQueries_ != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device_, timestampQueries_, nullptr);
    }
    if (inFlight_ != VK_NULL_HANDLE)
    {
        vkDestroyFence(device_, inFlight_, nullptr);
    }
    if (imageAvailable_ != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(device_, imageAvailable_, nullptr);
    }
    if (commandPool_ != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
    }
    timestampQueries_ = VK_NULL_HANDLE;
    inFlight_ = VK_NULL_HANDLE;
    imageAvailable_ = VK_NULL_HANDLE;
    commandBuffer_ = VK_NULL_HANDLE;
    commandPool_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}
}
