#pragma once

#include "vulkan_raii.hpp"

#include <danvulkan/ui.hpp>

#include <vulkan/vulkan.h>

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace danvulkan::vk {

struct UiContextCreateInfo
{
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    std::filesystem::path vertexShader;
    std::filesystem::path fragmentShader;
    std::size_t frameCount = 0;
    bool enableDebugNames = false;
};

// Renderer-owned Vulkan backend for ImmediateUi draw data. Each frame slot owns one mapped
// vertex arena so UI recording never waits on or overwrites another in-flight frame.
class UiContext
{
public:
    UiContext() = default;
    ~UiContext() { reset(); }

    UiContext(const UiContext&) = delete;
    UiContext& operator=(const UiContext&) = delete;
    UiContext(UiContext&&) = delete;
    UiContext& operator=(UiContext&&) = delete;

    void initialize(VkDevice device, VmaAllocator allocator,
        const UiContextCreateInfo& createInfo);
    void rebuild(VkFormat colorFormat, const std::filesystem::path& vertexShader,
        const std::filesystem::path& fragmentShader, bool enableDebugNames);
    void prepare(std::size_t frameIndex, std::span<const UiVertex> vertices);
    void record(VkCommandBuffer commandBuffer, std::size_t frameIndex,
        const UiDrawData& drawData, VkExtent2D extent) const;
    void reset() noexcept;

    [[nodiscard]] bool compatible(VkFormat colorFormat) const noexcept
    {
        return pipeline_ != VK_NULL_HANDLE && colorFormat_ == colorFormat;
    }
    explicit operator bool() const noexcept { return pipeline_ != VK_NULL_HANDLE; }

private:
    struct PipelineResources
    {
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    [[nodiscard]] PipelineResources buildPipeline(VkFormat colorFormat,
        const std::filesystem::path& vertexShader,
        const std::filesystem::path& fragmentShader, bool enableDebugNames) const;
    void destroyPipeline(PipelineResources resources) const noexcept;
    void ensureCapacity(std::size_t frameIndex, std::size_t vertexCount);

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::vector<Buffer> vertexBuffers_;
    std::vector<std::size_t> vertexCapacities_;
    bool enableDebugNames_ = false;
};

}
