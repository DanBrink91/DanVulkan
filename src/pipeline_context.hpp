#pragma once

#include "pipeline_planner.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <filesystem>

namespace danvulkan::vk
{
struct PipelineContextCreateInfo
{
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    std::filesystem::path vertexShader;
    std::filesystem::path fragmentShader;
    bool depthOnly = false;
    // Fullscreen backgrounds do not consume geometry or participate in depth. They still use
    // the active color/depth attachment formats so dynamic rendering remains compatible.
    bool background = false;
    bool enableDebugNames = false;
};

class PipelineContext
{
public:
    PipelineContext() = default;
    ~PipelineContext() { reset(); }

    PipelineContext(const PipelineContext&) = delete;
    PipelineContext& operator=(const PipelineContext&) = delete;
    PipelineContext(PipelineContext&&) = delete;
    PipelineContext& operator=(PipelineContext&&) = delete;

    void initialize(VkDevice device, const PipelineContextCreateInfo& createInfo);
    void rebuild(const PipelineContextCreateInfo& createInfo);
    void reset() noexcept;

    [[nodiscard]] bool compatible(VkDescriptorSetLayout descriptorLayout,
        VkFormat colorFormat, VkFormat depthFormat,
        VkSampleCountFlagBits samples) const noexcept;
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return layout_; }
    [[nodiscard]] VkPipeline pipeline(std::size_t variant) const;
    explicit operator bool() const noexcept { return layout_ != VK_NULL_HANDLE; }

private:
    struct BuiltPipelines
    {
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::array<VkPipeline, PipelineVariantCount> pipelines{};
    };

    [[nodiscard]] BuiltPipelines build(const PipelineContextCreateInfo& createInfo) const;
    static void destroy(VkDevice device, BuiltPipelines& built) noexcept;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout_ = VK_NULL_HANDLE;
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::array<VkPipeline, PipelineVariantCount> pipelines_{};
};
}
