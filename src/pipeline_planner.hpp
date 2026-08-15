#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace danvulkan::vk
{
enum class PipelineVariant : std::uint32_t
{
    opaque = 0,
    opaqueDoubleSided,
    mask,
    maskDoubleSided,
    blend,
    blendDoubleSided,
    count
};

inline constexpr std::size_t PipelineVariantCount =
    static_cast<std::size_t>(PipelineVariant::count);

struct PipelineVariantPlan
{
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkBool32 blendEnable = VK_FALSE;
    VkBool32 depthWriteEnable = VK_TRUE;
    VkBlendFactor sourceColorBlendFactor = VK_BLEND_FACTOR_ONE;
    VkBlendFactor destinationColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    VkBlendFactor sourceAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    VkBlendFactor destinationAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
};

struct GraphicsPipelinePlan
{
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    bool hasStencil = false;
    std::array<PipelineVariantPlan, PipelineVariantCount> variants{};
};

[[nodiscard]] bool depthFormatHasStencil(VkFormat format) noexcept;
[[nodiscard]] std::optional<GraphicsPipelinePlan> planGraphicsPipelines(
    VkFormat colorFormat, VkFormat depthFormat, VkSampleCountFlagBits samples) noexcept;
}
