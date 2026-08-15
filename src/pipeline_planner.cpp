#include "pipeline_planner.hpp"

#include <bit>

namespace danvulkan::vk
{
bool depthFormatHasStencil(VkFormat format) noexcept
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        format == VK_FORMAT_D24_UNORM_S8_UINT ||
        format == VK_FORMAT_D16_UNORM_S8_UINT;
}

std::optional<GraphicsPipelinePlan> planGraphicsPipelines(
    VkFormat colorFormat, VkFormat depthFormat, VkSampleCountFlagBits samples) noexcept
{
    const std::uint32_t sampleBits = static_cast<std::uint32_t>(samples);
    if (colorFormat == VK_FORMAT_UNDEFINED || depthFormat == VK_FORMAT_UNDEFINED ||
        sampleBits == 0 || !std::has_single_bit(sampleBits))
    {
        return std::nullopt;
    }

    GraphicsPipelinePlan plan;
    plan.colorFormat = colorFormat;
    plan.depthFormat = depthFormat;
    plan.samples = samples;
    plan.hasStencil = depthFormatHasStencil(depthFormat);

    for (std::size_t index = 0; index < plan.variants.size(); ++index)
    {
        PipelineVariantPlan& variant = plan.variants[index];
        const bool doubleSided = (index % 2U) != 0;
        const bool blended = index >= static_cast<std::size_t>(PipelineVariant::blend);
        variant.cullMode = doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
        if (blended)
        {
            variant.blendEnable = VK_TRUE;
            variant.depthWriteEnable = VK_FALSE;
            variant.sourceColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            variant.destinationColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            variant.sourceAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            variant.destinationAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        }
    }
    return plan;
}
}
