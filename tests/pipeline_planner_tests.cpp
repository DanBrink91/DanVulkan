#include "src/pipeline_planner.hpp"

#include <cstdlib>
#include <iostream>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
}

int main()
{
    using namespace danvulkan::vk;

    require(!planGraphicsPipelines(VK_FORMAT_UNDEFINED,
        VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT),
        "undefined color formats must be rejected");
    require(!planGraphicsPipelines(VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT),
        "undefined depth formats must be rejected");
    require(!planGraphicsPipelines(VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_D32_SFLOAT, static_cast<VkSampleCountFlagBits>(3)),
        "multiple sample-count bits must be rejected");

    const auto plan = planGraphicsPipelines(VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_D32_SFLOAT_S8_UINT, VK_SAMPLE_COUNT_4_BIT);
    require(plan.has_value(), "valid attachment configuration must produce a plan");
    require(plan->colorFormat == VK_FORMAT_B8G8R8A8_SRGB &&
        plan->depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT &&
        plan->samples == VK_SAMPLE_COUNT_4_BIT && plan->hasStencil,
        "attachment configuration was not preserved");

    for (std::size_t index = 0; index < plan->variants.size(); ++index)
    {
        const PipelineVariantPlan& variant = plan->variants[index];
        const bool doubleSided = (index % 2U) != 0;
        const bool blended = index >= static_cast<std::size_t>(PipelineVariant::blend);
        require(variant.cullMode == (doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT),
            "pipeline culling policy does not match its sidedness");
        require((variant.blendEnable == VK_TRUE) == blended,
            "pipeline blend policy does not match its alpha mode");
        require((variant.depthWriteEnable == VK_TRUE) != blended,
            "pipeline depth-write policy does not match its alpha mode");
        require(variant.sourceColorBlendFactor ==
                (blended ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE) &&
            variant.destinationColorBlendFactor ==
                (blended ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ZERO),
            "pipeline color blend factors do not match their alpha mode");
    }

    require(!depthFormatHasStencil(VK_FORMAT_D32_SFLOAT),
        "depth-only format must not request a stencil attachment");
    require(depthFormatHasStencil(VK_FORMAT_D24_UNORM_S8_UINT),
        "depth-stencil format must request a stencil attachment");
    require(depthFormatHasStencil(VK_FORMAT_D16_UNORM_S8_UINT),
        "16-bit depth-stencil format must request a stencil attachment");
    return EXIT_SUCCESS;
}
