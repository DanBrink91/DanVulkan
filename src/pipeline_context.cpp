#include "pipeline_context.hpp"

#include "vulkan_result.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(pipeline context)");
}

std::vector<std::uint32_t> readShader(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to open shader: " + path.string());
    }
    const std::streamoff byteCount = file.tellg();
    if (byteCount <= 0 || byteCount % static_cast<std::streamoff>(sizeof(std::uint32_t)) != 0)
    {
        throw std::runtime_error("shader bytecode has an invalid size: " + path.string());
    }
    std::vector<std::uint32_t> code(
        static_cast<std::size_t>(byteCount) / sizeof(std::uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), byteCount))
    {
        throw std::runtime_error("failed to read shader: " + path.string());
    }
    return code;
}

VkShaderModule createShaderModule(VkDevice device, const std::filesystem::path& path)
{
    const std::vector<std::uint32_t> code = readShader(path);
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(std::uint32_t);
    createInfo.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &createInfo, nullptr, &module),
        "vkCreateShaderModule");
    return module;
}
}

void PipelineContext::initialize(VkDevice device, const PipelineContextCreateInfo& createInfo)
{
    if (device_ != VK_NULL_HANDLE)
    {
        throw std::logic_error("pipeline context is already initialized");
    }
    if (device == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("pipeline context requires a device");
    }

    device_ = device;
    try
    {
        rebuild(createInfo);
    }
    catch (...)
    {
        device_ = VK_NULL_HANDLE;
        throw;
    }
}

PipelineContext::BuiltPipelines PipelineContext::build(
    const PipelineContextCreateInfo& createInfo) const
{
    if (createInfo.descriptorLayout == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("pipeline context requires a descriptor layout");
    }
    const std::optional<GraphicsPipelinePlan> plan = planGraphicsPipelines(
        createInfo.colorFormat, createInfo.depthFormat, createInfo.samples);
    if (!plan)
    {
        throw std::invalid_argument("pipeline context received invalid attachment configuration");
    }

    VkShaderModule vertexShader = VK_NULL_HANDLE;
    VkShaderModule fragmentShader = VK_NULL_HANDLE;
    BuiltPipelines built;
    try
    {
        vertexShader = createShaderModule(device_, createInfo.vertexShader);
        fragmentShader = createShaderModule(device_, createInfo.fragmentShader);
        if (createInfo.enableDebugNames)
        {
            setDebugName(device_, VK_OBJECT_TYPE_SHADER_MODULE, vertexShader,
                createInfo.vertexShader.string());
            setDebugName(device_, VK_OBJECT_TYPE_SHADER_MODULE, fragmentShader,
                createInfo.fragmentShader.string());
        }

        const std::array shaderStages{
            VkPipelineShaderStageCreateInfo{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                VK_SHADER_STAGE_VERTEX_BIT, vertexShader, "main", nullptr },
            VkPipelineShaderStageCreateInfo{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader, "main", nullptr }
        };

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &createInfo.descriptorLayout;
        check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &built.layout),
            "vkCreatePipelineLayout");
        if (createInfo.enableDebugNames)
        {
            setDebugName(device_, VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                built.layout, "main pipeline layout");
        }

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        constexpr std::array dynamicStates{
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_TRUE;
        multisampling.rasterizationSamples = plan->samples;
        multisampling.minSampleShading = 0.2f;

        VkPipelineColorBlendAttachmentState colorBlend{};
        colorBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlend.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlend.alphaBlendOp = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlend;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &plan->colorFormat;
        renderingInfo.depthAttachmentFormat = plan->depthFormat;
        renderingInfo.stencilAttachmentFormat = plan->hasStencil
            ? plan->depthFormat : VK_FORMAT_UNDEFINED;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = built.layout;

        for (std::size_t index = 0; index < built.pipelines.size(); ++index)
        {
            const PipelineVariantPlan& variant = plan->variants[index];
            rasterizer.cullMode = variant.cullMode;
            colorBlend.blendEnable = variant.blendEnable;
            colorBlend.srcColorBlendFactor = variant.sourceColorBlendFactor;
            colorBlend.dstColorBlendFactor = variant.destinationColorBlendFactor;
            colorBlend.srcAlphaBlendFactor = variant.sourceAlphaBlendFactor;
            colorBlend.dstAlphaBlendFactor = variant.destinationAlphaBlendFactor;
            depthStencil.depthWriteEnable = variant.depthWriteEnable;
            check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                nullptr, &built.pipelines[index]),
                "vkCreateGraphicsPipelines(material variant)");
            if (createInfo.enableDebugNames)
            {
                setDebugName(device_, VK_OBJECT_TYPE_PIPELINE, built.pipelines[index],
                    "material pipeline variant " + std::to_string(index));
            }
        }
    }
    catch (...)
    {
        if (fragmentShader != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device_, fragmentShader, nullptr);
        }
        if (vertexShader != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device_, vertexShader, nullptr);
        }
        destroy(device_, built);
        throw;
    }

    vkDestroyShaderModule(device_, fragmentShader, nullptr);
    vkDestroyShaderModule(device_, vertexShader, nullptr);
    return built;
}

void PipelineContext::rebuild(const PipelineContextCreateInfo& createInfo)
{
    if (device_ == VK_NULL_HANDLE)
    {
        throw std::logic_error("pipeline context must be initialized before rebuild");
    }
    BuiltPipelines replacement = build(createInfo);
    BuiltPipelines previous{ layout_, pipelines_ };
    layout_ = replacement.layout;
    pipelines_ = replacement.pipelines;
    descriptorLayout_ = createInfo.descriptorLayout;
    colorFormat_ = createInfo.colorFormat;
    depthFormat_ = createInfo.depthFormat;
    samples_ = createInfo.samples;
    destroy(device_, previous);
}

bool PipelineContext::compatible(VkDescriptorSetLayout descriptorLayout,
    VkFormat colorFormat, VkFormat depthFormat, VkSampleCountFlagBits samples) const noexcept
{
    return layout_ != VK_NULL_HANDLE && descriptorLayout_ == descriptorLayout &&
        colorFormat_ == colorFormat && depthFormat_ == depthFormat && samples_ == samples;
}

VkPipeline PipelineContext::pipeline(std::size_t variant) const
{
    if (variant >= pipelines_.size())
    {
        throw std::out_of_range("pipeline variant index is out of range");
    }
    return pipelines_[variant];
}

void PipelineContext::destroy(VkDevice device, BuiltPipelines& built) noexcept
{
    for (VkPipeline& pipeline : built.pipelines)
    {
        if (pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
    }
    if (built.layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, built.layout, nullptr);
        built.layout = VK_NULL_HANDLE;
    }
}

void PipelineContext::reset() noexcept
{
    BuiltPipelines built{ layout_, pipelines_ };
    if (device_ != VK_NULL_HANDLE)
    {
        destroy(device_, built);
    }
    layout_ = VK_NULL_HANDLE;
    pipelines_ = {};
    descriptorLayout_ = VK_NULL_HANDLE;
    colorFormat_ = VK_FORMAT_UNDEFINED;
    depthFormat_ = VK_FORMAT_UNDEFINED;
    samples_ = VK_SAMPLE_COUNT_1_BIT;
    device_ = VK_NULL_HANDLE;
}
}
