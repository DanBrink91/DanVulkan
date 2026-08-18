#include "ui_context.hpp"

#include "vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace danvulkan::vk {
namespace {
constexpr std::size_t initialVertexCapacity = 8192;

std::vector<std::uint32_t> readShader(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to open UI shader: " + path.string());
    }
    const std::streamoff byteCount = file.tellg();
    if (byteCount <= 0 || byteCount % static_cast<std::streamoff>(sizeof(std::uint32_t)) != 0)
    {
        throw std::runtime_error("UI shader bytecode has an invalid size: " + path.string());
    }
    std::vector<std::uint32_t> code(static_cast<std::size_t>(byteCount) /
        sizeof(std::uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), byteCount))
    {
        throw std::runtime_error("failed to read UI shader: " + path.string());
    }
    return code;
}

VkShaderModule createShaderModule(VkDevice device, const std::filesystem::path& path)
{
    const std::vector<std::uint32_t> code = readShader(path);
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = code.size() * sizeof(std::uint32_t);
    createInfo.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &createInfo, nullptr, &module),
        "vkCreateShaderModule(UI)");
    return module;
}

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
void setDebugName(VkDevice device, VkObjectType type, Handle handle, const char* name)
{
    const auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (setName == nullptr)
    {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = handleValue(handle);
    info.pObjectName = name;
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(UI)");
}
}

void UiContext::initialize(VkDevice device, VmaAllocator allocator,
    const UiContextCreateInfo& createInfo)
{
    if (device_ != VK_NULL_HANDLE)
    {
        throw std::logic_error("UI context is already initialized");
    }
    if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE ||
        createInfo.colorFormat == VK_FORMAT_UNDEFINED || createInfo.frameCount == 0)
    {
        throw std::invalid_argument("UI context received incomplete creation data");
    }
    device_ = device;
    allocator_ = allocator;
    enableDebugNames_ = createInfo.enableDebugNames;
    vertexBuffers_.resize(createInfo.frameCount);
    vertexCapacities_.resize(createInfo.frameCount, 0);
    try
    {
        const PipelineResources built = buildPipeline(createInfo.colorFormat,
            createInfo.vertexShader, createInfo.fragmentShader, createInfo.enableDebugNames);
        layout_ = built.layout;
        pipeline_ = built.pipeline;
        colorFormat_ = createInfo.colorFormat;
        for (std::size_t index = 0; index < createInfo.frameCount; ++index)
        {
            ensureCapacity(index, initialVertexCapacity);
        }
    }
    catch (...)
    {
        reset();
        throw;
    }
}

void UiContext::rebuild(VkFormat colorFormat, const std::filesystem::path& vertexShader,
    const std::filesystem::path& fragmentShader, bool enableDebugNames)
{
    if (device_ == VK_NULL_HANDLE)
    {
        throw std::logic_error("UI context must be initialized before rebuild");
    }
    const PipelineResources replacement = buildPipeline(
        colorFormat, vertexShader, fragmentShader, enableDebugNames);
    const PipelineResources previous{layout_, pipeline_};
    layout_ = replacement.layout;
    pipeline_ = replacement.pipeline;
    colorFormat_ = colorFormat;
    enableDebugNames_ = enableDebugNames;
    destroyPipeline(previous);
}

void UiContext::prepare(std::size_t frameIndex, std::span<const UiVertex> vertices)
{
    if (frameIndex >= vertexBuffers_.size())
    {
        throw std::out_of_range("UI frame index is out of range");
    }
    if (vertices.empty())
    {
        return;
    }
    ensureCapacity(frameIndex, vertices.size());
    const VkDeviceSize byteCount = static_cast<VkDeviceSize>(vertices.size_bytes());
    std::memcpy(vertexBuffers_[frameIndex].mapped(), vertices.data(), vertices.size_bytes());
    check(vertexBuffers_[frameIndex].flush(0, byteCount), "vmaFlushAllocation(UI vertices)");
}

void UiContext::record(VkCommandBuffer commandBuffer, std::size_t frameIndex,
    const UiDrawData& drawData, VkExtent2D extent) const
{
    if (drawData.vertices.empty() || drawData.commands.empty() || extent.width == 0 ||
        extent.height == 0)
    {
        return;
    }
    if (frameIndex >= vertexBuffers_.size())
    {
        throw std::out_of_range("UI frame index is out of range");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    const VkDeviceSize offset = 0;
    const VkBuffer buffer = vertexBuffers_[frameIndex].get();
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &buffer, &offset);
    const std::array<float, 2> viewportSize{
        static_cast<float>(extent.width), static_cast<float>(extent.height)};
    vkCmdPushConstants(commandBuffer, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
        sizeof(viewportSize), viewportSize.data());

    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    for (const UiDrawCommand& command : drawData.commands)
    {
        if (command.firstVertex > drawData.vertices.size() ||
            command.vertexCount > drawData.vertices.size() - command.firstVertex)
        {
            throw std::invalid_argument("UI draw command exceeds submitted vertex data");
        }
        VkRect2D scissor;
        scissor.offset = {static_cast<std::int32_t>(command.clipX),
            static_cast<std::int32_t>(command.clipY)};
        scissor.extent = {
            std::min(command.clipWidth, extent.width - std::min(command.clipX, extent.width)),
            std::min(command.clipHeight, extent.height - std::min(command.clipY, extent.height))};
        if (scissor.extent.width == 0 || scissor.extent.height == 0 ||
            command.vertexCount == 0)
        {
            continue;
        }
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdDraw(commandBuffer, command.vertexCount, 1, command.firstVertex, 0);
    }
}

void UiContext::reset() noexcept
{
    vertexBuffers_.clear();
    vertexCapacities_.clear();
    destroyPipeline({layout_, pipeline_});
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    colorFormat_ = VK_FORMAT_UNDEFINED;
    allocator_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

UiContext::PipelineResources UiContext::buildPipeline(VkFormat colorFormat,
    const std::filesystem::path& vertexShader,
    const std::filesystem::path& fragmentShader, bool enableDebugNames) const
{
    VkShaderModule vertexModule = VK_NULL_HANDLE;
    VkShaderModule fragmentModule = VK_NULL_HANDLE;
    PipelineResources result;
    try
    {
        vertexModule = createShaderModule(device_, vertexShader);
        fragmentModule = createShaderModule(device_, fragmentShader);
        const std::array stages{
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertexModule, "main", nullptr},
            VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main", nullptr}
        };

        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstant.size = sizeof(float) * 2;
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstant;
        check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &result.layout),
            "vkCreatePipelineLayout(UI)");

        const std::array bindings{
            VkVertexInputBindingDescription{0, sizeof(UiVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        };
        const std::array attributes{
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(UiVertex, position))},
            VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(UiVertex, glyphUv))},
            VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
                static_cast<std::uint32_t>(offsetof(UiVertex, color))},
            VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32_UINT,
                static_cast<std::uint32_t>(offsetof(UiVertex, glyphMask))}
        };
        VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInput.vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size());
        vertexInput.pVertexBindingDescriptions = bindings.data();
        vertexInput.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkPipelineRasterizationStateCreateInfo rasterization{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo colorBlend{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &blend;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &colorFormat;

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.pNext = &rendering;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewport;
        pipelineInfo.pRasterizationState = &rasterization;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pColorBlendState = &colorBlend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = result.layout;
        check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
            nullptr, &result.pipeline), "vkCreateGraphicsPipelines(UI)");

        if (enableDebugNames)
        {
            setDebugName(device_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, result.layout,
                "UI pipeline layout");
            setDebugName(device_, VK_OBJECT_TYPE_PIPELINE, result.pipeline, "UI pipeline");
        }
    }
    catch (...)
    {
        if (fragmentModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device_, fragmentModule, nullptr);
        }
        if (vertexModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device_, vertexModule, nullptr);
        }
        destroyPipeline(result);
        throw;
    }
    vkDestroyShaderModule(device_, fragmentModule, nullptr);
    vkDestroyShaderModule(device_, vertexModule, nullptr);
    return result;
}

void UiContext::destroyPipeline(PipelineResources resources) const noexcept
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }
    if (resources.pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, resources.pipeline, nullptr);
    }
    if (resources.layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, resources.layout, nullptr);
    }
}

void UiContext::ensureCapacity(std::size_t frameIndex, std::size_t vertexCount)
{
    if (vertexCount <= vertexCapacities_[frameIndex])
    {
        return;
    }
    const std::size_t replacementCapacity = std::max(initialVertexCapacity,
        std::max(vertexCount, vertexCapacities_[frameIndex] * 2));
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = replacementCapacity * sizeof(UiVertex);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo resultInfo{};
    check(vmaCreateBuffer(allocator_, &bufferInfo, &allocationInfo, &buffer, &allocation,
        &resultInfo), "vmaCreateBuffer(UI vertex arena)");
    Buffer replacement(allocator_, buffer, allocation, resultInfo.pMappedData, bufferInfo.size);
    const std::string name = "UI frame " + std::to_string(frameIndex) + " vertex arena";
    vmaSetAllocationName(allocator_, allocation, name.c_str());
    if (enableDebugNames_)
    {
        setDebugName(device_, VK_OBJECT_TYPE_BUFFER, buffer, name.c_str());
    }
    vertexBuffers_[frameIndex] = std::move(replacement);
    vertexCapacities_[frameIndex] = replacementCapacity;
}

}
