#include "descriptor_context.hpp"

#include "descriptor_planner.hpp"
#include "vulkan_result.hpp"

#include <array>
#include <cstdint>
#include <limits>
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
    VkDebugUtilsObjectNameInfoEXT info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    info.objectType = type;
    info.objectHandle = handleValue(handle);
    info.pObjectName = name.c_str();
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(descriptor context)");
}

constexpr std::uint32_t bindingIndex(DescriptorBinding binding) noexcept
{
    return static_cast<std::uint32_t>(binding);
}
}

void DescriptorContext::initialize(
    VkDevice device, std::uint32_t textureCapacity, bool enableDebugNames)
{
    if (layout_ != VK_NULL_HANDLE)
    {
        throw std::logic_error("descriptor context is already initialized");
    }
    if (device == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("descriptor context requires a device");
    }
    const std::optional<DescriptorPlan> plan = planDescriptors(textureCapacity, 1);
    if (!plan)
    {
        throw std::invalid_argument("descriptor context requires a non-zero texture capacity");
    }

    VkDescriptorSetLayoutCreateInfo createInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    createInfo.bindingCount = static_cast<std::uint32_t>(plan->bindings.size());
    createInfo.pBindings = plan->bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(device, &createInfo, nullptr, &layout),
        "vkCreateDescriptorSetLayout");
    try
    {
        if (enableDebugNames)
        {
            setDebugName(device, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                layout, "material descriptor layout");
        }
    }
    catch (...)
    {
        vkDestroyDescriptorSetLayout(device, layout, nullptr);
        throw;
    }
    device_ = device;
    layout_ = layout;
    textureCapacity_ = textureCapacity;
    enableDebugNames_ = enableDebugNames;
}

void DescriptorContext::validateBuffer(BufferDescriptor buffer, const char* role)
{
    if (buffer.buffer == VK_NULL_HANDLE || buffer.range == 0)
    {
        throw std::invalid_argument(std::string("invalid ") + role + " descriptor buffer");
    }
}

void DescriptorContext::writeSet(VkDescriptorSet set,
    const DescriptorSetBindings& bindings,
    std::span<const VkDescriptorImageInfo> textures,
    std::span<const VkDescriptorImageInfo> environment) const
{
    validateBuffer(bindings.uniform, "uniform");
    validateBuffer(bindings.material, "material");
    validateBuffer(bindings.draw, "draw");
    validateBuffer(bindings.transform, "transform");
    validateBuffer(bindings.vertex, "vertex");
    validateBuffer(bindings.joints, "joint");
    validateBuffer(bindings.pointLights, "point-light");
    if (textures.size() != textureCapacity_)
    {
        throw std::invalid_argument("texture descriptor count does not match context capacity");
    }
    for (const VkDescriptorImageInfo& texture : textures)
    {
        if (texture.sampler == VK_NULL_HANDLE || texture.imageView == VK_NULL_HANDLE)
        {
            throw std::invalid_argument("texture descriptors require a sampler and image view");
        }
    }
    if (environment.size() != 3)
    {
        throw std::invalid_argument("environment requires irradiance, specular, and BRDF maps");
    }
    for (const VkDescriptorImageInfo& image : environment)
    {
        if (image.sampler == VK_NULL_HANDLE || image.imageView == VK_NULL_HANDLE)
        {
            throw std::invalid_argument(
                "environment descriptors require a sampler and image view");
        }
    }

    const std::array bufferInfos{
        VkDescriptorBufferInfo{ bindings.uniform.buffer, bindings.uniform.offset,
            bindings.uniform.range },
        VkDescriptorBufferInfo{ bindings.material.buffer, bindings.material.offset,
            bindings.material.range },
        VkDescriptorBufferInfo{ bindings.draw.buffer, bindings.draw.offset,
            bindings.draw.range },
        VkDescriptorBufferInfo{ bindings.transform.buffer, bindings.transform.offset,
            bindings.transform.range },
        VkDescriptorBufferInfo{ bindings.vertex.buffer, bindings.vertex.offset,
            bindings.vertex.range },
        VkDescriptorBufferInfo{ bindings.joints.buffer, bindings.joints.offset,
            bindings.joints.range },
        VkDescriptorBufferInfo{ bindings.pointLights.buffer, bindings.pointLights.offset,
            bindings.pointLights.range }
    };
    const std::array roles{
        DescriptorBinding::uniform,
        DescriptorBinding::material,
        DescriptorBinding::draw,
        DescriptorBinding::transform,
        DescriptorBinding::vertex,
        DescriptorBinding::joints,
        DescriptorBinding::pointLights
    };
    const std::array types{
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    };
    std::array<VkWriteDescriptorSet, 11> writes{};
    for (std::size_t index = 0; index < bufferInfos.size(); ++index)
    {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set;
        writes[index].dstBinding = bindingIndex(roles[index]);
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = types[index];
        writes[index].pBufferInfo = &bufferInfos[index];
    }
    VkWriteDescriptorSet& textureWrite = writes[bufferInfos.size()];
    textureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    textureWrite.dstSet = set;
    textureWrite.dstBinding = bindingIndex(DescriptorBinding::textures);
    textureWrite.descriptorCount = static_cast<std::uint32_t>(textures.size());
    textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureWrite.pImageInfo = textures.data();
    constexpr std::array environmentRoles{DescriptorBinding::irradiance,
        DescriptorBinding::prefilteredSpecular, DescriptorBinding::environmentBrdf};
    for (std::size_t index = 0; index < environment.size(); ++index)
    {
        VkWriteDescriptorSet& environmentWrite =
            writes[bufferInfos.size() + 1U + index];
        environmentWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        environmentWrite.dstSet = set;
        environmentWrite.dstBinding = bindingIndex(environmentRoles[index]);
        environmentWrite.descriptorCount = 1;
        environmentWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        environmentWrite.pImageInfo = &environment[index];
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);
}

void DescriptorContext::allocateSets(std::span<const DescriptorSetBindings> bindings,
    std::span<const VkDescriptorImageInfo> textures,
    std::span<const VkDescriptorImageInfo> environment)
{
    if (layout_ == VK_NULL_HANDLE)
    {
        throw std::logic_error("descriptor context must be initialized before set allocation");
    }
    if (bindings.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::length_error("descriptor set count exceeds Vulkan's 32-bit limit");
    }
    const std::optional<DescriptorPlan> plan = planDescriptors(
        textureCapacity_, static_cast<std::uint32_t>(bindings.size()));
    if (!plan)
    {
        throw std::invalid_argument("descriptor set allocation requires at least one set");
    }

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(plan->poolSizes.size());
    poolInfo.pPoolSizes = plan->poolSizes.data();
    poolInfo.maxSets = plan->setCount;
    VkDescriptorPool replacementPool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &replacementPool),
        "vkCreateDescriptorPool");

    std::vector<VkDescriptorSetLayout> layouts(bindings.size(), layout_);
    VkDescriptorSetAllocateInfo allocateInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocateInfo.descriptorPool = replacementPool;
    allocateInfo.descriptorSetCount = plan->setCount;
    allocateInfo.pSetLayouts = layouts.data();
    std::vector<VkDescriptorSet> replacementSets(bindings.size());
    try
    {
        check(vkAllocateDescriptorSets(
            device_, &allocateInfo, replacementSets.data()), "vkAllocateDescriptorSets");
        for (std::size_t index = 0; index < replacementSets.size(); ++index)
        {
            writeSet(replacementSets[index], bindings[index], textures, environment);
            if (enableDebugNames_)
            {
                setDebugName(device_, VK_OBJECT_TYPE_DESCRIPTOR_SET, replacementSets[index],
                    "swapchain descriptor set " + std::to_string(index));
            }
        }
        if (enableDebugNames_)
        {
            setDebugName(device_, VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                replacementPool, "main descriptor pool");
        }
    }
    catch (...)
    {
        vkDestroyDescriptorPool(device_, replacementPool, nullptr);
        throw;
    }

    resetSets();
    pool_ = replacementPool;
    sets_ = std::move(replacementSets);
}

VkDescriptorSet DescriptorContext::set(std::size_t index) const
{
    if (index >= sets_.size())
    {
        throw std::out_of_range("descriptor set index is out of range");
    }
    return sets_[index];
}

void DescriptorContext::updateVertex(std::size_t setIndex, BufferDescriptor vertex)
{
    validateBuffer(vertex, "vertex");
    const VkDescriptorBufferInfo bufferInfo{ vertex.buffer, vertex.offset, vertex.range };
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set(setIndex);
    write.dstBinding = bindingIndex(DescriptorBinding::vertex);
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void DescriptorContext::updateTextures(std::size_t setIndex,
    std::span<const VkDescriptorImageInfo> textures)
{
    if (textures.size() != textureCapacity_)
    {
        throw std::invalid_argument("texture descriptor count does not match context capacity");
    }
    for (const VkDescriptorImageInfo& texture : textures)
    {
        if (texture.sampler == VK_NULL_HANDLE || texture.imageView == VK_NULL_HANDLE)
        {
            throw std::invalid_argument("texture descriptors require a sampler and image view");
        }
    }
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = set(setIndex);
    write.dstBinding = bindingIndex(DescriptorBinding::textures);
    write.descriptorCount = static_cast<std::uint32_t>(textures.size());
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = textures.data();
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void DescriptorContext::resetSets() noexcept
{
    sets_.clear();
    if (pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }
}

void DescriptorContext::reset() noexcept
{
    resetSets();
    if (layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
    textureCapacity_ = 0;
    enableDebugNames_ = false;
}
}
