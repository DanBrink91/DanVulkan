#include "descriptor_planner.hpp"

#include <algorithm>
#include <limits>

namespace danvulkan::vk
{
namespace
{
constexpr std::uint32_t bindingIndex(DescriptorBinding binding) noexcept
{
    return static_cast<std::uint32_t>(binding);
}

VkDescriptorSetLayoutBinding makeBinding(DescriptorBinding binding,
    VkDescriptorType type, std::uint32_t count, VkShaderStageFlags stages) noexcept
{
    VkDescriptorSetLayoutBinding result{};
    result.binding = bindingIndex(binding);
    result.descriptorType = type;
    result.descriptorCount = count;
    result.stageFlags = stages;
    return result;
}
}

std::optional<std::uint32_t> selectTextureDescriptorCapacity(
    const DescriptorCapacityLimits& limits) noexcept
{
    const std::uint32_t capacity = std::min({
        limits.requestedTextures,
        limits.maxPerStageSamplers,
        limits.maxDescriptorSetSamplers
    });
    if (capacity == 0 || limits.requiredTextures == 0 ||
        limits.requiredTextures > capacity)
    {
        return std::nullopt;
    }
    return capacity;
}

std::optional<DescriptorPlan> planDescriptors(
    std::uint32_t textureCapacity, std::uint32_t setCount) noexcept
{
    if (textureCapacity == 0 || setCount == 0)
    {
        return std::nullopt;
    }
    constexpr std::uint64_t storageBindingsPerSet = 5;
    const std::uint64_t storageCount =
        static_cast<std::uint64_t>(setCount) * storageBindingsPerSet;
    const std::uint64_t textureCount =
        static_cast<std::uint64_t>(setCount) * textureCapacity;
    if (storageCount > std::numeric_limits<std::uint32_t>::max() ||
        textureCount > std::numeric_limits<std::uint32_t>::max())
    {
        return std::nullopt;
    }

    DescriptorPlan plan;
    plan.textureCapacity = textureCapacity;
    plan.setCount = setCount;
    plan.bindings = {
        makeBinding(DescriptorBinding::uniform, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT),
        makeBinding(DescriptorBinding::material, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT),
        makeBinding(DescriptorBinding::draw, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_VERTEX_BIT),
        makeBinding(DescriptorBinding::transform, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_VERTEX_BIT),
        makeBinding(DescriptorBinding::vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_VERTEX_BIT),
        makeBinding(DescriptorBinding::textures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            textureCapacity, VK_SHADER_STAGE_FRAGMENT_BIT),
        makeBinding(DescriptorBinding::joints, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_VERTEX_BIT)
    };
    plan.poolSizes = {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, setCount },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            static_cast<std::uint32_t>(storageCount) },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            static_cast<std::uint32_t>(textureCount) }
    };
    return plan;
}
}
