#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <optional>

namespace danvulkan::vk
{
enum class DescriptorBinding : std::uint32_t
{
    uniform = 0,
    material = 1,
    draw = 2,
    transform = 3,
    vertex = 4,
    textures = 5,
    joints = 6,
    pointLights = 7,
    irradiance = 8,
    prefilteredSpecular = 9,
    environmentBrdf = 10
};

struct DescriptorCapacityLimits
{
    std::uint32_t requestedTextures = 0;
    std::uint32_t requiredTextures = 0;
    std::uint32_t maxPerStageSamplers = 0;
    std::uint32_t maxDescriptorSetSamplers = 0;
    std::uint32_t reservedSamplers = 3;
};

struct DescriptorPlan
{
    static constexpr std::size_t bindingCount = 11;
    static constexpr std::size_t poolSizeCount = 3;

    std::uint32_t textureCapacity = 0;
    std::uint32_t setCount = 0;
    std::array<VkDescriptorSetLayoutBinding, bindingCount> bindings{};
    std::array<VkDescriptorPoolSize, poolSizeCount> poolSizes{};
};

[[nodiscard]] std::optional<std::uint32_t> selectTextureDescriptorCapacity(
    const DescriptorCapacityLimits& limits) noexcept;
[[nodiscard]] std::optional<DescriptorPlan> planDescriptors(
    std::uint32_t textureCapacity, std::uint32_t setCount) noexcept;
}
