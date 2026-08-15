#pragma once

#include <danvulkan/assets.hpp>

#include "device_context.hpp"
#include "descriptor_context.hpp"
#include "environment_precompute.hpp"
#include "lighting_planner.hpp"
#include "upload_context.hpp"
#include "vulkan_raii.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace danvulkan::vk
{
class LightingContext
{
public:
    LightingContext() = default;
    ~LightingContext() { reset(); }

    LightingContext(const LightingContext&) = delete;
    LightingContext& operator=(const LightingContext&) = delete;
    LightingContext(LightingContext&&) = delete;
    LightingContext& operator=(LightingContext&&) = delete;

    void initialize(const DeviceContext& device, VmaAllocator allocator,
        UploadContext& uploads, const assets::TextureAsset* environment,
        std::uint32_t pointLightCapacity, std::size_t imageCount,
        bool enableDebugNames);
    void recreateBuffers(std::size_t imageCount);
    void writePointLights(std::size_t imageIndex,
        std::span<const PlannedPointLight> pointLights);
    void reset() noexcept;

    [[nodiscard]] BufferDescriptor lightBuffer(std::size_t imageIndex) const;
    [[nodiscard]] std::array<VkDescriptorImageInfo, 3> environmentDescriptors() const noexcept;
    [[nodiscard]] std::uint32_t pointLightCapacity() const noexcept
    {
        return pointLightCapacity_;
    }
    [[nodiscard]] std::uint32_t environmentMipLevels() const noexcept
    {
        return environmentMipLevels_;
    }

private:
    void createEnvironment(const assets::TextureAsset& source, UploadContext& uploads);
    void createEnvironmentImage(Image& destination, VkSampler& sampler,
        std::uint32_t width, std::uint32_t height, std::span<const EnvironmentMip> levels,
        std::span<const float> pixels, bool repeatHorizontally, UploadContext& uploads,
        const char* name);
    [[nodiscard]] Buffer createLightBuffer(std::size_t index) const;
    void setDebugName(VkObjectType type, std::uint64_t handle, const char* name) const;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkSampler irradianceSampler_ = VK_NULL_HANDLE;
    VkSampler specularSampler_ = VK_NULL_HANDLE;
    VkSampler brdfSampler_ = VK_NULL_HANDLE;
    Image irradianceImage_;
    Image specularImage_;
    Image brdfImage_;
    std::vector<Buffer> lightBuffers_;
    std::uint32_t pointLightCapacity_ = 0;
    std::uint32_t environmentMipLevels_ = 1;
    bool enableDebugNames_ = false;
};
}
