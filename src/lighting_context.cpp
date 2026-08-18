#include "lighting_context.hpp"

#include "vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

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

assets::TextureAsset defaultEnvironment()
{
    // A tiny neutral sky keeps the environment binding valid without creating another
    // asset-loading path. Bilinear filtering turns these rows into a soft horizon gradient.
    constexpr std::array<std::uint8_t, 32> pixels{
        105, 128, 158, 255, 105, 128, 158, 255, 105, 128, 158, 255, 105, 128, 158, 255,
        70,  73,  78,  255, 70,  73,  78,  255, 70,  73,  78,  255, 70,  73,  78,  255
    };
    assets::TextureAsset result;
    result.name = "default neutral environment";
    result.width = 4;
    result.height = 2;
    result.colorSpace = assets::ColorSpace::srgb;
    result.sampler.minFilter = assets::TextureFilter::linear;
    result.sampler.magFilter = assets::TextureFilter::linear;
    result.sampler.mipmapMode = assets::TextureMipmapMode::linear;
    result.sampler.wrapU = assets::TextureWrap::repeat;
    result.sampler.wrapV = assets::TextureWrap::clampToEdge;
    result.rgba8.resize(pixels.size());
    std::memcpy(result.rgba8.data(), pixels.data(), pixels.size());
    return result;
}

}

void LightingContext::initialize(const DeviceContext& device, VmaAllocator allocator,
    UploadContext& uploads, const assets::TextureAsset* environment,
    std::uint32_t pointLightCapacity, std::size_t imageCount,
    bool enableDebugNames)
{
    if (device_ != VK_NULL_HANDLE)
    {
        throw std::logic_error("lighting context is already initialized");
    }
    if (!device || allocator == VK_NULL_HANDLE || pointLightCapacity == 0 ||
        pointLightCapacity > MaxScenePointLights || imageCount == 0)
    {
        throw std::invalid_argument("lighting context received invalid configuration");
    }
    physicalDevice_ = device.physicalDevice();
    device_ = device;
    allocator_ = allocator;
    pointLightCapacity_ = pointLightCapacity;
    enableDebugNames_ = enableDebugNames;
    try
    {
        assets::TextureAsset fallback;
        if (environment == nullptr)
        {
            fallback = defaultEnvironment();
            environment = &fallback;
        }
        createEnvironment(*environment, uploads);
        recreateBuffers(imageCount);
    }
    catch (...)
    {
        reset();
        throw;
    }
}

void LightingContext::createEnvironment(const assets::TextureAsset& source,
    UploadContext& uploads)
{
    const EnvironmentPrecompute precomputed = precomputeEnvironment(source);
    const EnvironmentMip irradianceLevel{
        precomputed.irradianceWidth, precomputed.irradianceHeight, 0};
    createEnvironmentImage(irradianceImage_, irradianceSampler_,
        precomputed.irradianceWidth, precomputed.irradianceHeight,
        std::span(&irradianceLevel, 1), precomputed.irradianceRgba32f,
        true, uploads, "lighting irradiance");
    createEnvironmentImage(specularImage_, specularSampler_,
        precomputed.specularMips.front().width, precomputed.specularMips.front().height,
        precomputed.specularMips, precomputed.specularRgba32f,
        true, uploads, "lighting prefiltered specular");
    const EnvironmentMip brdfLevel{precomputed.brdfSize, precomputed.brdfSize, 0};
    createEnvironmentImage(brdfImage_, brdfSampler_, precomputed.brdfSize,
        precomputed.brdfSize, std::span(&brdfLevel, 1), precomputed.brdfRgba32f,
        false, uploads, "lighting BRDF lookup");
    environmentMipLevels_ = static_cast<std::uint32_t>(precomputed.specularMips.size());
}

void LightingContext::createEnvironmentImage(Image& destination, VkSampler& sampler,
    std::uint32_t width, std::uint32_t height, std::span<const EnvironmentMip> levels,
    std::span<const float> pixels, bool repeatHorizontally, UploadContext& uploads,
    const char* name)
{
    if (width == 0 || height == 0 || levels.empty() || pixels.empty())
    {
        throw std::invalid_argument("precomputed environment image is incomplete");
    }
    constexpr VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &formatProperties);
    constexpr VkFormatFeatureFlags requiredFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
        VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((formatProperties.optimalTilingFeatures & requiredFeatures) != requiredFeatures)
    {
        throw std::runtime_error("device does not support sampled RGBA32F environment images");
    }

    auto imageInfo = makeVulkanStructure<VkImageCreateInfo>(
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = static_cast<std::uint32_t>(levels.size());
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    check(vmaCreateImage(allocator_, &imageInfo, &allocationInfo, &image, &allocation, nullptr),
        "vmaCreateImage(precomputed environment)");
    destination = Image(device_, allocator_, image, allocation);
    vmaSetAllocationName(allocator_, allocation, name);
    setDebugName(VK_OBJECT_TYPE_IMAGE, handleValue(image), name);

    std::vector<ImageUploadLevel> uploadLevels;
    uploadLevels.reserve(levels.size());
    for (const EnvironmentMip& level : levels)
    {
        uploadLevels.push_back({static_cast<VkDeviceSize>(level.valueOffset * sizeof(float)),
            level.width, level.height});
    }
    uploads.uploadImageMipChain(destination, uploadLevels, pixels.data(),
        static_cast<VkDeviceSize>(pixels.size_bytes()), name);

    auto viewInfo = makeVulkanStructure<VkImageViewCreateInfo>(
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewInfo.image = destination;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = static_cast<std::uint32_t>(levels.size());
    viewInfo.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    check(vkCreateImageView(device_, &viewInfo, nullptr, &view),
        "vkCreateImageView(precomputed environment)");
    destination.setView(view);
    setDebugName(VK_OBJECT_TYPE_IMAGE_VIEW, handleValue(view), name);

    auto samplerInfo = makeVulkanStructure<VkSamplerCreateInfo>(
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = repeatHorizontally ? VK_SAMPLER_ADDRESS_MODE_REPEAT :
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(levels.size() - 1U);
    check(vkCreateSampler(device_, &samplerInfo, nullptr, &sampler),
        "vkCreateSampler(precomputed environment)");
    setDebugName(VK_OBJECT_TYPE_SAMPLER, handleValue(sampler), name);
}

Buffer LightingContext::createLightBuffer(std::size_t index) const
{
    auto bufferInfo = makeVulkanStructure<VkBufferCreateInfo>(
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    bufferInfo.size = sizeof(PlannedPointLight) * pointLightCapacity_;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    allocationInfo.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo resultInfo{};
    check(vmaCreateBuffer(allocator_, &bufferInfo, &allocationInfo, &buffer, &allocation,
        &resultInfo), "vmaCreateBuffer(point lights)");
    const std::string name = "point light buffer " + std::to_string(index);
    vmaSetAllocationName(allocator_, allocation, name.c_str());
    setDebugName(VK_OBJECT_TYPE_BUFFER, handleValue(buffer), name.c_str());
    return Buffer(allocator_, buffer, allocation, resultInfo.pMappedData, bufferInfo.size);
}

void LightingContext::recreateBuffers(std::size_t imageCount)
{
    if (device_ == VK_NULL_HANDLE || imageCount == 0)
    {
        throw std::logic_error("lighting context cannot create swapchain buffers");
    }
    std::vector<Buffer> replacement;
    replacement.reserve(imageCount);
    for (std::size_t index = 0; index < imageCount; ++index)
    {
        replacement.push_back(createLightBuffer(index));
    }
    lightBuffers_ = std::move(replacement);
}

void LightingContext::writePointLights(std::size_t imageIndex,
    std::span<const PlannedPointLight> pointLights)
{
    if (imageIndex >= lightBuffers_.size() || pointLights.size() > pointLightCapacity_)
    {
        throw std::out_of_range("point-light buffer write is out of range");
    }
    if (pointLights.empty())
    {
        return;
    }
    Buffer& destination = lightBuffers_[imageIndex];
    if (destination.mapped() == nullptr)
    {
        throw std::runtime_error("point-light buffer is not mapped");
    }
    const VkDeviceSize size = sizeof(PlannedPointLight) * pointLights.size();
    std::memcpy(destination.mapped(), pointLights.data(), static_cast<std::size_t>(size));
    check(destination.flush(0, size), "vmaFlushAllocation(point lights)");
}

BufferDescriptor LightingContext::lightBuffer(std::size_t imageIndex) const
{
    if (imageIndex >= lightBuffers_.size())
    {
        throw std::out_of_range("point-light buffer index is out of range");
    }
    return {lightBuffers_[imageIndex], 0, lightBuffers_[imageIndex].size()};
}

std::array<VkDescriptorImageInfo, 3> LightingContext::environmentDescriptors() const noexcept
{
    return {{
        {irradianceSampler_, irradianceImage_.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {specularSampler_, specularImage_.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {brdfSampler_, brdfImage_.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}
    }};
}

void LightingContext::reset() noexcept
{
    lightBuffers_.clear();
    if (irradianceSampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, irradianceSampler_, nullptr);
    }
    if (specularSampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, specularSampler_, nullptr);
    }
    if (brdfSampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, brdfSampler_, nullptr);
    }
    irradianceSampler_ = VK_NULL_HANDLE;
    specularSampler_ = VK_NULL_HANDLE;
    brdfSampler_ = VK_NULL_HANDLE;
    irradianceImage_.reset();
    specularImage_.reset();
    brdfImage_.reset();
    environmentMipLevels_ = 1;
    pointLightCapacity_ = 0;
    allocator_ = VK_NULL_HANDLE;
    physicalDevice_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    enableDebugNames_ = false;
}

void LightingContext::setDebugName(VkObjectType type, std::uint64_t handle,
    const char* name) const
{
    if (!enableDebugNames_ || handle == 0)
    {
        return;
    }
    const auto function = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));
    if (function == nullptr)
    {
        return;
    }
    auto info = makeVulkanStructure<VkDebugUtilsObjectNameInfoEXT>(
        VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT);
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    check(function(device_, &info), "vkSetDebugUtilsObjectNameEXT(lighting context)");
}
}
