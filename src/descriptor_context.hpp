#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace danvulkan::vk
{
struct BufferDescriptor
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = 0;
};

struct DescriptorSetBindings
{
    BufferDescriptor uniform;
    BufferDescriptor material;
    BufferDescriptor draw;
    BufferDescriptor transform;
    BufferDescriptor vertex;
    BufferDescriptor joints;
    BufferDescriptor pointLights;
};

class DescriptorContext
{
public:
    DescriptorContext() = default;
    ~DescriptorContext() { reset(); }

    DescriptorContext(const DescriptorContext&) = delete;
    DescriptorContext& operator=(const DescriptorContext&) = delete;
    DescriptorContext(DescriptorContext&&) = delete;
    DescriptorContext& operator=(DescriptorContext&&) = delete;

    void initialize(VkDevice device, std::uint32_t textureCapacity, bool enableDebugNames);
    void allocateSets(std::span<const DescriptorSetBindings> bindings,
        std::span<const VkDescriptorImageInfo> textures,
        std::span<const std::array<VkDescriptorImageInfo, 4>> lighting);
    void updateVertex(std::size_t setIndex, BufferDescriptor vertex);
    void updateTextures(std::size_t setIndex,
        std::span<const VkDescriptorImageInfo> textures);
    void resetSets() noexcept;
    void reset() noexcept;

    [[nodiscard]] VkDescriptorSetLayout layout() const noexcept { return layout_; }
    [[nodiscard]] VkDescriptorSet set(std::size_t index) const;
    [[nodiscard]] std::size_t setCount() const noexcept { return sets_.size(); }
    [[nodiscard]] std::uint32_t textureCapacity() const noexcept { return textureCapacity_; }
    explicit operator bool() const noexcept { return layout_ != VK_NULL_HANDLE; }

private:
    static void validateBuffer(BufferDescriptor buffer, const char* role);
    void writeSet(VkDescriptorSet set, const DescriptorSetBindings& bindings,
        std::span<const VkDescriptorImageInfo> textures,
        const std::array<VkDescriptorImageInfo, 4>& lighting) const;

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;
    std::uint32_t textureCapacity_ = 0;
    bool enableDebugNames_ = false;
};
}
