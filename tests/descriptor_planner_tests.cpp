#include "src/descriptor_planner.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
void require(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}
}

int main()
{
    using namespace danvulkan::vk;

    require(selectTextureDescriptorCapacity({ 256, 32, 128, 192 }) == 125,
        "texture capacity did not reserve the environment samplers");
    require(!selectTextureDescriptorCapacity({ 256, 126, 128, 192 }),
        "a scene larger than descriptor capacity was accepted");
    require(!selectTextureDescriptorCapacity({ 0, 1, 128, 192 }),
        "zero requested texture capacity was accepted");
    require(!selectTextureDescriptorCapacity({ 256, 0, 128, 192 }),
        "zero required textures were accepted");

    const std::optional<DescriptorPlan> plan = planDescriptors(128, 3);
    require(plan && plan->textureCapacity == 128 && plan->setCount == 3,
        "valid descriptor inputs did not produce a plan");
    require(plan->bindings[static_cast<std::size_t>(DescriptorBinding::uniform)].descriptorType ==
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        "uniform binding type is incorrect");
    const VkDescriptorSetLayoutBinding& textures =
        plan->bindings[static_cast<std::size_t>(DescriptorBinding::textures)];
    require(textures.binding == 5 && textures.descriptorCount == 128 &&
            textures.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
            textures.stageFlags == VK_SHADER_STAGE_FRAGMENT_BIT,
        "bindless texture layout is incorrect");
    require(plan->poolSizes[0].descriptorCount == 3,
        "uniform pool size is incorrect");
    require(plan->poolSizes[1].descriptorCount == 18,
        "lighting storage-buffer pool size is incorrect");
    require(plan->poolSizes[2].descriptorCount == 393,
        "texture pool size is incorrect");
    const auto& lights =
        plan->bindings[static_cast<std::size_t>(DescriptorBinding::pointLights)];
    const auto& irradiance =
        plan->bindings[static_cast<std::size_t>(DescriptorBinding::irradiance)];
    const auto& specular =
        plan->bindings[static_cast<std::size_t>(DescriptorBinding::prefilteredSpecular)];
    const auto& brdf =
        plan->bindings[static_cast<std::size_t>(DescriptorBinding::environmentBrdf)];
    require(lights.binding == 7 && lights.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
        irradiance.binding == 8 && specular.binding == 9 && brdf.binding == 10 &&
        irradiance.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
        specular.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
        brdf.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        "lighting/environment descriptor layout is incorrect");

    require(!planDescriptors(0, 3), "zero texture capacity produced a descriptor plan");
    require(!planDescriptors(128, 0), "zero descriptor sets produced a descriptor plan");
    require(!planDescriptors(std::numeric_limits<std::uint32_t>::max(), 2),
        "overflowing descriptor counts produced a plan");
}
