#include "device_context.hpp"

#include "vulkan_result.hpp"

#include <array>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace danvulkan::vk
{
namespace
{
constexpr std::array<const char*, 1> requiredExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };

struct ProbedDevice
{
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    DeviceCandidateCapabilities capabilities;
};

SwapchainSupportDetails querySwapchainSupport(
    VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    SwapchainSupportDetails details;
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        physicalDevice, surface, &details.capabilities),
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    std::uint32_t formatCount = 0;
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr),
        "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
    details.formats.resize(formatCount);
    if (formatCount > 0)
    {
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(
            physicalDevice, surface, &formatCount, details.formats.data()),
            "vkGetPhysicalDeviceSurfaceFormatsKHR");
        details.formats.resize(formatCount);
    }

    std::uint32_t presentModeCount = 0;
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, surface, &presentModeCount, nullptr),
        "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
    details.presentModes.resize(presentModeCount);
    if (presentModeCount > 0)
    {
        check(vkGetPhysicalDeviceSurfacePresentModesKHR(
            physicalDevice, surface, &presentModeCount, details.presentModes.data()),
            "vkGetPhysicalDeviceSurfacePresentModesKHR");
        details.presentModes.resize(presentModeCount);
    }
    return details;
}

QueueFamilyIndices queryQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    std::uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> properties(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &familyCount, properties.data());

    std::vector<QueueFamilyCapabilities> capabilities(familyCount);
    for (std::uint32_t index = 0; index < familyCount; ++index)
    {
        VkBool32 presentation = VK_FALSE;
        check(vkGetPhysicalDeviceSurfaceSupportKHR(
            physicalDevice, index, surface, &presentation),
            "vkGetPhysicalDeviceSurfaceSupportKHR");
        capabilities[index] = {
            (properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0,
            presentation == VK_TRUE
        };
    }
    return selectQueueFamilies(capabilities);
}

bool supportsRequiredExtensions(VkPhysicalDevice physicalDevice)
{
    std::uint32_t extensionCount = 0;
    check(vkEnumerateDeviceExtensionProperties(
        physicalDevice, nullptr, &extensionCount, nullptr),
        "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> available(extensionCount);
    if (extensionCount > 0)
    {
        check(vkEnumerateDeviceExtensionProperties(
            physicalDevice, nullptr, &extensionCount, available.data()),
            "vkEnumerateDeviceExtensionProperties");
    }

    for (const char* required : requiredExtensions)
    {
        bool found = false;
        for (const VkExtensionProperties& extension : available)
        {
            if (std::string_view(extension.extensionName) == required)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            return false;
        }
    }
    return true;
}

DeviceClass deviceClass(VkPhysicalDeviceType type) noexcept
{
    switch (type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return DeviceClass::integratedGpu;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return DeviceClass::discreteGpu;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return DeviceClass::virtualGpu;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return DeviceClass::cpu;
    default: return DeviceClass::other;
    }
}

ProbedDevice probeDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    ProbedDevice result;
    result.handle = physicalDevice;
    result.capabilities.queueFamilies = queryQueueFamilies(physicalDevice, surface);
    result.capabilities.requiredExtensions = supportsRequiredExtensions(physicalDevice);

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    result.capabilities.apiVersion = properties.apiVersion;
    result.capabilities.maxImageDimension2D = properties.limits.maxImageDimension2D;
    result.capabilities.deviceClass = deviceClass(properties.deviceType);

    if (result.capabilities.requiredExtensions)
    {
        const SwapchainSupportDetails support =
            querySwapchainSupport(physicalDevice, surface);
        result.capabilities.swapchainAdequate =
            !support.formats.empty() && !support.presentModes.empty();
    }

    VkPhysicalDeviceVulkan11Features vulkan11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    VkPhysicalDeviceVulkan12Features vulkan12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceVulkan13Features vulkan13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    features.pNext = &vulkan11;
    vulkan11.pNext = &vulkan12;
    vulkan12.pNext = &vulkan13;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);

    result.capabilities.features = {
        features.features.samplerAnisotropy == VK_TRUE,
        features.features.sampleRateShading == VK_TRUE,
        features.features.multiDrawIndirect == VK_TRUE,
        features.features.drawIndirectFirstInstance == VK_TRUE,
        vulkan11.shaderDrawParameters == VK_TRUE,
        vulkan12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE,
        vulkan12.runtimeDescriptorArray == VK_TRUE,
        vulkan13.shaderDemoteToHelperInvocation == VK_TRUE,
        vulkan13.synchronization2 == VK_TRUE,
        vulkan13.dynamicRendering == VK_TRUE
    };
    return result;
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
    VkDebugUtilsObjectNameInfoEXT info{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    info.objectType = type;
    info.objectHandle = handleValue(handle);
    info.pObjectName = name;
    check(setName(device, &info), "vkSetDebugUtilsObjectNameEXT(device context)");
}
}

void DeviceContext::initialize(
    VkInstance instance, VkSurfaceKHR surface, bool enableDebugNames)
{
    if (device_)
    {
        throw std::logic_error("device context is already initialized");
    }
    if (instance == VK_NULL_HANDLE || surface == VK_NULL_HANDLE)
    {
        throw std::invalid_argument("device context requires an instance and surface");
    }

    std::uint32_t deviceCount = 0;
    check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr),
        "vkEnumeratePhysicalDevices(count)");
    if (deviceCount == 0)
    {
        throw std::runtime_error("no Vulkan physical devices were found");
    }
    std::vector<VkPhysicalDevice> handles(deviceCount);
    check(vkEnumeratePhysicalDevices(instance, &deviceCount, handles.data()),
        "vkEnumeratePhysicalDevices");
    handles.resize(deviceCount);

    std::vector<ProbedDevice> probed;
    std::vector<DeviceCandidateCapabilities> candidates;
    probed.reserve(handles.size());
    candidates.reserve(handles.size());
    for (VkPhysicalDevice handle : handles)
    {
        probed.push_back(probeDevice(handle, surface));
        candidates.push_back(probed.back().capabilities);
    }
    const std::optional<std::size_t> selected =
        selectBestDeviceCandidate(candidates, VK_API_VERSION_1_4);
    if (!selected)
    {
        throw std::runtime_error(
            "no physical device satisfies the Vulkan 1.4 renderer requirements");
    }

    physicalDevice_ = probed[*selected].handle;
    queueFamilies_ = probed[*selected].capabilities.queueFamilies;
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties_);

    const float priority = 1.0f;
    const std::set<std::uint32_t> uniqueFamilies{
        *queueFamilies_.graphicsFamily, *queueFamilies_.presentFamily };
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(uniqueFamilies.size());
    for (std::uint32_t family : uniqueFamilies)
    {
        VkDeviceQueueCreateInfo info{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        info.queueFamilyIndex = family;
        info.queueCount = 1;
        info.pQueuePriorities = &priority;
        queueInfos.push_back(info);
    }

    VkPhysicalDeviceVulkan11Features vulkan11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    vulkan11.shaderDrawParameters = VK_TRUE;
    VkPhysicalDeviceVulkan12Features vulkan12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    vulkan12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    vulkan12.runtimeDescriptorArray = VK_TRUE;
    VkPhysicalDeviceVulkan13Features vulkan13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    vulkan13.shaderDemoteToHelperInvocation = VK_TRUE;
    vulkan13.synchronization2 = VK_TRUE;
    vulkan13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    features.features.samplerAnisotropy = VK_TRUE;
    features.features.sampleRateShading = VK_TRUE;
    features.features.multiDrawIndirect = VK_TRUE;
    features.features.drawIndirectFirstInstance = VK_TRUE;
    features.pNext = &vulkan11;
    vulkan11.pNext = &vulkan12;
    vulkan12.pNext = &vulkan13;

    VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    createInfo.pNext = &features;
    createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();
    check(vkCreateDevice(physicalDevice_, &createInfo, nullptr, device_.put()),
        "vkCreateDevice");

    vkGetDeviceQueue(device_, *queueFamilies_.graphicsFamily, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, *queueFamilies_.presentFamily, 0, &presentQueue_);
    if (enableDebugNames)
    {
        setDebugName(device_, VK_OBJECT_TYPE_DEVICE, device_.get(), "DanVulkan logical device");
        if (graphicsQueue_ == presentQueue_)
        {
            setDebugName(device_, VK_OBJECT_TYPE_QUEUE, graphicsQueue_,
                "graphics/present queue");
        }
        else
        {
            setDebugName(device_, VK_OBJECT_TYPE_QUEUE, graphicsQueue_, "graphics queue");
            setDebugName(device_, VK_OBJECT_TYPE_QUEUE, presentQueue_, "present queue");
        }
    }
}

void DeviceContext::reset() noexcept
{
    device_.reset();
    physicalDevice_ = VK_NULL_HANDLE;
    graphicsQueue_ = VK_NULL_HANDLE;
    presentQueue_ = VK_NULL_HANDLE;
    queueFamilies_ = {};
    properties_ = {};
}

VkSampleCountFlagBits DeviceContext::maxUsableSampleCount() const noexcept
{
    const VkSampleCountFlags counts = properties_.limits.framebufferColorSampleCounts &
        properties_.limits.framebufferDepthSampleCounts;
    if ((counts & VK_SAMPLE_COUNT_64_BIT) != 0) return VK_SAMPLE_COUNT_64_BIT;
    if ((counts & VK_SAMPLE_COUNT_32_BIT) != 0) return VK_SAMPLE_COUNT_32_BIT;
    if ((counts & VK_SAMPLE_COUNT_16_BIT) != 0) return VK_SAMPLE_COUNT_16_BIT;
    if ((counts & VK_SAMPLE_COUNT_8_BIT) != 0) return VK_SAMPLE_COUNT_8_BIT;
    if ((counts & VK_SAMPLE_COUNT_4_BIT) != 0) return VK_SAMPLE_COUNT_4_BIT;
    if ((counts & VK_SAMPLE_COUNT_2_BIT) != 0) return VK_SAMPLE_COUNT_2_BIT;
    return VK_SAMPLE_COUNT_1_BIT;
}

SwapchainSupportDetails DeviceContext::querySwapchainSupport(VkSurfaceKHR surface) const
{
    if (!device_ || surface == VK_NULL_HANDLE)
    {
        throw std::logic_error("swapchain support requires an initialized device context");
    }
    return danvulkan::vk::querySwapchainSupport(physicalDevice_, surface);
}
}
