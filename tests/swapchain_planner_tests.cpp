#include "src/swapchain_planner.hpp"

#include <array>
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

VkSurfaceCapabilitiesKHR variableCapabilities()
{
    VkSurfaceCapabilitiesKHR capabilities{};
    capabilities.minImageCount = 2;
    capabilities.maxImageCount = 3;
    capabilities.currentExtent = {
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max()
    };
    capabilities.minImageExtent = { 320, 240 };
    capabilities.maxImageExtent = { 1920, 1080 };
    capabilities.currentTransform = VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR;
    capabilities.supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    capabilities.supportedCompositeAlpha =
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR |
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    return capabilities;
}
}

int main()
{
    using namespace danvulkan::vk;

    constexpr std::array formats{
        VkSurfaceFormatKHR{ VK_FORMAT_R8_UNORM, VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT },
        VkSurfaceFormatKHR{ VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        VkSurfaceFormatKHR{ VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }
    };
    require(selectSurfaceFormat(formats)->format == VK_FORMAT_B8G8R8A8_SRGB,
        "BGRA sRGB was not preferred");

    constexpr std::array rgbaOnly{
        VkSurfaceFormatKHR{ VK_FORMAT_R8_UNORM, VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT },
        VkSurfaceFormatKHR{ VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }
    };
    require(selectSurfaceFormat(rgbaOnly)->format == VK_FORMAT_R8G8B8A8_SRGB,
        "RGBA sRGB fallback was not selected");
    require(!selectSurfaceFormat({}), "an empty surface-format list produced a format");
    constexpr std::array undefinedFormat{
        VkSurfaceFormatKHR{ VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR }
    };
    require(selectSurfaceFormat(undefinedFormat)->format == VK_FORMAT_B8G8R8A8_SRGB,
        "an unrestricted surface format did not select the preferred BGRA format");

    constexpr std::array modes{ VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR };
    require(selectPresentMode(modes) == VK_PRESENT_MODE_MAILBOX_KHR,
        "mailbox presentation was not preferred");
    constexpr std::array fifoOnly{ VK_PRESENT_MODE_FIFO_KHR };
    require(selectPresentMode(fifoOnly) == VK_PRESENT_MODE_FIFO_KHR,
        "FIFO presentation fallback was not selected");

    VkSurfaceCapabilitiesKHR capabilities = variableCapabilities();
    const VkExtent2D clamped = selectSwapchainExtent(capabilities, { 4096, 100 });
    require(clamped.width == 1920 && clamped.height == 240,
        "variable surface extent was not clamped");
    capabilities.currentExtent = { 1280, 720 };
    const VkExtent2D fixed = selectSwapchainExtent(capabilities, { 640, 480 });
    require(fixed.width == 1280 && fixed.height == 720,
        "fixed surface extent was not preserved");

    capabilities = variableCapabilities();
    require(selectSwapchainImageCount(capabilities) == 3,
        "one image beyond the surface minimum was not requested");
    capabilities.maxImageCount = 2;
    require(selectSwapchainImageCount(capabilities) == 2,
        "image count did not respect the surface maximum");
    require(selectCompositeAlpha(capabilities.supportedCompositeAlpha) ==
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        "supported composite-alpha fallback was not selected");

    capabilities = variableCapabilities();
    const std::optional<SwapchainPlan> combined = planSwapchain(capabilities, formats, modes,
        { 1600, 900 }, QueueFamilyIndices{ 4, 4 });
    require(combined && combined->sharingMode == VK_SHARING_MODE_EXCLUSIVE &&
            combined->queueFamilyIndexCount == 0,
        "combined queues did not produce exclusive sharing");
    require(combined->extent.width == 1600 && combined->extent.height == 900 &&
            combined->preTransform == capabilities.currentTransform,
        "swapchain plan did not retain its extent and transform");

    const std::optional<SwapchainPlan> split = planSwapchain(capabilities, formats, fifoOnly,
        { 800, 600 }, QueueFamilyIndices{ 2, 7 });
    require(split && split->sharingMode == VK_SHARING_MODE_CONCURRENT &&
            split->queueFamilyIndexCount == 2 && split->queueFamilyIndices[0] == 2 &&
            split->queueFamilyIndices[1] == 7,
        "split queues did not produce concurrent sharing");

    require(!planSwapchain(capabilities, {}, fifoOnly, { 800, 600 }, { 0, 0 }),
        "a plan was produced without a surface format");
    require(!planSwapchain(capabilities, formats, {}, { 800, 600 }, { 0, 0 }),
        "a plan was produced without a present mode");
    require(!planSwapchain(capabilities, formats, fifoOnly, { 800, 600 }, { 0, std::nullopt }),
        "a plan was produced with incomplete queue families");
    capabilities.supportedUsageFlags = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    require(!planSwapchain(capabilities, formats, fifoOnly, { 800, 600 }, { 0, 0 }),
        "a plan was produced without color-attachment image support");
}
