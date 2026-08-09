#include "src/device_planner.hpp"

#include <array>
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

danvulkan::vk::DeviceCandidateCapabilities suitableCandidate()
{
    danvulkan::vk::DeviceCandidateCapabilities candidate;
    candidate.queueFamilies = { 0, 0 };
    candidate.features = { true, true, true, true, true, true, true, true, true, true };
    candidate.apiVersion = 140;
    candidate.maxImageDimension2D = 8192;
    candidate.deviceClass = danvulkan::vk::DeviceClass::integratedGpu;
    candidate.requiredExtensions = true;
    candidate.swapchainAdequate = true;
    return candidate;
}
}

int main()
{
    using namespace danvulkan::vk;

    constexpr std::array combinedFamilies{
        QueueFamilyCapabilities{ true, false },
        QueueFamilyCapabilities{ false, true },
        QueueFamilyCapabilities{ true, true }
    };
    const QueueFamilyIndices combined = selectQueueFamilies(combinedFamilies);
    require(combined.graphicsFamily == 2U && combined.presentFamily == 2U,
        "a combined graphics/present queue was not preferred");

    constexpr std::array splitFamilies{
        QueueFamilyCapabilities{ true, false },
        QueueFamilyCapabilities{ false, true }
    };
    const QueueFamilyIndices split = selectQueueFamilies(splitFamilies);
    require(split.graphicsFamily == 0U && split.presentFamily == 1U,
        "split graphics/present queues were not selected");

    constexpr std::array incompleteFamilies{
        QueueFamilyCapabilities{ true, false }
    };
    require(!selectQueueFamilies(incompleteFamilies).complete(),
        "an incomplete queue plan was accepted");

    DeviceCandidateCapabilities suitable = suitableCandidate();
    require(scoreDeviceCandidate(suitable, 140).has_value(),
        "a suitable device was rejected");

    DeviceCandidateCapabilities missingPresent = suitable;
    missingPresent.queueFamilies.presentFamily.reset();
    require(!scoreDeviceCandidate(missingPresent, 140),
        "a device without presentation support was accepted");

    DeviceCandidateCapabilities oldApi = suitable;
    oldApi.apiVersion = 139;
    require(!scoreDeviceCandidate(oldApi, 140), "an old API version was accepted");

    DeviceCandidateCapabilities missingFeature = suitable;
    missingFeature.features.dynamicRendering = false;
    require(!scoreDeviceCandidate(missingFeature, 140),
        "a device missing a required feature was accepted");

    DeviceCandidateCapabilities discrete = suitable;
    discrete.deviceClass = DeviceClass::discreteGpu;
    discrete.maxImageDimension2D = 4096;
    DeviceCandidateCapabilities largerIntegrated = suitable;
    largerIntegrated.maxImageDimension2D = 16384;
    const std::array candidates{ largerIntegrated, missingFeature, discrete };
    require(selectBestDeviceCandidate(candidates, 140) == 2,
        "device scoring did not prefer the discrete GPU");

    DeviceCandidateCapabilities largerDiscrete = discrete;
    largerDiscrete.maxImageDimension2D = 8192;
    const std::array discreteCandidates{ discrete, largerDiscrete };
    require(selectBestDeviceCandidate(discreteCandidates, 140) == 1,
        "same-class device scoring did not use the capability limit");

    const std::array unsuitableCandidates{ oldApi, missingFeature, missingPresent };
    require(!selectBestDeviceCandidate(unsuitableCandidates, 140),
        "an unsuitable candidate set produced a selection");
}
