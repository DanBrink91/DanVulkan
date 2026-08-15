#include "src/memory_planner.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
}

int main()
{
    using namespace danvulkan::vk;

    constexpr VkSampleCountFlags supported =
        VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT;
    const auto automatic = planMemoryPolicy(supported, 0, 2, true);
    require(automatic && automatic->samples == VK_SAMPLE_COUNT_4_BIT &&
        automatic->attachmentSetCount == 2 && automatic->preferLazilyAllocatedAttachments,
        "automatic policy must select the highest supported count and frame-owned targets");

    const auto disabled = planMemoryPolicy(supported, 1, 3, false);
    require(disabled && disabled->samples == VK_SAMPLE_COUNT_1_BIT &&
        disabled->attachmentSetCount == 3 && !disabled->preferLazilyAllocatedAttachments,
        "a one-sample limit must disable multisampling and preserve the lazy preference");

    const auto capped = planMemoryPolicy(
        VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT, 2, 2, true);
    require(capped && capped->samples == VK_SAMPLE_COUNT_1_BIT,
        "selection must fall back to an actually supported count below the requested cap");

    require(!planMemoryPolicy(supported, 3, 2, true),
        "non-power-of-two sample limits must be rejected");
    require(!planMemoryPolicy(0, 0, 2, true),
        "an empty supported sample mask must be rejected");
    require(!planMemoryPolicy(supported, 0, 0, true),
        "a zero frame count must be rejected");

    const auto initialArena = planArenaCapacity(0, 4);
    require(initialArena && *initialArena == 64U * 1024U,
        "the first arena allocation must use the minimum capacity");
    const auto retainedArena = planArenaCapacity(*initialArena, 1024);
    require(retainedArena && *retainedArena == *initialArena,
        "an arena with enough capacity must not grow");
    const auto grownArena = planArenaCapacity(*initialArena, 70U * 1024U);
    require(grownArena && *grownArena == 128U * 1024U,
        "arena capacity must grow geometrically");
    const auto largeArena = planArenaCapacity(
        std::uint64_t{1} << 63U, std::numeric_limits<std::uint64_t>::max());
    require(largeArena && *largeArena == std::numeric_limits<std::uint64_t>::max(),
        "arena growth must saturate safely near the integer limit");
    require(!planArenaCapacity(0, 0) && !planArenaCapacity(0, 1, 0),
        "zero-sized arena requirements and policies must be rejected");
    return EXIT_SUCCESS;
}
