#pragma once

#include <danvulkan/assets.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace danvulkan
{
struct EnvironmentMip
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Offset into the RGBA32F value array, not a byte offset.
    std::size_t valueOffset = 0;
};

struct EnvironmentPrecompute
{
    std::uint32_t irradianceWidth = 0;
    std::uint32_t irradianceHeight = 0;
    std::vector<float> irradianceRgba32f;
    std::vector<EnvironmentMip> specularMips;
    std::vector<float> specularRgba32f;
    std::uint32_t brdfSize = 0;
    std::vector<float> brdfRgba32f;
};

// Validates an RGBA8 or linear RGBA32F equirectangular source, then generates diffuse
// irradiance, GGX-prefiltered specular levels, and a split-sum BRDF lookup table.
[[nodiscard]] EnvironmentPrecompute precomputeEnvironment(
    const assets::TextureAsset& source);
}
