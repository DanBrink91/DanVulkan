#include "src/environment_precompute.hpp"

#include <cmath>
#include <cstdlib>
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

bool near(float left, float right, float tolerance = 0.025f)
{
    return std::abs(left - right) <= tolerance;
}
}

int main()
{
    danvulkan::assets::TextureAsset source;
    source.name = "constant HDR fixture";
    source.width = 2;
    source.height = 1;
    source.colorSpace = danvulkan::assets::ColorSpace::linear;
    source.rgba32f = {4.0f, 2.0f, 1.0f, 1.0f, 4.0f, 2.0f, 1.0f, 1.0f};

    const danvulkan::EnvironmentPrecompute result =
        danvulkan::precomputeEnvironment(source);
    require(result.irradianceWidth == 8 && result.irradianceHeight == 4 &&
        result.irradianceRgba32f.size() == 8U * 4U * 4U,
        "irradiance dimensions or payload are incorrect");
    require(near(result.irradianceRgba32f[0], 4.0f * 3.14159265f) &&
        near(result.irradianceRgba32f[1], 2.0f * 3.14159265f) &&
        near(result.irradianceRgba32f[2], 3.14159265f),
        "constant HDR radiance was not preserved by diffuse convolution");
    require(result.specularMips.size() == 5 && result.specularMips.front().width == 16 &&
        result.specularMips.back().width == 1 && result.specularMips.back().height == 1,
        "specular mip chain is incomplete");
    for (std::size_t index = 0; index < result.specularRgba32f.size(); index += 4U)
    {
        require(near(result.specularRgba32f[index], 4.0f) &&
            near(result.specularRgba32f[index + 1U], 2.0f) &&
            near(result.specularRgba32f[index + 2U], 1.0f),
            "constant radiance changed during GGX prefiltering");
    }
    require(result.brdfSize == 64 && result.brdfRgba32f.size() == 64U * 64U * 4U,
        "BRDF lookup dimensions or payload are incorrect");
    for (float value : result.brdfRgba32f)
    {
        require(std::isfinite(value) && value >= 0.0f,
            "BRDF lookup contains an invalid value");
    }

    source.rgba8.resize(source.rgba32f.size());
    bool rejectedAmbiguousPayload = false;
    try
    {
        static_cast<void>(danvulkan::precomputeEnvironment(source));
    }
    catch (const std::invalid_argument&)
    {
        rejectedAmbiguousPayload = true;
    }
    require(rejectedAmbiguousPayload, "ambiguous environment payload was accepted");
    source.rgba8.clear();
    source.rgba32f[0] = std::numeric_limits<float>::infinity();
    bool rejectedNonFinitePayload = false;
    try
    {
        static_cast<void>(danvulkan::precomputeEnvironment(source));
    }
    catch (const std::invalid_argument&)
    {
        rejectedNonFinitePayload = true;
    }
    require(rejectedNonFinitePayload, "non-finite HDR environment was accepted");
    return EXIT_SUCCESS;
}
