#include "environment_precompute.hpp"

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace danvulkan
{
namespace
{
constexpr float Pi = 3.14159265358979323846f;
constexpr std::uint32_t convolutionSampleCount = 32;
constexpr std::uint32_t brdfSize = 64;

float srgbToLinear(float value) noexcept
{
    return value <= 0.04045f ? value / 12.92f :
        std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float radicalInverse(std::uint32_t bits) noexcept
{
    bits = (bits << 16U) | (bits >> 16U);
    bits = ((bits & 0x55555555U) << 1U) | ((bits & 0xAAAAAAAAU) >> 1U);
    bits = ((bits & 0x33333333U) << 2U) | ((bits & 0xCCCCCCCCU) >> 2U);
    bits = ((bits & 0x0F0F0F0FU) << 4U) | ((bits & 0xF0F0F0F0U) >> 4U);
    bits = ((bits & 0x00FF00FFU) << 8U) | ((bits & 0xFF00FF00U) >> 8U);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

glm::vec2 hammersley(std::uint32_t index, std::uint32_t count) noexcept
{
    return {static_cast<float>(index) / static_cast<float>(count), radicalInverse(index)};
}

glm::vec3 directionFromUv(float u, float v) noexcept
{
    const float phi = (u - 0.5f) * 2.0f * Pi;
    const float theta = v * Pi;
    const float sineTheta = std::sin(theta);
    return {std::cos(phi) * sineTheta, std::cos(theta), std::sin(phi) * sineTheta};
}

void tangentBasis(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent) noexcept
{
    const glm::vec3 up = std::abs(normal.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) :
        glm::vec3(1.0f, 0.0f, 0.0f);
    tangent = glm::normalize(glm::cross(up, normal));
    bitangent = glm::cross(normal, tangent);
}

glm::vec3 sampleEquirect(const std::vector<float>& pixels, std::uint32_t width,
    std::uint32_t height, const glm::vec3& direction) noexcept
{
    const glm::vec3 normalized = glm::normalize(direction);
    const float u = std::atan2(normalized.z, normalized.x) / (2.0f * Pi) + 0.5f;
    const float v = std::acos(std::clamp(normalized.y, -1.0f, 1.0f)) / Pi;
    const float x = u * static_cast<float>(width) - 0.5f;
    const float y = v * static_cast<float>(height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    const auto texel = [&](int sourceX, int sourceY) {
        const int wrappedX = ((sourceX % static_cast<int>(width)) +
            static_cast<int>(width)) % static_cast<int>(width);
        const int clampedY = std::clamp(sourceY, 0, static_cast<int>(height) - 1);
        const std::size_t offset = (static_cast<std::size_t>(clampedY) * width +
            static_cast<std::uint32_t>(wrappedX)) * 4U;
        return glm::vec3(pixels[offset], pixels[offset + 1U], pixels[offset + 2U]);
    };
    return glm::mix(glm::mix(texel(x0, y0), texel(x0 + 1, y0), tx),
        glm::mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), tx), ty);
}

glm::vec3 hemisphereSample(glm::vec2 sequence, const glm::vec3& normal) noexcept
{
    const float phi = 2.0f * Pi * sequence.x;
    const float sineTheta = std::sqrt(sequence.y);
    const float cosineTheta = std::sqrt(1.0f - sequence.y);
    glm::vec3 tangent;
    glm::vec3 bitangent;
    tangentBasis(normal, tangent, bitangent);
    return glm::normalize(tangent * (std::cos(phi) * sineTheta) +
        bitangent * (std::sin(phi) * sineTheta) + normal * cosineTheta);
}

glm::vec3 importanceSampleGgx(glm::vec2 sequence, const glm::vec3& normal,
    float roughness) noexcept
{
    const float alpha = roughness * roughness;
    const float phi = 2.0f * Pi * sequence.x;
    const float cosineTheta = std::sqrt((1.0f - sequence.y) /
        (1.0f + (alpha * alpha - 1.0f) * sequence.y));
    const float sineTheta = std::sqrt(std::max(1.0f - cosineTheta * cosineTheta, 0.0f));
    glm::vec3 tangent;
    glm::vec3 bitangent;
    tangentBasis(normal, tangent, bitangent);
    return glm::normalize(tangent * (std::cos(phi) * sineTheta) +
        bitangent * (std::sin(phi) * sineTheta) + normal * cosineTheta);
}

float geometrySchlickGgx(float nDotDirection, float roughness) noexcept
{
    const float k = roughness * roughness * 0.5f;
    return nDotDirection / (nDotDirection * (1.0f - k) + k);
}

glm::vec2 integrateBrdf(float nDotV, float roughness) noexcept
{
    const glm::vec3 view(std::sqrt(std::max(1.0f - nDotV * nDotV, 0.0f)), 0.0f, nDotV);
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    float scale = 0.0f;
    float bias = 0.0f;
    for (std::uint32_t index = 0; index < convolutionSampleCount; ++index)
    {
        const glm::vec3 halfway = importanceSampleGgx(
            hammersley(index, convolutionSampleCount), normal, roughness);
        const glm::vec3 light = glm::normalize(2.0f * glm::dot(view, halfway) * halfway - view);
        const float nDotL = std::max(light.z, 0.0f);
        const float nDotH = std::max(halfway.z, 0.0f);
        const float vDotH = std::max(glm::dot(view, halfway), 0.0f);
        if (nDotL > 0.0f)
        {
            const float geometry = geometrySchlickGgx(nDotV, roughness) *
                geometrySchlickGgx(nDotL, roughness);
            const float visibility = geometry * vDotH /
                std::max(nDotH * nDotV, 0.0001f);
            const float fresnel = std::pow(1.0f - vDotH, 5.0f);
            scale += (1.0f - fresnel) * visibility;
            bias += fresnel * visibility;
        }
    }
    return {scale / static_cast<float>(convolutionSampleCount),
        bias / static_cast<float>(convolutionSampleCount)};
}

std::vector<float> linearPixels(const assets::TextureAsset& source)
{
    if (source.width == 0 || source.height == 0 ||
        static_cast<std::size_t>(source.height) >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(source.width) / 4U)
    {
        throw std::invalid_argument("environment dimensions are invalid or overflowed");
    }
    const std::size_t valueCount = static_cast<std::size_t>(source.width) * source.height * 4U;
    const bool hasBytes = source.rgba8.size() == valueCount;
    const bool hasFloats = source.rgba32f.size() == valueCount;
    if (hasBytes == hasFloats)
    {
        throw std::invalid_argument(
            "environment requires exactly one complete RGBA8 or RGBA32F payload");
    }
    std::vector<float> result(valueCount);
    if (hasFloats)
    {
        for (std::size_t index = 0; index < valueCount; ++index)
        {
            if (!std::isfinite(source.rgba32f[index]) || source.rgba32f[index] < 0.0f)
            {
                throw std::invalid_argument(
                    "environment RGBA32F values must be finite and nonnegative");
            }
            result[index] = source.rgba32f[index];
        }
        return result;
    }
    for (std::size_t index = 0; index < valueCount; ++index)
    {
        const float value = static_cast<float>(std::to_integer<std::uint8_t>(source.rgba8[index])) /
            255.0f;
        result[index] = source.colorSpace == assets::ColorSpace::srgb && index % 4U != 3U
            ? srgbToLinear(value) : value;
    }
    return result;
}

void appendRgba(std::vector<float>& output, glm::vec3 color)
{
    output.insert(output.end(), {color.r, color.g, color.b, 1.0f});
}
} // namespace

EnvironmentPrecompute precomputeEnvironment(const assets::TextureAsset& source)
{
    const std::vector<float> pixels = linearPixels(source);
    EnvironmentPrecompute result;
    result.irradianceWidth = std::min(32U, std::max(8U, source.width));
    result.irradianceHeight = result.irradianceWidth / 2U;
    result.irradianceRgba32f.reserve(static_cast<std::size_t>(result.irradianceWidth) *
        result.irradianceHeight * 4U);
    for (std::uint32_t y = 0; y < result.irradianceHeight; ++y)
    {
        for (std::uint32_t x = 0; x < result.irradianceWidth; ++x)
        {
            const glm::vec3 normal = directionFromUv(
                (static_cast<float>(x) + 0.5f) / static_cast<float>(result.irradianceWidth),
                (static_cast<float>(y) + 0.5f) / static_cast<float>(result.irradianceHeight));
            glm::vec3 irradiance(0.0f);
            for (std::uint32_t sample = 0; sample < convolutionSampleCount; ++sample)
            {
                irradiance += sampleEquirect(pixels, source.width, source.height,
                    hemisphereSample(hammersley(sample, convolutionSampleCount), normal));
            }
            appendRgba(result.irradianceRgba32f,
                irradiance * (Pi / static_cast<float>(convolutionSampleCount)));
        }
    }

    const std::uint32_t specularWidth = std::min(128U, std::max(16U, source.width));
    const std::uint32_t mipCount = std::bit_width(specularWidth);
    for (std::uint32_t mip = 0; mip < mipCount; ++mip)
    {
        const std::uint32_t width = std::max(specularWidth >> mip, 1U);
        const std::uint32_t height = std::max((specularWidth / 2U) >> mip, 1U);
        result.specularMips.push_back({width, height, result.specularRgba32f.size()});
        const float roughness = mipCount > 1 ? static_cast<float>(mip) /
            static_cast<float>(mipCount - 1U) : 0.0f;
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                const glm::vec3 normal = directionFromUv(
                    (static_cast<float>(x) + 0.5f) / static_cast<float>(width),
                    (static_cast<float>(y) + 0.5f) / static_cast<float>(height));
                glm::vec3 color(0.0f);
                float weight = 0.0f;
                for (std::uint32_t sample = 0; sample < convolutionSampleCount; ++sample)
                {
                    const glm::vec3 halfway = importanceSampleGgx(
                        hammersley(sample, convolutionSampleCount), normal, roughness);
                    const glm::vec3 light = glm::normalize(
                        2.0f * glm::dot(normal, halfway) * halfway - normal);
                    const float nDotL = std::max(glm::dot(normal, light), 0.0f);
                    if (nDotL > 0.0f)
                    {
                        color += sampleEquirect(pixels, source.width, source.height, light) * nDotL;
                        weight += nDotL;
                    }
                }
                appendRgba(result.specularRgba32f, color / std::max(weight, 0.0001f));
            }
        }
    }

    result.brdfSize = brdfSize;
    result.brdfRgba32f.reserve(static_cast<std::size_t>(brdfSize) * brdfSize * 4U);
    for (std::uint32_t y = 0; y < brdfSize; ++y)
    {
        const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(brdfSize);
        for (std::uint32_t x = 0; x < brdfSize; ++x)
        {
            const float nDotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(brdfSize);
            const glm::vec2 integrated = integrateBrdf(nDotV, roughness);
            result.brdfRgba32f.insert(result.brdfRgba32f.end(),
                {integrated.x, integrated.y, 0.0f, 1.0f});
        }
    }
    return result;
}
}
