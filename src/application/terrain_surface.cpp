#include "terrain_surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace danvulkan::application {
namespace {
constexpr float pi = 3.14159265358979323846f;

std::uint32_t hash(std::int32_t x, std::int32_t z, std::uint32_t seed) noexcept
{
    std::uint32_t value = static_cast<std::uint32_t>(x) * 0x8da6b343U ^
        static_cast<std::uint32_t>(z) * 0xd8163841U ^ seed * 0xcb1ab31fU;
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

std::uint32_t rehash(std::uint32_t value) noexcept
{
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

float unit(std::uint32_t value) noexcept
{
    return static_cast<float>(value & 0x00ffffffU) / 16777215.0f;
}

float smoothstep(float edge0, float edge1, float value) noexcept
{
    if (edge1 <= edge0)
    {
        return value < edge0 ? 0.0f : 1.0f;
    }
    const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

glm::vec2 directionalFlow(const glm::vec2& point, std::uint32_t seed) noexcept
{
    // A slowly varying analytic field gives neighboring Voronoi sites a shared large-scale
    // sweep. Sampling at the site (rather than per blade) preserves exact clump coherence.
    const float seedPhase = unit(rehash(seed ^ 0x6a09e667U)) * 2.0f * pi;
    const float primary = std::sin(point.x * 1.17f + point.y * 0.73f + seedPhase);
    const float secondary = std::sin(
        point.x * -0.41f + point.y * 0.89f - seedPhase * 0.61f);
    const float angle = seedPhase * 0.37f + primary * 0.72f + secondary * 0.34f;
    return {std::cos(angle), std::sin(angle)};
}

GrassSurfaceTraits traitsForSite(std::int32_t cellX, std::int32_t cellZ,
    const glm::vec2& point, std::uint32_t seed) noexcept
{
    const std::uint32_t id = hash(cellX, cellZ, seed ^ 0xa511e9b3U);
    const float height = unit(rehash(id ^ 0x68bc21ebU));
    const float direction = unit(rehash(id ^ 0x02e5be93U)) * 2.0f * pi;
    const float color = unit(rehash(id ^ 0x967a889bU));
    const float warmth = unit(rehash(id ^ 0x4f1bbcdcU));

    GrassSurfaceTraits result;
    result.clumpPoint = point;
    const glm::vec2 clumpDirection(std::cos(direction), std::sin(direction));
    result.bendDirection = glm::normalize(glm::mix(
        directionalFlow(point, seed), clumpDirection, 0.27f));
    const glm::vec3 deepGreen(0.075f, 0.31f, 0.025f);
    const glm::vec3 freshGreen(0.24f, 0.62f, 0.085f);
    const glm::vec3 warmGreen(0.37f, 0.55f, 0.075f);
    result.color = glm::mix(glm::mix(deepGreen, freshGreen, color), warmGreen,
        warmth * 0.28f);
    result.heightScale = std::lerp(0.72f, 1.34f, height);
    result.bend = std::lerp(0.10f, 0.38f, unit(rehash(id ^ 0xb5297a4dU)));
    result.stiffness = std::lerp(0.62f, 1.0f, unit(rehash(id ^ 0x1b56c4e9U)));
    result.windPhase = unit(rehash(id ^ 0x7f4a7c15U)) * 2.0f * pi;
    result.clumpId = id == 0U ? 1U : id;
    return result;
}
}

ProceduralTerrainSurfaceField::ProceduralTerrainSurfaceField(
    ProceduralTerrainSurfaceConfig config)
    : config_(config)
{
    const bool finite = std::isfinite(config_.pathOrigin.x) &&
        std::isfinite(config_.pathOrigin.y) &&
        std::isfinite(config_.pathDirection.x) &&
        std::isfinite(config_.pathDirection.y) &&
        std::isfinite(config_.pathHalfWidth) && std::isfinite(config_.pathFeather) &&
        std::isfinite(config_.pathMeanderAmplitude) &&
        std::isfinite(config_.pathMeanderFrequency) &&
        std::isfinite(config_.clumpCellSize);
    if (!finite || glm::dot(config_.pathDirection, config_.pathDirection) < 0.000001f ||
        config_.pathHalfWidth < 0.0f || config_.pathFeather < 0.0f ||
        config_.pathMeanderAmplitude < 0.0f || config_.pathMeanderFrequency <= 0.0f ||
        config_.clumpCellSize <= 0.0f)
    {
        throw std::invalid_argument("procedural terrain surface configuration is invalid");
    }
    pathForward_ = glm::normalize(config_.pathDirection);
    pathRight_ = {-pathForward_.y, pathForward_.x};
}

TerrainSurfaceSample ProceduralTerrainSurfaceField::sample(
    const glm::vec2& worldPosition) const
{
    const float cellSize = config_.clumpCellSize;
    const std::int32_t centerX = static_cast<std::int32_t>(
        std::floor(worldPosition.x / cellSize));
    const std::int32_t centerZ = static_cast<std::int32_t>(
        std::floor(worldPosition.y / cellSize));
    float nearestDistanceSquared = std::numeric_limits<float>::max();
    GrassSurfaceTraits nearest;
    for (std::int32_t z = centerZ - 1; z <= centerZ + 1; ++z)
    {
        for (std::int32_t x = centerX - 1; x <= centerX + 1; ++x)
        {
            const std::uint32_t siteHash = hash(x, z, config_.seed);
            const glm::vec2 jitter(
                std::lerp(0.12f, 0.88f, unit(rehash(siteHash ^ 0x243f6a88U))),
                std::lerp(0.12f, 0.88f, unit(rehash(siteHash ^ 0x85a308d3U))));
            const glm::vec2 point = (glm::vec2(x, z) + jitter) * cellSize;
            const glm::vec2 offset = worldPosition - point;
            const float distanceSquared = glm::dot(offset, offset);
            if (distanceSquared < nearestDistanceSquared)
            {
                nearestDistanceSquared = distanceSquared;
                nearest = traitsForSite(x, z, point, config_.seed);
            }
        }
    }

    const glm::vec2 pathOffset = worldPosition - config_.pathOrigin;
    const float along = glm::dot(pathOffset, pathForward_);
    const float across = glm::dot(pathOffset, pathRight_);
    const float seedPhase = unit(rehash(config_.seed ^ 0x3c6ef372U)) * 2.0f * pi;
    const float centerLine = config_.pathMeanderAmplitude *
        (std::sin(along * config_.pathMeanderFrequency + seedPhase) * 0.72f +
            std::sin(along * config_.pathMeanderFrequency * 0.43f - seedPhase * 0.7f) *
                0.28f);
    const float pathDistance = std::abs(across - centerLine);
    const float dirtWeight = 1.0f - smoothstep(config_.pathHalfWidth,
        config_.pathHalfWidth + config_.pathFeather, pathDistance);

    TerrainSurfaceSample result;
    result.dirtWeight = dirtWeight;
    result.grassCoverage = 1.0f - dirtWeight;
    result.grass = nearest;
    return result;
}

TerrainSurfaceSample blendTerrainSurfaceSamples(const TerrainSurfaceSample& base,
    const TerrainSurfaceSample& overlay, float opacity) noexcept
{
    const float weight = std::clamp(opacity, 0.0f, 1.0f);
    TerrainSurfaceSample result;
    result.grassCoverage = std::lerp(base.grassCoverage, overlay.grassCoverage, weight);
    result.dirtWeight = std::lerp(base.dirtWeight, overlay.dirtWeight, weight);
    result.grass.clumpPoint = glm::mix(base.grass.clumpPoint,
        overlay.grass.clumpPoint, weight);
    const glm::vec2 direction = glm::mix(base.grass.bendDirection,
        overlay.grass.bendDirection, weight);
    result.grass.bendDirection = glm::dot(direction, direction) > 0.000001f
        ? glm::normalize(direction) : base.grass.bendDirection;
    result.grass.color = glm::mix(base.grass.color, overlay.grass.color, weight);
    result.grass.heightScale = std::lerp(
        base.grass.heightScale, overlay.grass.heightScale, weight);
    result.grass.bend = std::lerp(base.grass.bend, overlay.grass.bend, weight);
    result.grass.stiffness = std::lerp(
        base.grass.stiffness, overlay.grass.stiffness, weight);
    result.grass.windPhase = std::lerp(
        base.grass.windPhase, overlay.grass.windPhase, weight);
    result.grass.clumpId = weight >= 0.5f ? overlay.grass.clumpId : base.grass.clumpId;
    return result;
}

}
