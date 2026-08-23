#include "generated_terrain.hpp"

#include "terrain_surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace danvulkan::application {
namespace {
constexpr std::uint32_t minimumResolution = 2;
constexpr std::uint32_t maximumResolution = 2048;

std::uint32_t hash(std::uint32_t x, std::uint32_t z, std::uint32_t seed) noexcept
{
    std::uint32_t value = x * 0x8da6b343U ^ z * 0xd8163841U ^ seed * 0xcb1ab31fU;
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

float randomSigned(std::int32_t x, std::int32_t z, std::uint32_t seed) noexcept
{
    const std::uint32_t bits = hash(static_cast<std::uint32_t>(x),
        static_cast<std::uint32_t>(z), seed);
    return static_cast<float>(bits & 0x00ffffffU) / 8388607.5f - 1.0f;
}

float smooth(float value) noexcept
{
    return value * value * (3.0f - 2.0f * value);
}

float valueNoise(float x, float z, std::uint32_t seed) noexcept
{
    const auto x0 = static_cast<std::int32_t>(std::floor(x));
    const auto z0 = static_cast<std::int32_t>(std::floor(z));
    const float tx = smooth(x - static_cast<float>(x0));
    const float tz = smooth(z - static_cast<float>(z0));
    const float top = std::lerp(randomSigned(x0, z0, seed),
        randomSigned(x0 + 1, z0, seed), tx);
    const float bottom = std::lerp(randomSigned(x0, z0 + 1, seed),
        randomSigned(x0 + 1, z0 + 1, seed), tx);
    return std::lerp(top, bottom, tz);
}

float fractalNoise(float x, float z, std::uint32_t seed) noexcept
{
    float result = 0.0f;
    float amplitude = 0.516129f;
    float frequency = 1.0f;
    for (std::uint32_t octave = 0; octave < 5; ++octave)
    {
        result += valueNoise(x * frequency, z * frequency,
            seed + octave * 0x9e3779b9U) * amplitude;
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }
    return result;
}

bool finite(const GeneratedTerrainConfig& config) noexcept
{
    return std::isfinite(config.center.x) && std::isfinite(config.center.y) &&
        std::isfinite(config.size) && std::isfinite(config.baseHeight) &&
        std::isfinite(config.heightScale) && std::isfinite(config.noiseScale);
}

float generatedHeight(const GeneratedTerrainConfig& config, float x, float z) noexcept
{
    return config.baseHeight +
        fractalNoise(x * config.noiseScale, z * config.noiseScale, config.seed) *
            config.heightScale;
}
}

GeneratedTerrain::GeneratedTerrain(GeneratedTerrainConfig config,
    const TerrainSurfaceField* surfaceField)
    : config_(config)
{
    if (!finite(config_) || config_.size <= 0.0f || config_.heightScale < 0.0f ||
        config_.noiseScale <= 0.0f || config_.resolution < minimumResolution ||
        config_.resolution > maximumResolution)
    {
        throw std::invalid_argument("generated terrain configuration is invalid");
    }

    const std::size_t resolution = config_.resolution;
    if (resolution > std::numeric_limits<std::size_t>::max() / resolution)
    {
        throw std::length_error("generated terrain resolution is too large");
    }
    const std::size_t vertexCount = resolution * resolution;
    cellSize_ = config_.size / static_cast<float>(config_.resolution - 1U);
    heights_.resize(vertexCount);
    vertices_.resize(vertexCount);

    const glm::vec2 terrainMinimum = minimum();
    for (std::uint32_t row = 0; row < config_.resolution; ++row)
    {
        for (std::uint32_t column = 0; column < config_.resolution; ++column)
        {
            const std::size_t index = static_cast<std::size_t>(row) * resolution + column;
            const glm::vec2 normalized(
                static_cast<float>(column) / static_cast<float>(config_.resolution - 1U),
                static_cast<float>(row) / static_cast<float>(config_.resolution - 1U));
            danvulkan::assets::Vertex& vertex = vertices_[index];
            const float worldX = terrainMinimum.x + static_cast<float>(column) * cellSize_;
            const float worldZ = terrainMinimum.y + static_cast<float>(row) * cellSize_;
            heights_[index] = generatedHeight(config_, worldX, worldZ);
            vertex.pos = {worldX, heights_[index], worldZ};
            vertex.texCoord = normalized * config_.noiseScale;
            // Magnitude three identifies generated terrain to the shared fragment shader while
            // the sign continues to carry ordinary tangent handedness.
            vertex.tangentSign = -3.0f;
            vertex.unused0 = 1.0f;
        }
    }

    for (std::uint32_t row = 0; row < config_.resolution; ++row)
    {
        for (std::uint32_t column = 0; column < config_.resolution; ++column)
        {
            danvulkan::assets::Vertex& vertex =
                vertices_[static_cast<std::size_t>(row) * resolution + column];
            // Sample beyond the patch boundary so adjacent chunks calculate identical seam
            // normals instead of each using a different one-sided derivative.
            const float left = generatedHeight(
                config_, vertex.pos.x - cellSize_, vertex.pos.z);
            const float right = generatedHeight(
                config_, vertex.pos.x + cellSize_, vertex.pos.z);
            const float back = generatedHeight(
                config_, vertex.pos.x, vertex.pos.z - cellSize_);
            const float front = generatedHeight(
                config_, vertex.pos.x, vertex.pos.z + cellSize_);
            const float slopeX = (right - left) / (cellSize_ * 2.0f);
            const float slopeZ = (front - back) / (cellSize_ * 2.0f);
            vertex.normal = glm::normalize(glm::vec3(-slopeX, 1.0f, -slopeZ));
            vertex.tangent = glm::normalize(glm::vec3(1.0f, slopeX, 0.0f));

            const float relativeHeight = config_.heightScale > 0.0f
                ? glm::clamp((vertex.pos.y - config_.baseHeight) /
                        config_.heightScale * 0.5f + 0.5f,
                    0.0f, 1.0f)
                : 0.5f;
            const glm::vec3 lowGrass(0.25f, 0.48f, 0.12f);
            const glm::vec3 dryGrass(0.58f, 0.52f, 0.22f);
            const glm::vec3 exposedEarth(0.42f, 0.36f, 0.27f);
            const glm::vec3 groundColor = glm::mix(lowGrass, dryGrass,
                smooth(relativeHeight));
            const float slopeColor = glm::smoothstep(0.035f, 0.22f,
                1.0f - vertex.normal.y);
            vertex.color = glm::mix(groundColor, exposedEarth, slopeColor);
            if (surfaceField != nullptr)
            {
                const TerrainSurfaceSample surface = surfaceField->sample(
                    {vertex.pos.x, vertex.pos.z});
                vertex.unused0 = glm::clamp(surface.grassCoverage, 0.0f, 1.0f);
                const glm::vec3 pathColor(0.56f, 0.43f, 0.19f);
                // Retain a little terrain-height variation inside the path so the first
                // vertex-color version does not read as a perfectly flat painted stripe.
                const glm::vec3 variedPath = glm::mix(pathColor,
                    glm::vec3(0.69f, 0.56f, 0.27f), relativeHeight * 0.22f);
                vertex.color = glm::mix(vertex.color, variedPath,
                    glm::clamp(surface.dirtWeight, 0.0f, 1.0f));
            }
        }
    }

    const std::size_t quadCount = static_cast<std::size_t>(config_.resolution - 1U) *
        static_cast<std::size_t>(config_.resolution - 1U);
    indices_.reserve(quadCount * 6U);
    for (std::uint32_t row = 0; row + 1U < config_.resolution; ++row)
    {
        for (std::uint32_t column = 0; column + 1U < config_.resolution; ++column)
        {
            const std::uint32_t v00 = row * config_.resolution + column;
            const std::uint32_t v01 = v00 + 1U;
            const std::uint32_t v10 = v00 + config_.resolution;
            const std::uint32_t v11 = v10 + 1U;
            indices_.insert(indices_.end(), {v00, v10, v01, v01, v10, v11});
        }
    }
}

glm::vec2 GeneratedTerrain::minimum() const noexcept
{
    return config_.center - glm::vec2(config_.size * 0.5f);
}

glm::vec2 GeneratedTerrain::maximum() const noexcept
{
    return config_.center + glm::vec2(config_.size * 0.5f);
}

bool GeneratedTerrain::contains(const glm::vec2& position) const noexcept
{
    const glm::vec2 terrainMinimum = minimum();
    const glm::vec2 terrainMaximum = maximum();
    return position.x >= terrainMinimum.x && position.y >= terrainMinimum.y &&
        position.x <= terrainMaximum.x && position.y <= terrainMaximum.y;
}

glm::vec2 GeneratedTerrain::clampToBounds(const glm::vec2& position) const noexcept
{
    return glm::clamp(position, minimum(), maximum());
}

std::optional<TerrainSample> GeneratedTerrain::sample(
    const glm::vec2& position) const noexcept
{
    if (!contains(position))
    {
        return std::nullopt;
    }

    const glm::vec2 grid = (position - minimum()) / cellSize_;
    const std::uint32_t column = std::min(static_cast<std::uint32_t>(grid.x),
        config_.resolution - 2U);
    const std::uint32_t row = std::min(static_cast<std::uint32_t>(grid.y),
        config_.resolution - 2U);
    const float u = glm::clamp(grid.x - static_cast<float>(column), 0.0f, 1.0f);
    const float v = glm::clamp(grid.y - static_cast<float>(row), 0.0f, 1.0f);
    const std::size_t resolution = config_.resolution;
    const danvulkan::assets::Vertex& v00 = vertices_[row * resolution + column];
    const danvulkan::assets::Vertex& v01 = vertices_[row * resolution + column + 1U];
    const danvulkan::assets::Vertex& v10 = vertices_[(row + 1U) * resolution + column];
    const danvulkan::assets::Vertex& v11 =
        vertices_[(row + 1U) * resolution + column + 1U];

    TerrainSample result;
    if (u + v <= 1.0f)
    {
        result.height = v00.pos.y * (1.0f - u - v) + v01.pos.y * u + v10.pos.y * v;
        result.normal = glm::normalize(
            v00.normal * (1.0f - u - v) + v01.normal * u + v10.normal * v);
    }
    else
    {
        const float v11Weight = u + v - 1.0f;
        const float v01Weight = 1.0f - v;
        const float v10Weight = 1.0f - u;
        result.height = v11.pos.y * v11Weight + v01.pos.y * v01Weight +
            v10.pos.y * v10Weight;
        result.normal = glm::normalize(v11.normal * v11Weight + v01.normal * v01Weight +
            v10.normal * v10Weight);
    }
    return result;
}

}
