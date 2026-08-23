#pragma once

#include <cstdint>

#include <glm/vec2.hpp>

namespace danvulkan::application {

struct TerrainChunkCoordinate
{
    std::int32_t x = 0;
    std::int32_t z = 0;

    friend constexpr bool operator==(
        TerrainChunkCoordinate, TerrainChunkCoordinate) noexcept = default;
};

struct TerrainStreamingConfig
{
    glm::vec2 origin{0.0f};
    float chunkSize = 1.0f;
    float prefetchOffset = 0.32f;
};

[[nodiscard]] TerrainChunkCoordinate terrainChunkAt(
    const glm::vec2& position, const TerrainStreamingConfig& config) noexcept;
[[nodiscard]] TerrainChunkCoordinate requestedTerrainStreamCenter(
    const glm::vec2& position, const glm::vec2& velocity,
    const TerrainStreamingConfig& config) noexcept;
[[nodiscard]] bool terrainChunkInWindow(TerrainChunkCoordinate coordinate,
    TerrainChunkCoordinate center, std::int32_t radius) noexcept;

}
