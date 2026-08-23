#include "terrain_streaming.hpp"

#include <cmath>

namespace danvulkan::application {
namespace {
std::int32_t chunkCoordinate(float position, float origin, float chunkSize) noexcept
{
    return static_cast<std::int32_t>(
        std::floor((position - origin) / chunkSize + 0.5f));
}
}

TerrainChunkCoordinate terrainChunkAt(
    const glm::vec2& position, const TerrainStreamingConfig& config) noexcept
{
    return {chunkCoordinate(position.x, config.origin.x, config.chunkSize),
        chunkCoordinate(position.y, config.origin.y, config.chunkSize)};
}

TerrainChunkCoordinate requestedTerrainStreamCenter(const glm::vec2& position,
    const glm::vec2& velocity, const TerrainStreamingConfig& config) noexcept
{
    TerrainChunkCoordinate center = terrainChunkAt(position, config);
    const glm::vec2 chunkCenter = config.origin +
        glm::vec2(static_cast<float>(center.x), static_cast<float>(center.z)) *
            config.chunkSize;
    const glm::vec2 localPosition = position - chunkCenter;
    const auto prefetchDirection = [&](float offset, float speed) noexcept
    {
        if (std::abs(offset) < config.prefetchOffset || offset * speed < 0.0f)
        {
            return 0;
        }
        return offset > 0.0f ? 1 : -1;
    };
    center.x += prefetchDirection(localPosition.x, velocity.x);
    center.z += prefetchDirection(localPosition.y, velocity.y);
    return center;
}

bool terrainChunkInWindow(TerrainChunkCoordinate coordinate,
    TerrainChunkCoordinate center, std::int32_t radius) noexcept
{
    return std::abs(coordinate.x - center.x) <= radius &&
        std::abs(coordinate.z - center.z) <= radius;
}

}
