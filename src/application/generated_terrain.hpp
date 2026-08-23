#pragma once

#include <danvulkan/assets.hpp>

#include <cstdint>
#include <optional>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace danvulkan::application {

class TerrainSurfaceField;

struct GeneratedTerrainConfig
{
    glm::vec2 center{0.0f};
    float size = 1.0f;
    float baseHeight = 0.0f;
    float heightScale = 0.1f;
    float noiseScale = 3.0f;
    std::uint32_t resolution = 65;
    std::uint32_t seed = 1;
};

struct TerrainSample
{
    float height = 0.0f;
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

// A deterministic CPU height field. The same samples drive both the rendered mesh and gameplay,
// preventing the character controller from drifting away from the visible surface.
class GeneratedTerrain
{
public:
    explicit GeneratedTerrain(GeneratedTerrainConfig config = {},
        const TerrainSurfaceField* surfaceField = nullptr);

    [[nodiscard]] const GeneratedTerrainConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<danvulkan::assets::Vertex>& vertices() const noexcept
    {
        return vertices_;
    }
    [[nodiscard]] const std::vector<std::uint32_t>& indices() const noexcept
    {
        return indices_;
    }
    [[nodiscard]] glm::vec2 minimum() const noexcept;
    [[nodiscard]] glm::vec2 maximum() const noexcept;
    [[nodiscard]] bool contains(const glm::vec2& position) const noexcept;
    [[nodiscard]] glm::vec2 clampToBounds(const glm::vec2& position) const noexcept;
    [[nodiscard]] std::optional<TerrainSample> sample(
        const glm::vec2& position) const noexcept;

private:
    GeneratedTerrainConfig config_;
    float cellSize_ = 1.0f;
    std::vector<float> heights_;
    std::vector<danvulkan::assets::Vertex> vertices_;
    std::vector<std::uint32_t> indices_;
};

}
