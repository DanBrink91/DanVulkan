#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace danvulkan::application {

struct GrassSurfaceTraits
{
    glm::vec2 clumpPoint{0.0f};
    glm::vec2 bendDirection{1.0f, 0.0f};
    glm::vec3 color{0.16f, 0.46f, 0.07f};
    float heightScale = 1.0f;
    float bend = 0.2f;
    float stiffness = 0.8f;
    float windPhase = 0.0f;
    std::uint32_t clumpId = 0;
};

// The renderer and generators consume this semantic surface description rather than knowing
// whether it came from procedural rules, an authored paint map, or a blend of both.
struct TerrainSurfaceSample
{
    float grassCoverage = 1.0f;
    float dirtWeight = 0.0f;
    GrassSurfaceTraits grass;
};

class TerrainSurfaceField
{
public:
    virtual ~TerrainSurfaceField() = default;
    [[nodiscard]] virtual TerrainSurfaceSample sample(const glm::vec2& worldPosition) const = 0;
};

struct ProceduralTerrainSurfaceConfig
{
    glm::vec2 pathOrigin{0.0f};
    glm::vec2 pathDirection{1.0f, 0.18f};
    float pathHalfWidth = 0.035f;
    float pathFeather = 0.025f;
    float pathMeanderAmplitude = 0.055f;
    float pathMeanderFrequency = 3.2f;
    float clumpCellSize = 0.14f;
    std::uint32_t seed = 0x464f5245U;
};

// Infinite, deterministic world-space field. Voronoi sites and paths do not restart at chunk
// boundaries, so independently streamed chunks agree at their seams.
class ProceduralTerrainSurfaceField final : public TerrainSurfaceField
{
public:
    explicit ProceduralTerrainSurfaceField(ProceduralTerrainSurfaceConfig config = {});

    [[nodiscard]] const ProceduralTerrainSurfaceConfig& config() const noexcept
    {
        return config_;
    }
    [[nodiscard]] TerrainSurfaceSample sample(
        const glm::vec2& worldPosition) const override;

private:
    ProceduralTerrainSurfaceConfig config_;
    glm::vec2 pathForward_{1.0f, 0.0f};
    glm::vec2 pathRight_{0.0f, 1.0f};
};

// Authored paint providers can blend their sample over the procedural base with this operation.
// Keeping the composition semantic avoids coupling future editor storage to runtime rendering.
[[nodiscard]] TerrainSurfaceSample blendTerrainSurfaceSamples(
    const TerrainSurfaceSample& base, const TerrainSurfaceSample& overlay,
    float opacity) noexcept;

}
