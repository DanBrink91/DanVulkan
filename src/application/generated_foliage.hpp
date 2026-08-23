#pragma once

#include <danvulkan/assets.hpp>
#include <danvulkan/grass.hpp>

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>

namespace danvulkan::application {

class GeneratedTerrain;
class TerrainSurfaceField;

struct GeneratedFoliageConfig
{
    glm::vec2 clearingCenter{0.0f};
    float grassClearingRadius = 0.055f;
    float treeClearingRadius = 0.2f;
    float grassMinimumHeight = 0.006f;
    float grassMaximumHeight = 0.014f;
    float treeMinimumHeight = 0.055f;
    float treeMaximumHeight = 0.095f;
    float treeMinimumSpacing = 0.045f;
    float minimumGroundNormalY = 0.92f;
    std::uint32_t grassBladeCount = 40000U;
    std::uint32_t treeCount = 28U;
    std::uint32_t grassSeed = 0x47524153U;
    std::uint32_t seed = 0x464f5245U;
};

struct FoliageMesh
{
    std::vector<danvulkan::assets::Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

// Generates compact terrain-grounded Bezier blade records plus batched tree geometry.
class GeneratedFoliage
{
public:
    GeneratedFoliage(const GeneratedTerrain& terrain, GeneratedFoliageConfig config = {},
        const TerrainSurfaceField* surfaceField = nullptr,
        const std::function<bool()>& cancelled = {});

    [[nodiscard]] const GeneratedFoliageConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<RuntimeGrassBlade>& grassBlades() const noexcept
    {
        return grassBlades_;
    }
    [[nodiscard]] const FoliageMesh& trunks() const noexcept { return trunks_; }
    [[nodiscard]] const FoliageMesh& canopies() const noexcept { return canopies_; }
    [[nodiscard]] std::uint32_t grassBladeCount() const noexcept
    {
        return static_cast<std::uint32_t>(grassBlades_.size());
    }
    [[nodiscard]] std::uint32_t treeCount() const noexcept { return treeCount_; }
    [[nodiscard]] std::vector<RuntimeGrassBlade> releaseGrassBlades() noexcept
    {
        return std::move(grassBlades_);
    }
    [[nodiscard]] FoliageMesh releaseTrunks() noexcept { return std::move(trunks_); }
    [[nodiscard]] FoliageMesh releaseCanopies() noexcept { return std::move(canopies_); }

private:
    GeneratedFoliageConfig config_;
    std::vector<RuntimeGrassBlade> grassBlades_;
    FoliageMesh trunks_;
    FoliageMesh canopies_;
    std::uint32_t treeCount_ = 0;
};

}
