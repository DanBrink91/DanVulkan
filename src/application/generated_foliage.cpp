#include "generated_foliage.hpp"

#include "generated_terrain.hpp"
#include "terrain_surface.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

namespace danvulkan::application {
namespace {
constexpr float pi = 3.14159265358979323846f;
constexpr std::uint32_t trunkSides = 7U;
constexpr std::uint32_t crownSides = 9U;
constexpr std::uint32_t maximumPlacementAttempts = 64U;
constexpr float minimumTriangleAreaSquared = 1.0e-20f;

struct Random
{
    explicit Random(std::uint32_t seed) : state(seed == 0U ? 1U : seed) {}

    [[nodiscard]] std::uint32_t next() noexcept
    {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        return state;
    }

    [[nodiscard]] float unit() noexcept
    {
        return static_cast<float>(next() & 0x00ffffffU) / 16777215.0f;
    }

    [[nodiscard]] float range(float minimum, float maximum) noexcept
    {
        return std::lerp(minimum, maximum, unit());
    }

    std::uint32_t state;
};

struct TreePlacement
{
    glm::vec3 base{0.0f};
    float height = 0.0f;
    float rotation = 0.0f;
    float colorVariation = 1.0f;
};

bool finite(const GeneratedFoliageConfig& config) noexcept
{
    return std::isfinite(config.clearingCenter.x) &&
        std::isfinite(config.clearingCenter.y) &&
        std::isfinite(config.grassClearingRadius) &&
        std::isfinite(config.treeClearingRadius) &&
        std::isfinite(config.grassMinimumHeight) &&
        std::isfinite(config.grassMaximumHeight) &&
        std::isfinite(config.treeMinimumHeight) &&
        std::isfinite(config.treeMaximumHeight) &&
        std::isfinite(config.treeMinimumSpacing) &&
        std::isfinite(config.minimumGroundNormalY);
}

danvulkan::assets::Vertex makeVertex(const glm::vec3& position, const glm::vec3& normal,
    const glm::vec3& color, const glm::vec3& tangent)
{
    danvulkan::assets::Vertex vertex;
    vertex.pos = position;
    vertex.normal = normal;
    vertex.color = color;
    vertex.tangent = tangent;
    vertex.tangentSign = 1.0f;
    return vertex;
}

void appendTriangle(FoliageMesh& mesh, const glm::vec3& a, const glm::vec3& b,
    const glm::vec3& c, const glm::vec3& colorA, const glm::vec3& colorB,
    const glm::vec3& colorC)
{
    const glm::vec3 edge = b - a;
    const glm::vec3 face = glm::cross(edge, c - a);
    if (glm::dot(face, face) <= minimumTriangleAreaSquared)
    {
        return;
    }
    const glm::vec3 normal = glm::normalize(face);
    const glm::vec3 tangent = glm::normalize(edge);
    const std::uint32_t first = static_cast<std::uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(makeVertex(a, normal, colorA, tangent));
    mesh.vertices.push_back(makeVertex(b, normal, colorB, tangent));
    mesh.vertices.push_back(makeVertex(c, normal, colorC, tangent));
    mesh.indices.insert(mesh.indices.end(), {first, first + 1U, first + 2U});
}

void appendTrunk(FoliageMesh& mesh, const TreePlacement& tree)
{
    const float trunkHeight = tree.height * 0.62f;
    const float bottomRadius = tree.height * 0.045f;
    const float topRadius = bottomRadius * 0.68f;
    const glm::vec3 bottomColor = glm::clamp(
        glm::vec3(0.24f, 0.105f, 0.035f) * tree.colorVariation, 0.0f, 1.0f);
    const glm::vec3 topColor = glm::clamp(
        glm::vec3(0.38f, 0.19f, 0.065f) * tree.colorVariation, 0.0f, 1.0f);
    for (std::uint32_t side = 0; side < trunkSides; ++side)
    {
        const float angle0 = tree.rotation + 2.0f * pi * static_cast<float>(side) /
            static_cast<float>(trunkSides);
        const float angle1 = tree.rotation + 2.0f * pi * static_cast<float>(side + 1U) /
            static_cast<float>(trunkSides);
        const glm::vec3 bottom0 = tree.base +
            glm::vec3(std::cos(angle0) * bottomRadius, 0.0f, std::sin(angle0) * bottomRadius);
        const glm::vec3 bottom1 = tree.base +
            glm::vec3(std::cos(angle1) * bottomRadius, 0.0f, std::sin(angle1) * bottomRadius);
        const glm::vec3 top0 = tree.base +
            glm::vec3(std::cos(angle0) * topRadius, trunkHeight, std::sin(angle0) * topRadius);
        const glm::vec3 top1 = tree.base +
            glm::vec3(std::cos(angle1) * topRadius, trunkHeight, std::sin(angle1) * topRadius);
        appendTriangle(mesh, bottom0, top0, bottom1,
            bottomColor, topColor, bottomColor);
        appendTriangle(mesh, bottom1, top0, top1,
            bottomColor, topColor, topColor);
    }
}

void appendCrown(FoliageMesh& mesh, const TreePlacement& tree, float verticalOffset,
    float radiusScale, float heightScale)
{
    const float centerY = tree.base.y + tree.height * (0.72f + verticalOffset);
    const float radius = tree.height * radiusScale;
    const float crownHeight = tree.height * heightScale;
    const glm::vec3 top(tree.base.x, centerY + crownHeight * 0.55f, tree.base.z);
    const glm::vec3 bottom(tree.base.x, centerY - crownHeight * 0.45f, tree.base.z);
    const glm::vec3 lowerColor = glm::clamp(
        glm::vec3(0.07f, 0.29f, 0.045f) * tree.colorVariation, 0.0f, 1.0f);
    const glm::vec3 ringColor = glm::clamp(
        glm::vec3(0.12f, 0.47f, 0.075f) * tree.colorVariation, 0.0f, 1.0f);
    const glm::vec3 topColor = glm::clamp(
        glm::vec3(0.26f, 0.62f, 0.11f) * tree.colorVariation, 0.0f, 1.0f);
    for (std::uint32_t side = 0; side < crownSides; ++side)
    {
        const float angle0 = tree.rotation + 2.0f * pi * static_cast<float>(side) /
            static_cast<float>(crownSides);
        const float angle1 = tree.rotation + 2.0f * pi * static_cast<float>(side + 1U) /
            static_cast<float>(crownSides);
        const glm::vec3 ring0(tree.base.x + std::cos(angle0) * radius, centerY,
            tree.base.z + std::sin(angle0) * radius);
        const glm::vec3 ring1(tree.base.x + std::cos(angle1) * radius, centerY,
            tree.base.z + std::sin(angle1) * radius);
        appendTriangle(mesh, ring0, top, ring1, ringColor, topColor, ringColor);
        appendTriangle(mesh, ring0, ring1, bottom, ringColor, ringColor, lowerColor);
    }
}

float distanceSquared(const glm::vec2& a, const glm::vec2& b) noexcept
{
    const glm::vec2 offset = a - b;
    return glm::dot(offset, offset);
}

std::uint32_t coordinateHash(std::int32_t x, std::int32_t z, std::uint32_t seed) noexcept
{
    std::uint32_t value = static_cast<std::uint32_t>(x) * 0x8da6b343U ^
        static_cast<std::uint32_t>(z) * 0xd8163841U ^ seed * 0xcb1ab31fU;
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

float hashUnit(std::uint32_t value) noexcept
{
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return static_cast<float>(value & 0x00ffffffU) / 16777215.0f;
}
}

GeneratedFoliage::GeneratedFoliage(
    const GeneratedTerrain& terrain, GeneratedFoliageConfig config,
    const TerrainSurfaceField* surfaceField,
    const std::function<bool()>& cancelled)
    : config_(config)
{
    constexpr std::uint32_t maximumGrassBlades = 200000U;
    constexpr std::uint32_t maximumTrees = 10000U;
    if (!finite(config_) || config_.grassClearingRadius < 0.0f ||
        config_.treeClearingRadius < 0.0f || config_.grassMinimumHeight <= 0.0f ||
        config_.grassMaximumHeight < config_.grassMinimumHeight ||
        config_.treeMinimumHeight <= 0.0f ||
        config_.treeMaximumHeight < config_.treeMinimumHeight ||
        config_.treeMinimumSpacing < 0.0f || config_.minimumGroundNormalY < 0.0f ||
        config_.minimumGroundNormalY > 1.0f ||
        config_.grassBladeCount > maximumGrassBlades ||
        config_.treeCount > maximumTrees)
    {
        throw std::invalid_argument("generated foliage configuration is invalid");
    }

    Random random(config_.seed);
    const glm::vec2 terrainMinimum = terrain.minimum();
    const glm::vec2 terrainMaximum = terrain.maximum();
    const float treeMargin = config_.treeMaximumHeight * 0.28f;
    if (terrainMaximum.x - terrainMinimum.x <= treeMargin * 2.0f ||
        terrainMaximum.y - terrainMinimum.y <= treeMargin * 2.0f)
    {
        throw std::invalid_argument("generated foliage does not fit within the terrain");
    }

    std::vector<TreePlacement> trees;
    trees.reserve(config_.treeCount);
    const float minimumTreeSpacingSquared =
        config_.treeMinimumSpacing * config_.treeMinimumSpacing;
    for (std::uint32_t attempt = 0;
         trees.size() < config_.treeCount &&
         attempt < config_.treeCount * maximumPlacementAttempts;
         ++attempt)
    {
        if ((attempt & 63U) == 0U && cancelled && cancelled())
        {
            return;
        }
        const glm::vec2 position(
            random.range(terrainMinimum.x + treeMargin, terrainMaximum.x - treeMargin),
            random.range(terrainMinimum.y + treeMargin, terrainMaximum.y - treeMargin));
        if (distanceSquared(position, config_.clearingCenter) <
            config_.treeClearingRadius * config_.treeClearingRadius)
        {
            continue;
        }
        if (surfaceField != nullptr &&
            surfaceField->sample(position).dirtWeight > 0.12f)
        {
            continue;
        }
        const std::optional<TerrainSample> ground = terrain.sample(position);
        if (!ground || ground->normal.y < config_.minimumGroundNormalY)
        {
            continue;
        }
        bool separated = true;
        for (const TreePlacement& tree : trees)
        {
            if (distanceSquared(position, {tree.base.x, tree.base.z}) <
                minimumTreeSpacingSquared)
            {
                separated = false;
                break;
            }
        }
        if (!separated)
        {
            continue;
        }
        trees.push_back({{position.x, ground->height, position.y},
            random.range(config_.treeMinimumHeight, config_.treeMaximumHeight),
            random.range(0.0f, 2.0f * pi), random.range(0.82f, 1.14f)});
    }
    treeCount_ = static_cast<std::uint32_t>(trees.size());

    trunks_.vertices.reserve(static_cast<std::size_t>(treeCount_) * trunkSides * 6U);
    trunks_.indices.reserve(trunks_.vertices.capacity());
    canopies_.vertices.reserve(static_cast<std::size_t>(treeCount_) * crownSides * 12U);
    canopies_.indices.reserve(canopies_.vertices.capacity());
    for (const TreePlacement& tree : trees)
    {
        appendTrunk(trunks_, tree);
        appendCrown(canopies_, tree, 0.0f, 0.24f, 0.55f);
        appendCrown(canopies_, tree, 0.13f, 0.17f, 0.38f);
    }

    grassBlades_.reserve(config_.grassBladeCount);
    if (config_.grassBladeCount == 0U)
    {
        return;
    }

    // A globally aligned jittered grid supplies continuous ground coverage. Procedural or
    // authored surface coverage removes candidates; Voronoi traits control the survivors.
    const float spacing = terrain.config().size /
        std::sqrt(static_cast<float>(config_.grassBladeCount));
    const std::int32_t minimumCellX = static_cast<std::int32_t>(
        std::floor(terrainMinimum.x / spacing));
    const std::int32_t maximumCellX = static_cast<std::int32_t>(
        std::ceil(terrainMaximum.x / spacing));
    const std::int32_t minimumCellZ = static_cast<std::int32_t>(
        std::floor(terrainMinimum.y / spacing));
    const std::int32_t maximumCellZ = static_cast<std::int32_t>(
        std::ceil(terrainMaximum.y / spacing));
    for (std::int32_t cellZ = minimumCellZ; cellZ < maximumCellZ; ++cellZ)
    {
        if (cancelled && cancelled())
        {
            return;
        }
        for (std::int32_t cellX = minimumCellX; cellX < maximumCellX; ++cellX)
        {
            const std::uint32_t bladeHash = coordinateHash(
                cellX, cellZ, config_.grassSeed);
            const glm::vec2 jitter(
                std::lerp(0.08f, 0.92f, hashUnit(bladeHash ^ 0x243f6a88U)),
                std::lerp(0.08f, 0.92f, hashUnit(bladeHash ^ 0x85a308d3U)));
            const glm::vec2 position = (glm::vec2(cellX, cellZ) + jitter) * spacing;
            if (position.x < terrainMinimum.x || position.y < terrainMinimum.y ||
                position.x >= terrainMaximum.x || position.y >= terrainMaximum.y)
            {
                continue;
            }
            if (distanceSquared(position, config_.clearingCenter) <
                config_.grassClearingRadius * config_.grassClearingRadius)
            {
                continue;
            }
            bool clearOfTrees = true;
            for (const TreePlacement& tree : trees)
            {
                const float crownRadius = tree.height * 0.045f;
                if (distanceSquared(position, {tree.base.x, tree.base.z}) <
                    crownRadius * crownRadius)
                {
                    clearOfTrees = false;
                    break;
                }
            }
            if (!clearOfTrees)
            {
                continue;
            }
            const std::optional<TerrainSample> ground = terrain.sample(position);
            if (!ground || ground->normal.y < config_.minimumGroundNormalY)
            {
                continue;
            }
            TerrainSurfaceSample surface;
            if (surfaceField != nullptr)
            {
                surface = surfaceField->sample(position);
            }
            if (hashUnit(bladeHash ^ 0x13198a2eU) >=
                glm::clamp(surface.grassCoverage, 0.0f, 1.0f))
            {
                continue;
            }

            const float baseHeight = std::lerp(config_.grassMinimumHeight,
                config_.grassMaximumHeight, hashUnit(bladeHash ^ 0x03707344U));
            const float heightJitter = std::lerp(0.91f, 1.09f,
                hashUnit(bladeHash ^ 0xa4093822U));
            glm::vec3 bendDirection(surface.grass.bendDirection.x, 0.0f,
                surface.grass.bendDirection.y);
            bendDirection -= ground->normal * glm::dot(bendDirection, ground->normal);
            if (glm::dot(bendDirection, bendDirection) < 0.000001f)
            {
                bendDirection = glm::normalize(glm::cross(
                    ground->normal, glm::vec3(0.0f, 0.0f, 1.0f)));
            }
            else
            {
                bendDirection = glm::normalize(bendDirection);
            }
            const float directionOffset = std::lerp(-0.18f, 0.18f,
                hashUnit(bladeHash ^ 0xc0ac29b7U));
            bendDirection = glm::normalize(bendDirection * std::cos(directionOffset) +
                glm::cross(ground->normal, bendDirection) * std::sin(directionOffset));

            RuntimeGrassBlade blade;
            blade.base = {position.x, ground->height + 0.00008f, position.y};
            blade.height = baseHeight * surface.grass.heightScale * heightJitter;
            blade.groundNormal = ground->normal;
            blade.halfWidth = blade.height * std::lerp(0.048f, 0.075f,
                hashUnit(bladeHash ^ 0x299f31d0U));
            const float brightness = std::lerp(0.94f, 1.06f,
                hashUnit(bladeHash ^ 0x082efa98U));
            const float colorTone = std::lerp(-1.0f, 1.0f,
                hashUnit(bladeHash ^ 0x6c8e9cf5U));
            blade.color = glm::clamp(surface.grass.color * brightness +
                glm::vec3(colorTone * 0.022f, -std::abs(colorTone) * 0.004f,
                    -colorTone * 0.012f), 0.0f, 1.0f);
            blade.bend = surface.grass.bend * std::lerp(0.88f, 1.12f,
                hashUnit(bladeHash ^ 0xec4e6c89U));
            blade.bendDirection = bendDirection;
            blade.windPhase = surface.grass.windPhase;
            blade.stiffness = glm::clamp(surface.grass.stiffness * std::lerp(0.96f, 1.04f,
                hashUnit(bladeHash ^ 0xd3a2646cU)), 0.0f, 1.0f);
            blade.orientation = hashUnit(bladeHash ^ 0x38d01377U) * pi;
            blade.lodRank = hashUnit(bladeHash ^ 0xbe5466cfU);
            blade.curveBias = std::lerp(0.36f, 0.72f,
                hashUnit(bladeHash ^ 0x9e3779b9U));
            blade.taper = std::lerp(1.05f, 1.65f,
                hashUnit(bladeHash ^ 0x7f4a7c15U));
            blade.flutter = std::lerp(0.55f, 1.15f,
                hashUnit(bladeHash ^ 0xf39cc060U));
            blade.bladePhase = hashUnit(bladeHash ^ 0x106aa070U) * 2.0f * pi;
            blade.camber = std::lerp(-0.32f, 0.32f,
                hashUnit(bladeHash ^ 0x19a4c116U));
            grassBlades_.push_back(blade);
        }
    }
}

}
