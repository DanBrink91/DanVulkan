#include <danvulkan/grass.hpp>

#include <glm/geometric.hpp>
#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace
{
glm::vec2 octEncode(glm::vec3 direction) noexcept
{
    direction = glm::normalize(direction);
    direction /= std::abs(direction.x) + std::abs(direction.y) +
        std::abs(direction.z);
    glm::vec2 encoded(direction.x, direction.y);
    if (direction.z < 0.0f)
    {
        const glm::vec2 sign(encoded.x >= 0.0f ? 1.0f : -1.0f,
            encoded.y >= 0.0f ? 1.0f : -1.0f);
        encoded = (glm::vec2(1.0f) - glm::abs(glm::vec2(encoded.y, encoded.x))) * sign;
    }
    return encoded;
}

glm::vec3 octDecode(glm::vec2 encoded) noexcept
{
    glm::vec3 direction(encoded.x, encoded.y,
        1.0f - std::abs(encoded.x) - std::abs(encoded.y));
    if (direction.z < 0.0f)
    {
        const glm::vec2 sign(direction.x >= 0.0f ? 1.0f : -1.0f,
            direction.y >= 0.0f ? 1.0f : -1.0f);
        const glm::vec2 folded =
            (glm::vec2(1.0f) - glm::abs(glm::vec2(direction.y, direction.x))) * sign;
        direction.x = folded.x;
        direction.y = folded.y;
    }
    return glm::normalize(direction);
}

bool validLod(const RuntimeGrassLodDescription& lod) noexcept
{
    return std::isfinite(lod.highDetailDistance) &&
        std::isfinite(lod.mediumDetailDistance) && std::isfinite(lod.maximumDistance) &&
        std::isfinite(lod.mediumPopulation) && std::isfinite(lod.lowPopulation) &&
        std::isfinite(lod.tileSize) && std::isfinite(lod.hysteresis) &&
        std::isfinite(lod.transitionBand) && lod.highDetailDistance > 0.0f &&
        lod.mediumDetailDistance > lod.highDetailDistance &&
        lod.maximumDistance > lod.mediumDetailDistance && lod.mediumPopulation > 0.0f &&
        lod.mediumPopulation <= 1.0f && lod.lowPopulation > 0.0f &&
        lod.lowPopulation <= lod.mediumPopulation && lod.tileSize > 0.0f &&
        lod.hysteresis >= 0.0f &&
        lod.hysteresis < (lod.mediumDetailDistance - lod.highDetailDistance) * 0.5f &&
        lod.transitionBand >= 0.0f &&
        lod.transitionBand < std::min(lod.mediumDetailDistance - lod.highDetailDistance,
            lod.maximumDistance - lod.mediumDetailDistance) * 0.5f;
}

bool validBlade(const RuntimeGrassBlade& blade) noexcept
{
    const bool finite = std::isfinite(blade.base.x) && std::isfinite(blade.base.y) &&
        std::isfinite(blade.base.z) && std::isfinite(blade.height) &&
        std::isfinite(blade.halfWidth) && std::isfinite(blade.bend) &&
        std::isfinite(blade.windPhase) && std::isfinite(blade.stiffness) &&
        std::isfinite(blade.orientation) && std::isfinite(blade.lodRank) &&
        std::isfinite(blade.curveBias) && std::isfinite(blade.taper) &&
        std::isfinite(blade.flutter) && std::isfinite(blade.bladePhase) &&
        std::isfinite(blade.camber) && std::isfinite(blade.groundNormal.x) &&
        std::isfinite(blade.groundNormal.y) && std::isfinite(blade.groundNormal.z) &&
        std::isfinite(blade.bendDirection.x) &&
        std::isfinite(blade.bendDirection.y) &&
        std::isfinite(blade.bendDirection.z);
    return finite && blade.height > 0.0f && blade.halfWidth > 0.0f &&
        blade.curveBias >= 0.0f && blade.curveBias <= 1.0f && blade.taper > 0.0f &&
        blade.flutter >= 0.0f && std::abs(blade.camber) <= 1.0f &&
        glm::dot(blade.groundNormal, blade.groundNormal) >= 0.000001f &&
        glm::dot(blade.bendDirection, blade.bendDirection) >= 0.000001f;
}
}

PackedRuntimeGrassBlade packRuntimeGrassBlade(const RuntimeGrassBlade& blade) noexcept
{
    PackedRuntimeGrassBlade packed;
    packed.base = blade.base;
    packed.groundNormalOct = glm::packSnorm2x16(octEncode(blade.groundNormal));
    packed.bendDirectionOct = glm::packSnorm2x16(octEncode(blade.bendDirection));
    packed.colorUnorm = glm::packUnorm4x8(glm::vec4(glm::clamp(blade.color, 0.0f, 1.0f), 1.0f));
    packed.heightHalfWidth = glm::packHalf2x16({blade.height, blade.halfWidth});
    packed.curveBiasTaper = glm::packHalf2x16({blade.curveBias, blade.taper});
    packed.flutterCamber = glm::packHalf2x16({blade.flutter, blade.camber});
    packed.windBladePhase = glm::packHalf2x16({blade.windPhase, blade.bladePhase});
    packed.stiffnessOrientation = glm::packHalf2x16({blade.stiffness, blade.orientation});
    packed.bendLodRank = glm::packHalf2x16({blade.bend, blade.lodRank});
    return packed;
}

RuntimeGrassBlade unpackRuntimeGrassBlade(const PackedRuntimeGrassBlade& packed) noexcept
{
    RuntimeGrassBlade blade;
    blade.base = packed.base;
    blade.groundNormal = octDecode(glm::unpackSnorm2x16(packed.groundNormalOct));
    blade.bendDirection = octDecode(glm::unpackSnorm2x16(packed.bendDirectionOct));
    blade.color = glm::vec3(glm::unpackUnorm4x8(packed.colorUnorm));
    const glm::vec2 dimensions = glm::unpackHalf2x16(packed.heightHalfWidth);
    blade.height = dimensions.x;
    blade.halfWidth = dimensions.y;
    const glm::vec2 curve = glm::unpackHalf2x16(packed.curveBiasTaper);
    blade.curveBias = curve.x;
    blade.taper = curve.y;
    const glm::vec2 shape = glm::unpackHalf2x16(packed.flutterCamber);
    blade.flutter = shape.x;
    blade.camber = shape.y;
    const glm::vec2 phases = glm::unpackHalf2x16(packed.windBladePhase);
    blade.windPhase = phases.x;
    blade.bladePhase = phases.y;
    const glm::vec2 controls = glm::unpackHalf2x16(packed.stiffnessOrientation);
    blade.stiffness = controls.x;
    blade.orientation = controls.y;
    const glm::vec2 bendRank = glm::unpackHalf2x16(packed.bendLodRank);
    blade.bend = bendRank.x;
    blade.lodRank = bendRank.y;
    return blade;
}

PreparedRuntimeGrass prepareRuntimeGrass(
    std::vector<RuntimeGrassBlade> blades, const RuntimeGrassLodDescription& lod)
{
    if (blades.empty() || !validLod(lod))
    {
        throw std::invalid_argument(
            "prepareRuntimeGrass requires blades and valid ordered LOD settings");
    }
    if (!std::ranges::all_of(blades, validBlade))
    {
        throw std::invalid_argument("prepareRuntimeGrass contains an invalid blade");
    }

    const auto tileCoordinate = [&](const RuntimeGrassBlade& blade)
    {
        return std::pair{
            static_cast<std::int32_t>(std::floor(blade.base.x / lod.tileSize)),
            static_cast<std::int32_t>(std::floor(blade.base.z / lod.tileSize))};
    };
    std::ranges::sort(blades,
        [&](const RuntimeGrassBlade& left, const RuntimeGrassBlade& right)
        {
            const auto leftTile = tileCoordinate(left);
            const auto rightTile = tileCoordinate(right);
            return std::tie(leftTile.first, leftTile.second, left.lodRank) <
                std::tie(rightTile.first, rightTile.second, right.lodRank);
        });

    PreparedRuntimeGrass prepared;
    prepared.lod = lod;
    prepared.records.resize(blades.size());
    for (std::size_t index = 0; index < blades.size(); ++index)
    {
        const RuntimeGrassBlade& blade = blades[index];
        prepared.records[index] = packRuntimeGrassBlade(blade);
        prepared.maximumHeight = std::max(prepared.maximumHeight, blade.height);
    }

    for (std::size_t first = 0; first < blades.size();)
    {
        const auto coordinate = tileCoordinate(blades[first]);
        std::size_t end = first + 1U;
        while (end < blades.size() && tileCoordinate(blades[end]) == coordinate)
        {
            ++end;
        }
        RuntimeGrassTileDescription tile;
        tile.firstBlade = static_cast<std::uint32_t>(first);
        tile.bladeCount = static_cast<std::uint32_t>(end - first);
        tile.boundsMinimum = blades[first].base;
        tile.boundsMaximum = blades[first].base;
        float maximumHeight = blades[first].height;
        for (std::size_t index = first + 1U; index < end; ++index)
        {
            tile.boundsMinimum = glm::min(tile.boundsMinimum, blades[index].base);
            tile.boundsMaximum = glm::max(tile.boundsMaximum, blades[index].base);
            maximumHeight = std::max(maximumHeight, blades[index].height);
        }
        const glm::vec3 padding(maximumHeight * 0.5f);
        tile.boundsMinimum -= padding;
        tile.boundsMaximum += padding;
        tile.boundsMaximum.y += maximumHeight;
        prepared.tiles.push_back(tile);
        first = end;
    }

    prepared.indices.reserve(grassTemplateIndexCount);
    const auto appendTemplate = [&](std::uint32_t firstVertex, std::uint32_t segmentCount)
    {
        for (std::uint32_t segment = 0; segment < segmentCount; ++segment)
        {
            const std::uint32_t left0 = firstVertex + segment * 2U;
            const std::uint32_t right0 = left0 + 1U;
            const std::uint32_t left1 = left0 + 2U;
            const std::uint32_t right1 = left0 + 3U;
            prepared.indices.insert(prepared.indices.end(),
                {left0, right0, left1, right0, right1, left1});
        }
    };
    // High: one six-segment hero leaf and two four-segment companion leaves.
    appendTemplate(0U, 6U);
    appendTemplate(14U, 4U);
    appendTemplate(24U, 4U);
    // Medium: two three-segment leaves.
    appendTemplate(grassMediumTemplateFirstVertex, 3U);
    appendTemplate(grassMediumTemplateFirstVertex + 8U, 3U);
    // Low: one two-segment silhouette leaf.
    appendTemplate(grassLowTemplateFirstVertex, 2U);
    if (prepared.indices.size() != grassTemplateIndexCount)
    {
        throw std::logic_error("grass tuft template ABI is inconsistent");
    }
    return prepared;
}
