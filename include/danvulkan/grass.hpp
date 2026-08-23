#pragma once

#include <danvulkan/assets.hpp>

#include <glm/vec3.hpp>

#include <cstddef>
#include <cstdint>
#include <array>
#include <type_traits>
#include <vector>

// Renderer-independent semantic records shared by procedural generation and the Vulkan upload
// path. Keeping these free of Vulkan types lets CPU generation and tests remain lightweight.
struct RuntimeGrassBlade
{
    glm::vec3 base{0.0f};
    float height = 0.01f;
    glm::vec3 groundNormal{0.0f, 1.0f, 0.0f};
    float halfWidth = 0.0006f;
    glm::vec3 color{0.16f, 0.46f, 0.07f};
    float bend = 0.2f;
    glm::vec3 bendDirection{1.0f, 0.0f, 0.0f};
    float windPhase = 0.0f;
    float stiffness = 0.8f;
    float orientation = 0.0f;
    float lodRank = 0.0f;
    // Close-detail shape controls. These occupy padding fields in the existing packed GPU
    // vertex record, so richer blades do not increase streaming or device-memory cost.
    float curveBias = 0.55f;
    float taper = 1.3f;
    float flutter = 0.7f;
    float bladePhase = 0.0f;
    float camber = 0.0f;
};
static_assert(sizeof(RuntimeGrassBlade) == 96,
    "RuntimeGrassBlade memory audit changed; revisit grass generation working-set costs");

// Device-facing grass ABI. The base position remains full precision so neighboring streamed
// chunks meet exactly. Unit vectors use octahedral SNORM16 encoding, color uses UNORM8, and the
// remaining scalar controls use IEEE half pairs. Its field order matches the raw uint words read
// by shaders/grass_vert.grass_vert.
struct PackedRuntimeGrassBlade
{
    glm::vec3 base{0.0f};
    std::uint32_t groundNormalOct = 0;
    std::uint32_t bendDirectionOct = 0;
    std::uint32_t colorUnorm = 0;
    std::uint32_t heightHalfWidth = 0;
    std::uint32_t curveBiasTaper = 0;
    std::uint32_t flutterCamber = 0;
    std::uint32_t windBladePhase = 0;
    std::uint32_t stiffnessOrientation = 0;
    std::uint32_t bendLodRank = 0;
};
static_assert(std::is_trivially_copyable_v<PackedRuntimeGrassBlade>);
static_assert(sizeof(PackedRuntimeGrassBlade) == 48,
    "packed grass ABI must remain twelve 32-bit words");

[[nodiscard]] constexpr std::size_t packedGrassGeometryUnits(std::size_t bladeCount) noexcept
{
    const std::size_t bytes = bladeCount * sizeof(PackedRuntimeGrassBlade);
    return (bytes + sizeof(danvulkan::assets::Vertex) - 1U) /
        sizeof(danvulkan::assets::Vertex);
}
static_assert(packedGrassGeometryUnits(50000U) == 21429U);

// Virtual ribbon templates consumed by the grass vertex shader. Close tufts contain three
// leaves, medium tufts contain two, and distant tufts collapse to one silhouette leaf.
inline constexpr std::uint32_t grassHighTemplateVertexCount = 34U;
inline constexpr std::uint32_t grassMediumTemplateFirstVertex = 34U;
inline constexpr std::uint32_t grassMediumTemplateVertexCount = 16U;
inline constexpr std::uint32_t grassLowTemplateFirstVertex = 50U;
inline constexpr std::uint32_t grassLowTemplateVertexCount = 6U;
inline constexpr std::uint32_t grassTemplateVertexCount = 56U;
inline constexpr std::array<std::uint32_t, 3> grassTemplateIndexCounts{84U, 36U, 12U};
inline constexpr std::array<std::uint32_t, 3> grassTemplateFirstIndexOffsets{0U, 84U, 120U};
inline constexpr std::uint32_t grassTemplateIndexCount = 132U;

[[nodiscard]] PackedRuntimeGrassBlade packRuntimeGrassBlade(
    const RuntimeGrassBlade& blade) noexcept;
[[nodiscard]] RuntimeGrassBlade unpackRuntimeGrassBlade(
    const PackedRuntimeGrassBlade& blade) noexcept;

struct RuntimeGrassLodDescription
{
    float highDetailDistance = 0.65f;
    float mediumDetailDistance = 1.35f;
    float maximumDistance = 4.0f;
    float mediumPopulation = 0.5f;
    float lowPopulation = 0.18f;
    // Blades remain in one GPU allocation, but are grouped into world-aligned tiles for
    // independent frustum culling and LOD selection. World alignment avoids seams when terrain
    // chunks stream in and out around the player.
    float tileSize = 0.25f;
    // Segment-count LOD changes use this dead band so camera jitter cannot make a tile chatter.
    float hysteresis = 0.08f;
    // Stable blade ranks progressively thin the population through this distance band instead
    // of making every blade in a tile appear or disappear on the same frame.
    float transitionBand = 0.12f;
};
static_assert(sizeof(RuntimeGrassLodDescription) == 32,
    "RuntimeGrassLodDescription memory audit changed");

struct RuntimeGrassTileDescription
{
    glm::vec3 boundsMinimum{0.0f};
    glm::vec3 boundsMaximum{0.0f};
    std::uint32_t firstBlade = 0;
    std::uint32_t bladeCount = 0;
};
static_assert(sizeof(RuntimeGrassTileDescription) == 32,
    "RuntimeGrassTileDescription memory audit changed");

// Fully validated, tile-ordered GPU input produced without touching Vulkan. Taking the semantic
// blade vector by value lets a background generator transfer ownership without another 50,000
// semantic-record copy. The renderer only allocates raw byte ranges and stages these compact
// records.
struct PreparedRuntimeGrass
{
    std::vector<PackedRuntimeGrassBlade> records;
    std::vector<std::uint32_t> indices;
    std::vector<RuntimeGrassTileDescription> tiles;
    RuntimeGrassLodDescription lod;
    float maximumHeight = 0.0f;
};

[[nodiscard]] PreparedRuntimeGrass prepareRuntimeGrass(
    std::vector<RuntimeGrassBlade> blades,
    const RuntimeGrassLodDescription& lod = {});
