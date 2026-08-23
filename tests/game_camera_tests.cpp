#include "application/game_camera.hpp"
#include "application/game_character.hpp"
#include "application/generated_foliage.hpp"
#include "application/generated_terrain.hpp"
#include "application/terrain_surface.hpp"
#include "application/terrain_streaming.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <glm/geometric.hpp>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

bool finite(const glm::mat4& matrix)
{
    for (glm::length_t column = 0; column < 4; ++column)
    {
        for (glm::length_t row = 0; row < 4; ++row)
        {
            if (!std::isfinite(matrix[column][row]))
            {
                return false;
            }
        }
    }
    return true;
}

void testSceneFraming()
{
    danvulkan::application::GameCamera camera;
    const glm::vec3 minimum(-4.0f, -2.0f, -1.0f);
    const glm::vec3 maximum(6.0f, 4.0f, 3.0f);
    camera.frame(minimum, maximum, 1280, 720);

    const glm::vec3 center = (minimum + maximum) * 0.5f;
    require(glm::distance(camera.position(), center) > 1.0f,
        "framed camera remained inside the scene center");
    require(camera.movementSpeed() > 0.0f, "framed camera has no movement speed");
    require(finite(camera.view()) && finite(camera.projection()),
        "framed camera produced a non-finite matrix");
}

void testNormalizedMovement()
{
    danvulkan::application::GameCamera forwardCamera;
    danvulkan::application::GameCamera diagonalCamera;
    forwardCamera.frame(glm::vec3(-1.0f), glm::vec3(1.0f), 1280, 720);
    diagonalCamera.frame(glm::vec3(-1.0f), glm::vec3(1.0f), 1280, 720);
    const glm::vec3 initialPosition = forwardCamera.position();

    danvulkan::application::InputState forward;
    forward.moveForward = true;
    forwardCamera.update(forward, 0.5f);

    danvulkan::application::InputState diagonal;
    diagonal.moveForward = true;
    diagonal.moveRight = true;
    diagonalCamera.update(diagonal, 0.5f);

    const float forwardDistance = glm::distance(initialPosition, forwardCamera.position());
    const float diagonalDistance = glm::distance(initialPosition, diagonalCamera.position());
    require(std::abs(forwardDistance - diagonalDistance) < 0.0001f,
        "diagonal camera movement was not normalized");
}

void testLookAndZeroViewport()
{
    danvulkan::application::GameCamera camera;
    camera.frame(glm::vec3(-1.0f), glm::vec3(1.0f), 1280, 720);
    const glm::mat4 previousView = camera.view();

    danvulkan::application::InputState input;
    input.lookDeltaX = 20.0f;
    input.lookDeltaY = -10.0f;
    camera.update(input, 0.0f);
    camera.setViewport(0, 0);

    require(camera.view() != previousView, "look input did not change the camera view");
    require(finite(camera.view()) && finite(camera.projection()),
        "zero viewport invalidated the camera matrices");
}

void testFollowCameraToggle()
{
    danvulkan::application::GameCamera camera;
    camera.setFollowTarget(glm::vec3(2.0f, 0.0f, -3.0f),
        glm::vec3(0.0f, 0.0f, 1.0f), 0.08f, 0.016f);

    danvulkan::application::InputState toggle;
    toggle.toggleCameraMode = true;
    camera.update(toggle, 0.0f);

    require(camera.mode() == danvulkan::application::CameraMode::follow,
        "camera did not enter follow mode");
    require(glm::distance(camera.position(), glm::vec3(2.0f, 0.016f, -3.08f)) < 0.0001f,
        "follow camera was not placed behind its target");
    require(finite(camera.view()), "follow camera produced a non-finite view matrix");

    camera.update(toggle, 0.0f);
    require(camera.mode() == danvulkan::application::CameraMode::free,
        "camera did not return to free mode");
}

void testCharacterMovement()
{
    danvulkan::application::GameCharacter forwardCharacter(
        glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2.0f);
    danvulkan::application::InputState forward;
    forward.moveForward = true;
    forwardCharacter.updateMovement(forward, 0.5f);
    require(glm::distance(forwardCharacter.position(), glm::vec3(2.0f, 0.0f, -2.0f)) <
            0.0001f,
        "W did not move the followed character forward");

    danvulkan::application::GameCharacter diagonalCharacter(
        glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2.0f);
    danvulkan::application::InputState diagonal;
    diagonal.moveForward = true;
    diagonal.moveRight = true;
    diagonalCharacter.updateMovement(diagonal, 0.5f);
    require(std::abs(glm::distance(glm::vec3(2.0f, 0.0f, -3.0f),
                         diagonalCharacter.position()) - 1.0f) < 0.0001f,
        "diagonal character movement was not normalized");
    require(diagonalCharacter.position().x < 2.0f &&
            diagonalCharacter.position().z > -3.0f,
        "W/D did not move the character along its forward and right axes");
    const glm::vec3 transformedOrigin = glm::vec3(diagonalCharacter.worldOffset() *
        glm::vec4(2.0f, 0.0f, -3.0f, 1.0f));
    require(glm::distance(transformedOrigin, diagonalCharacter.position()) < 0.0001f,
        "character world offset did not preserve its gameplay pivot");

    danvulkan::application::GameCharacter leftCharacter(
        glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2.0f);
    danvulkan::application::InputState left;
    left.moveLeft = true;
    leftCharacter.updateMovement(left, 0.5f);
    require(glm::distance(leftCharacter.position(), glm::vec3(3.0f, 0.0f, -3.0f)) <
            0.0001f,
        "A did not move the followed character toward the left side of the screen");
}

void testInitialFollowCamera()
{
    danvulkan::application::GameCamera camera(
        danvulkan::application::CameraMode::follow);
    camera.setViewport(1280, 720);
    camera.setFollowTarget(glm::vec3(2.0f, 0.0f, -3.0f),
        glm::vec3(0.0f, 0.0f, 1.0f), 0.08f, 0.016f);

    require(camera.mode() == danvulkan::application::CameraMode::follow,
        "camera did not preserve its requested initial mode");
    require(glm::distance(camera.position(), glm::vec3(2.0f, 0.016f, -3.08f)) < 0.0001f,
        "initial follow camera was not placed behind its target");
    require(finite(camera.view()) && finite(camera.projection()),
        "initial follow camera produced a non-finite matrix");
}

void testStableCharacterControlBasis()
{
    danvulkan::application::GameCharacter character(
        glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2.0f);
    danvulkan::application::InputState left;
    left.moveLeft = true;
    character.updateMovement(left, 0.2f);
    require(character.forward().x > 0.9f,
        "character did not face its leftward travel direction");
    require(glm::distance(character.controlForward(), glm::vec3(0.0f, 0.0f, 1.0f)) <
            0.0001f,
        "turning the actor also rotated its WASD control basis");

    danvulkan::application::GameCamera camera;
    camera.setFollowTarget(character.position(), character.controlForward(), 0.08f, 0.016f);
    danvulkan::application::InputState toggle;
    toggle.toggleCameraMode = true;
    camera.update(toggle, 0.0f);
    require(std::abs(camera.position().x - character.position().x) < 0.0001f &&
            std::abs(camera.position().z - (character.position().z - 0.08f)) < 0.0001f,
        "follow camera rotated away from the character control basis");

    const glm::vec3 positionBeforeForward = character.position();
    danvulkan::application::InputState forward;
    forward.moveForward = true;
    character.updateMovement(forward, 0.2f);
    require(character.position().z > positionBeforeForward.z &&
            std::abs(character.position().x - positionBeforeForward.x) < 0.0001f,
        "W did not remain aligned with the visible follow-camera heading after a turn");
}

void testGeneratedTerrain()
{
    const danvulkan::application::GeneratedTerrainConfig config{
        {3.0f, -2.0f}, 4.0f, 1.0f, 0.5f, 2.5f, 9U, 42U};
    const danvulkan::application::GeneratedTerrain terrain(config);
    const danvulkan::application::GeneratedTerrain matching(config);
    require(terrain.vertices().size() == 81U,
        "terrain did not generate one vertex per grid sample");
    require(terrain.indices().size() == 8U * 8U * 6U,
        "terrain did not generate two triangles per grid cell");
    require(terrain.vertices().front().pos == matching.vertices().front().pos,
        "terrain generation is not deterministic for a fixed seed");

    bool variedVertexColors = false;

    for (const danvulkan::assets::Vertex& vertex : terrain.vertices())
    {
        require(std::isfinite(vertex.pos.y) && std::isfinite(vertex.normal.x) &&
                std::isfinite(vertex.normal.y) && std::isfinite(vertex.normal.z),
            "terrain generated a non-finite vertex");
        require(vertex.normal.y > 0.0f &&
                std::abs(glm::length(vertex.normal) - 1.0f) < 0.0001f,
            "terrain generated an invalid surface normal");
        require(std::abs(vertex.tangentSign) == 3.0f && vertex.unused0 == 1.0f,
            "generated terrain lost its underlayer surface marker or default coverage");
        variedVertexColors = variedVertexColors ||
            glm::distance(vertex.color, terrain.vertices().front().color) > 0.001f;
    }
    require(variedVertexColors, "terrain did not generate visual surface variation");

    danvulkan::application::ProceduralTerrainSurfaceConfig surfaceConfig;
    surfaceConfig.pathOrigin = config.center;
    surfaceConfig.pathDirection = {1.0f, 0.0f};
    surfaceConfig.pathMeanderAmplitude = 0.0f;
    const danvulkan::application::ProceduralTerrainSurfaceField surface(surfaceConfig);
    const danvulkan::application::GeneratedTerrain surfacedTerrain(config, &surface);
    const auto& pathVertex = surfacedTerrain.vertices()[4U * 9U + 4U];
    const auto& forestVertex = surfacedTerrain.vertices()[5U * 9U + 4U];
    require(pathVertex.unused0 < 0.001f && forestVertex.unused0 > 0.999f,
        "terrain underlayer coverage did not preserve the procedural dirt path");

    const glm::vec3 centerVertex = terrain.vertices()[4U * 9U + 4U].pos;
    const auto centerSample = terrain.sample({centerVertex.x, centerVertex.z});
    require(centerSample && std::abs(centerSample->height - centerVertex.y) < 0.0001f,
        "terrain sampling disagrees with its rendered grid");
    require(!terrain.sample(terrain.maximum() + glm::vec2(0.1f)),
        "terrain sampling accepted a point outside its bounds");

    const danvulkan::application::GeneratedTerrain left({
        {0.0f, 0.0f}, 2.0f, 0.0f, 0.08f, 2.5f, 17U, 123U});
    const danvulkan::application::GeneratedTerrain right({
        {2.0f, 0.0f}, 2.0f, 0.0f, 0.08f, 2.5f, 17U, 123U});
    for (std::uint32_t row = 0; row < 17U; ++row)
    {
        const auto& leftSeam = left.vertices()[row * 17U + 16U];
        const auto& rightSeam = right.vertices()[row * 17U];
        require(glm::distance(leftSeam.pos, rightSeam.pos) < 0.000001f &&
                glm::distance(leftSeam.normal, rightSeam.normal) < 0.000001f,
            "adjacent generated terrain chunks have a visible seam");
    }
}

void testGeneratedFoliage()
{
    const danvulkan::application::GeneratedTerrain terrain({
        {0.0f, 0.0f}, 2.0f, 0.0f, 0.04f, 3.0f, 65U, 73U});
    danvulkan::application::GeneratedFoliageConfig config;
    config.clearingCenter = {0.0f, 0.0f};
    config.grassClearingRadius = 0.0f;
    config.treeClearingRadius = 0.0f;
    config.grassBladeCount = 4000U;
    config.treeCount = 8U;
    config.seed = 99U;
    danvulkan::application::ProceduralTerrainSurfaceConfig surfaceConfig;
    surfaceConfig.pathOrigin = {0.0f, 0.0f};
    surfaceConfig.seed = 1234U;
    const danvulkan::application::ProceduralTerrainSurfaceField surface(surfaceConfig);
    const danvulkan::application::GeneratedFoliage foliage(terrain, config, &surface);
    const danvulkan::application::GeneratedFoliage matching(terrain, config, &surface);

    require(foliage.grassBladeCount() > config.grassBladeCount * 3U / 4U &&
            foliage.grassBladeCount() <= config.grassBladeCount + 256U &&
            foliage.treeCount() == config.treeCount,
        "foliage generator did not produce dense path-filtered coverage");
    require(foliage.trunks().vertices.size() == config.treeCount * 42U &&
            foliage.canopies().vertices.size() == config.treeCount * 108U,
        "tree trunk or canopy batch geometry is incomplete");
    require(foliage.grassBlades().front().base == matching.grassBlades().front().base &&
            foliage.grassBlades().front().curveBias ==
                matching.grassBlades().front().curveBias &&
            foliage.grassBlades().front().bladePhase ==
                matching.grassBlades().front().bladePhase &&
            foliage.canopies().vertices.front().pos ==
                matching.canopies().vertices.front().pos,
        "foliage generation is not deterministic for a fixed seed");

    bool sameClumpVariation = false;
    std::unordered_map<std::uint32_t, RuntimeGrassBlade> firstBladeByClump;
    for (const RuntimeGrassBlade& blade : foliage.grassBlades())
    {
        require(std::isfinite(blade.base.x) && std::isfinite(blade.base.y) &&
                std::isfinite(blade.base.z) && blade.height > 0.0f &&
                blade.halfWidth > 0.0f &&
                blade.curveBias >= 0.36f && blade.curveBias <= 0.72f &&
                blade.taper >= 1.05f && blade.taper <= 1.65f &&
                blade.flutter >= 0.55f && blade.flutter <= 1.15f &&
                blade.bladePhase >= 0.0f && blade.bladePhase <= 2.0f * 3.14159266f &&
                std::abs(blade.camber) <= 0.32f &&
                std::abs(glm::length(blade.groundNormal) - 1.0f) < 0.0001f,
            "foliage generated an invalid compact grass blade");
        require(blade.base.x >= terrain.minimum().x &&
                blade.base.z >= terrain.minimum().y &&
                blade.base.x < terrain.maximum().x &&
                blade.base.z < terrain.maximum().y,
            "grass blade escaped its terrain patch");
        require(surface.sample({blade.base.x, blade.base.z}).dirtWeight < 0.999f,
            "grass was generated in the solid center of a dirt path");
        const danvulkan::application::TerrainSurfaceSample bladeSurface =
            surface.sample({blade.base.x, blade.base.z});
        const auto [existing, inserted] = firstBladeByClump.emplace(
            bladeSurface.grass.clumpId, blade);
        if (!inserted)
        {
            require(std::abs(existing->second.windPhase - blade.windPhase) < 0.00001f,
                "blades in one Voronoi clump lost their coherent wind phase");
            const float directionAgreement = glm::dot(
                existing->second.bendDirection, blade.bendDirection);
            sameClumpVariation = sameClumpVariation ||
                (glm::length(existing->second.color - blade.color) > 0.00001f &&
                 std::abs(existing->second.curveBias - blade.curveBias) > 0.00001f &&
                 directionAgreement < 0.99999f);
        }
    }
    require(sameClumpVariation,
        "grass clumps did not retain subtle per-blade color, curve, and direction variation");

    const PreparedRuntimeGrass prepared = prepareRuntimeGrass(
        std::vector<RuntimeGrassBlade>(
            foliage.grassBlades().begin(), foliage.grassBlades().end()));
    require(prepared.records.size() == foliage.grassBlades().size() &&
            prepared.indices.size() == grassTemplateIndexCount && !prepared.tiles.empty() &&
            *std::ranges::max_element(prepared.indices) < grassTemplateVertexCount,
        "background grass preparation did not produce packed tiled GPU input");
    for (const RuntimeGrassTileDescription& tile : prepared.tiles)
    {
        float previousRank = -1.0f;
        for (std::uint32_t index = tile.firstBlade;
             index < tile.firstBlade + tile.bladeCount; ++index)
        {
            const RuntimeGrassBlade unpacked =
                unpackRuntimeGrassBlade(prepared.records[index]);
            require(unpacked.lodRank >= previousRank,
                "prepared grass tile was not stable-ranked for nested LOD populations");
            previousRank = unpacked.lodRank;
        }
    }

    const RuntimeGrassBlade& sourceBlade = foliage.grassBlades().front();
    const RuntimeGrassBlade unpackedBlade =
        unpackRuntimeGrassBlade(packRuntimeGrassBlade(sourceBlade));
    require(sizeof(PackedRuntimeGrassBlade) == 48U &&
            unpackedBlade.base == sourceBlade.base &&
            glm::dot(unpackedBlade.groundNormal,
                glm::normalize(sourceBlade.groundNormal)) > 0.9999f &&
            glm::dot(unpackedBlade.bendDirection,
                glm::normalize(sourceBlade.bendDirection)) > 0.9999f &&
            glm::length(unpackedBlade.color - sourceBlade.color) < 0.007f &&
            std::abs(unpackedBlade.height - sourceBlade.height) < 0.00002f &&
            std::abs(unpackedBlade.halfWidth - sourceBlade.halfWidth) < 0.000002f &&
            std::abs(unpackedBlade.orientation - sourceBlade.orientation) < 0.002f,
        "compact grass packing exceeded its visual precision budget");

    const std::array<const danvulkan::application::FoliageMesh*, 2> meshes{{
        &foliage.trunks(), &foliage.canopies()}};
    for (const danvulkan::application::FoliageMesh* mesh : meshes)
    {
        require(!mesh->vertices.empty() && !mesh->indices.empty(),
            "foliage produced an empty render batch");
        for (const danvulkan::assets::Vertex& vertex : mesh->vertices)
        {
            require(std::isfinite(vertex.pos.x) && std::isfinite(vertex.pos.y) &&
                    std::isfinite(vertex.pos.z) &&
                    std::abs(glm::length(vertex.normal) - 1.0f) < 0.0001f,
                "foliage generated invalid geometry");
            require(vertex.pos.x >= terrain.minimum().x - 0.0001f &&
                    vertex.pos.z >= terrain.minimum().y - 0.0001f &&
                    vertex.pos.x <= terrain.maximum().x + 0.0001f &&
                    vertex.pos.z <= terrain.maximum().y + 0.0001f,
                "foliage geometry escaped the terrain patch");
        }
    }
}

void testTerrainSurfaceField()
{
    danvulkan::application::ProceduralTerrainSurfaceConfig config;
    config.pathOrigin = {2.0f, -3.0f};
    config.pathDirection = {1.0f, 0.0f};
    config.pathMeanderAmplitude = 0.0f;
    config.seed = 9182U;
    const danvulkan::application::ProceduralTerrainSurfaceField field(config);

    const auto path = field.sample(config.pathOrigin);
    const auto forest = field.sample(config.pathOrigin + glm::vec2(0.0f, 0.2f));
    require(path.dirtWeight > 0.999f && path.grassCoverage < 0.001f,
        "procedural path did not fully suppress grass at its center");
    require(forest.dirtWeight < 0.001f && forest.grassCoverage > 0.999f,
        "procedural forest did not restore full coverage away from its path");

    const glm::vec2 probe(1.37f, -2.41f);
    const auto first = field.sample(probe);
    const auto second = field.sample(probe);
    require(first.grass.clumpId == second.grass.clumpId &&
            first.grass.clumpPoint == second.grass.clumpPoint &&
            first.grass.color == second.grass.color,
        "Voronoi surface sampling is not deterministic");

    bool foundSharedClump = false;
    bool foundFlowAlignedNeighbor = false;
    for (int z = 0; z < 20 && !foundSharedClump; ++z)
    {
        for (int x = 0; x < 20 && !foundSharedClump; ++x)
        {
            const glm::vec2 position = probe + glm::vec2(x, z) * 0.002f;
            const auto nearby = field.sample(position);
            if (nearby.grass.clumpId == first.grass.clumpId)
            {
                foundSharedClump = nearby.grass.heightScale == first.grass.heightScale &&
                    nearby.grass.bendDirection == first.grass.bendDirection &&
                    nearby.grass.color == first.grass.color;
            }
        }
    }
    require(foundSharedClump,
        "nearby blades did not inherit coherent traits from a shared Voronoi clump");

    for (int z = -10; z <= 10 && !foundFlowAlignedNeighbor; ++z)
    {
        for (int x = -10; x <= 10 && !foundFlowAlignedNeighbor; ++x)
        {
            const auto nearby = field.sample(probe + glm::vec2(x, z) * 0.02f);
            if (nearby.grass.clumpId != first.grass.clumpId)
            {
                foundFlowAlignedNeighbor = glm::dot(nearby.grass.bendDirection,
                    first.grass.bendDirection) > 0.35f;
            }
        }
    }
    require(foundFlowAlignedNeighbor,
        "neighboring Voronoi clumps lost their shared low-frequency flow direction");

    const auto blended = danvulkan::application::blendTerrainSurfaceSamples(
        forest, path, 0.5f);
    require(blended.grassCoverage > 0.45f && blended.grassCoverage < 0.55f &&
            blended.dirtWeight > 0.45f && blended.dirtWeight < 0.55f,
        "surface composition cannot blend a future authored paint overlay");
}

void testTerrainStreamingPlan()
{
    const danvulkan::application::TerrainStreamingConfig config{
        {2.15f, -2.85f}, 1.0f, 0.32f};
    using Coordinate = danvulkan::application::TerrainChunkCoordinate;

    require(danvulkan::application::terrainChunkAt({2.64f, -2.85f}, config) ==
            Coordinate{0, 0} &&
            danvulkan::application::terrainChunkAt({2.66f, -2.85f}, config) ==
            Coordinate{1, 0},
        "terrain chunk selection did not change at the half-chunk boundary");
    require(danvulkan::application::requestedTerrainStreamCenter(
                {2.46f, -2.85f}, {1.0f, 0.0f}, config) == Coordinate{0, 0},
        "terrain streaming prefetched before its configured lead distance");
    require(danvulkan::application::requestedTerrainStreamCenter(
                {2.48f, -2.85f}, {1.0f, 0.0f}, config) == Coordinate{1, 0},
        "terrain streaming did not prefetch ahead of forward movement");
    require(danvulkan::application::requestedTerrainStreamCenter(
                {2.48f, -2.85f}, {-1.0f, 0.0f}, config) == Coordinate{0, 0},
        "terrain streaming prefetched opposite the movement direction");
    require(danvulkan::application::requestedTerrainStreamCenter(
                {2.48f, -2.52f}, {1.0f, 1.0f}, config) == Coordinate{1, 1},
        "terrain streaming did not predict a diagonal chunk transition");
    require(danvulkan::application::terrainChunkInWindow(
                {3, -2}, {1, 0}, 2) &&
            !danvulkan::application::terrainChunkInWindow({4, -2}, {1, 0}, 2),
        "terrain streaming window bounds are incorrect");
}

void testTerrainCharacterController()
{
    const danvulkan::application::GeneratedTerrain terrain({
        {0.0f, 0.0f}, 2.0f, 0.25f, 0.05f, 2.0f, 17U, 7U});
    danvulkan::application::CharacterMovementConfig movement;
    movement.maximumSpeed = 1.0f;
    movement.acceleration = 2.0f;
    movement.braking = 4.0f;
    movement.turnSpeedRadians = 20.0f;
    danvulkan::application::GameCharacter character(
        glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), movement);
    character.placeOnGround(terrain);

    danvulkan::application::InputState right;
    right.moveRight = true;
    character.updateMovement(right, 0.1f, terrain);
    require(character.moving() && glm::length(character.velocity()) < movement.maximumSpeed,
        "character controller did not accelerate from rest");
    const auto ground = terrain.sample({character.position().x, character.position().z});
    require(ground && std::abs(character.position().y - ground->height) < 0.0001f,
        "character did not stay grounded on generated terrain");
    require(character.forward().x < 0.0f,
        "character did not turn toward its movement direction");

    for (int step = 0; step < 30; ++step)
    {
        character.updateMovement(right, 0.1f, terrain);
    }
    require(terrain.contains({character.position().x, character.position().z}),
        "character escaped the generated terrain bounds");

    const danvulkan::application::InputState idle;
    for (int step = 0; step < 10; ++step)
    {
        character.updateMovement(idle, 0.1f, terrain);
    }
    require(!character.moving(), "character did not brake to a stop");

    const danvulkan::application::GeneratedTerrain leftChunk({
        {0.0f, 0.0f}, 1.0f, 0.0f, 0.02f, 2.0f, 17U, 51U});
    const danvulkan::application::GeneratedTerrain rightChunk({
        {1.0f, 0.0f}, 1.0f, 0.0f, 0.02f, 2.0f, 17U, 51U});
    danvulkan::application::GameCharacter crossingCharacter(
        glm::vec3(0.45f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), 1.0f);
    crossingCharacter.placeOnGround(leftChunk);
    danvulkan::application::InputState moveAcrossSeam;
    moveAcrossSeam.moveLeft = true;
    crossingCharacter.updateMovementOnSurface(moveAcrossSeam, 0.2f,
        [&](const glm::vec2& position)
        {
            return position.x < 0.5f ? leftChunk.sample(position) :
                rightChunk.sample(position);
        });
    require(crossingCharacter.position().x > 0.5f,
        "sampled character movement was clamped at a streamed terrain seam");
}

}

int main()
{
    try
    {
        testSceneFraming();
        testNormalizedMovement();
        testLookAndZeroViewport();
        testFollowCameraToggle();
        testInitialFollowCamera();
        testCharacterMovement();
        testStableCharacterControlBasis();
        testGeneratedTerrain();
        testTerrainSurfaceField();
        testGeneratedFoliage();
        testTerrainStreamingPlan();
        testTerrainCharacterController();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
