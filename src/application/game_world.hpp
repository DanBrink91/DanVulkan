#pragma once

#include "game_camera.hpp"
#include "game_character.hpp"
#include "generated_foliage.hpp"
#include "generated_terrain.hpp"
#include "terrain_streaming.hpp"
#include "terrain_surface.hpp"

#include <danvulkan/renderer.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace danvulkan::application {

struct TerrainStreamingDiagnostics
{
    glm::vec3 characterPosition{0.0f};
    TerrainChunkCoordinate playerChunk{};
    TerrainChunkCoordinate committedCenter{};
    TerrainChunkCoordinate requestedCenter{};
    std::uint32_t residentChunkCount = 0;
    std::uint32_t completeChunkCount = 0;
    bool committedCenterValid = false;
    bool requestedCenterValid = false;
    bool requestedWindowComplete = false;
};

// Owns demo/gameplay state and produces the renderer's per-frame scene description.
class GameWorld
{
public:
    GameWorld();
    ~GameWorld();

    static void configureRenderer(RendererConfig& config);
    static void configureStreamingCapacity(RendererConfig& config);

    void initialize(VulkanRenderer& renderer, std::uint32_t viewportWidth,
        std::uint32_t viewportHeight);
    void update(const InputState& input, float deltaSeconds, std::uint32_t viewportWidth,
        std::uint32_t viewportHeight);
    // Resource mutations must happen between renderer frames. The app calls this after endFrame
    // so the next frame sees a chunk window centered on the latest player position.
    void streamTerrain(VulkanRenderer& renderer);
    void setEnvironment(const SceneEnvironment& environment) noexcept;
    void setKeyLightIntensity(float intensity) noexcept;
    [[nodiscard]] TerrainStreamingDiagnostics streamingDiagnostics() const noexcept;

    [[nodiscard]] const SceneSubmission& sceneSubmission() const noexcept
    {
        return submission_;
    }

private:
    using ChunkCoordinate = TerrainChunkCoordinate;

    struct PreparedTerrainChunk
    {
        ChunkCoordinate coordinate;
        std::unique_ptr<GeneratedTerrain> terrain;
        PreparedRuntimeGrass grass;
        FoliageMesh trunks;
        FoliageMesh canopies;
        bool foliagePrepared = false;
    };

    struct RuntimeTerrainChunk
    {
        std::int32_t x = 0;
        std::int32_t z = 0;
        std::unique_ptr<GeneratedTerrain> terrain;
        std::optional<SceneMeshHandle> terrainMesh;
        std::optional<SceneInstanceHandle> terrainInstance;
        std::optional<SceneMeshHandle> grassMesh;
        std::optional<SceneInstanceHandle> grassInstance;
        std::optional<SceneMeshHandle> trunkMesh;
        std::optional<SceneInstanceHandle> trunkInstance;
        std::optional<SceneMeshHandle> canopyMesh;
        std::optional<SceneInstanceHandle> canopyInstance;

        [[nodiscard]] bool complete() const noexcept
        {
            return grassMesh.has_value() && trunkMesh.has_value() &&
                canopyMesh.has_value();
        }
    };

    class ChunkGenerationPool;

    void updateCameraSubmission();
    [[nodiscard]] std::optional<TerrainSample> sampleTerrain(
        const glm::vec2& position) const noexcept;
    [[nodiscard]] static PreparedTerrainChunk prepareTerrainChunk(
        ChunkCoordinate coordinate,
        std::shared_ptr<const TerrainSurfaceField> surfaceField,
        bool includeFoliage = true,
        const std::function<bool()>& cancelled = {});
    void publishTerrainChunk(
        VulkanRenderer& renderer, PreparedTerrainChunk prepared);
    void publishInitialTerrain(
        VulkanRenderer& renderer, ChunkCoordinate center);
    static void destroyTerrainChunk(
        VulkanRenderer& renderer, RuntimeTerrainChunk& chunk);
    [[nodiscard]] ChunkCoordinate requestedStreamCenter() const noexcept;
    [[nodiscard]] bool windowComplete(ChunkCoordinate center) const noexcept;
    void requestStreamWindow(VulkanRenderer& renderer, ChunkCoordinate center);
    void rebuildPendingChunks(ChunkCoordinate center);
    void retireChunksOutsideWindow(VulkanRenderer& renderer, ChunkCoordinate center);

    GameCamera camera_;
    GameCharacter ninja_;
    std::optional<SceneAnimationActorHandle> ninjaActor_;
    std::optional<SceneTextureHandle> terrainAlbedoTexture_;
    std::optional<SceneTextureHandle> terrainPathAlbedoTexture_;
    std::optional<SceneMaterialHandle> terrainMaterial_;
    std::optional<SceneMaterialHandle> grassMaterial_;
    std::optional<SceneMaterialHandle> trunkMaterial_;
    std::optional<SceneMaterialHandle> canopyMaterial_;
    std::shared_ptr<const TerrainSurfaceField> surfaceField_;
    std::vector<RuntimeTerrainChunk> terrainChunks_;
    ChunkCoordinate streamCenter_;
    ChunkCoordinate requestedCenter_;
    std::unique_ptr<ChunkGenerationPool> chunkWorkers_;
    std::uint64_t streamRequestEpoch_ = 0;
    bool streamCenterValid_ = false;
    bool requestedCenterValid_ = false;
    SceneSubmission submission_;
};

}
