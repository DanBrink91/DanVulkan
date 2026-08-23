#include "game_world.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <ranges>
#include <string>
#include <thread>

namespace danvulkan::application {
namespace {
constexpr glm::vec3 ninjaPosition(2.15f, -0.05f, -2.85f);
// glTF character assets face world +Z in the demo scene.
constexpr glm::vec3 ninjaForward(0.0f, 0.0f, 1.0f);
constexpr float ninjaScale = 0.01f;
constexpr float ninjaMovementSpeed = 0.08f;
constexpr float terrainChunkSize = 1.0f;
constexpr float terrainHeightScale = 0.032f;
constexpr float terrainNoiseScale = 2.8f;
constexpr std::uint32_t terrainChunkResolution = 97U;
constexpr std::int32_t terrainStreamRadius = 2;
constexpr std::uint32_t grassBladesPerChunk = 50000U;
constexpr std::uint32_t treesPerChunk = 20U;
constexpr std::uint32_t terrainSeed = 0x4e415255U;
constexpr float terrainPrefetchOffset = 0.32f;
// A straight transition temporarily retains 30 chunks; a diagonal transition retains 34.
// Include the character scene and some allocator headroom in the initial geometry reservation.
// Grass records occupy 48 raw bytes in the shared 112-byte geometry arena. This reservation
// covers the 34-chunk diagonal overlap, terrain/tree geometry, the character, and headroom.
constexpr std::uint32_t streamedVertexCapacity = 1450000U;
constexpr std::uint32_t streamedIndexCapacity = 4250000U;
// The imported origin sits in front of and below the rendered character. Aim at the upper torso
// and leave enough depth behind the bind-pose bounds for the running animation.
constexpr glm::vec3 ninjaFollowTargetOffset(0.1f, 2.0f, -4.0f);
constexpr float ninjaFollowDistance = 8.0f;
constexpr float ninjaFollowHeight = 1.6f;
constexpr float ninjaGrassInteractionRadius = 0.045f;

std::uint32_t chunkSeed(std::int32_t x, std::int32_t z) noexcept
{
    std::uint32_t value = static_cast<std::uint32_t>(x) * 0x8da6b343U ^
        static_cast<std::uint32_t>(z) * 0xd8163841U ^ 0x464f5245U;
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    return value == 0U ? 1U : value;
}

TerrainStreamingConfig streamingConfig() noexcept
{
    return {{ninjaPosition.x, ninjaPosition.z}, terrainChunkSize, terrainPrefetchOffset};
}
}

class GameWorld::ChunkGenerationPool
{
public:
    explicit ChunkGenerationPool(
        std::shared_ptr<const TerrainSurfaceField> surfaceField,
        std::size_t workerCount = 2U)
        : surfaceField_(std::move(surfaceField))
    {
        workers_.reserve(workerCount);
        for (std::size_t index = 0; index < workerCount; ++index)
        {
            workers_.emplace_back([this](std::stop_token stop) { workerLoop(stop); });
        }
    }

    ~ChunkGenerationPool()
    {
        for (std::jthread& worker : workers_)
        {
            worker.request_stop();
        }
        condition_.notify_all();
        // jthread members would otherwise be destroyed after the mutex and condition variable
        // because of reverse member-destruction order. Join them explicitly while every wait and
        // cancellation dependency is still alive.
        workers_.clear();
    }

    void replace(std::uint64_t epoch, std::vector<ChunkCoordinate> coordinates)
    {
        activeEpoch_.store(epoch, std::memory_order_release);
        std::scoped_lock lock(mutex_);
        jobs_.clear();
        completed_.clear();
        jobs_.reserve(coordinates.size());
        for (const ChunkCoordinate coordinate : coordinates)
        {
            jobs_.push_back({coordinate, epoch});
        }
        condition_.notify_all();
    }

    [[nodiscard]] std::optional<PreparedTerrainChunk> takeCompleted(std::uint64_t epoch)
    {
        std::scoped_lock lock(mutex_);
        while (!completed_.empty())
        {
            Completed result = std::move(completed_.front());
            completed_.pop_front();
            condition_.notify_all();
            if (result.epoch == epoch)
            {
                if (result.error)
                {
                    std::rethrow_exception(result.error);
                }
                return std::move(result.chunk);
            }
        }
        return std::nullopt;
    }

private:
    struct Job
    {
        ChunkCoordinate coordinate;
        std::uint64_t epoch = 0;
    };

    struct Completed
    {
        std::uint64_t epoch = 0;
        PreparedTerrainChunk chunk;
        std::exception_ptr error;
    };

    void workerLoop(std::stop_token stop)
    {
        while (!stop.stop_requested())
        {
            Job job;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, stop,
                    [this] { return !jobs_.empty(); });
                if (stop.stop_requested())
                {
                    return;
                }
                job = jobs_.front();
                jobs_.erase(jobs_.begin());
            }

            const auto cancelled = [this, epoch = job.epoch, stop]
            {
                return stop.stop_requested() ||
                    activeEpoch_.load(std::memory_order_acquire) != epoch;
            };
            try
            {
                PreparedTerrainChunk prepared = GameWorld::prepareTerrainChunk(
                    job.coordinate, surfaceField_, true, cancelled);
                if (cancelled())
                {
                    continue;
                }
                std::unique_lock lock(mutex_);
                condition_.wait(lock, stop,
                    [this, epoch = job.epoch]
                    {
                        return completed_.size() < maximumCompletedChunks ||
                            activeEpoch_.load(std::memory_order_acquire) != epoch;
                    });
                if (activeEpoch_.load(std::memory_order_acquire) == job.epoch)
                {
                    completed_.push_back({job.epoch, std::move(prepared), {}});
                }
            }
            catch (...)
            {
                if (!cancelled())
                {
                    std::scoped_lock lock(mutex_);
                    completed_.push_back({job.epoch, {}, std::current_exception()});
                }
            }
        }
    }

    std::shared_ptr<const TerrainSurfaceField> surfaceField_;
    std::vector<std::jthread> workers_;
    std::vector<Job> jobs_;
    std::deque<Completed> completed_;
    std::mutex mutex_;
    std::condition_variable_any condition_;
    std::atomic<std::uint64_t> activeEpoch_{0};
    static constexpr std::size_t maximumCompletedChunks = 4U;
};

GameWorld::GameWorld()
    : camera_(CameraMode::follow),
      ninja_(ninjaPosition, ninjaForward,
          CharacterMovementConfig{ninjaMovementSpeed, 0.5f, 0.8f, 8.0f,
              0.959931f, 0.0f})
{
    ProceduralTerrainSurfaceConfig surfaceConfig;
    surfaceConfig.pathOrigin = {ninjaPosition.x, ninjaPosition.z};
    surfaceConfig.seed = terrainSeed ^ 0x1f123bb5U;
    surfaceField_ = std::make_shared<ProceduralTerrainSurfaceField>(surfaceConfig);
    submission_.environment = {
        {0.95f, 1.0f, 1.08f}, 0.45f, 0.0f, 1.25f, 0.8f};
    submission_.pointLights = {
        {{ninjaPosition.x - 0.35f, ninjaPosition.y + 0.55f, ninjaPosition.z + 0.25f},
            2.0f, {1.0f, 0.86f, 0.68f}, 3.5f},
        {{ninjaPosition.x + 0.4f, ninjaPosition.y + 0.25f, ninjaPosition.z - 0.3f},
            1.5f, {0.48f, 0.68f, 1.0f}, 0.8f}
    };
}

GameWorld::~GameWorld() = default;

void GameWorld::configureRenderer(RendererConfig& config)
{
    config.modelPath.clear();
    config.additionalScenes.push_back({
        "models/ninja_run_free_fire_emote.glb",
        glm::translate(glm::mat4(1.0f), ninjaPosition) *
            glm::scale(glm::mat4(1.0f), glm::vec3(ninjaScale))
    });
}

void GameWorld::configureStreamingCapacity(RendererConfig& config)
{
    config.initialVertexCapacity =
        std::max(config.initialVertexCapacity, streamedVertexCapacity);
    config.initialIndexCapacity =
        std::max(config.initialIndexCapacity, streamedIndexCapacity);
}

void GameWorld::initialize(VulkanRenderer& renderer, std::uint32_t viewportWidth,
    std::uint32_t viewportHeight)
{
    chunkWorkers_.reset();
    ninjaActor_.reset();
    terrainChunks_.clear();
    streamCenterValid_ = false;
    requestedCenterValid_ = false;
    streamRequestEpoch_ = 0;

    danvulkan::assets::TextureAsset terrainAlbedo = danvulkan::assets::loadTexture(
        "textures/forest_floor_albedo.png", danvulkan::assets::ColorSpace::srgb);
    terrainAlbedo.name = "generated forest floor albedo";
    // The terrain shader mirrors the capture in world space. Clamp sampling keeps both mirror
    // turning points continuous even if the generated source's opposite borders differ slightly.
    terrainAlbedo.sampler.wrapU = danvulkan::assets::TextureWrap::clampToEdge;
    terrainAlbedo.sampler.wrapV = danvulkan::assets::TextureWrap::clampToEdge;
    terrainAlbedoTexture_ = renderer.uploadTexture(terrainAlbedo);

    danvulkan::assets::TextureAsset pathAlbedo = danvulkan::assets::loadTexture(
        "textures/forest_path_albedo.png", danvulkan::assets::ColorSpace::srgb);
    pathAlbedo.name = "generated forest path albedo";
    pathAlbedo.sampler.wrapU = danvulkan::assets::TextureWrap::clampToEdge;
    pathAlbedo.sampler.wrapV = danvulkan::assets::TextureWrap::clampToEdge;
    terrainPathAlbedoTexture_ = renderer.uploadTexture(pathAlbedo);

    RuntimeMaterialDescription terrainMaterialDescription;
    terrainMaterialDescription.name = "generated terrain material";
    terrainMaterialDescription.properties.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
    terrainMaterialDescription.properties.roughnessFactor = 0.94f;
    terrainMaterialDescription.textures.baseColor = *terrainAlbedoTexture_;
    // Generated terrain uses the otherwise-unused occlusion binding as its second albedo layer.
    // The terrain shader bypasses ordinary occlusion sampling for this semantic material.
    terrainMaterialDescription.textures.occlusion = *terrainPathAlbedoTexture_;
    terrainMaterial_ = renderer.createMaterial(terrainMaterialDescription);

    RuntimeMaterialDescription grassMaterialDescription;
    grassMaterialDescription.name = "generated grass material";
    grassMaterialDescription.properties.roughnessFactor = 0.84f;
    grassMaterialDescription.doubleSided = true;
    grassMaterial_ = renderer.createMaterial(grassMaterialDescription);

    RuntimeMaterialDescription trunkMaterialDescription;
    trunkMaterialDescription.name = "generated tree trunk material";
    trunkMaterialDescription.properties.roughnessFactor = 0.92f;
    trunkMaterial_ = renderer.createMaterial(trunkMaterialDescription);

    RuntimeMaterialDescription canopyMaterialDescription;
    canopyMaterialDescription.name = "generated tree canopy material";
    canopyMaterialDescription.properties.roughnessFactor = 0.9f;
    canopyMaterialDescription.doubleSided = true;
    canopyMaterial_ = renderer.createMaterial(canopyMaterialDescription);

    // Publish inexpensive terrain around the spawn plus full vegetation only under the player.
    // The remaining vegetation and outer terrain stream after the first frame.
    const ChunkCoordinate initialCenter = terrainChunkAt(
        {ninja_.position().x, ninja_.position().z}, streamingConfig());
    publishInitialTerrain(renderer, initialCenter);
    streamCenter_ = initialCenter;
    requestedCenter_ = initialCenter;
    streamCenterValid_ = true;
    requestedCenterValid_ = true;
    chunkWorkers_ = std::make_unique<ChunkGenerationPool>(surfaceField_);
    ++streamRequestEpoch_;
    rebuildPendingChunks(initialCenter);
    const glm::vec2 ninjaGroundPosition(ninja_.position().x, ninja_.position().z);
    const auto centerChunk = std::find_if(terrainChunks_.begin(), terrainChunks_.end(),
        [&](const RuntimeTerrainChunk& chunk)
        {
            return chunk.terrain->contains(ninjaGroundPosition);
        });
    if (centerChunk != terrainChunks_.end())
    {
        ninja_.placeOnGround(*centerChunk->terrain);
    }

    if (const std::optional<SceneBounds> bounds = renderer.sceneBounds())
    {
        camera_.frame(bounds->minimum, bounds->maximum, viewportWidth, viewportHeight);
    }
    else
    {
        camera_.setViewport(viewportWidth, viewportHeight);
    }
    camera_.setFollowTarget(ninja_.position() + ninjaFollowTargetOffset * ninjaScale,
        ninja_.controlForward(), ninjaFollowDistance * ninjaScale,
        ninjaFollowHeight * ninjaScale);

    const std::vector<SceneAnimationInfo> animations = renderer.sceneAnimations();
    if (!animations.empty())
    {
        renderer.playAnimation(animations.back().handle, false);
        const std::vector<SceneAnimationActorInfo> actors = renderer.sceneAnimationActors();
        for (auto actor = actors.rbegin(); actor != actors.rend(); ++actor)
        {
            if (actor->animation == animations.back().handle)
            {
                ninjaActor_ = actor->handle;
                break;
            }
        }
    }
    updateCameraSubmission();
}

void GameWorld::update(const InputState& input, float deltaSeconds,
    std::uint32_t viewportWidth, std::uint32_t viewportHeight)
{
    const bool followControls = input.toggleCameraMode ?
        camera_.mode() != CameraMode::follow : camera_.mode() == CameraMode::follow;
    if (followControls)
    {
        ninja_.updateMovementOnSurface(input, deltaSeconds,
            [this](const glm::vec2& position)
            {
                return sampleTerrain(position);
            });
    }
    camera_.setViewport(viewportWidth, viewportHeight);
    camera_.setFollowTarget(ninja_.position() + ninjaFollowTargetOffset * ninjaScale,
        ninja_.controlForward(), ninjaFollowDistance * ninjaScale,
        ninjaFollowHeight * ninjaScale);
    camera_.update(input, deltaSeconds);
    if (submission_.pointLights.size() >= 2U)
    {
        submission_.pointLights[0].position = ninja_.position() +
            glm::vec3(-0.35f, 0.55f, 0.25f);
        submission_.pointLights[1].position = ninja_.position() +
            glm::vec3(0.4f, 0.25f, -0.3f);
    }
    updateCameraSubmission();
}

void GameWorld::streamTerrain(VulkanRenderer& renderer)
{
    if (!terrainMaterial_ || !grassMaterial_ || !trunkMaterial_ || !canopyMaterial_)
    {
        return;
    }

    const ChunkCoordinate desiredCenter = requestedStreamCenter();
    if (!requestedCenterValid_ || desiredCenter != requestedCenter_)
    {
        requestStreamWindow(renderer, desiredCenter);
    }

    // Publication is budgeted to one chunk per frame and never waits for a transfer fence.
    // A second prepared chunk remains in the worker completion queue until the retained staging
    // slot from the previous batch is reusable.
    if (chunkWorkers_ && renderer.runtimeUploadReady())
    {
        if (std::optional<PreparedTerrainChunk> prepared =
                chunkWorkers_->takeCompleted(streamRequestEpoch_))
        {
            const bool stillRequested = terrainChunkInWindow(
                prepared->coordinate, requestedCenter_, terrainStreamRadius);
            const auto existing = std::find_if(terrainChunks_.begin(), terrainChunks_.end(),
                [&](const RuntimeTerrainChunk& chunk)
                {
                    return chunk.x == prepared->coordinate.x &&
                        chunk.z == prepared->coordinate.z;
                });
            if (stillRequested &&
                (existing == terrainChunks_.end() || !existing->complete()))
            {
                publishTerrainChunk(renderer, std::move(*prepared));
            }
        }
    }

    if (windowComplete(requestedCenter_))
    {
        retireChunksOutsideWindow(renderer, requestedCenter_);
        streamCenter_ = requestedCenter_;
        streamCenterValid_ = true;
    }
}

std::optional<TerrainSample> GameWorld::sampleTerrain(
    const glm::vec2& position) const noexcept
{
    const ChunkCoordinate coordinate = terrainChunkAt(position, streamingConfig());
    for (const RuntimeTerrainChunk& chunk : terrainChunks_)
    {
        if (chunk.x == coordinate.x && chunk.z == coordinate.z)
        {
            return chunk.terrain->sample(position);
        }
    }
    return std::nullopt;
}

GameWorld::PreparedTerrainChunk GameWorld::prepareTerrainChunk(
    ChunkCoordinate coordinate,
    std::shared_ptr<const TerrainSurfaceField> surfaceField,
    bool includeFoliage,
    const std::function<bool()>& cancelled)
{
    PreparedTerrainChunk prepared;
    prepared.coordinate = coordinate;
    const glm::vec2 center = {
        ninjaPosition.x + static_cast<float>(coordinate.x) * terrainChunkSize,
        ninjaPosition.z + static_cast<float>(coordinate.z) * terrainChunkSize};
    prepared.terrain = std::make_unique<GeneratedTerrain>(GeneratedTerrainConfig{
        center, terrainChunkSize, ninjaPosition.y, terrainHeightScale, terrainNoiseScale,
        terrainChunkResolution, terrainSeed}, surfaceField.get());
    if (!includeFoliage || (cancelled && cancelled()))
    {
        return prepared;
    }

    GeneratedFoliageConfig foliageConfig;
    foliageConfig.clearingCenter = {ninjaPosition.x, ninjaPosition.z};
    foliageConfig.grassMinimumHeight = 0.008f;
    foliageConfig.grassMaximumHeight = 0.017f;
    foliageConfig.grassClearingRadius = 0.0f;
    foliageConfig.treeClearingRadius = 0.0f;
    foliageConfig.grassBladeCount = grassBladesPerChunk;
    foliageConfig.treeCount = treesPerChunk;
    foliageConfig.seed = chunkSeed(coordinate.x, coordinate.z);
    GeneratedFoliage foliage(
        *prepared.terrain, foliageConfig, surfaceField.get(), cancelled);
    if (cancelled && cancelled())
    {
        return prepared;
    }
    prepared.grass = prepareRuntimeGrass(
        foliage.releaseGrassBlades(), RuntimeGrassLodDescription{});
    prepared.trunks = foliage.releaseTrunks();
    prepared.canopies = foliage.releaseCanopies();
    prepared.foliagePrepared = true;
    return prepared;
}

void GameWorld::publishTerrainChunk(
    VulkanRenderer& renderer, PreparedTerrainChunk prepared)
{
    auto existing = std::find_if(terrainChunks_.begin(), terrainChunks_.end(),
        [&](const RuntimeTerrainChunk& chunk)
        {
            return chunk.x == prepared.coordinate.x && chunk.z == prepared.coordinate.z;
        });
    const bool uploadTerrain = existing == terrainChunks_.end();
    if (!prepared.foliagePrepared)
    {
        throw std::logic_error("streamed chunk reached publication without foliage");
    }
    const std::string suffix = " [" + std::to_string(prepared.coordinate.x) + "," +
        std::to_string(prepared.coordinate.z) + "]";
    std::vector<RuntimeMeshUploadDescription> uploads;
    uploads.reserve(uploadTerrain ? 4U : 3U);
    if (uploadTerrain)
    {
        uploads.push_back({prepared.terrain->vertices(), prepared.terrain->indices(),
            *terrainMaterial_, "terrain chunk mesh" + suffix});
    }
    uploads.push_back({{}, prepared.grass.indices,
        *grassMaterial_, "dense Bezier grass chunk mesh" + suffix, &prepared.grass});
    uploads.push_back({prepared.trunks.vertices, prepared.trunks.indices,
        *trunkMaterial_, "tree trunk chunk mesh" + suffix});
    uploads.push_back({prepared.canopies.vertices, prepared.canopies.indices,
        *canopyMaterial_, "tree canopy chunk mesh" + suffix});
    const std::vector<SceneMeshHandle> handles = renderer.uploadMeshBatch(uploads);

    if (uploadTerrain)
    {
        RuntimeTerrainChunk chunk;
        chunk.x = prepared.coordinate.x;
        chunk.z = prepared.coordinate.z;
        chunk.terrain = std::move(prepared.terrain);
        terrainChunks_.push_back(std::move(chunk));
        existing = std::prev(terrainChunks_.end());
    }
    std::size_t handle = 0;
    if (uploadTerrain)
    {
        existing->terrainMesh = handles[handle++];
        existing->terrainInstance = renderer.createMeshInstance(
            *existing->terrainMesh, glm::mat4(1.0f), "terrain chunk" + suffix);
    }
    existing->grassMesh = handles[handle++];
    existing->grassInstance = renderer.createMeshInstance(
        *existing->grassMesh, glm::mat4(1.0f), "dense grass chunk" + suffix);
    existing->trunkMesh = handles[handle++];
    existing->trunkInstance = renderer.createMeshInstance(
        *existing->trunkMesh, glm::mat4(1.0f), "tree trunk chunk" + suffix);
    existing->canopyMesh = handles[handle];
    existing->canopyInstance = renderer.createMeshInstance(
        *existing->canopyMesh, glm::mat4(1.0f), "tree canopy chunk" + suffix);
}

void GameWorld::publishInitialTerrain(
    VulkanRenderer& renderer, ChunkCoordinate center)
{
    std::vector<PreparedTerrainChunk> prepared;
    prepared.reserve(9U);
    prepared.push_back(prepareTerrainChunk(center, surfaceField_, true));
    for (std::int32_t z = center.z - 1; z <= center.z + 1; ++z)
    {
        for (std::int32_t x = center.x - 1; x <= center.x + 1; ++x)
        {
            if (x != center.x || z != center.z)
            {
                prepared.push_back(prepareTerrainChunk({x, z}, surfaceField_, false));
            }
        }
    }

    std::vector<RuntimeMeshUploadDescription> uploads;
    uploads.reserve(12U);
    for (PreparedTerrainChunk& chunk : prepared)
    {
        const std::string suffix = " [" + std::to_string(chunk.coordinate.x) + "," +
            std::to_string(chunk.coordinate.z) + "]";
        uploads.push_back({chunk.terrain->vertices(), chunk.terrain->indices(),
            *terrainMaterial_, "terrain chunk mesh" + suffix});
        if (chunk.foliagePrepared)
        {
            uploads.push_back({{}, chunk.grass.indices,
                *grassMaterial_, "dense Bezier grass chunk mesh" + suffix, &chunk.grass});
            uploads.push_back({chunk.trunks.vertices, chunk.trunks.indices,
                *trunkMaterial_, "tree trunk chunk mesh" + suffix});
            uploads.push_back({chunk.canopies.vertices, chunk.canopies.indices,
                *canopyMaterial_, "tree canopy chunk mesh" + suffix});
        }
    }
    const std::vector<SceneMeshHandle> handles = renderer.uploadMeshBatch(uploads);

    std::size_t handle = 0;
    for (PreparedTerrainChunk& source : prepared)
    {
        RuntimeTerrainChunk chunk;
        chunk.x = source.coordinate.x;
        chunk.z = source.coordinate.z;
        chunk.terrain = std::move(source.terrain);
        const std::string suffix = " [" + std::to_string(chunk.x) + "," +
            std::to_string(chunk.z) + "]";
        chunk.terrainMesh = handles[handle++];
        chunk.terrainInstance = renderer.createMeshInstance(
            *chunk.terrainMesh, glm::mat4(1.0f), "terrain chunk" + suffix);
        if (source.foliagePrepared)
        {
            chunk.grassMesh = handles[handle++];
            chunk.grassInstance = renderer.createMeshInstance(
                *chunk.grassMesh, glm::mat4(1.0f), "dense grass chunk" + suffix);
            chunk.trunkMesh = handles[handle++];
            chunk.trunkInstance = renderer.createMeshInstance(
                *chunk.trunkMesh, glm::mat4(1.0f), "tree trunk chunk" + suffix);
            chunk.canopyMesh = handles[handle++];
            chunk.canopyInstance = renderer.createMeshInstance(
                *chunk.canopyMesh, glm::mat4(1.0f), "tree canopy chunk" + suffix);
        }
        terrainChunks_.push_back(std::move(chunk));
    }
}

GameWorld::ChunkCoordinate GameWorld::requestedStreamCenter() const noexcept
{
    return requestedTerrainStreamCenter(
        {ninja_.position().x, ninja_.position().z},
        {ninja_.velocity().x, ninja_.velocity().z}, streamingConfig());
}

bool GameWorld::windowComplete(ChunkCoordinate center) const noexcept
{
    for (std::int32_t z = center.z - terrainStreamRadius;
         z <= center.z + terrainStreamRadius; ++z)
    {
        for (std::int32_t x = center.x - terrainStreamRadius;
             x <= center.x + terrainStreamRadius; ++x)
        {
            const auto chunk = std::find_if(terrainChunks_.begin(), terrainChunks_.end(),
                [&](const RuntimeTerrainChunk& candidate)
                {
                    return candidate.x == x && candidate.z == z;
                });
            if (chunk == terrainChunks_.end() || !chunk->complete())
            {
                return false;
            }
        }
    }
    return true;
}

void GameWorld::requestStreamWindow(VulkanRenderer& renderer, ChunkCoordinate center)
{
    requestedCenter_ = center;
    requestedCenterValid_ = true;
    ++streamRequestEpoch_;
    rebuildPendingChunks(center);

    // An abandoned directional prefetch may have published chunks that belong to neither the
    // committed window nor the new request. They can be retired immediately without exposing a
    // hole around the player.
    for (auto chunk = terrainChunks_.begin(); chunk != terrainChunks_.end();)
    {
        const ChunkCoordinate coordinate{chunk->x, chunk->z};
        const bool inCommittedWindow = streamCenterValid_ && terrainChunkInWindow(
            coordinate, streamCenter_, terrainStreamRadius);
        const bool inRequestedWindow = terrainChunkInWindow(
            coordinate, requestedCenter_, terrainStreamRadius);
        if (!inCommittedWindow && !inRequestedWindow)
        {
            destroyTerrainChunk(renderer, *chunk);
            chunk = terrainChunks_.erase(chunk);
        }
        else
        {
            ++chunk;
        }
    }
}

void GameWorld::rebuildPendingChunks(ChunkCoordinate center)
{
    std::vector<ChunkCoordinate> missing;
    for (std::int32_t z = center.z - terrainStreamRadius;
         z <= center.z + terrainStreamRadius; ++z)
    {
        for (std::int32_t x = center.x - terrainStreamRadius;
             x <= center.x + terrainStreamRadius; ++x)
        {
            const ChunkCoordinate coordinate{x, z};
            const auto chunk = std::find_if(terrainChunks_.begin(), terrainChunks_.end(),
                [&](const RuntimeTerrainChunk& candidate)
                {
                    return candidate.x == coordinate.x && candidate.z == coordinate.z;
                });
            if (chunk == terrainChunks_.end() || !chunk->complete())
            {
                missing.push_back(coordinate);
            }
        }
    }
    std::ranges::sort(missing,
        [center](ChunkCoordinate left, ChunkCoordinate right)
        {
            const std::int32_t leftDistance =
                std::abs(left.x - center.x) + std::abs(left.z - center.z);
            const std::int32_t rightDistance =
                std::abs(right.x - center.x) + std::abs(right.z - center.z);
            return leftDistance < rightDistance;
        });
    if (chunkWorkers_)
    {
        chunkWorkers_->replace(streamRequestEpoch_, std::move(missing));
    }
}

void GameWorld::retireChunksOutsideWindow(
    VulkanRenderer& renderer, ChunkCoordinate center)
{
    for (auto chunk = terrainChunks_.begin(); chunk != terrainChunks_.end();)
    {
        if (!terrainChunkInWindow({chunk->x, chunk->z}, center, terrainStreamRadius))
        {
            destroyTerrainChunk(renderer, *chunk);
            chunk = terrainChunks_.erase(chunk);
        }
        else
        {
            ++chunk;
        }
    }
}

void GameWorld::destroyTerrainChunk(
    VulkanRenderer& renderer, RuntimeTerrainChunk& chunk)
{
    const std::array instances{&chunk.canopyInstance, &chunk.trunkInstance,
        &chunk.grassInstance, &chunk.terrainInstance};
    for (std::optional<SceneInstanceHandle>* instance : instances)
    {
        if (*instance)
        {
            renderer.destroyInstance(**instance);
            instance->reset();
        }
    }
    const std::array meshes{&chunk.canopyMesh, &chunk.trunkMesh,
        &chunk.grassMesh, &chunk.terrainMesh};
    for (std::optional<SceneMeshHandle>* mesh : meshes)
    {
        if (*mesh)
        {
            renderer.destroyMesh(**mesh);
            mesh->reset();
        }
    }
}

void GameWorld::setEnvironment(const SceneEnvironment& environment) noexcept
{
    submission_.environment = environment;
}

void GameWorld::setKeyLightIntensity(float intensity) noexcept
{
    if (!submission_.pointLights.empty())
    {
        submission_.pointLights.front().intensity = intensity;
    }
}

TerrainStreamingDiagnostics GameWorld::streamingDiagnostics() const noexcept
{
    TerrainStreamingDiagnostics result;
    result.characterPosition = ninja_.position();
    result.playerChunk = terrainChunkAt(
        {ninja_.position().x, ninja_.position().z}, streamingConfig());
    result.committedCenter = streamCenter_;
    result.requestedCenter = requestedCenter_;
    result.residentChunkCount = static_cast<std::uint32_t>(terrainChunks_.size());
    result.completeChunkCount = static_cast<std::uint32_t>(std::ranges::count_if(
        terrainChunks_, [](const RuntimeTerrainChunk& chunk) { return chunk.complete(); }));
    result.committedCenterValid = streamCenterValid_;
    result.requestedCenterValid = requestedCenterValid_;
    result.requestedWindowComplete = requestedCenterValid_ && windowComplete(requestedCenter_);
    return result;
}

void GameWorld::updateCameraSubmission()
{
    submission_.view = camera_.view();
    submission_.projection = camera_.projection();
    submission_.cameraPosition = camera_.position();
    submission_.vegetationInteractorPositionRadius =
        glm::vec4(ninja_.position(), ninjaGrassInteractionRadius);
    submission_.animationActorTransforms.clear();
    if (ninjaActor_)
    {
        submission_.animationActorTransforms.push_back(
            {*ninjaActor_, ninja_.worldOffset()});
    }
}

}
