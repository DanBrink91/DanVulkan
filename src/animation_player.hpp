#pragma once

#include <danvulkan/assets.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace danvulkan
{
// CPU clip evaluation and hierarchy propagation. Vulkan resources consume only the resulting
// world transforms and joint matrices, keeping animation authoring data out of the GPU layer.
class AnimationPlayer
{
public:
    enum class Status
    {
        stopped,
        playing,
        paused,
        finished
    };

    static constexpr std::size_t invalidClip = std::numeric_limits<std::size_t>::max();

    struct NodePose
    {
        glm::vec3 translation{};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 scale{1.0f};
    };

    struct EvaluationTimings
    {
        double samplingMilliseconds = 0.0;
        double poseResetMilliseconds = 0.0;
        double timelineResolutionMilliseconds = 0.0;
        double vectorSamplingMilliseconds = 0.0;
        double rotationSamplingMilliseconds = 0.0;
        double transformPropagationMilliseconds = 0.0;
        std::uint32_t evaluatedInstances = 0;
        std::uint32_t sampledChannels = 0;
        std::uint32_t sampledVectorChannels = 0;
        std::uint32_t sampledRotationChannels = 0;
        std::uint32_t nlerpRotationChannels = 0;
        std::uint32_t propagatedNodes = 0;
        std::uint32_t poseComposedNodes = 0;
        std::uint32_t cachedLocalNodes = 0;
    };

    struct SamplingSettings
    {
        // Zero retains exact spherical interpolation. Positive values allow normalized linear
        // interpolation when the angular distance between adjacent keys does not exceed this
        // threshold. The value is expressed in radians and may not exceed pi.
        float adaptiveNlerpMaxAngleRadians = 0.0f;
    };

    struct InstanceUpdatePolicy
    {
        bool evaluate = true;
        // Zero evaluates every update. A positive interval retains the last pose until enough
        // playback time has elapsed, while the playback clock itself continues advancing.
        float minimumEvaluationIntervalSeconds = 0.0f;
    };

    explicit AnimationPlayer(const assets::SceneAsset& scene);
    AnimationPlayer(const assets::SceneAsset& scene, SamplingSettings settings);

    [[nodiscard]] bool animated() const noexcept;
    [[nodiscard]] std::size_t clipCount() const noexcept;
    [[nodiscard]] std::size_t instanceCount() const noexcept { return playback_.size(); }
    [[nodiscard]] std::string_view clipName(std::size_t clip) const;
    [[nodiscard]] float clipDuration(std::size_t clip) const;
    [[nodiscard]] std::size_t compiledTimelineCount(std::size_t clip) const;
    [[nodiscard]] std::size_t compiledChannelCount(std::size_t clip) const;
    [[nodiscard]] std::size_t foldedConstantChannelCount(std::size_t clip) const;
    [[nodiscard]] std::size_t currentClip() const noexcept { return currentClip_; }
    [[nodiscard]] Status status() const noexcept;
    [[nodiscard]] bool looping() const noexcept;
    [[nodiscard]] float playbackSpeed() const noexcept;
    [[nodiscard]] float position() const noexcept;
    [[nodiscard]] float instancePosition(std::size_t instance) const;
    [[nodiscard]] bool instanceEvaluated(std::size_t instance) const;
    [[nodiscard]] std::optional<std::size_t> instanceForNode(
        assets::NodeHandle node) const;
    [[nodiscard]] std::optional<std::size_t> instanceForSkin(
        assets::SkinHandle skin) const;
    [[nodiscard]] EvaluationTimings evaluationTimings() const noexcept
    {
        return evaluationTimings_;
    }

    void play(std::size_t clip, bool restart = true);
    void pause() noexcept;
    void resume() noexcept;
    void stop();
    void seek(float seconds);
    void setLooping(bool looping) noexcept;
    void setPlaybackSpeed(float speed);
    void update(float deltaSeconds);
    void update(float deltaSeconds, std::span<const InstanceUpdatePolicy> policies);

    [[nodiscard]] const glm::mat4& worldTransform(assets::NodeHandle node) const;
    // Emits world-space skinning matrices. Every mesh node using the same skin can share this
    // palette; the vertex shader must not apply the mesh model transform a second time.
    void appendSkinMatrices(
        assets::SkinHandle skin, std::vector<glm::mat4>& destination) const;
    void writeSkinMatrices(assets::SkinHandle skin, std::span<glm::mat4> destination) const;

private:
    struct NodeDefinition
    {
        glm::mat4 baseLocal{1.0f};
        NodePose basePose;
        bool transformIsTrs = false;
        bool baseLocalIsAffine = true;
        std::vector<assets::NodeHandle> children;
    };

    struct NodeState
    {
        NodePose pose;
        glm::mat4 world{1.0f};
    };

    struct PropagationEntry
    {
        enum class LocalSource : std::uint8_t
        {
            pose,
            cachedAffine,
            cachedGeneral
        };

        std::uint32_t node = 0;
        std::uint32_t parent = std::numeric_limits<std::uint32_t>::max();
        LocalSource localSource = LocalSource::cachedAffine;
    };

    struct PropagationPlan
    {
        std::vector<PropagationEntry> entries;
        std::uint32_t poseComposedNodes = 0;
    };

    struct PropagationRange
    {
        std::size_t offset = 0;
        std::size_t count = 0;
        std::uint32_t poseComposedNodes = 0;
    };

    struct InstanceDefinition
    {
        std::size_t clip = invalidClip;
        std::array<std::vector<std::size_t>, 6> channelTargets;
        std::vector<std::size_t> constantChannelTargets;
        std::vector<std::size_t> animatedNodes;
        PropagationRange propagationRange;
        std::size_t evaluationRoot = invalidClip;
        bool requiresFullPropagation = false;
        float initialPosition = 0.0f;
        float playbackSpeed = 1.0f;
        bool looping = true;
    };

    struct TimelineDefinition
    {
        std::vector<float> times;
        std::vector<float> inverseDurations;
        bool uniform = false;
        float uniformStepInverse = 0.0f;
    };

    struct ChannelDefinition
    {
        std::size_t sourceTarget = 0;
        std::size_t timeline = 0;
        std::size_t valueOffset = 0;
        std::size_t valueCount = 0;
    };

    struct ConstantChannelDefinition
    {
        std::size_t sourceTarget = 0;
        glm::vec4 value{};
        assets::AnimationTarget path = assets::AnimationTarget::translation;
    };

    struct ClipDefinition
    {
        std::string name;
        float startTime = 0.0f;
        float endTime = 0.0f;
        std::vector<TimelineDefinition> timelines;
        // Translation-linear, translation-step, scale-linear, scale-step, rotation-linear,
        // rotation-step. Keeping these paths separate removes type/interpolation branches from
        // the per-channel evaluation loops.
        std::array<std::vector<ChannelDefinition>, 6> channelGroups;
        std::vector<ConstantChannelDefinition> constantChannels;
        std::vector<glm::vec4> values;
    };

    struct TimelineSample
    {
        std::size_t previous = 0;
        std::size_t next = 0;
        float alpha = 0.0f;
    };

    struct Definition
    {
        std::vector<NodeDefinition> nodes;
        std::vector<assets::NodeHandle> roots;
        std::vector<assets::SkinAsset> skins;
        std::vector<ClipDefinition> clips;
        std::vector<InstanceDefinition> instances;
        std::vector<std::size_t> parents;
        std::vector<std::optional<std::size_t>> nodeInstances;
        std::vector<std::optional<std::size_t>> skinInstances;
        PropagationPlan fullPropagationPlan;
        bool explicitInstances = false;
    };

    struct PlaybackState
    {
        std::size_t definition = 0;
        Status status = Status::playing;
        float position = 0.0f;
        float playbackSpeed = 1.0f;
        bool looping = true;
        float unevaluatedSeconds = 0.0f;
        bool evaluationEnabledLastUpdate = true;
        std::vector<std::size_t> timelineCursors;
        std::vector<TimelineSample> timelineSamples;
    };

    [[nodiscard]] std::size_t nodeIndex(assets::NodeHandle node) const;
    [[nodiscard]] const ClipDefinition& clip(std::size_t index) const;
    [[nodiscard]] const assets::SkinAsset& skin(assets::SkinHandle handle) const;
    [[nodiscard]] const PlaybackState* selectedPlayback() const noexcept;
    [[nodiscard]] PlaybackState* selectedPlayback() noexcept;
    void evaluatePose();
    void evaluateInstances(std::span<const std::size_t> instances, bool resetAllNodes);
    void applyConstantChannels(std::size_t instanceIndex);
    [[nodiscard]] TimelineSample sampleTimeline(
        const TimelineDefinition& timeline, float time, std::size_t& cursor) const;
    [[nodiscard]] glm::vec4 sampleVector(const ClipDefinition& clip,
        const ChannelDefinition& channel, const TimelineSample& timelineSample,
        bool step) const;
    [[nodiscard]] glm::quat sampleRotation(const ClipDefinition& clip,
        const ChannelDefinition& channel, const TimelineSample& timelineSample,
        bool step, bool& usedNlerp) const;
    void propagate(std::span<const PropagationEntry> entries,
        std::uint32_t poseComposedNodes);

    std::shared_ptr<const Definition> definition_;
    std::vector<NodeState> nodes_;
    std::vector<PlaybackState> playback_;
    std::vector<std::uint8_t> evaluatedInstances_;
    std::vector<std::size_t> dueInstancesScratch_;
    std::size_t currentClip_ = invalidClip;
    float adaptiveNlerpDotThreshold_ = 1.0f;
    bool adaptiveNlerpEnabled_ = false;
    EvaluationTimings evaluationTimings_;
};
}
