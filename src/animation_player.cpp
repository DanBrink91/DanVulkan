#define GLM_ENABLE_EXPERIMENTAL

#include "animation_player.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace danvulkan
{
namespace
{
constexpr std::size_t translationLinearGroup = 0;
constexpr std::size_t translationStepGroup = 1;
constexpr std::size_t scaleLinearGroup = 2;
constexpr std::size_t scaleStepGroup = 3;
constexpr std::size_t rotationLinearGroup = 4;
constexpr std::size_t rotationStepGroup = 5;

glm::mat4 composeLocal(const AnimationPlayer::NodePose& pose)
{
    const glm::mat3 rotation = glm::mat3_cast(pose.rotation);
    return {
        glm::vec4(rotation[0] * pose.scale.x, 0.0f),
        glm::vec4(rotation[1] * pose.scale.y, 0.0f),
        glm::vec4(rotation[2] * pose.scale.z, 0.0f),
        glm::vec4(pose.translation, 1.0f)
    };
}

glm::mat4 composeWorld(const glm::mat4& parent, const AnimationPlayer::NodePose& pose)
{
    const glm::mat3 rotation = glm::mat3_cast(pose.rotation);
    return {
        parent[0] * (rotation[0].x * pose.scale.x) +
            parent[1] * (rotation[0].y * pose.scale.x) +
            parent[2] * (rotation[0].z * pose.scale.x),
        parent[0] * (rotation[1].x * pose.scale.y) +
            parent[1] * (rotation[1].y * pose.scale.y) +
            parent[2] * (rotation[1].z * pose.scale.y),
        parent[0] * (rotation[2].x * pose.scale.z) +
            parent[1] * (rotation[2].y * pose.scale.z) +
            parent[2] * (rotation[2].z * pose.scale.z),
        parent[0] * pose.translation.x + parent[1] * pose.translation.y +
            parent[2] * pose.translation.z + parent[3]
    };
}

glm::mat4 multiplyAffineLocal(const glm::mat4& parent, const glm::mat4& local)
{
    return {
        parent[0] * local[0].x + parent[1] * local[0].y + parent[2] * local[0].z,
        parent[0] * local[1].x + parent[1] * local[1].y + parent[2] * local[1].z,
        parent[0] * local[2].x + parent[1] * local[2].y + parent[2] * local[2].z,
        parent[0] * local[3].x + parent[1] * local[3].y + parent[2] * local[3].z + parent[3]
    };
}

bool affine(const glm::mat4& transform) noexcept
{
    return transform[0].w == 0.0f && transform[1].w == 0.0f &&
        transform[2].w == 0.0f && transform[3].w == 1.0f;
}

std::size_t timelineHash(const std::vector<float>& times) noexcept
{
    std::size_t result = times.size();
    for (const float time : times)
    {
        result ^= static_cast<std::size_t>(std::bit_cast<std::uint32_t>(time)) +
            0x9e3779b9U + (result << 6U) + (result >> 2U);
    }
    return result;
}

bool uniformlySampled(const std::vector<float>& times) noexcept
{
    if (times.size() < 2)
    {
        return false;
    }
    const float step = times[1] - times[0];
    if (!(step > 0.0f))
    {
        return false;
    }
    const float tolerance = std::max(std::abs(step) * 1.0e-4f, 1.0e-6f);
    for (std::size_t index = 2; index < times.size(); ++index)
    {
        if (std::abs((times[index] - times[index - 1]) - step) > tolerance)
        {
            return false;
        }
    }
    return true;
}

bool exactlyEqual(const glm::vec4& left, const glm::vec4& right) noexcept
{
    return left.x == right.x && left.y == right.y && left.z == right.z &&
        left.w == right.w;
}

std::size_t channelGroup(
    assets::AnimationTarget path, assets::AnimationInterpolation interpolation) noexcept
{
    const bool step = interpolation == assets::AnimationInterpolation::step;
    switch (path)
    {
    case assets::AnimationTarget::translation:
        return step ? translationStepGroup : translationLinearGroup;
    case assets::AnimationTarget::scale:
        return step ? scaleStepGroup : scaleLinearGroup;
    case assets::AnimationTarget::rotation:
        return step ? rotationStepGroup : rotationLinearGroup;
    }
    return translationLinearGroup;
}
}

AnimationPlayer::AnimationPlayer(const assets::SceneAsset& scene)
    : AnimationPlayer(scene, SamplingSettings{})
{
}

AnimationPlayer::AnimationPlayer(const assets::SceneAsset& scene, SamplingSettings settings)
{
    if (!std::isfinite(settings.adaptiveNlerpMaxAngleRadians) ||
        settings.adaptiveNlerpMaxAngleRadians < 0.0f ||
        settings.adaptiveNlerpMaxAngleRadians > std::numbers::pi_v<float>)
    {
        throw std::invalid_argument("adaptive nlerp angle must be between zero and pi radians");
    }
    adaptiveNlerpEnabled_ = settings.adaptiveNlerpMaxAngleRadians > 0.0f;
    adaptiveNlerpDotThreshold_ = adaptiveNlerpEnabled_ ?
        std::cos(settings.adaptiveNlerpMaxAngleRadians * 0.5f) : 1.0f;

    auto definition = std::make_shared<Definition>();
    definition->roots.assign(scene.rootNodes().begin(), scene.rootNodes().end());
    definition->skins.assign(scene.skins().begin(), scene.skins().end());
    definition->explicitInstances = !scene.animationInstances().empty();
    definition->nodes.reserve(scene.nodes().size());
    for (const assets::NodeAsset& node : scene.nodes())
    {
        NodeDefinition state;
        state.basePose.translation = node.translation;
        state.basePose.rotation = glm::normalize(glm::quat(
            node.rotation.w, node.rotation.x, node.rotation.y, node.rotation.z));
        state.basePose.scale = node.scale;
        state.transformIsTrs = node.transformIsTrs;
        state.baseLocal = state.transformIsTrs ? composeLocal(state.basePose) :
            node.localTransform;
        state.baseLocalIsAffine = affine(state.baseLocal);
        state.children = node.children;
        definition->nodes.push_back(std::move(state));
    }
    definition->parents.assign(definition->nodes.size(), invalidClip);
    for (std::size_t parent = 0; parent < definition->nodes.size(); ++parent)
    {
        for (const assets::NodeHandle child : definition->nodes[parent].children)
        {
            if (!child || child.generation != 1U || child.slot >= definition->nodes.size() ||
                definition->parents[child.slot] != invalidClip)
            {
                throw std::runtime_error("animation hierarchy contains an invalid child");
            }
            definition->parents[child.slot] = parent;
        }
    }
    const auto definitionNodeIndex = [&](assets::NodeHandle node)
    {
        if (!node || node.generation != 1U || node.slot >= definition->nodes.size())
        {
            throw std::out_of_range("animation references an invalid node handle");
        }
        return static_cast<std::size_t>(node.slot);
    };

    definition->clips.reserve(scene.animations().size());
    for (const assets::AnimationClipAsset& sourceClip : scene.animations())
    {
        ClipDefinition compiled;
        compiled.name = sourceClip.name;
        compiled.startTime = sourceClip.startTime;
        compiled.endTime = sourceClip.endTime;
        std::unordered_map<std::size_t, std::vector<std::size_t>> timelineCandidates;
        for (const assets::AnimationChannelAsset& sourceChannel : sourceClip.channels)
        {
            const std::size_t sourceTarget = definitionNodeIndex(sourceChannel.target);
            if (!definition->nodes[sourceTarget].transformIsTrs)
            {
                throw std::runtime_error("animation targets a matrix-authored node");
            }
            std::vector<glm::vec4> values = sourceChannel.values;
            if (sourceChannel.path == assets::AnimationTarget::rotation)
            {
                for (glm::vec4& value : values)
                {
                    const glm::quat normalized = glm::normalize(
                        glm::quat(value.w, value.x, value.y, value.z));
                    value = {normalized.x, normalized.y, normalized.z, normalized.w};
                }
                for (std::size_t index = 1; index < values.size(); ++index)
                {
                    if (glm::dot(values[index - 1], values[index]) < 0.0f)
                    {
                        values[index] = -values[index];
                    }
                }
            }
            const bool constant = !values.empty() && std::ranges::all_of(values,
                [&](const glm::vec4& value) { return exactlyEqual(value, values.front()); });
            if (constant)
            {
                compiled.constantChannels.push_back(
                    {sourceTarget, values.front(), sourceChannel.path});
                continue;
            }

            const std::size_t hash = timelineHash(sourceChannel.times);
            std::size_t timelineIndex = invalidClip;
            for (const std::size_t candidate : timelineCandidates[hash])
            {
                if (compiled.timelines[candidate].times == sourceChannel.times)
                {
                    timelineIndex = candidate;
                    break;
                }
            }
            if (timelineIndex == invalidClip)
            {
                timelineIndex = compiled.timelines.size();
                TimelineDefinition timeline;
                timeline.times = sourceChannel.times;
                timeline.uniform = uniformlySampled(timeline.times);
                if (timeline.uniform)
                {
                    timeline.uniformStepInverse = 1.0f /
                        (timeline.times[1] - timeline.times[0]);
                }
                timeline.inverseDurations.reserve(timeline.times.size() - 1U);
                for (std::size_t index = 1; index < timeline.times.size(); ++index)
                {
                    const float duration = timeline.times[index] - timeline.times[index - 1];
                    timeline.inverseDurations.push_back(
                        duration > 0.0f ? 1.0f / duration : 0.0f);
                }
                compiled.timelines.push_back(std::move(timeline));
                timelineCandidates[hash].push_back(timelineIndex);
            }

            ChannelDefinition channel;
            channel.sourceTarget = sourceTarget;
            channel.timeline = timelineIndex;
            channel.valueOffset = compiled.values.size();
            channel.valueCount = values.size();
            compiled.values.insert(compiled.values.end(), values.begin(), values.end());
            compiled.channelGroups[channelGroup(
                sourceChannel.path, sourceChannel.interpolation)].push_back(channel);
        }
        definition->clips.push_back(std::move(compiled));
    }

    const auto populateInstanceTargets = [&](InstanceDefinition& instance,
        const ClipDefinition& selected, const auto& resolveTarget)
    {
        for (std::size_t group = 0; group < selected.channelGroups.size(); ++group)
        {
            std::vector<std::size_t>& targets = instance.channelTargets[group];
            targets.reserve(selected.channelGroups[group].size());
            for (const ChannelDefinition& channel : selected.channelGroups[group])
            {
                const std::size_t target = resolveTarget(channel.sourceTarget);
                if (!definition->nodes[target].transformIsTrs)
                {
                    throw std::runtime_error("animation targets a matrix-authored node");
                }
                targets.push_back(target);
            }
        }
        instance.constantChannelTargets.reserve(selected.constantChannels.size());
        for (const ConstantChannelDefinition& channel : selected.constantChannels)
        {
            const std::size_t target = resolveTarget(channel.sourceTarget);
            if (!definition->nodes[target].transformIsTrs)
            {
                throw std::runtime_error("animation targets a matrix-authored node");
            }
            instance.constantChannelTargets.push_back(target);
        }
    };
    const auto identityInstance = [&](std::size_t clipIndex)
    {
        InstanceDefinition instance;
        instance.clip = clipIndex;
        const ClipDefinition& selected = definition->clips.at(clipIndex);
        populateInstanceTargets(instance, selected,
            [](std::size_t sourceTarget) { return sourceTarget; });
        return instance;
    };
    if (definition->explicitInstances)
    {
        definition->instances.reserve(scene.animationInstances().size());
        for (const assets::AnimationInstanceAsset& source : scene.animationInstances())
        {
            if (!source.clip || source.clip.generation != 1U ||
                source.clip.slot >= definition->clips.size() ||
                !std::isfinite(source.initialPositionSeconds) ||
                source.initialPositionSeconds < 0.0f || !std::isfinite(source.playbackSpeed) ||
                source.playbackSpeed <= 0.0f)
            {
                throw std::runtime_error("scene contains an invalid animation instance");
            }
            std::unordered_map<std::uint32_t, std::size_t> targets;
            targets.reserve(source.nodeBindings.size());
            for (const assets::AnimationNodeBindingAsset& binding : source.nodeBindings)
            {
                const std::size_t sourceIndex = definitionNodeIndex(binding.source);
                const std::size_t targetIndex = definitionNodeIndex(binding.target);
                if (!targets.emplace(static_cast<std::uint32_t>(sourceIndex), targetIndex).second)
                {
                    throw std::runtime_error(
                        "animation instance contains duplicate source-node bindings");
                }
            }
            InstanceDefinition instance;
            instance.clip = source.clip.slot;
            instance.initialPosition = source.initialPositionSeconds;
            instance.playbackSpeed = source.playbackSpeed;
            instance.looping = source.looping;
            const ClipDefinition& selected = definition->clips[instance.clip];
            populateInstanceTargets(instance, selected, [&](std::size_t sourceTarget)
            {
                const auto target = targets.find(static_cast<std::uint32_t>(sourceTarget));
                if (target == targets.end())
                {
                    throw std::runtime_error(
                        "animation instance does not bind every clip target node");
                }
                return target->second;
            });
            definition->instances.push_back(std::move(instance));
        }
    }
    else
    {
        definition->instances.reserve(definition->clips.size());
        for (std::size_t clipIndex = 0; clipIndex < definition->clips.size(); ++clipIndex)
        {
            definition->instances.push_back(identityInstance(clipIndex));
        }
    }

    definition->nodeInstances.resize(definition->nodes.size());
    definition->skinInstances.resize(definition->skins.size());
    const auto assignNode = [&](std::size_t node, std::size_t instance)
    {
        std::optional<std::size_t>& owner = definition->nodeInstances.at(node);
        if (owner && *owner != instance)
        {
            throw std::runtime_error("animation instances contain overlapping node bindings");
        }
        owner = instance;
    };
    if (definition->explicitInstances)
    {
        for (std::size_t instanceIndex = 0;
             instanceIndex < scene.animationInstances().size(); ++instanceIndex)
        {
            for (const assets::AnimationNodeBindingAsset& binding :
                 scene.animationInstances()[instanceIndex].nodeBindings)
            {
                assignNode(definitionNodeIndex(binding.target), instanceIndex);
            }
        }
    }
    else if (!definition->instances.empty())
    {
        for (const InstanceDefinition& instance : definition->instances)
        {
            for (const std::vector<std::size_t>& targets : instance.channelTargets)
            {
                for (const std::size_t target : targets)
                {
                    assignNode(target, 0);
                }
            }
            for (const std::size_t target : instance.constantChannelTargets)
            {
                assignNode(target, 0);
            }
        }
    }

    for (std::size_t instanceIndex = 0; instanceIndex < definition->instances.size();
         ++instanceIndex)
    {
        InstanceDefinition& instance = definition->instances[instanceIndex];
        for (const std::vector<std::size_t>& targets : instance.channelTargets)
        {
            instance.animatedNodes.insert(
                instance.animatedNodes.end(), targets.begin(), targets.end());
        }
        instance.animatedNodes.insert(instance.animatedNodes.end(),
            instance.constantChannelTargets.begin(), instance.constantChannelTargets.end());
        std::ranges::sort(instance.animatedNodes);
        const auto uniqueEnd = std::ranges::unique(instance.animatedNodes).begin();
        instance.animatedNodes.erase(uniqueEnd, instance.animatedNodes.end());
        if (!instance.animatedNodes.empty())
        {
            std::unordered_set<std::size_t> ancestors;
            for (std::size_t node = instance.animatedNodes.front(); node != invalidClip;
                 node = definition->parents[node])
            {
                ancestors.insert(node);
            }
            instance.evaluationRoot = instance.animatedNodes.front();
            for (const std::size_t target : instance.animatedNodes)
            {
                std::size_t node = target;
                while (node != invalidClip && !ancestors.contains(node))
                {
                    node = definition->parents[node];
                }
                if (node == invalidClip)
                {
                    instance.evaluationRoot = invalidClip;
                    break;
                }
                instance.evaluationRoot = node;
                ancestors.clear();
                for (; node != invalidClip; node = definition->parents[node])
                {
                    ancestors.insert(node);
                }
            }
        }
    }

    std::vector<std::uint8_t> poseDrivenNodes(definition->nodes.size(), 0U);
    for (const InstanceDefinition& instance : definition->instances)
    {
        for (const std::size_t node : instance.animatedNodes)
        {
            poseDrivenNodes[node] = 1U;
        }
    }
    const auto appendSubtree = [&](std::size_t root, PropagationPlan& plan)
    {
        std::vector<std::size_t> pending{root};
        while (!pending.empty())
        {
            const std::size_t node = pending.back();
            pending.pop_back();
            const std::size_t parent = definition->parents[node];
            const bool poseDriven = poseDrivenNodes[node] != 0U;
            plan.entries.push_back({static_cast<std::uint32_t>(node),
                parent == invalidClip ? std::numeric_limits<std::uint32_t>::max() :
                    static_cast<std::uint32_t>(parent),
                poseDriven ? PropagationEntry::LocalSource::pose :
                    (definition->nodes[node].baseLocalIsAffine ?
                        PropagationEntry::LocalSource::cachedAffine :
                        PropagationEntry::LocalSource::cachedGeneral)});
            plan.poseComposedNodes += poseDriven ? 1U : 0U;
            const std::vector<assets::NodeHandle>& children = definition->nodes[node].children;
            for (auto child = children.rbegin(); child != children.rend(); ++child)
            {
                pending.push_back(child->slot);
            }
        }
    };
    definition->fullPropagationPlan.entries.reserve(definition->nodes.size());
    for (const assets::NodeHandle root : definition->roots)
    {
        appendSubtree(definitionNodeIndex(root), definition->fullPropagationPlan);
    }
    std::vector<std::size_t> propagationOffsets(definition->nodes.size(), invalidClip);
    std::vector<std::size_t> subtreeSizes(definition->nodes.size(), 0U);
    for (std::size_t index = 0;
         index < definition->fullPropagationPlan.entries.size(); ++index)
    {
        const std::size_t node = definition->fullPropagationPlan.entries[index].node;
        if (propagationOffsets[node] != invalidClip)
        {
            throw std::runtime_error("animation hierarchy contains a duplicate root");
        }
        propagationOffsets[node] = index;
    }
    for (auto entry = definition->fullPropagationPlan.entries.rbegin();
         entry != definition->fullPropagationPlan.entries.rend(); ++entry)
    {
        std::size_t size = 1U;
        for (const assets::NodeHandle child : definition->nodes[entry->node].children)
        {
            size += subtreeSizes[child.slot];
        }
        subtreeSizes[entry->node] = size;
    }
    std::vector<std::uint32_t> poseCountPrefix(
        definition->fullPropagationPlan.entries.size() + 1U, 0U);
    for (std::size_t index = 0;
         index < definition->fullPropagationPlan.entries.size(); ++index)
    {
        poseCountPrefix[index + 1U] = poseCountPrefix[index] +
            (definition->fullPropagationPlan.entries[index].localSource ==
                PropagationEntry::LocalSource::pose ? 1U : 0U);
    }
    for (InstanceDefinition& instance : definition->instances)
    {
        if (instance.animatedNodes.empty())
        {
            continue;
        }
        instance.requiresFullPropagation = instance.evaluationRoot == invalidClip;
        if (!instance.requiresFullPropagation)
        {
            const std::size_t offset = propagationOffsets[instance.evaluationRoot];
            if (offset == invalidClip)
            {
                throw std::runtime_error(
                    "animation target is outside the active scene hierarchy");
            }
            const std::size_t count = subtreeSizes[instance.evaluationRoot];
            instance.propagationRange = {offset, count,
                poseCountPrefix[offset + count] - poseCountPrefix[offset]};
        }
        instance.animatedNodes.clear();
        instance.animatedNodes.shrink_to_fit();
    }
    for (std::size_t skinIndex = 0; skinIndex < definition->skins.size(); ++skinIndex)
    {
        for (const assets::NodeHandle joint : definition->skins[skinIndex].joints)
        {
            const std::optional<std::size_t> owner = definition->nodeInstances[joint.slot];
            if (!owner)
            {
                continue;
            }
            if (definition->skinInstances[skinIndex] &&
                definition->skinInstances[skinIndex] != owner)
            {
                throw std::runtime_error("skin joints span multiple animation instances");
            }
            definition->skinInstances[skinIndex] = owner;
        }
    }

    definition_ = std::move(definition);
    nodes_.resize(definition_->nodes.size());
    evaluatedInstances_.resize(definition_->explicitInstances ?
        definition_->instances.size() : (definition_->clips.empty() ? 0U : 1U));
    if (definition_->explicitInstances)
    {
        playback_.reserve(definition_->instances.size());
        for (std::size_t index = 0; index < definition_->instances.size(); ++index)
        {
            const InstanceDefinition& instance = definition_->instances[index];
            const float duration = clipDuration(instance.clip);
            const float position = duration > 0.0f && instance.looping
                ? std::fmod(instance.initialPosition, duration)
                : std::min(instance.initialPosition, duration);
            PlaybackState state;
            state.definition = index;
            state.status = Status::playing;
            state.position = position;
            state.playbackSpeed = instance.playbackSpeed;
            state.looping = instance.looping;
            state.timelineCursors.resize(definition_->clips[instance.clip].timelines.size());
            state.timelineSamples.resize(definition_->clips[instance.clip].timelines.size());
            playback_.push_back(std::move(state));
        }
        currentClip_ = playback_.empty() ? invalidClip :
            definition_->instances[playback_.front().definition].clip;
    }
    else if (!definition_->clips.empty())
    {
        PlaybackState state;
        state.timelineCursors.resize(definition_->clips.front().timelines.size());
        state.timelineSamples.resize(definition_->clips.front().timelines.size());
        playback_.push_back(std::move(state));
        currentClip_ = 0;
    }
    dueInstancesScratch_.reserve(playback_.size());
    evaluatePose();
}

bool AnimationPlayer::animated() const noexcept
{
    return definition_ && !definition_->clips.empty();
}

std::size_t AnimationPlayer::clipCount() const noexcept
{
    return definition_ ? definition_->clips.size() : 0;
}

std::string_view AnimationPlayer::clipName(std::size_t clipIndex) const
{
    return clip(clipIndex).name;
}

float AnimationPlayer::clipDuration(std::size_t clipIndex) const
{
    const ClipDefinition& selected = clip(clipIndex);
    return std::max(selected.endTime - selected.startTime, 0.0f);
}

std::size_t AnimationPlayer::compiledTimelineCount(std::size_t clipIndex) const
{
    return clip(clipIndex).timelines.size();
}

std::size_t AnimationPlayer::compiledChannelCount(std::size_t clipIndex) const
{
    const ClipDefinition& selected = clip(clipIndex);
    std::size_t result = 0;
    for (const std::vector<ChannelDefinition>& channels : selected.channelGroups)
    {
        result += channels.size();
    }
    return result;
}

std::size_t AnimationPlayer::foldedConstantChannelCount(std::size_t clipIndex) const
{
    return clip(clipIndex).constantChannels.size();
}

void AnimationPlayer::play(std::size_t clipIndex, bool restart)
{
    (void)clip(clipIndex);
    bool found = false;
    if (definition_->explicitInstances)
    {
        for (PlaybackState& state : playback_)
        {
            const InstanceDefinition& instance = definition_->instances[state.definition];
            if (instance.clip != clipIndex)
            {
                continue;
            }
            found = true;
            if (restart || state.status == Status::stopped || state.status == Status::finished)
            {
                const float duration = clipDuration(clipIndex);
                state.position = duration > 0.0f && state.looping
                    ? std::fmod(instance.initialPosition, duration)
                    : std::min(instance.initialPosition, duration);
            }
            state.status = Status::playing;
        }
    }
    else
    {
        PlaybackState& state = playback_.front();
        found = true;
        if (currentClip_ != clipIndex || restart || state.status == Status::stopped ||
            state.status == Status::finished)
        {
            state.definition = clipIndex;
            state.position = 0.0f;
            state.timelineCursors.assign(definition_->clips[clipIndex].timelines.size(), 0U);
            state.timelineSamples.resize(definition_->clips[clipIndex].timelines.size());
        }
        state.status = Status::playing;
    }
    if (!found)
    {
        throw std::logic_error("animation clip has no scene instances");
    }
    currentClip_ = clipIndex;
    evaluatePose();
}

void AnimationPlayer::pause() noexcept
{
    for (PlaybackState& state : playback_)
    {
        if (definition_->instances[state.definition].clip == currentClip_ &&
            state.status == Status::playing)
        {
            state.status = Status::paused;
        }
    }
}

void AnimationPlayer::resume() noexcept
{
    for (PlaybackState& state : playback_)
    {
        if (definition_->instances[state.definition].clip == currentClip_ &&
            state.status == Status::paused)
        {
            state.status = Status::playing;
        }
    }
}

void AnimationPlayer::stop()
{
    if (currentClip_ == invalidClip)
    {
        return;
    }
    for (PlaybackState& state : playback_)
    {
        const InstanceDefinition& instance = definition_->instances[state.definition];
        if (instance.clip == currentClip_)
        {
            state.status = Status::stopped;
            const float duration = clipDuration(instance.clip);
            state.position = definition_->explicitInstances && duration > 0.0f && state.looping
                ? std::fmod(instance.initialPosition, duration)
                : std::min(definition_->explicitInstances ? instance.initialPosition : 0.0f,
                    duration);
        }
    }
    evaluatePose();
}

void AnimationPlayer::seek(float seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0f)
    {
        throw std::invalid_argument("animation seek time must be finite and non-negative");
    }
    if (currentClip_ == invalidClip)
    {
        throw std::logic_error("cannot seek without an animation clip");
    }
    const float duration = clipDuration(currentClip_);
    for (PlaybackState& state : playback_)
    {
        if (definition_->instances[state.definition].clip != currentClip_)
        {
            continue;
        }
        state.position = std::min(seconds, duration);
        if (state.status == Status::finished && state.position < duration)
        {
            state.status = Status::paused;
        }
    }
    evaluatePose();
}

void AnimationPlayer::setPlaybackSpeed(float speed)
{
    if (!std::isfinite(speed) || speed <= 0.0f)
    {
        throw std::invalid_argument("animation playback speed must be finite and greater than zero");
    }
    for (PlaybackState& state : playback_)
    {
        if (definition_->instances[state.definition].clip == currentClip_)
        {
            state.playbackSpeed = speed;
        }
    }
}

void AnimationPlayer::setLooping(bool looping) noexcept
{
    for (PlaybackState& state : playback_)
    {
        if (definition_->instances[state.definition].clip == currentClip_)
        {
            state.looping = looping;
        }
    }
}

void AnimationPlayer::update(float deltaSeconds)
{
    update(deltaSeconds, {});
}

void AnimationPlayer::update(float deltaSeconds,
    std::span<const InstanceUpdatePolicy> policies)
{
    if (!std::isfinite(deltaSeconds) || deltaSeconds < 0.0f)
    {
        throw std::invalid_argument("animation delta time must be finite and non-negative");
    }
    if (!policies.empty() && policies.size() != playback_.size())
    {
        throw std::invalid_argument("animation update policy count does not match instances");
    }
    std::fill(evaluatedInstances_.begin(), evaluatedInstances_.end(), 0U);
    dueInstancesScratch_.clear();
    for (std::size_t playbackIndex = 0; playbackIndex < playback_.size(); ++playbackIndex)
    {
        PlaybackState& state = playback_[playbackIndex];
        const InstanceUpdatePolicy policy = policies.empty() ? InstanceUpdatePolicy{} :
            policies[playbackIndex];
        if (!std::isfinite(policy.minimumEvaluationIntervalSeconds) ||
            policy.minimumEvaluationIntervalSeconds < 0.0f)
        {
            throw std::invalid_argument(
                "animation evaluation interval must be finite and non-negative");
        }
        const bool newlyEnabled = policy.evaluate && !state.evaluationEnabledLastUpdate;
        if (state.status != Status::playing)
        {
            if (state.status == Status::finished && newlyEnabled)
            {
                dueInstancesScratch_.push_back(playbackIndex);
                state.unevaluatedSeconds = 0.0f;
            }
            state.evaluationEnabledLastUpdate = policy.evaluate;
            continue;
        }
        const std::size_t clipIndex = definition_->instances[state.definition].clip;
        const float duration = clipDuration(clipIndex);
        const float advance = deltaSeconds * state.playbackSpeed;
        if (!std::isfinite(advance))
        {
            throw std::overflow_error("animation time advancement overflowed");
        }
        const float previousPosition = state.position;
        state.position += advance;
        if (state.looping && duration > 0.0f)
        {
            state.position = std::fmod(state.position, duration);
        }
        else if (state.position >= duration)
        {
            state.position = duration;
            state.status = Status::finished;
        }
        if (state.position < previousPosition)
        {
            std::ranges::fill(state.timelineCursors, 0U);
        }

        state.unevaluatedSeconds += deltaSeconds;
        const bool intervalElapsed = policy.minimumEvaluationIntervalSeconds == 0.0f ||
            state.unevaluatedSeconds + 1.0e-6f >= policy.minimumEvaluationIntervalSeconds;
        if (policy.evaluate && (newlyEnabled || intervalElapsed || state.status == Status::finished))
        {
            dueInstancesScratch_.push_back(playbackIndex);
            state.unevaluatedSeconds = 0.0f;
        }
        state.evaluationEnabledLastUpdate = policy.evaluate;
    }
    if (!dueInstancesScratch_.empty())
    {
        evaluateInstances(dueInstancesScratch_, false);
    }
    else
    {
        evaluationTimings_ = {};
    }
}

AnimationPlayer::Status AnimationPlayer::status() const noexcept
{
    const PlaybackState* state = selectedPlayback();
    return state != nullptr ? state->status : Status::stopped;
}

bool AnimationPlayer::looping() const noexcept
{
    const PlaybackState* state = selectedPlayback();
    return state != nullptr && state->looping;
}

float AnimationPlayer::playbackSpeed() const noexcept
{
    const PlaybackState* state = selectedPlayback();
    return state != nullptr ? state->playbackSpeed : 1.0f;
}

float AnimationPlayer::position() const noexcept
{
    const PlaybackState* state = selectedPlayback();
    return state != nullptr ? state->position : 0.0f;
}

float AnimationPlayer::instancePosition(std::size_t instance) const
{
    return playback_.at(instance).position;
}

std::size_t AnimationPlayer::instanceClip(std::size_t instance) const
{
    const PlaybackState& state = playback_.at(instance);
    return definition_->instances.at(state.definition).clip;
}

bool AnimationPlayer::instanceEvaluated(std::size_t instance) const
{
    return evaluatedInstances_.at(instance) != 0;
}

std::optional<std::size_t> AnimationPlayer::instanceForNode(assets::NodeHandle node) const
{
    std::size_t index = nodeIndex(node);
    while (index != invalidClip)
    {
        if (definition_->nodeInstances[index])
        {
            return definition_->nodeInstances[index];
        }
        index = definition_->parents[index];
    }
    return std::nullopt;
}

std::optional<std::size_t> AnimationPlayer::instanceForSkin(assets::SkinHandle skinHandle) const
{
    static_cast<void>(skin(skinHandle));
    return definition_->skinInstances.at(skinHandle.slot);
}

const AnimationPlayer::PlaybackState* AnimationPlayer::selectedPlayback() const noexcept
{
    const auto selected = std::find_if(playback_.begin(), playback_.end(), [&](const auto& state)
    {
        return definition_->instances[state.definition].clip == currentClip_;
    });
    return selected != playback_.end() ? &*selected : nullptr;
}

AnimationPlayer::PlaybackState* AnimationPlayer::selectedPlayback() noexcept
{
    return const_cast<PlaybackState*>(std::as_const(*this).selectedPlayback());
}

const glm::mat4& AnimationPlayer::worldTransform(assets::NodeHandle node) const
{
    return nodes_.at(nodeIndex(node)).world;
}

void AnimationPlayer::appendSkinMatrices(
    assets::SkinHandle skinHandle, std::vector<glm::mat4>& destination) const
{
    const assets::SkinAsset& selectedSkin = skin(skinHandle);
    if (selectedSkin.joints.size() != selectedSkin.inverseBindMatrices.size())
    {
        throw std::runtime_error("skin joint and inverse-bind-matrix counts differ");
    }
    const std::size_t offset = destination.size();
    destination.resize(offset + selectedSkin.joints.size());
    writeSkinMatrices(skinHandle,
        std::span<glm::mat4>(destination).subspan(offset, selectedSkin.joints.size()));
}

void AnimationPlayer::writeSkinMatrices(
    assets::SkinHandle skinHandle, std::span<glm::mat4> destination) const
{
    const assets::SkinAsset& selectedSkin = skin(skinHandle);
    if (selectedSkin.joints.size() != selectedSkin.inverseBindMatrices.size() ||
        destination.size() != selectedSkin.joints.size())
    {
        throw std::runtime_error("skin palette destination has an invalid size");
    }
    for (std::size_t index = 0; index < selectedSkin.joints.size(); ++index)
    {
        destination[index] = worldTransform(selectedSkin.joints[index]) *
            selectedSkin.inverseBindMatrices[index];
    }
}

std::size_t AnimationPlayer::nodeIndex(assets::NodeHandle node) const
{
    if (!node || node.generation != 1U || node.slot >= nodes_.size())
    {
        throw std::out_of_range("animation references an invalid node handle");
    }
    return node.slot;
}

const assets::SkinAsset& AnimationPlayer::skin(assets::SkinHandle handle) const
{
    if (!handle || handle.generation != 1U || handle.slot >= definition_->skins.size() ||
        definition_->skins[handle.slot].handle != handle)
    {
        throw std::out_of_range("animation references an invalid skin handle");
    }
    return definition_->skins[handle.slot];
}

const AnimationPlayer::ClipDefinition& AnimationPlayer::clip(std::size_t index) const
{
    if (index >= definition_->clips.size())
    {
        throw std::out_of_range("animation clip index is out of range");
    }
    return definition_->clips[index];
}

void AnimationPlayer::evaluatePose()
{
    dueInstancesScratch_.clear();
    for (std::size_t index = 0; index < playback_.size(); ++index)
    {
        dueInstancesScratch_.push_back(index);
    }
    evaluateInstances(dueInstancesScratch_, true);
}

AnimationPlayer::TimelineSample AnimationPlayer::sampleTimeline(
    const TimelineDefinition& timeline, float time, std::size_t& cursor) const
{
    if (time <= timeline.times.front() || timeline.times.size() == 1)
    {
        cursor = 0;
        return {};
    }
    if (time >= timeline.times.back())
    {
        cursor = timeline.times.size() > 1 ? timeline.times.size() - 2U : 0U;
        const std::size_t last = timeline.times.size() - 1U;
        return {last, last, 0.0f};
    }

    if (timeline.uniform)
    {
        cursor = std::min(static_cast<std::size_t>(
            (time - timeline.times.front()) * timeline.uniformStepInverse),
            timeline.times.size() - 2U);
        while (cursor + 1U < timeline.times.size() - 1U &&
               time >= timeline.times[cursor + 1U])
        {
            ++cursor;
        }
    }
    else
    {
        cursor = std::min(cursor, timeline.times.size() - 2U);
        if (time < timeline.times[cursor])
        {
            const auto upper = std::upper_bound(timeline.times.begin(), timeline.times.end(), time);
            cursor = static_cast<std::size_t>(upper - timeline.times.begin()) - 1U;
        }
        else
        {
            while (cursor + 1U < timeline.times.size() - 1U &&
                   time >= timeline.times[cursor + 1U])
            {
                ++cursor;
            }
        }
    }

    const std::size_t next = cursor + 1U;
    const float alpha = (time - timeline.times[cursor]) *
        timeline.inverseDurations[cursor];
    return {cursor, next, alpha};
}

glm::vec4 AnimationPlayer::sampleVector(const ClipDefinition& selected,
    const ChannelDefinition& channel, const TimelineSample& timelineSample, bool step) const
{
    const glm::vec4* values = selected.values.data() + channel.valueOffset;
    if (timelineSample.previous == timelineSample.next || step)
    {
        return values[timelineSample.previous];
    }
    return glm::mix(values[timelineSample.previous], values[timelineSample.next],
        timelineSample.alpha);
}

glm::quat AnimationPlayer::sampleRotation(const ClipDefinition& selected,
    const ChannelDefinition& channel, const TimelineSample& timelineSample, bool step,
    bool& usedNlerp) const
{
    usedNlerp = false;
    const glm::vec4* values = selected.values.data() + channel.valueOffset;
    const glm::vec4& previous = values[timelineSample.previous];
    const glm::quat a(previous.w, previous.x, previous.y, previous.z);
    if (timelineSample.previous == timelineSample.next || step)
    {
        return a;
    }
    const glm::vec4& next = values[timelineSample.next];
    const glm::quat b(next.w, next.x, next.y, next.z);
    if (adaptiveNlerpEnabled_ && glm::dot(a, b) >= adaptiveNlerpDotThreshold_)
    {
        usedNlerp = true;
        return glm::normalize(a * (1.0f - timelineSample.alpha) +
            b * timelineSample.alpha);
    }
    return glm::normalize(glm::slerp(a, b, timelineSample.alpha));
}

void AnimationPlayer::applyConstantChannels(std::size_t instanceIndex)
{
    const PlaybackState& state = playback_[instanceIndex];
    const InstanceDefinition& instance = definition_->instances[state.definition];
    const ClipDefinition& selected = clip(instance.clip);
    for (std::size_t index = 0; index < selected.constantChannels.size(); ++index)
    {
        const ConstantChannelDefinition& channel = selected.constantChannels[index];
        NodePose& pose = nodes_[instance.constantChannelTargets[index]].pose;
        switch (channel.path)
        {
        case assets::AnimationTarget::translation:
            pose.translation = glm::vec3(channel.value);
            break;
        case assets::AnimationTarget::rotation:
            pose.rotation = glm::quat(
                channel.value.w, channel.value.x, channel.value.y, channel.value.z);
            break;
        case assets::AnimationTarget::scale:
            pose.scale = glm::vec3(channel.value);
            break;
        }
    }
}

void AnimationPlayer::evaluateInstances(
    std::span<const std::size_t> instances, bool resetAllNodes)
{
    evaluationTimings_ = {};
    std::fill(evaluatedInstances_.begin(), evaluatedInstances_.end(), 0U);
    const auto samplingBegin = std::chrono::steady_clock::now();
    if (resetAllNodes)
    {
        const auto resetBegin = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < nodes_.size(); ++index)
        {
            nodes_[index].pose = definition_->nodes[index].basePose;
        }
        for (const std::size_t playbackIndex : instances)
        {
            applyConstantChannels(playbackIndex);
        }
        evaluationTimings_.poseResetMilliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - resetBegin).count();
    }

    const auto timelineBegin = std::chrono::steady_clock::now();
    for (const std::size_t instance : instances)
    {
        PlaybackState& state = playback_[instance];
        const InstanceDefinition& instanceDefinition =
            definition_->instances[state.definition];
        const ClipDefinition& selected = clip(instanceDefinition.clip);
        if (state.timelineCursors.size() != selected.timelines.size() ||
            state.timelineSamples.size() != selected.timelines.size())
        {
            state.timelineCursors.assign(selected.timelines.size(), 0U);
            state.timelineSamples.resize(selected.timelines.size());
        }
        const float time = selected.startTime + state.position;
        for (std::size_t timelineIndex = 0; timelineIndex < selected.timelines.size();
             ++timelineIndex)
        {
            state.timelineSamples[timelineIndex] = sampleTimeline(
                selected.timelines[timelineIndex], time, state.timelineCursors[timelineIndex]);
        }
    }
    evaluationTimings_.timelineResolutionMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - timelineBegin).count();

    const auto vectorBegin = std::chrono::steady_clock::now();
    const auto applyVectorGroup = [&]<typename Assign>(
        std::size_t group, bool step, Assign&& assign)
    {
        for (const std::size_t instanceIndex : instances)
        {
            const PlaybackState& state = playback_[instanceIndex];
            const InstanceDefinition& instance = definition_->instances[state.definition];
            const ClipDefinition& selected = clip(instance.clip);
            const std::vector<ChannelDefinition>& channels = selected.channelGroups[group];
            const std::vector<std::size_t>& targets = instance.channelTargets[group];
            for (std::size_t channelIndex = 0; channelIndex < channels.size(); ++channelIndex)
            {
                const ChannelDefinition& channel = channels[channelIndex];
                assign(nodes_[targets[channelIndex]].pose,
                    sampleVector(selected, channel, state.timelineSamples[channel.timeline], step));
            }
            evaluationTimings_.sampledVectorChannels +=
                static_cast<std::uint32_t>(channels.size());
        }
    };
    applyVectorGroup(translationLinearGroup, false,
        [](NodePose& pose, const glm::vec4& value) { pose.translation = glm::vec3(value); });
    applyVectorGroup(translationStepGroup, true,
        [](NodePose& pose, const glm::vec4& value) { pose.translation = glm::vec3(value); });
    applyVectorGroup(scaleLinearGroup, false,
        [](NodePose& pose, const glm::vec4& value) { pose.scale = glm::vec3(value); });
    applyVectorGroup(scaleStepGroup, true,
        [](NodePose& pose, const glm::vec4& value) { pose.scale = glm::vec3(value); });
    evaluationTimings_.vectorSamplingMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - vectorBegin).count();

    const auto rotationBegin = std::chrono::steady_clock::now();
    const auto applyRotationGroup = [&](std::size_t group, bool step)
    {
        for (const std::size_t instanceIndex : instances)
        {
            const PlaybackState& state = playback_[instanceIndex];
            const InstanceDefinition& instance = definition_->instances[state.definition];
            const ClipDefinition& selected = clip(instance.clip);
            const std::vector<ChannelDefinition>& channels = selected.channelGroups[group];
            const std::vector<std::size_t>& targets = instance.channelTargets[group];
            for (std::size_t channelIndex = 0; channelIndex < channels.size(); ++channelIndex)
            {
                const ChannelDefinition& channel = channels[channelIndex];
                bool usedNlerp = false;
                nodes_[targets[channelIndex]].pose.rotation = sampleRotation(
                    selected, channel, state.timelineSamples[channel.timeline], step, usedNlerp);
                evaluationTimings_.nlerpRotationChannels += usedNlerp ? 1U : 0U;
            }
            evaluationTimings_.sampledRotationChannels +=
                static_cast<std::uint32_t>(channels.size());
        }
    };
    applyRotationGroup(rotationLinearGroup, false);
    applyRotationGroup(rotationStepGroup, true);
    evaluationTimings_.rotationSamplingMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - rotationBegin).count();

    for (const std::size_t instance : instances)
    {
        evaluatedInstances_[instance] = 1U;
    }
    evaluationTimings_.evaluatedInstances = static_cast<std::uint32_t>(instances.size());
    evaluationTimings_.sampledChannels = evaluationTimings_.sampledVectorChannels +
        evaluationTimings_.sampledRotationChannels;
    evaluationTimings_.samplingMilliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - samplingBegin).count();

    const auto propagationBegin = std::chrono::steady_clock::now();
    if (resetAllNodes)
    {
        propagate(definition_->fullPropagationPlan.entries,
            definition_->fullPropagationPlan.poseComposedNodes);
    }
    else
    {
        bool evaluateWholeScene = false;
        for (const std::size_t playbackIndex : instances)
        {
            const InstanceDefinition& instance =
                definition_->instances[playback_[playbackIndex].definition];
            if (instance.requiresFullPropagation)
            {
                evaluateWholeScene = true;
                break;
            }
        }
        if (evaluateWholeScene)
        {
            propagate(definition_->fullPropagationPlan.entries,
                definition_->fullPropagationPlan.poseComposedNodes);
        }
        else
        {
            for (const std::size_t playbackIndex : instances)
            {
                const PropagationRange& range = definition_->instances[
                    playback_[playbackIndex].definition].propagationRange;
                propagate(std::span(definition_->fullPropagationPlan.entries).subspan(
                    range.offset, range.count), range.poseComposedNodes);
            }
        }
    }
    evaluationTimings_.transformPropagationMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - propagationBegin).count();
}

void AnimationPlayer::propagate(std::span<const PropagationEntry> entries,
    std::uint32_t poseComposedNodes)
{
    constexpr std::uint32_t noParent = std::numeric_limits<std::uint32_t>::max();
    for (const PropagationEntry& entry : entries)
    {
        NodeState& node = nodes_[entry.node];
        const NodeDefinition& definition = definition_->nodes[entry.node];
        switch (entry.localSource)
        {
        case PropagationEntry::LocalSource::pose:
            node.world = entry.parent == noParent ? composeLocal(node.pose) :
                composeWorld(nodes_[entry.parent].world, node.pose);
            break;
        case PropagationEntry::LocalSource::cachedAffine:
            node.world = entry.parent == noParent ? definition.baseLocal :
                multiplyAffineLocal(nodes_[entry.parent].world, definition.baseLocal);
            break;
        case PropagationEntry::LocalSource::cachedGeneral:
            node.world = entry.parent == noParent ? definition.baseLocal :
                nodes_[entry.parent].world * definition.baseLocal;
            break;
        }
    }
    evaluationTimings_.propagatedNodes += static_cast<std::uint32_t>(entries.size());
    evaluationTimings_.poseComposedNodes += poseComposedNodes;
    evaluationTimings_.cachedLocalNodes +=
        static_cast<std::uint32_t>(entries.size()) - poseComposedNodes;
}
}
