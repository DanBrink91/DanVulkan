#define GLM_ENABLE_EXPERIMENTAL

#include "animation_player.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace danvulkan
{
namespace
{
glm::mat4 compose(const AnimationPlayer::NodePose& pose)
{
    return glm::translate(glm::mat4(1.0f), pose.translation) * glm::mat4_cast(pose.rotation) *
        glm::scale(glm::mat4(1.0f), pose.scale);
}

glm::vec4 sample(const assets::AnimationChannelAsset& channel, float time)
{
    if (time <= channel.times.front())
    {
        return channel.values.front();
    }
    if (time >= channel.times.back())
    {
        return channel.values.back();
    }
    const auto upper = std::upper_bound(channel.times.begin(), channel.times.end(), time);
    const std::size_t next = static_cast<std::size_t>(upper - channel.times.begin());
    const std::size_t previous = next - 1;
    if (channel.interpolation == assets::AnimationInterpolation::step)
    {
        return channel.values[previous];
    }
    const float duration = channel.times[next] - channel.times[previous];
    const float alpha = duration > 0.0f ? (time - channel.times[previous]) / duration : 0.0f;
    if (channel.path == assets::AnimationTarget::rotation)
    {
        const glm::vec4& a = channel.values[previous];
        const glm::vec4& b = channel.values[next];
        const glm::quat start(a.w, a.x, a.y, a.z);
        const glm::quat end(b.w, b.x, b.y, b.z);
        const glm::quat value = glm::normalize(glm::slerp(start, end, alpha));
        return { value.x, value.y, value.z, value.w };
    }
    return glm::mix(channel.values[previous], channel.values[next], alpha);
}
}

AnimationPlayer::AnimationPlayer(const assets::SceneAsset& scene)
    : roots_(scene.rootNodes().begin(), scene.rootNodes().end()),
      skins_(scene.skins().begin(), scene.skins().end()),
      clips_(scene.animations().begin(), scene.animations().end())
{
    nodes_.reserve(scene.nodes().size());
    for (const assets::NodeAsset& node : scene.nodes())
    {
        NodeState state;
        state.baseLocal = node.localTransform;
        state.basePose.translation = node.translation;
        state.basePose.rotation = glm::normalize(glm::quat(
            node.rotation.w, node.rotation.x, node.rotation.y, node.rotation.z));
        state.basePose.scale = node.scale;
        state.pose = state.basePose;
        state.transformIsTrs = node.transformIsTrs;
        state.children = node.children;
        nodes_.push_back(std::move(state));
    }
    if (!clips_.empty())
    {
        currentClip_ = 0;
        status_ = Status::playing;
    }
    evaluatePose();
}

std::string_view AnimationPlayer::clipName(std::size_t clipIndex) const
{
    return clip(clipIndex).name;
}

float AnimationPlayer::clipDuration(std::size_t clipIndex) const
{
    const assets::AnimationClipAsset& selected = clip(clipIndex);
    return std::max(selected.endTime - selected.startTime, 0.0f);
}

void AnimationPlayer::play(std::size_t clipIndex, bool restart)
{
    (void)clip(clipIndex);
    if (currentClip_ != clipIndex || restart || status_ == Status::stopped ||
        status_ == Status::finished)
    {
        currentClip_ = clipIndex;
        position_ = 0.0f;
        evaluatePose();
    }
    status_ = Status::playing;
}

void AnimationPlayer::pause() noexcept
{
    if (status_ == Status::playing)
    {
        status_ = Status::paused;
    }
}

void AnimationPlayer::resume() noexcept
{
    if (status_ == Status::paused)
    {
        status_ = Status::playing;
    }
}

void AnimationPlayer::stop()
{
    if (currentClip_ == invalidClip)
    {
        return;
    }
    status_ = Status::stopped;
    position_ = 0.0f;
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
    position_ = std::min(seconds, clipDuration(currentClip_));
    if (status_ == Status::finished && position_ < clipDuration(currentClip_))
    {
        status_ = Status::paused;
    }
    evaluatePose();
}

void AnimationPlayer::setPlaybackSpeed(float speed)
{
    if (!std::isfinite(speed) || speed <= 0.0f)
    {
        throw std::invalid_argument("animation playback speed must be finite and greater than zero");
    }
    playbackSpeed_ = speed;
}

void AnimationPlayer::update(float deltaSeconds)
{
    if (!std::isfinite(deltaSeconds) || deltaSeconds < 0.0f)
    {
        throw std::invalid_argument("animation delta time must be finite and non-negative");
    }
    if (currentClip_ != invalidClip && status_ == Status::playing)
    {
        const float duration = clipDuration(currentClip_);
        const float advance = deltaSeconds * playbackSpeed_;
        if (!std::isfinite(advance))
        {
            throw std::overflow_error("animation time advancement overflowed");
        }
        position_ += advance;
        if (looping_ && duration > 0.0f)
        {
            position_ = std::fmod(position_, duration);
        }
        else if (position_ >= duration)
        {
            position_ = duration;
            status_ = Status::finished;
        }
        evaluatePose();
    }
}

const glm::mat4& AnimationPlayer::worldTransform(assets::NodeHandle node) const
{
    return nodes_.at(nodeIndex(node)).world;
}

void AnimationPlayer::appendSkinMatrices(assets::SkinHandle skinHandle,
    assets::NodeHandle meshNode, std::vector<glm::mat4>& destination) const
{
    const assets::SkinAsset& selectedSkin = skin(skinHandle);
    if (selectedSkin.joints.size() != selectedSkin.inverseBindMatrices.size())
    {
        throw std::runtime_error("skin joint and inverse-bind-matrix counts differ");
    }
    const glm::mat4 meshInverse = glm::inverse(worldTransform(meshNode));
    destination.reserve(destination.size() + selectedSkin.joints.size());
    for (std::size_t index = 0; index < selectedSkin.joints.size(); ++index)
    {
        destination.push_back(meshInverse * worldTransform(selectedSkin.joints[index]) *
            selectedSkin.inverseBindMatrices[index]);
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
    if (!handle || handle.generation != 1U || handle.slot >= skins_.size() ||
        skins_[handle.slot].handle != handle)
    {
        throw std::out_of_range("animation references an invalid skin handle");
    }
    return skins_[handle.slot];
}

const assets::AnimationClipAsset& AnimationPlayer::clip(std::size_t index) const
{
    if (index >= clips_.size())
    {
        throw std::out_of_range("animation clip index is out of range");
    }
    return clips_[index];
}

void AnimationPlayer::evaluatePose()
{
    for (NodeState& node : nodes_)
    {
        node.pose = node.basePose;
    }
    if (currentClip_ != invalidClip)
    {
        const assets::AnimationClipAsset& selected = clip(currentClip_);
        const float time = selected.startTime + position_;
        for (const assets::AnimationChannelAsset& channel : selected.channels)
        {
            NodePose& pose = nodes_.at(nodeIndex(channel.target)).pose;
            if (!nodes_.at(nodeIndex(channel.target)).transformIsTrs)
            {
                throw std::runtime_error("animation targets a matrix-authored node");
            }
            const glm::vec4 value = sample(channel, time);
            switch (channel.path)
            {
            case assets::AnimationTarget::translation:
                pose.translation = glm::vec3(value);
                break;
            case assets::AnimationTarget::rotation:
                pose.rotation = glm::normalize(glm::quat(value.w, value.x, value.y, value.z));
                break;
            case assets::AnimationTarget::scale:
                pose.scale = glm::vec3(value);
                break;
            }
        }
    }
    for (const assets::NodeHandle root : roots_)
    {
        evaluateWorld(root, glm::mat4(1.0f));
    }
}

void AnimationPlayer::evaluateWorld(assets::NodeHandle nodeHandle, const glm::mat4& parent)
{
    NodeState& node = nodes_.at(nodeIndex(nodeHandle));
    node.world = parent * (node.transformIsTrs ? compose(node.pose) : node.baseLocal);
    for (const assets::NodeHandle child : node.children)
    {
        evaluateWorld(child, node.world);
    }
}
}
