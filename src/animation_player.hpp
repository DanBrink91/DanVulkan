#pragma once

#include <danvulkan/assets.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstddef>
#include <limits>
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

    explicit AnimationPlayer(const assets::SceneAsset& scene);

    [[nodiscard]] bool animated() const noexcept { return !clips_.empty(); }
    [[nodiscard]] std::size_t clipCount() const noexcept { return clips_.size(); }
    [[nodiscard]] std::string_view clipName(std::size_t clip) const;
    [[nodiscard]] float clipDuration(std::size_t clip) const;
    [[nodiscard]] std::size_t currentClip() const noexcept { return currentClip_; }
    [[nodiscard]] Status status() const noexcept { return status_; }
    [[nodiscard]] bool looping() const noexcept { return looping_; }
    [[nodiscard]] float playbackSpeed() const noexcept { return playbackSpeed_; }
    [[nodiscard]] float position() const noexcept { return position_; }

    void play(std::size_t clip, bool restart = true);
    void pause() noexcept;
    void resume() noexcept;
    void stop();
    void seek(float seconds);
    void setLooping(bool looping) noexcept { looping_ = looping; }
    void setPlaybackSpeed(float speed);
    void update(float deltaSeconds);

    [[nodiscard]] const glm::mat4& worldTransform(assets::NodeHandle node) const;
    void appendSkinMatrices(assets::SkinHandle skin, assets::NodeHandle meshNode,
        std::vector<glm::mat4>& destination) const;

    struct NodePose
    {
        glm::vec3 translation{};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 scale{1.0f};
    };

private:
    struct NodeState
    {
        glm::mat4 baseLocal{1.0f};
        NodePose basePose;
        NodePose pose;
        bool transformIsTrs = false;
        std::vector<assets::NodeHandle> children;
        glm::mat4 world{1.0f};
    };

    [[nodiscard]] std::size_t nodeIndex(assets::NodeHandle node) const;
    [[nodiscard]] const assets::AnimationClipAsset& clip(std::size_t index) const;
    [[nodiscard]] const assets::SkinAsset& skin(assets::SkinHandle handle) const;
    void evaluatePose();
    void evaluateWorld(assets::NodeHandle node, const glm::mat4& parent);

    std::vector<NodeState> nodes_;
    std::vector<assets::NodeHandle> roots_;
    std::vector<assets::SkinAsset> skins_;
    std::vector<assets::AnimationClipAsset> clips_;
    std::size_t currentClip_ = invalidClip;
    Status status_ = Status::stopped;
    bool looping_ = true;
    float playbackSpeed_ = 1.0f;
    float position_ = 0.0f;
};
}
