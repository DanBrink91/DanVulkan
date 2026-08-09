#include "src/animation_player.hpp"

#include <danvulkan/assets.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
void require(bool condition, std::string_view message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string(message));
    }
}

bool nearlyEqual(float left, float right)
{
    return std::abs(left - right) < 0.0001f;
}
}

int main()
{
    danvulkan::assets::SceneAsset scene =
        danvulkan::assets::loadScene("models/ninja_run_free_fire_emote.glb");
    danvulkan::assets::AnimationClipAsset duplicate = scene.animations().front();
    duplicate.handle = {};
    duplicate.name = "alternate run";
    scene.addAnimation(std::move(duplicate));

    danvulkan::AnimationPlayer player(scene);
    require(player.clipCount() == 2, "expected both animation clips");
    require(player.currentClip() == 0, "expected the first clip to be selected");
    require(player.status() == danvulkan::AnimationPlayer::Status::playing,
        "expected imported animation to autoplay");
    require(player.looping(), "expected looping to default on");
    require(nearlyEqual(player.playbackSpeed(), 1.0f), "unexpected default playback speed");

    const float duration = player.clipDuration(0);
    require(duration > 0.0f, "expected a non-empty animation duration");
    player.pause();
    player.update(duration * 0.5f);
    require(nearlyEqual(player.position(), 0.0f), "paused animation advanced");

    player.setPlaybackSpeed(2.0f);
    player.resume();
    player.update(duration * 0.25f);
    require(nearlyEqual(player.position(), duration * 0.5f),
        "playback speed was not applied");

    player.seek(duration * 0.75f);
    require(nearlyEqual(player.position(), duration * 0.75f), "seek did not update position");
    player.setLooping(false);
    player.update(duration);
    require(player.status() == danvulkan::AnimationPlayer::Status::finished,
        "non-looping animation did not finish");
    require(nearlyEqual(player.position(), duration), "finished animation did not clamp to its end");

    player.seek(duration * 0.25f);
    require(player.status() == danvulkan::AnimationPlayer::Status::paused,
        "seeking a finished clip should leave it paused");
    player.play(1);
    require(player.currentClip() == 1 && nearlyEqual(player.position(), 0.0f),
        "clip selection did not restart the new clip");
    require(player.clipName(1) == "alternate run", "selected clip name is incorrect");

    player.update(duration * 0.1f);
    const float preservedPosition = player.position();
    player.play(1, false);
    require(nearlyEqual(player.position(), preservedPosition),
        "play without restart rewound the selected clip");
    player.stop();
    require(player.status() == danvulkan::AnimationPlayer::Status::stopped,
        "stop did not update playback status");
    require(nearlyEqual(player.position(), 0.0f), "stop did not rewind the clip");

    bool invalidSpeedRejected = false;
    try
    {
        player.setPlaybackSpeed(0.0f);
    }
    catch (const std::invalid_argument&)
    {
        invalidSpeedRejected = true;
    }
    require(invalidSpeedRejected, "invalid playback speed was accepted");

    bool invalidClipRejected = false;
    try
    {
        player.play(99);
    }
    catch (const std::out_of_range&)
    {
        invalidClipRejected = true;
    }
    require(invalidClipRejected, "invalid animation clip was accepted");
}
