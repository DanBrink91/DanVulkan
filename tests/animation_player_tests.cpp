#include "src/animation_player.hpp"

#include <danvulkan/assets.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <numbers>
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

bool nearlyEqual(const glm::mat4& left, const glm::mat4& right)
{
    for (glm::length_t column = 0; column < 4; ++column)
    {
        for (glm::length_t row = 0; row < 4; ++row)
        {
            if (!nearlyEqual(left[column][row], right[column][row]))
            {
                return false;
            }
        }
    }
    return true;
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
    require(player.compiledChannelCount(0) == 149 &&
        player.foldedConstantChannelCount(0) == 3,
        "character clip did not retain its expected compiled channel layout");

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

    danvulkan::assets::SceneAsset propagationScene;
    danvulkan::assets::NodeAsset cachedTrsNode;
    cachedTrsNode.name = "cached TRS child";
    cachedTrsNode.translation = {0.0f, 2.0f, 0.0f};
    const glm::quat cachedTrsRotation = glm::angleAxis(
        glm::radians(15.0f), glm::normalize(glm::vec3(1.0f, 0.5f, 0.25f)));
    cachedTrsNode.rotation = {cachedTrsRotation.x, cachedTrsRotation.y,
        cachedTrsRotation.z, cachedTrsRotation.w};
    cachedTrsNode.scale = {0.75f, 1.25f, 0.5f};
    cachedTrsNode.transformIsTrs = true;
    const danvulkan::assets::NodeHandle cachedTrsHandle =
        propagationScene.addNode(std::move(cachedTrsNode));
    danvulkan::assets::NodeAsset cachedMatrixNode;
    cachedMatrixNode.name = "cached matrix child";
    cachedMatrixNode.localTransform = glm::translate(
        glm::mat4(1.0f), glm::vec3(-1.0f, 0.5f, 0.25f));
    cachedMatrixNode.localTransform[0].w = 0.125f;
    const glm::mat4 cachedMatrixLocal = cachedMatrixNode.localTransform;
    const danvulkan::assets::NodeHandle cachedMatrixHandle =
        propagationScene.addNode(std::move(cachedMatrixNode));
    danvulkan::assets::NodeAsset propagatedNode;
    propagatedNode.name = "animated parent";
    propagatedNode.translation = {1.0f, 0.0f, 0.0f};
    const glm::quat authoredPropagatedRotation = glm::angleAxis(
        glm::radians(35.0f), glm::normalize(glm::vec3(0.25f, 1.0f, 0.5f)));
    propagatedNode.rotation = {authoredPropagatedRotation.x, authoredPropagatedRotation.y,
        authoredPropagatedRotation.z, authoredPropagatedRotation.w};
    propagatedNode.scale = {1.5f, 0.5f, 2.0f};
    propagatedNode.transformIsTrs = true;
    propagatedNode.children = {cachedTrsHandle, cachedMatrixHandle};
    const glm::quat propagatedRotation(propagatedNode.rotation.w, propagatedNode.rotation.x,
        propagatedNode.rotation.y, propagatedNode.rotation.z);
    const glm::vec3 propagatedScale = propagatedNode.scale;
    const danvulkan::assets::NodeHandle propagatedHandle =
        propagationScene.addNode(std::move(propagatedNode));
    danvulkan::assets::NodeAsset matrixRoot;
    matrixRoot.name = "matrix root";
    matrixRoot.localTransform = glm::translate(
        glm::mat4(1.0f), glm::vec3(4.0f, -2.0f, 3.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(20.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    matrixRoot.children.push_back(propagatedHandle);
    const glm::mat4 matrixRootLocal = matrixRoot.localTransform;
    const danvulkan::assets::NodeHandle matrixRootHandle =
        propagationScene.addNode(std::move(matrixRoot));
    propagationScene.addRootNode(matrixRootHandle);
    danvulkan::assets::NodeAsset staticSibling;
    staticSibling.name = "unrelated static root";
    const danvulkan::assets::NodeHandle staticSiblingHandle =
        propagationScene.addNode(std::move(staticSibling));
    propagationScene.addRootNode(staticSiblingHandle);
    danvulkan::assets::AnimationClipAsset propagationClip;
    propagationClip.endTime = 1.0f;
    danvulkan::assets::AnimationChannelAsset propagatedTranslation;
    propagatedTranslation.target = propagatedHandle;
    propagatedTranslation.path = danvulkan::assets::AnimationTarget::translation;
    propagatedTranslation.times = {0.0f, 1.0f};
    propagatedTranslation.values = {
        glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), glm::vec4(3.0f, 0.0f, 0.0f, 0.0f)};
    propagationClip.channels.push_back(std::move(propagatedTranslation));
    propagationScene.addAnimation(std::move(propagationClip));

    danvulkan::AnimationPlayer propagationPlayer(propagationScene);
    require(propagationPlayer.instanceForNode(cachedTrsHandle) == 0U &&
        !propagationPlayer.instanceForNode(staticSiblingHandle),
        "implicit animation ownership leaked into an unrelated static hierarchy");
    require(propagationPlayer.instanceClip(0) == 0U,
        "implicit animation actor did not report its selected clip");
    propagationPlayer.update(0.5f);
    const glm::mat4 expectedPropagated = matrixRootLocal *
        glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)) *
        glm::mat4_cast(propagatedRotation) * glm::scale(glm::mat4(1.0f), propagatedScale);
    const glm::mat4 expectedCachedTrs = expectedPropagated *
        glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f)) *
        glm::mat4_cast(cachedTrsRotation) *
        glm::scale(glm::mat4(1.0f), glm::vec3(0.75f, 1.25f, 0.5f));
    require(nearlyEqual(propagationPlayer.worldTransform(propagatedHandle), expectedPropagated) &&
        nearlyEqual(propagationPlayer.worldTransform(cachedTrsHandle), expectedCachedTrs) &&
        nearlyEqual(propagationPlayer.worldTransform(cachedMatrixHandle),
            expectedPropagated * cachedMatrixLocal),
        "flat affine propagation changed mixed TRS and matrix world transforms");
    const danvulkan::AnimationPlayer::EvaluationTimings propagationTimings =
        propagationPlayer.evaluationTimings();
    require(propagationTimings.propagatedNodes == 3 &&
        propagationTimings.poseComposedNodes == 1 &&
        propagationTimings.cachedLocalNodes == 2,
        "flat propagation did not reuse static local matrices");

    danvulkan::assets::SceneAsset instancedScene;
    danvulkan::assets::NodeAsset sourceNode;
    sourceNode.name = "source";
    sourceNode.transformIsTrs = true;
    const danvulkan::assets::NodeHandle sourceHandle =
        instancedScene.addNode(std::move(sourceNode));
    danvulkan::assets::NodeAsset targetNode;
    targetNode.name = "target";
    targetNode.transformIsTrs = true;
    const danvulkan::assets::NodeHandle targetHandle =
        instancedScene.addNode(std::move(targetNode));
    danvulkan::assets::NodeAsset sourceRoot;
    sourceRoot.children.push_back(sourceHandle);
    const danvulkan::assets::NodeHandle sourceRootHandle =
        instancedScene.addNode(std::move(sourceRoot));
    danvulkan::assets::NodeAsset targetRoot;
    targetRoot.children.push_back(targetHandle);
    const danvulkan::assets::NodeHandle targetRootHandle =
        instancedScene.addNode(std::move(targetRoot));
    instancedScene.addRootNode(sourceRootHandle);
    instancedScene.addRootNode(targetRootHandle);

    danvulkan::assets::AnimationClipAsset sharedClip;
    sharedClip.name = "shared translation";
    sharedClip.endTime = 1.0f;
    danvulkan::assets::AnimationChannelAsset translation;
    translation.target = sourceHandle;
    translation.path = danvulkan::assets::AnimationTarget::translation;
    translation.times = {0.0f, 1.0f};
    translation.values = {glm::vec4(0.0f), glm::vec4(10.0f, 0.0f, 0.0f, 0.0f)};
    sharedClip.channels.push_back(std::move(translation));
    danvulkan::assets::AnimationChannelAsset scale;
    scale.target = sourceHandle;
    scale.path = danvulkan::assets::AnimationTarget::scale;
    scale.times = {0.0f, 1.0f};
    scale.values = {glm::vec4(1.0f), glm::vec4(2.0f)};
    sharedClip.channels.push_back(std::move(scale));
    danvulkan::assets::AnimationChannelAsset constantRotation;
    constantRotation.target = sourceHandle;
    constantRotation.path = danvulkan::assets::AnimationTarget::rotation;
    constantRotation.times = {0.0f, 1.0f};
    constantRotation.values = {
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)};
    sharedClip.channels.push_back(std::move(constantRotation));
    const danvulkan::assets::AnimationHandle sharedClipHandle =
        instancedScene.addAnimation(std::move(sharedClip));

    danvulkan::assets::AnimationInstanceAsset firstInstance;
    firstInstance.clip = sharedClipHandle;
    firstInstance.initialPositionSeconds = 0.25f;
    firstInstance.nodeBindings.push_back({sourceHandle, sourceHandle});
    instancedScene.addAnimationInstance(std::move(firstInstance));
    danvulkan::assets::AnimationInstanceAsset secondInstance;
    secondInstance.clip = sharedClipHandle;
    secondInstance.initialPositionSeconds = 0.75f;
    secondInstance.nodeBindings.push_back({sourceHandle, targetHandle});
    instancedScene.addAnimationInstance(std::move(secondInstance));

    danvulkan::AnimationPlayer instancedPlayer(instancedScene);
    require(instancedPlayer.instanceCount() == 2,
        "shared clip did not create independent playback instances");
    require(instancedPlayer.compiledTimelineCount(0) == 1,
        "identical channel timelines were not compiled into shared sampling data");
    require(instancedPlayer.compiledChannelCount(0) == 2 &&
        instancedPlayer.foldedConstantChannelCount(0) == 1,
        "constant animation channels were not removed from steady-state sampling");
    require(nearlyEqual(instancedPlayer.instancePosition(0), 0.25f) &&
        nearlyEqual(instancedPlayer.instancePosition(1), 0.75f),
        "animation instance phase offsets were not preserved");
    require(nearlyEqual(instancedPlayer.worldTransform(sourceHandle)[3].x, 2.5f) &&
        nearlyEqual(instancedPlayer.worldTransform(targetHandle)[3].x, 7.5f),
        "shared clip was not evaluated against both node mappings");
    instancedPlayer.update(0.1f);
    require(nearlyEqual(instancedPlayer.instancePosition(0), 0.35f) &&
        nearlyEqual(instancedPlayer.instancePosition(1), 0.85f),
        "independent playback cursors did not advance");
    require(instancedPlayer.evaluationTimings().poseResetMilliseconds == 0.0,
        "steady-state animation evaluation reset base poses");
    instancedPlayer.play(0);
    require(nearlyEqual(instancedPlayer.instancePosition(0), 0.25f) &&
        nearlyEqual(instancedPlayer.instancePosition(1), 0.75f),
        "restarting a shared clip lost per-instance phase offsets");

    const float retainedPosition = instancedPlayer.worldTransform(sourceHandle)[3].x;
    const std::array<danvulkan::AnimationPlayer::InstanceUpdatePolicy, 2> culledPolicies{{
        {false, 0.0f}, {false, 0.0f}
    }};
    instancedPlayer.update(0.1f, culledPolicies);
    require(nearlyEqual(instancedPlayer.instancePosition(0), 0.35f) &&
        nearlyEqual(instancedPlayer.worldTransform(sourceHandle)[3].x, retainedPosition) &&
        !instancedPlayer.instanceEvaluated(0),
        "culled animation did not advance its clock while retaining its last pose");

    const std::array<danvulkan::AnimationPlayer::InstanceUpdatePolicy, 2> limitedPolicies{{
        {true, 0.5f}, {true, 0.5f}
    }};
    instancedPlayer.update(0.1f, limitedPolicies);
    require(nearlyEqual(instancedPlayer.worldTransform(sourceHandle)[3].x, 4.5f) &&
        instancedPlayer.instanceEvaluated(0),
        "a newly visible rate-limited animation was not evaluated immediately");
    instancedPlayer.update(0.1f, limitedPolicies);
    require(nearlyEqual(instancedPlayer.worldTransform(sourceHandle)[3].x, 4.5f) &&
        !instancedPlayer.instanceEvaluated(0),
        "animation update-rate limiting did not retain the previous pose");
    for (std::size_t updateIndex = 0; updateIndex < 4; ++updateIndex)
    {
        instancedPlayer.update(0.1f, limitedPolicies);
    }
    require(nearlyEqual(instancedPlayer.worldTransform(sourceHandle)[3].x, 9.5f) &&
        instancedPlayer.instanceEvaluated(0) &&
        instancedPlayer.evaluationTimings().sampledChannels == 4,
        "rate-limited animation did not evaluate on its configured cadence");

    instancedPlayer.play(0);
    instancedPlayer.setLooping(false);
    instancedPlayer.update(2.0f, culledPolicies);
    require(instancedPlayer.status() == danvulkan::AnimationPlayer::Status::finished &&
        nearlyEqual(instancedPlayer.worldTransform(sourceHandle)[3].x, 2.5f),
        "a hidden non-looping animation did not retain its previous pose at completion");
    instancedPlayer.update(0.0f, limitedPolicies);
    require(nearlyEqual(instancedPlayer.worldTransform(sourceHandle)[3].x, 10.0f),
        "a completed hidden animation did not publish its final pose when visible");

    danvulkan::assets::SceneAsset composedInstances;
    danvulkan::assets::NodeAsset prefixNode;
    prefixNode.transformIsTrs = true;
    const danvulkan::assets::NodeHandle prefixHandle =
        composedInstances.addNode(std::move(prefixNode));
    composedInstances.addRootNode(prefixHandle);
    const std::vector<danvulkan::assets::NodeHandle> appendedRoots =
        composedInstances.append(instancedScene);
    danvulkan::AnimationPlayer appendedPlayer(composedInstances);
    const danvulkan::assets::NodeHandle appendedSource{sourceHandle.slot + 1U, 1U};
    const danvulkan::assets::NodeHandle appendedTarget{targetHandle.slot + 1U, 1U};
    require(appendedRoots.size() == 2 &&
        nearlyEqual(appendedPlayer.worldTransform(appendedSource)[3].x, 2.5f) &&
        nearlyEqual(appendedPlayer.worldTransform(appendedTarget)[3].x, 7.5f),
        "scene composition did not remap animation instance bindings");

    danvulkan::assets::SceneAsset rotationScene;
    danvulkan::assets::NodeAsset rotatingNode;
    rotatingNode.transformIsTrs = true;
    const danvulkan::assets::NodeHandle rotatingHandle =
        rotationScene.addNode(std::move(rotatingNode));
    rotationScene.addRootNode(rotatingHandle);
    danvulkan::assets::AnimationClipAsset rotationClip;
    rotationClip.endTime = 1.0f;
    danvulkan::assets::AnimationChannelAsset rotationChannel;
    rotationChannel.target = rotatingHandle;
    rotationChannel.path = danvulkan::assets::AnimationTarget::rotation;
    rotationChannel.times = {0.0f, 1.0f};
    const glm::quat rotationEnd = glm::angleAxis(
        glm::radians(120.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    rotationChannel.values = {
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(rotationEnd.x, rotationEnd.y, rotationEnd.z, rotationEnd.w)};
    rotationClip.channels.push_back(std::move(rotationChannel));
    rotationScene.addAnimation(std::move(rotationClip));
    danvulkan::assets::AnimationClipAsset translationOnlyClip;
    translationOnlyClip.endTime = 1.0f;
    danvulkan::assets::AnimationChannelAsset translationOnlyChannel;
    translationOnlyChannel.target = rotatingHandle;
    translationOnlyChannel.path = danvulkan::assets::AnimationTarget::translation;
    translationOnlyChannel.times = {0.0f, 1.0f};
    translationOnlyChannel.values = {
        glm::vec4(0.0f), glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)};
    translationOnlyClip.channels.push_back(std::move(translationOnlyChannel));
    rotationScene.addAnimation(std::move(translationOnlyClip));

    danvulkan::AnimationPlayer exactRotation(rotationScene);
    danvulkan::AnimationPlayer adaptiveRotation(rotationScene,
        {std::numbers::pi_v<float>});
    danvulkan::AnimationPlayer guardedRotation(rotationScene, {glm::radians(20.0f)});
    exactRotation.seek(0.25f);
    adaptiveRotation.seek(0.25f);
    guardedRotation.seek(0.25f);
    const glm::quat exactPose = glm::quat_cast(exactRotation.worldTransform(rotatingHandle));
    const glm::quat adaptivePose = glm::quat_cast(adaptiveRotation.worldTransform(rotatingHandle));
    const glm::quat guardedPose = glm::quat_cast(guardedRotation.worldTransform(rotatingHandle));
    const auto angularDifference = [](const glm::quat& left, const glm::quat& right)
    {
        return 2.0f * std::acos(std::clamp(std::abs(glm::dot(left, right)), 0.0f, 1.0f));
    };
    const float adaptiveError = angularDifference(exactPose, adaptivePose);
    require(adaptiveError > 0.0f && adaptiveError < glm::radians(3.0f) &&
        adaptiveRotation.evaluationTimings().nlerpRotationChannels == 1,
        "adaptive nlerp did not take its bounded approximation path");
    require(angularDifference(exactPose, guardedPose) < 1.0e-5f &&
        guardedRotation.evaluationTimings().nlerpRotationChannels == 0,
        "adaptive nlerp ignored its angular threshold");
    exactRotation.play(1);
    require(angularDifference(glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                glm::quat_cast(exactRotation.worldTransform(rotatingHandle))) < 1.0e-5f,
        "changing clips retained a component animated only by the previous clip");

    bool invalidNlerpRejected = false;
    try
    {
        danvulkan::AnimationPlayer invalidNlerp(rotationScene,
            {std::numbers::pi_v<float> + 0.01f});
    }
    catch (const std::invalid_argument&)
    {
        invalidNlerpRejected = true;
    }
    require(invalidNlerpRejected, "an invalid adaptive nlerp threshold was accepted");
}
