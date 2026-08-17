#include "game_world.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace danvulkan::application {
namespace {
constexpr glm::vec3 ninjaPosition(2.15f, -0.05f, -2.85f);
// glTF character assets face world +Z in the demo scene.
constexpr glm::vec3 ninjaForward(0.0f, 0.0f, 1.0f);
constexpr float ninjaScale = 0.01f;
constexpr float ninjaMovementSpeed = 0.08f;
// The imported origin sits in front of and below the rendered character. Aim at the upper torso
// and leave enough depth behind the bind-pose bounds for the running animation.
constexpr glm::vec3 ninjaFollowTargetOffset(0.1f, 2.0f, -4.0f);
constexpr float ninjaFollowDistance = 8.0f;
constexpr float ninjaFollowHeight = 1.6f;
}

GameWorld::GameWorld()
    : ninja_(ninjaPosition, ninjaForward, ninjaMovementSpeed)
{
}

void GameWorld::configureRenderer(RendererConfig& config)
{
    config.additionalScenes.push_back({
        "models/ninja_run_free_fire_emote.glb",
        glm::translate(glm::mat4(1.0f), ninjaPosition) *
            glm::scale(glm::mat4(1.0f), glm::vec3(ninjaScale))
    });
}

void GameWorld::initialize(VulkanRenderer& renderer, std::uint32_t viewportWidth,
    std::uint32_t viewportHeight)
{
    ninjaActor_.reset();
    if (const std::optional<SceneBounds> bounds = renderer.sceneBounds())
    {
        camera_.frame(bounds->minimum, bounds->maximum, viewportWidth, viewportHeight);
    }
    else
    {
        camera_.setViewport(viewportWidth, viewportHeight);
    }
    camera_.setFollowTarget(ninja_.position() + ninjaFollowTargetOffset * ninjaScale,
        ninjaForward, ninjaFollowDistance * ninjaScale, ninjaFollowHeight * ninjaScale);

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
        ninja_.updateMovement(input, deltaSeconds);
    }
    camera_.setViewport(viewportWidth, viewportHeight);
    camera_.setFollowTarget(ninja_.position() + ninjaFollowTargetOffset * ninjaScale,
        ninjaForward, ninjaFollowDistance * ninjaScale, ninjaFollowHeight * ninjaScale);
    camera_.update(input, deltaSeconds);
    updateCameraSubmission();
}

void GameWorld::updateCameraSubmission()
{
    submission_.view = camera_.view();
    submission_.projection = camera_.projection();
    submission_.cameraPosition = camera_.position();
    submission_.animationActorTransforms.clear();
    if (ninjaActor_)
    {
        submission_.animationActorTransforms.push_back(
            {*ninjaActor_, ninja_.worldOffset()});
    }
}

}
