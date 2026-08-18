#pragma once

#include "game_camera.hpp"
#include "game_character.hpp"

#include <danvulkan/renderer.hpp>

#include <cstdint>
#include <optional>

namespace danvulkan::application {

// Owns demo/gameplay state and produces the renderer's per-frame scene description.
class GameWorld
{
public:
    GameWorld();

    static void configureRenderer(RendererConfig& config);

    void initialize(VulkanRenderer& renderer, std::uint32_t viewportWidth,
        std::uint32_t viewportHeight);
    void update(const InputState& input, float deltaSeconds, std::uint32_t viewportWidth,
        std::uint32_t viewportHeight);
    void setEnvironment(const SceneEnvironment& environment) noexcept;
    void setKeyLightIntensity(float intensity) noexcept;

    [[nodiscard]] const SceneSubmission& sceneSubmission() const noexcept
    {
        return submission_;
    }

private:
    void updateCameraSubmission();

    GameCamera camera_;
    GameCharacter ninja_;
    std::optional<SceneAnimationActorHandle> ninjaActor_;
    SceneSubmission submission_;
};

}
