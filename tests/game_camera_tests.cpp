#include "application/game_camera.hpp"
#include "application/game_character.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <glm/geometric.hpp>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

bool finite(const glm::mat4& matrix)
{
    for (glm::length_t column = 0; column < 4; ++column)
    {
        for (glm::length_t row = 0; row < 4; ++row)
        {
            if (!std::isfinite(matrix[column][row]))
            {
                return false;
            }
        }
    }
    return true;
}

void testSceneFraming()
{
    danvulkan::application::GameCamera camera;
    const glm::vec3 minimum(-4.0f, -2.0f, -1.0f);
    const glm::vec3 maximum(6.0f, 4.0f, 3.0f);
    camera.frame(minimum, maximum, 1280, 720);

    const glm::vec3 center = (minimum + maximum) * 0.5f;
    require(glm::distance(camera.position(), center) > 1.0f,
        "framed camera remained inside the scene center");
    require(camera.movementSpeed() > 0.0f, "framed camera has no movement speed");
    require(finite(camera.view()) && finite(camera.projection()),
        "framed camera produced a non-finite matrix");
}

void testNormalizedMovement()
{
    danvulkan::application::GameCamera forwardCamera;
    danvulkan::application::GameCamera diagonalCamera;
    forwardCamera.frame(glm::vec3(-1.0f), glm::vec3(1.0f), 1280, 720);
    diagonalCamera.frame(glm::vec3(-1.0f), glm::vec3(1.0f), 1280, 720);
    const glm::vec3 initialPosition = forwardCamera.position();

    danvulkan::application::InputState forward;
    forward.moveForward = true;
    forwardCamera.update(forward, 0.5f);

    danvulkan::application::InputState diagonal;
    diagonal.moveForward = true;
    diagonal.moveRight = true;
    diagonalCamera.update(diagonal, 0.5f);

    const float forwardDistance = glm::distance(initialPosition, forwardCamera.position());
    const float diagonalDistance = glm::distance(initialPosition, diagonalCamera.position());
    require(std::abs(forwardDistance - diagonalDistance) < 0.0001f,
        "diagonal camera movement was not normalized");
}

void testLookAndZeroViewport()
{
    danvulkan::application::GameCamera camera;
    camera.frame(glm::vec3(-1.0f), glm::vec3(1.0f), 1280, 720);
    const glm::mat4 previousView = camera.view();

    danvulkan::application::InputState input;
    input.lookDeltaX = 20.0f;
    input.lookDeltaY = -10.0f;
    camera.update(input, 0.0f);
    camera.setViewport(0, 0);

    require(camera.view() != previousView, "look input did not change the camera view");
    require(finite(camera.view()) && finite(camera.projection()),
        "zero viewport invalidated the camera matrices");
}

void testFollowCameraToggle()
{
    danvulkan::application::GameCamera camera;
    camera.setFollowTarget(glm::vec3(2.0f, 0.0f, -3.0f),
        glm::vec3(0.0f, 0.0f, 1.0f), 0.08f, 0.016f);

    danvulkan::application::InputState toggle;
    toggle.toggleCameraMode = true;
    camera.update(toggle, 0.0f);

    require(camera.mode() == danvulkan::application::CameraMode::follow,
        "camera did not enter follow mode");
    require(glm::distance(camera.position(), glm::vec3(2.0f, 0.016f, -3.08f)) < 0.0001f,
        "follow camera was not placed behind its target");
    require(finite(camera.view()), "follow camera produced a non-finite view matrix");

    camera.update(toggle, 0.0f);
    require(camera.mode() == danvulkan::application::CameraMode::free,
        "camera did not return to free mode");
}

void testCharacterMovement()
{
    danvulkan::application::GameCharacter forwardCharacter(
        glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2.0f);
    danvulkan::application::InputState forward;
    forward.moveForward = true;
    forwardCharacter.updateMovement(forward, 0.5f);
    require(glm::distance(forwardCharacter.position(), glm::vec3(2.0f, 0.0f, -2.0f)) <
            0.0001f,
        "W did not move the followed character forward");

    danvulkan::application::GameCharacter diagonalCharacter(
        glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2.0f);
    danvulkan::application::InputState diagonal;
    diagonal.moveForward = true;
    diagonal.moveRight = true;
    diagonalCharacter.updateMovement(diagonal, 0.5f);
    require(std::abs(glm::distance(glm::vec3(2.0f, 0.0f, -3.0f),
                         diagonalCharacter.position()) - 1.0f) < 0.0001f,
        "diagonal character movement was not normalized");
    require(diagonalCharacter.position().x < 2.0f &&
            diagonalCharacter.position().z > -3.0f,
        "W/D did not move the character along its forward and right axes");
    require(glm::distance(glm::vec3(diagonalCharacter.worldOffset()[3]),
                diagonalCharacter.position() - glm::vec3(2.0f, 0.0f, -3.0f)) < 0.0001f,
        "character world offset did not match gameplay movement");

    danvulkan::application::GameCharacter leftCharacter(
        glm::vec3(2.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), 2.0f);
    danvulkan::application::InputState left;
    left.moveLeft = true;
    leftCharacter.updateMovement(left, 0.5f);
    require(glm::distance(leftCharacter.position(), glm::vec3(3.0f, 0.0f, -3.0f)) <
            0.0001f,
        "A did not move the followed character toward the left side of the screen");
}

}

int main()
{
    try
    {
        testSceneFraming();
        testNormalizedMovement();
        testLookAndZeroViewport();
        testFollowCameraToggle();
        testCharacterMovement();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
