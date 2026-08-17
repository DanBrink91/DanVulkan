#pragma once

namespace danvulkan::application {

// One frame of application input. The window backend translates native events into this
// renderer-independent state; gameplay systems can replace that translation without changing
// the renderer or camera.
struct InputState
{
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool toggleCameraMode = false;
    float lookDeltaX = 0.0f;
    float lookDeltaY = 0.0f;
    float scrollDelta = 0.0f;
};

}
