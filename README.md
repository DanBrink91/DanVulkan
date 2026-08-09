# DanVulkan

DanVulkan is a Vulkan 1.4 rendering sandbox being refactored from an older Visual Studio-only prototype into a reusable C++ rendering module.

## Requirements

- Windows 10 or 11
- Visual Studio 2022 with the **Desktop development with C++** workload
- A Vulkan 1.4-capable driver
- The Vulkan SDK 1.4.x with `VULKAN_SDK` set
- CMake 3.25 or newer
- Git (CMake downloads pinned third-party dependencies on the first configure)

The current baseline uses Vulkan 1.4, GLFW 3.4, GLM 1.0.1, Vulkan Memory Allocator 3.3.0, fastgltf 0.9.0, tinyobjloader 2.0.0-rc13, and stb_image. Dependencies are placed under `build/`; there are no machine-specific include or library paths.

## VS Code

Install the recommended CMake Tools and C/C++ extensions, then:

1. Open this repository as the workspace.
2. Run **CMake: Configure** if configuration does not start automatically.
3. Press `Ctrl+Shift+B` to compile the Debug build and the GLSL shaders.
4. Press `F5` to build and debug, or run the **DanVulkan: run** task.

From a terminal, the equivalent commands are:

```powershell
cmake --preset windows
cmake --build --preset debug
& .\build\windows\Debug\DanVulkan.exe
```

Run the CPU asset, animation-player, and device-planning tests plus validation-layer standalone-loop, step-driven, and resize smoke tests with `ctest --preset debug`.

Run the executable with the repository root as its working directory so it can find `models/` and `textures/`.

## Module boundary

Applications include `<danvulkan/renderer.hpp>`, create a `RendererConfig`, and own a `VulkanRenderer`. The public header hides Vulkan and windowing implementation details behind a private implementation. Applications can either call `run()` for the standalone demo or own the loop through `initialize()`, `beginFrame()`, `submitScene()`, `endFrame()`, and `shutdown()`:

```cpp
VulkanRenderer renderer(config);
renderer.initialize();
const SceneInstanceHandle player = renderer.sceneInstances().front().handle;
while (!renderer.shouldClose())
{
    // Resource changes happen between frames.
    renderer.updateInstanceTransform(player, gameWorld.playerTransform());
    if (!renderer.beginFrame())
        break;

    SceneSubmission scene;
    scene.view = gameCamera.view();
    scene.projection = gameCamera.projection();
    scene.cameraPosition = gameCamera.position();
    scene.lightPosition = gameLight.position();
    renderer.submitScene(scene);
    renderer.endFrame();
}
renderer.shutdown();
```

The renderer loads the scene named by `RendererConfig::modelPath` during initialization. `SceneSubmission` supplies per-frame view and lighting state. `sceneMeshes()`, `sceneInstances()`, and `sceneMaterials()` expose generation-tagged handles plus their current values. Between frames, games can create and remove PBR materials, upload and remove meshes, create and remove instances, and update world transforms or material factors. Destroyed material, mesh, and instance slots are reusable, but their generations advance so stale handles are rejected.

Internally, physical-device probing and logical-device ownership live in a focused device context. Selection requires Vulkan 1.4, every renderer feature, complete graphics/presentation queues, and adequate swapchain support; suitable candidates are scored deterministically rather than accepted in driver enumeration order. Queue-family selection and candidate scoring are Vulkan-free planning functions with dedicated CPU tests.

Swapchain selection and ownership also live behind a focused context. Pure planning chooses the surface format, presentation mode, extent, image count, sharing mode, transform, and supported composite alpha; CPU tests cover those decisions. The context owns the swapchain and its image views, uses the previous handle during recreation, and receives framebuffer pixels from the platform adapter so it remains independent of GLFW.

Descriptor policy and ownership are similarly isolated. Pure planning clamps bindless texture capacity, describes the renderer's shader bindings, and calculates overflow-safe pool sizes. The descriptor context owns the layout, pool, and per-swapchain sets; it performs initial buffer/texture writes plus the generation-gated vertex and texture migrations used by runtime resource updates.

`RendererConfig::additionalScenes` composes more glTF/GLB assets above the primary scene with a caller-supplied root transform. The demo uses it to place `ninja_run_free_fire_emote.glb` at village scale and automatically loops that file's first animation. Skinned vertices retain four joint indices and normalized weights; a CPU animation player evaluates linear or step translation, rotation, and scale channels, propagates the node hierarchy, and writes a per-frame joint palette consumed by the vertex shader. This temporary character proves the animation path while `naruto.glb` remains static and cannot yet receive the clip without being rigged.

`sceneAnimations()` exposes the active scene's clips through generation-tagged handles. The first clip still autoplays for convenience, but a host can select and control it between frames:

```cpp
const SceneAnimationHandle run = renderer.sceneAnimations().front().handle;
renderer.playAnimation(run);
renderer.setAnimationLooping(true);
renderer.setAnimationPlaybackSpeed(1.2f);

// These are also valid between frames.
renderer.pauseAnimation();
renderer.seekAnimation(0.15f);
renderer.resumeAnimation();
const AnimationPlaybackState playback = renderer.animationPlaybackState();
```

`stopAnimation()` rewinds to the selected clip's first pose, while a non-looping clip reports `AnimationPlaybackStatus::finished` at its final pose. `playAnimation(handle, false)` preserves the cursor of an already-selected active or paused clip; selecting a different clip or replaying a stopped or finished clip starts from its beginning. Scene replacement invalidates old animation handles just like other scene-owned handles. The active scene currently has one global playback cursor, so independent playback for multiple animated actors and clip blending remain future work.

The convenience demo captures the mouse for first-person look, uses WASD for movement, and closes with Escape. Its initial camera position, clip planes, and movement speed are derived from the loaded scene bounds, so changing the configured model does not require another hard-coded camera pose.

Leaving `RendererConfig::platform` null selects the convenient renderer-owned GLFW backend. A host application can instead supply a shared `RendererPlatform` implementation around its existing native window. The adapter provides event pumping, close state, framebuffer extent and resize notification, Vulkan instance extensions, and surface creation without exposing GLFW types. The renderer owns the returned `VkSurfaceKHR`, while native-window and input ownership stay with the adapter/host. An adapter may make `pollEvents()` a no-op when the host pumps events before `beginFrame()`. `makeGlfwRendererPlatform()` exposes the default backend explicitly when desired.

`replaceScene()` accepts a complete decoded `SceneAsset` between frames. It validates and builds the replacement CPU metadata, textures, and device-local geometry before committing any live state. A successful commit advances the scene generation, invalidating every old handle at once. Old texture and geometry owners remain in the existing per-swapchain-image retirement queues until descriptors and command buffers have safely migrated, so replacement does not call `vkDeviceWaitIdle`.

`RendererConfig::initialVertexCapacity` and `initialIndexCapacity` reserve device-local geometry ranges (4096 vertices and 8192 indices by default). Runtime uploads allocate a pair of free ranges and transfer only the new vertex/index bytes. `destroyMesh()` rejects meshes with live instances and defers returning their ranges until every swapchain image that could contain an older draw has completed. Adjacent free ranges are coalesced. If no contiguous range fits, capacity grows geometrically; swapchain images migrate to the replacement vertex buffer after their fences complete, and both old buffers remain alive until migration finishes.

`RendererConfig::maxTextures` reserves bindless descriptor capacity (256 by default, clamped to the physical device limits). `RendererConfig::textureMipLodBias` controls the renderer-wide sampler bias; its mild `-0.5` default keeps atlas-backed terrain a little sharper while generated mipmaps, trilinear filtering, and anisotropy still suppress minification shimmer. `uploadTexture()` accepts a decoded `TextureAsset`, creates its Vulkan image/view/sampler, and returns a persistent texture handle. `updateMaterialTextures()` can rebind the five core glTF texture roles between frames. `destroyTexture()` rejects textures still referenced by a material, replaces the freed descriptor slot with a valid fallback, and advances that slot's generation before reuse. Each swapchain image receives new descriptors only after its fence completes; retired images, views, and samplers stay alive until every descriptor set has migrated.

`RendererConfig::maxMaterials` reserves per-swapchain-image material storage (2048 by default, clamped to the shader ABI limit). A `RuntimeMaterialDescription` combines PBR factors, texture handles, alpha mode, and sidedness; newly created materials can be passed directly to later mesh uploads without rebuilding descriptors or waiting for the device to become idle. `destroyMaterial()` rejects materials still referenced by a mesh, removes the slot from scene queries, and advances its generation before reuse. Per-image material buffers make the reused GPU index visible only after the corresponding in-flight fence completes.

The renderer uses explicit frame resources, synchronization2, dynamic rendering, and VMA-backed resource owners. Its dedicated upload subsystem owns staging allocation, transfer commands, synchronization2 barriers, GPU mip generation, and fence-backed submission; static scene geometry and decoded textures reach device-local memory without scene code managing upload mechanics.

The GPU material path implements glTF metallic-roughness shading with base color, normal scale, packed metallic/roughness, occlusion strength, and emissive inputs. Texture filtering and wrap modes are imported per glTF texture. Opaque, alpha-mask, and alpha-blended materials use explicit single- and double-sided pipeline variants; transparent draws are sorted back-to-front within each culling variant.

CPU asset types live in the separate Vulkan-free `DanVulkan::Assets` library and `<danvulkan/assets.hpp>`. `.gltf`, `.glb`, and compatibility OBJ imports produce a `SceneAsset` with generation-tagged resource handles, node hierarchies, transforms, mesh instances, decoded textures, and PBR metallic-roughness inputs. Only the renderer upload boundary creates GPU resources. The bundled demo now loads `models/naruto_hiddenly_village.glb` by default. The next stages are described in [docs/MODERNIZATION.md](docs/MODERNIZATION.md).

The bundled Sketchfab-derived Naruto, village, and ninja-run assets declare CC-BY-4.0 metadata inside their GLB files. Their embedded attribution names and source URLs must be retained when redistributing the assets.
