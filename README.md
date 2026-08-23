# DanVulkan

DanVulkan is a Vulkan 1.4 rendering sandbox being refactored from an older Visual Studio-only prototype into a reusable C++ rendering module.

## Requirements

- Windows 10 or 11 with Visual Studio 2022 and the **Desktop development with C++** workload, or macOS 15+ with Apple Clang
- A Vulkan 1.4-capable driver
- The Vulkan SDK 1.4.x with `VULKAN_SDK` set (the macOS SDK includes MoltenVK)
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

Run the CPU asset, animation-player, and planning tests plus validation-layer standalone-loop,
step-driven, resize, minimize/restore, and animated stress paths with `ctest --preset debug`.

Run the executable with the repository root as its working directory so it can find `models/`,
`textures/`, and the looping `audio/bg.mp3` background track.

## macOS with MoltenVK

Install the current macOS Vulkan SDK from LunarG, then source its `setup-env.sh` in each new terminal so CMake, the Vulkan loader, MoltenVK, validation layers, and `glslc` can be found. Configure and build the native Debug preset with:

```bash
source /path/to/VulkanSDK/1.4.x.x/setup-env.sh
cmake --preset macos
cmake --build --preset macos-debug
ctest --preset macos-debug
./build/macos/DanVulkan
```

Use `cmake --preset macos-release` followed by `cmake --build --preset macos-release` for an optimized build. The renderer enables portability enumeration when the loader exposes it, enables MoltenVK's required portability-subset device extension, and disables the configured sampler LOD bias only when that portability feature is unavailable.

## Module boundary

The executable's application-facing code lives in `src/application/`. `GameApp` owns the window,
input translation, clock, main loop, and `GameAudio` lifetime; `GameWorld` owns demo/gameplay state
and creates each `SceneSubmission`; `GameCamera` owns navigation and camera matrices; and `GameUi`
builds the debug overlay. These are the primary files to extend when building a demo or gameplay
feature. The normal no-argument launch uses this layer, while command-line validation and profiling
modes stay isolated in `main.cpp`.

The application demo maintains a 5-by-5 window of seeded, five-octave `GeneratedTerrain` chunks
around the player. Startup publishes a 3-by-3 terrain safety window and full center vegetation in
one non-blocking transfer batch, then renders while the rest fills progressively. Directional
prefetch uses two persistent priority workers with stale-request cancellation. Workers generate,
validate, tile-sort, and pack grass into its final GPU records; the render thread publishes at most
one ready chunk between frames. A whole chunk's terrain, grass, trunks, and canopy share one
asynchronous transfer submission instead of eight fence-waiting range uploads. The committed
window remains visible until its replacement is complete, after which the old edge is retired.
Run `DanVulkan --terrain-traversal-check` (or CTest's `danvulkan_terrain_traversal`) for a
scripted Vulkan check that crosses two terrain chunk boundaries, waits for the requested 5-by-5
vegetation window, and reports maximum render/publish times plus stall counts.
Application startup reserves enough geometry for the active window plus a diagonal transition, so
streaming does not resize the combined GPU buffers. Each 97-by-97 indexed patch is generated from
world-space noise, keeping positions and normals continuous across seams while the steady live
geometry budget remains fixed as the playable world expands. Mesh vertices, interpolated gameplay
heights, slope normals, and elevation/slope colors all come from the same CPU samples. A shared
world-space `TerrainSurfaceField` adds a meandering dirt-yellow path and seamless Voronoi grass
clumps. The procedural provider is intentionally interchangeable with a future authored paint-map
provider: both produce grass coverage, dirt weight, and grass traits, and their samples can be
blended. Each forest chunk considers 50,000 globally jittered blade positions, suppresses them on
the path, and gives every surviving blade the height, direction, color, bend, stiffness, and wind
phase of its nearest Voronoi site. Grass is stored as one compact record per blade and expanded by
a dedicated vertex shader into a lit cubic-Bezier ribbon. World-aligned internal tiles are culled
and assigned six-, three-, or two-segment curve LOD independently. Hysteresis stabilizes segment
changes while ranked populations transition smoothly between 100%, 50%, 18%, and zero density.
The close curve has a broadened root, a true pointed tip, per-blade taper, curvature, and camber,
plus corrected ribbon normals. A low-frequency flow field aligns neighboring Voronoi clumps,
traveling gust fronts move coherently across the field, and restrained individual flutter keeps
the close leaves from looking synchronized. The shared fragment shader adds center highlights,
soft two-sided diffuse response, and a path-aware short-growth underlayer without allocating more
grass records. Beneath it, a photographed-style forest-floor albedo is mirrored in world space
for seamless chunk-independent mapping and mixed with macro tint variation, procedural bump and
roughness detail, and root darkening. Exposed earth, moss, tiny plants, bark, twigs, and needles
remain recognizable wherever the grass canopy opens.
Packed dirt paths use a companion photographed-style albedo with their own mirrored world-space
scale, shallow irregularity, feathered moss edges, and sparse embedded pebbles. A per-frame
vegetation interactor also pushes
and lowers nearby blades around the character entirely on the GPU.
Trees remain spaced low-poly trunks with layered faceted canopies. A warm key and cool fill follow
the player over a brighter neutral environment. In follow-camera mode, the character controller
accelerates and brakes, turns the animated actor toward travel, follows streamed terrain height,
rejects steep slopes, and crosses chunk boundaries without changing its WASD basis.

Reusable hosts include `<danvulkan/renderer.hpp>`, create a `RendererConfig`, and own a `VulkanRenderer`. The public header hides Vulkan and windowing implementation details behind a private implementation. Applications can either use the compatibility `run()` loop or own the loop through `initialize()`, `beginFrame()`, `submitScene()`, `endFrame()`, and `shutdown()`:

```cpp
VulkanRenderer renderer(config);
renderer.initialize();
const SceneInstanceHandle player = renderer.sceneInstances().front().handle;
SceneSubmission scene;
while (!renderer.shouldClose())
{
    // Resource changes happen between frames.
    renderer.updateInstanceTransform(player, gameWorld.playerTransform());
    if (!renderer.beginFrame())
        break;

    scene.view = gameCamera.view();
    scene.projection = gameCamera.projection();
    scene.cameraPosition = gameCamera.position();
    scene.pointLights.front().position = gameLight.position();
    scene.pointLights.front().color = gameLight.color();
    scene.pointLights.front().intensity = gameLight.intensity();
    scene.environment.intensity = 0.05f;
    renderer.submitScene(scene);
    renderer.endFrame();
}
renderer.shutdown();
```

The renderer loads the scene named by `RendererConfig::modelPath` during initialization. The path
may be empty when `additionalScenes` supplies the initial renderable content, as it does in the
application demo. `SceneSubmission` supplies per-frame view, point-light, and environment controls.
Additional `ScenePointLight` values can be appended independently, up to
`RendererConfig::maxPointLights`. `sceneBounds()` supplies optional combined world-space bounds for
app-owned camera framing. `sceneMeshes()`, `sceneInstances()`, and `sceneMaterials()` expose
generation-tagged handles plus their current values. Between frames, games can create and remove
PBR materials, upload and remove meshes, create and remove instances, and update world transforms
or material factors. Destroyed material, mesh, and instance slots are reusable, but their
generations advance so stale handles are rejected.

Internally, physical-device probing and logical-device ownership live in a focused device context. Selection requires Vulkan 1.4, every renderer feature, complete graphics/presentation queues, and adequate swapchain support; suitable candidates are scored deterministically rather than accepted in driver enumeration order. Queue-family selection and candidate scoring are Vulkan-free planning functions with dedicated CPU tests.

Swapchain selection and ownership also live behind focused contexts. Pure planning chooses the surface format, presentation mode, extent, image count, sharing mode, transform, and supported composite alpha; CPU tests cover those decisions. The swapchain context owns the swapchain and its image views, the attachment context owns per-frame depth and optional MSAA color targets, and the presentation context owns render-finished semaphores plus image-fence tracking. Recreation receives framebuffer pixels from the platform adapter, waits only frame fences and the presentation queue, and preserves format-compatible pipelines and same-count presentation semaphores.

Descriptor policy and ownership are similarly isolated. Pure planning clamps bindless texture capacity, describes the renderer's shader bindings, and calculates overflow-safe pool sizes. The descriptor context owns the layout, pool, and per-swapchain sets; it performs initial buffer/texture writes plus the generation-gated vertex and texture migrations used by runtime resource updates.

Graphics-pipeline policy and ownership live in a transactional pipeline context. Pure planning defines the culling, blending, and depth-write state for all six material variants. Shader modules, the pipeline layout, and pipeline handles are replaced as one complete set, and extent-only swapchain recreation retains compatible pipelines. Per-frame command pools, command buffers, acquire semaphores, fences, and timestamp queries are likewise owned by frame contexts; completed-frame timestamps are collected after their fence wait without serializing every submitted frame.

Uploaded-scene ownership is isolated in a scene context. It holds scene metadata and handles, geometry and textures, per-swapchain scene buffers, free ranges, generation-gated retirement queues, and reusable animation/culling/indirect scratch. It also synchronizes animated poses, transforms bounds, culls and orders visible draws, and builds per-pipeline indirect batches. The top-level renderer now coordinates these focused owners and records frames without owning scene storage directly.

Lighting ownership is isolated in a lighting context. It holds renderer-owned RGBA32F irradiance, GGX-prefiltered specular, and split-sum BRDF images plus persistently mapped point-light storage for each swapchain image. `RendererConfig::environmentMap` accepts an optional decoded RGBA8 or linear RGBA32F equirectangular environment without exposing Vulkan; `assets::loadEnvironment("sky.hdr")` preserves HDR values, while omitting the map selects a small neutral fallback. A Vulkan-free startup stage builds the three IBL inputs, and the shader combines them with bounded multi-light Cook-Torrance direct lighting. Directional/spot lights and shadows remain future refinements.

`RendererConfig::additionalScenes` composes glTF/GLB assets with a caller-supplied root transform,
with or without a primary scene. The demo uses it to place `ninja_run_free_fire_emote.glb` at
gameplay scale above the generated terrain and automatically loops that file's first animation.
Skinned vertices retain four joint indices and normalized weights; a CPU animation player evaluates
linear or step translation, rotation, and scale channels, propagates the node hierarchy, and writes
a per-frame joint palette consumed by the vertex shader. This temporary character proves the
animation path while `naruto.glb` remains static and cannot yet receive the clip without being
rigged.

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

`stopAnimation()` rewinds to the selected clip's authored starting poses, while a non-looping clip reports `AnimationPlaybackStatus::finished` at its final pose. `playAnimation(handle, false)` preserves the cursors of already-selected active or paused instances; selecting a different clip or replaying stopped or finished instances restores their authored phase offsets. Scene replacement invalidates old animation handles just like other scene-owned handles. `AnimationInstanceAsset` can map one immutable clip onto independent actor hierarchies with separate position, speed, looping, and status state. Clip blending remains future work.

Animation clips compile into packed channels and values, shared timelines, precomputed interval reciprocals, and per-instance advancing cursors. `RendererConfig::animation` controls actor-level pose culling plus full-, medium-, and far-rate distance tiers. Culled or rate-limited actors retain their last pose and palette while their playback clocks continue advancing, and an actor evaluates immediately when it becomes visible again.

Compiled clips fold exactly constant tracks out of steady evaluation and split translation, scale,
and rotation tracks by interpolation mode. Steady updates overwrite animated components directly;
base poses are restored only for initialization and playback operations such as changing clips or
stopping. `RendererConfig::animation.adaptiveNlerpMaxAngleRadians` optionally substitutes normalized
linear quaternion interpolation for adjacent keys within a caller-selected angular threshold. Its
zero default retains exact slerp. Hierarchies also compile into one packed parent-before-child
propagation plan, with allocation-free per-instance subtree ranges. Pose-driven nodes compose TRS
directly into parent space, while static TRS and affine matrix locals reuse a cached matrix;
arbitrary matrix-authored locals retain the general multiplication path.

The application demo captures the mouse for first-person look, uses WASD to move the free camera,
scrolls forward/backward, toggles between free and ninja-follow cameras when C is released, and
closes with Escape. The application starts in follow mode, where WASD moves the ninja while the
chase camera tracks it.
F1 toggles the application-owned immediate-mode control and statistics panel; opening it releases
the pointer and suspends gameplay input. Clicking outside the panel or pressing WASD/C returns
keyboard and captured-mouse control to gameplay while leaving the panel visible. Tab re-enters UI
interaction; F1 hides the panel. The wheel scrolls overflowing content, Tab and Shift-Tab move
keyboard focus, Enter or Space activates the focused button/checkbox, and Left/Right adjusts the
focused slider. Its initial free-camera position, clip planes, and movement speed are
derived from `sceneBounds()`, so changing the configured model does not require another hard-coded
camera pose.

`ImmediateUi` provides panels, text, separators, buttons, checkboxes, float sliders, persistent
collapsing headers, and multi-series history plots. It emits renderer-neutral colored/glyph
triangles through `UiDrawData`; `submitUi()` copies those triangles into a persistently mapped arena
for the current frame and renders them in a single-sample overlay pass after scene resolve. The
built-in 5x7 shader font keeps the UI asset-free. `UiInteractionResult` separately reports panel hit
testing, pointer and keyboard ownership, and active/focused widget IDs so the application does not
infer capture from panel visibility. `GameUi` retains 120 frame samples even while hidden and graphs
frame/CPU/GPU timing, animation stages, draw counts, and memory alongside its background-music,
animation, and lighting controls. Editable text, nested layout, movable or resizable panels,
gamepad navigation, and multiple native windows are not yet implemented.

Leaving `RendererConfig::platform` null selects the convenient renderer-owned GLFW backend. A host application can instead supply a shared `RendererPlatform` implementation around its existing native window. The adapter provides event pumping, close state, framebuffer extent and resize notification, Vulkan instance extensions, and surface creation without exposing GLFW types. The renderer owns the returned `VkSurfaceKHR`, while native-window and input ownership stay with the adapter/host. An adapter may make `pollEvents()` a no-op when the host pumps events before `beginFrame()`. `makeGlfwRendererPlatform()` exposes the default backend explicitly when desired.

`replaceScene()` accepts a complete decoded `SceneAsset` between frames. It validates and builds the replacement CPU metadata, textures, and device-local geometry before committing any live state. A successful commit advances the scene generation, invalidating every old handle at once. Old texture and geometry owners remain in the existing per-swapchain-image retirement queues until descriptors and command buffers have safely migrated, so replacement does not call `vkDeviceWaitIdle`.

A Vulkan-free scene planner now validates resource references, graph ownership, animation data,
capacities, skin indices, and decoded textures while packing geometry and flattening hierarchy draws.
Startup and transactional replacement consume the same plan, including animated replacements.
`RendererPerformanceStats` exposes separate rolling end-to-end frame wall time, render-thread CPU
time, GPU time, detailed animation stage and actor-policy, culling, mapped-buffer-write, and
command-recording measurements, including propagated, pose-composed, and cached-local node counts.
Frame wall time includes synchronization and presentation pacing; render-thread CPU time excludes
blocked and descheduled time. Run
`DanVulkan --animated-stress-test` to exercise 100 independently phased characters sharing one
uploaded character resource set and immutable 152-channel clip. The stress run uses 120 warm-up
frames followed by 480 measured frames, then reports means and selected variability across the
post-warm-up rolling timing snapshots. `--animated-stress-test-nlerp` runs the same workload with
the separately measured 20-degree adaptive-nlerp policy.

`RendererConfig::memory.maxMsaaSamples` caps multisampling with a Vulkan-free numeric policy;
zero selects the device maximum and one disables MSAA. Transient MSAA color and depth targets are
allocated per frame in flight rather than per swapchain image, and lazy attachment memory is
preferred when the device exposes it. `RendererMemoryStats` reports current and renderer-sampled peak
VMA block/allocation counts and bytes, aggregate driver heap usage/budget, attachment-set count, and
selected sample count. It also exposes staging-arena growth/submission counters, retained CPU geometry
bytes, and persistent frame-scratch capacity. Swapchain-owned presentation images and ordinary CPU
allocations are outside the VMA totals.

`RendererConfig::initialVertexCapacity` and `initialIndexCapacity` reserve device-local geometry ranges (4096 vertices and 8192 indices by default). Runtime uploads allocate a pair of free ranges and transfer only the new vertex/index bytes. `destroyMesh()` rejects meshes with live instances and defers returning their ranges until every swapchain image that could contain an older draw has completed. Adjacent free ranges are coalesced. If no contiguous range fits, capacity grows geometrically; existing geometry is copied GPU-to-GPU, swapchain images migrate to the replacement vertex buffer after their fences complete, and both old buffers remain alive until migration finishes. Packed scene geometry exists on the CPU only while the initial or replacement upload is being prepared; spare GPU capacity and later growth do not retain a full CPU mirror.

`uploadGrass()` uses those same generation-safe geometry ranges for semantic blade records and a
small shared-in-upload curve template. Each upload is internally sorted into world-aligned 0.25-unit
tiles, keeping one allocation and transfer per streamed terrain chunk while letting the draw planner
frustum-cull and select curve detail independently per tile. Segment LOD uses a hysteresis dead band,
and stable per-blade ranks progressively thin density through transition bands rather than popping a
whole tile. The grass vertex pipeline evaluates the curve and wind without materializing its rendered
vertices on the CPU. Five padding scalars in the existing blade record carry curvature bias, taper,
flutter strength, flutter phase, and camber, so the close-quality upgrade adds no GPU record size or
transfer bandwidth; the semantic CPU generation record temporarily carries those controls.

`RendererConfig::maxTextures` reserves bindless descriptor capacity (256 by default, clamped to the physical device limits). `RendererConfig::textureMipLodBias` controls the renderer-wide sampler bias; its mild `-0.5` default keeps atlas-backed terrain a little sharper while generated mipmaps, trilinear filtering, and anisotropy still suppress minification shimmer. `uploadTexture()` accepts a decoded `TextureAsset`, creates its Vulkan image/view/sampler, and returns a persistent texture handle. `updateMaterialTextures()` can rebind the five core glTF texture roles between frames. `destroyTexture()` rejects textures still referenced by a material, replaces the freed descriptor slot with a valid fallback, and advances that slot's generation before reuse. Each swapchain image receives new descriptors only after its fence completes; retired images, views, and samplers stay alive until every descriptor set has migrated.

`RendererConfig::maxMaterials` reserves per-swapchain-image material storage (2048 by default, clamped to the shader ABI limit). A `RuntimeMaterialDescription` combines PBR factors, texture handles, alpha mode, and sidedness; newly created materials can be passed directly to later mesh uploads without rebuilding descriptors or waiting for the device to become idle. `destroyMaterial()` rejects materials still referenced by a mesh, removes the slot from scene queries, and advances its generation before reuse. Per-image material buffers make the reused GPU index visible only after the corresponding in-flight fence completes.

The renderer uses explicit frame resources, synchronization2, dynamic rendering, and VMA-backed resource owners. Its dedicated upload subsystem owns a single persistently mapped, geometrically grown staging arena, transfer commands, synchronization2 barriers, GPU mip generation, and fence-backed submission. Because submissions are currently synchronous, each payload safely reuses the arena from offset zero. Static scene geometry and decoded textures reach device-local memory without scene code managing upload mechanics.

The GPU material path implements glTF metallic-roughness shading with base color, normal scale, packed metallic/roughness, occlusion strength, and emissive inputs. Texture filtering and wrap modes are imported per glTF texture. Opaque, alpha-mask, and alpha-blended materials use explicit single- and double-sided pipeline variants; transparent draws are sorted back-to-front within each culling variant.

CPU asset types live in the separate Vulkan-free `DanVulkan::Assets` library and `<danvulkan/assets.hpp>`. `.gltf`, `.glb`, and compatibility OBJ imports produce a `SceneAsset` with generation-tagged resource handles, node hierarchies, transforms, mesh instances, decoded textures, and PBR metallic-roughness inputs. Only the renderer upload boundary creates GPU resources. The renderer default is the animated ninja asset; the application demo starts without the village and combines the scaled character with generated terrain. The next stages are described in [docs/MODERNIZATION.md](docs/MODERNIZATION.md).

The bundled Sketchfab-derived Naruto, village, and ninja-run assets declare CC-BY-4.0 metadata inside their GLB files. Their embedded attribution names and source URLs must be retained when redistributing the assets.
