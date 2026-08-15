# Modernization roadmap

The legacy renderer was a single 2,700-line application tied to Vulkan 1.1.130 and absolute paths. Modernization is being done in validated slices so that ownership, synchronization, and rendering changes remain independently testable.

## Completed

### Portable Vulkan 1.4 baseline

- CMake presets replace the machine-specific Visual Studio project.
- VS Code can configure, build, run, debug, and launch validation tests.
- Shaders compile automatically for Vulkan 1.4.
- The renderer has a public PIMPL boundary and a self-contained default asset.

### Lifetime and diagnostics

- Move-only owners manage `Instance`, `Surface`, `Device`, `Allocator`, `Swapchain`, `Buffer`, and `Image` lifetimes. Focused `DeviceContext` and `SwapchainContext` subsystems now contain the logical-device and presentation-resource owners.
- Allocated images own their views and release the view before returning the image allocation to VMA.
- Allocated buffers return their buffer allocation to VMA automatically.
- Checked Vulkan operations report the operation, symbolic `VkResult`, and numeric result.
- `VK_EXT_debug_utils` is enabled in validation builds, with a debug messenger and names for queues, pipelines, descriptors, synchronization objects, and GPU resources.

Texture samplers remain explicitly destroyed by the renderer while their image allocations are
RAII-owned. Pipeline, frame, descriptor, swapchain attachment, presentation, and upload
command/synchronization objects are owned by their focused subsystems.

### Device selection and context

- Pure device-planning code selects queue families and scores already-probed device capabilities without requiring a Vulkan instance, surface, window, or GPU. A combined graphics/presentation family is preferred; split families remain supported.
- Candidate validation requires a complete graphics/presentation queue plan, swapchain extension and surface formats/modes, Vulkan 1.4, and every feature used by the shaders and command paths. The former first-match path could accept a device without a presentation queue even though logical-device creation required one.
- Suitable devices are ranked deterministically by device class and image-dimension capability instead of accepting enumeration order. Discrete, integrated, virtual, CPU, and other devices retain explicit relative priorities.
- `DeviceContext` owns logical-device lifetime and retains the selected physical device, immutable properties, queue-family indices, and graphics/presentation queues. It performs the Vulkan feature-chain setup and offers swapchain-support queries without owning the surface or window.
- Renderer code now consumes cached device limits and context accessors rather than rediscovering queue families or repeatedly querying physical-device properties.
- A CPU-only device-planner test covers combined and split queue selection, incomplete queues, API/feature rejection, unsuitable candidate sets, device-class scoring, and same-class capability scoring.

### Swapchain planning and context

- Pure swapchain-planning functions choose the preferred sRGB surface format, mailbox/FIFO presentation mode, bounded image count, fixed or clamped framebuffer extent, combined/split queue sharing, surface transform, and a supported composite-alpha mode.
- `SwapchainContext` owns the swapchain handle, borrowed swapchain images, and every corresponding image view. Renderer code consumes read-only spans and cached format/extent values instead of managing those raw handles directly.
- The platform adapter still owns framebuffer-size queries, minimized-window event waiting, and resize notifications. The swapchain subsystem receives the resulting pixel extent without depending on GLFW or another window system.
- Recreation supplies the active handle through `oldSwapchain`, builds the replacement images and views first, then retires the previous views and handle after the new set is ready.
- A CPU-only planner test covers format and presentation preferences, fixed and clamped extents, image-count limits, composite-alpha fallback, queue sharing, and rejection of incomplete surface inputs.

### Descriptor planning and context

- Pure descriptor planning clamps bindless texture capacity against the requested value and both relevant device sampler limits, rejects scenes that cannot fit, and produces the seven-binding shader layout plus overflow-checked pool counts for any swapchain image count.
- `DescriptorContext` owns the descriptor-set layout, current pool, and per-image sets. It validates complete buffer and texture bindings, allocates replacement pools transactionally, initializes every bindless slot with the renderer-supplied fallback texture, and names its Vulkan objects in validation builds.
- Initial uniform, material, draw, transform, vertex, joint, and texture writes now cross a compact binding description instead of being assembled in `renderer.cpp`.
- Runtime geometry-buffer growth and bindless texture changes retain the existing fence/generation checks, but their per-image Vulkan writes now go through focused `updateVertex()` and `updateTextures()` operations.
- Swapchain image-count changes rebuild the context's pool and sets alongside the corresponding per-image buffers. The immutable layout remains stable for the graphics pipeline.
- A CPU-only descriptor-planner test covers capacity clamping/rejection, shader binding roles, pool sizing, zero-input rejection, and count-overflow rejection.

### Graphics pipeline planning and context

- Pure pipeline planning defines the six opaque, masked, and blended single- and double-sided variants, including their culling, blending, and depth-write policy plus dynamic-rendering attachment compatibility.
- `PipelineContext` owns shader-module loading, the pipeline layout, and every graphics-pipeline handle. Creation and shader reload build a complete replacement before retiring the active set, so a failed rebuild leaves the previous pipelines intact.
- Swapchain recreation retains pipelines when the descriptor layout, color/depth formats, and sample count remain compatible. Dynamic viewport and scissor state means an extent-only resize does not rebuild them.
- A CPU-only pipeline-planner test covers invalid attachment inputs, stencil selection, multisampling, and all six material variants.

### Frame and rendering model

- `FrameContext` owns the command pool, command buffer, acquire semaphore, fence, and timestamp query pool for each frame in flight.
- `PresentationContext` owns render-finished semaphores and image-fence tracking indexed by swapchain image. Same-count recreation retains its semaphores after presentation completion.
- `AttachmentContext` transactionally owns per-frame depth and optional MSAA color images plus their views. It retains the active set when extent, count, formats, samples, and allocation policy are unchanged.
- Both frame submissions and upload submissions use synchronization2 (`vkQueueSubmit2`).
- Resource transitions use synchronization2 barriers (`vkCmdPipelineBarrier2`).
- Dynamic rendering replaces render passes and framebuffers.
- Viewport and scissor are dynamic pipeline state.
- MSAA color and depth attachments are allocated per frame in flight. Fence-gated frame reuse prevents concurrent submissions from sharing writable attachments without tying their count to presentation buffering.
- GPU timestamps are read only after that frame context's fence completes. The renderer no longer requests blocking query results immediately after presentation, so its two frames in flight can overlap.
- Swapchain recreation waits for each frame fence and then the presentation queue before retiring old image views and dependent resources. It no longer stalls unrelated device queues with `vkDeviceWaitIdle`.

### Renderer decomposition

- `renderer.cpp` is now the lifecycle and frame-orchestration layer. Device, swapchain, attachment, presentation, descriptor, pipeline, frame, upload, and uploaded-scene lifetimes are held by focused contexts in separate translation units.
- `SceneContext` owns the active scene's CPU metadata, device-local vertex/index buffers, texture images and samplers, per-image material/transform/draw/joint/indirect buffers, handle generations, free geometry ranges, and persistent draw scratch as one lifetime domain.
- Per-image geometry and texture descriptor migration, old-buffer/texture retirement, deferred geometry-range reclamation, swapchain-wide retirement completion, and scene teardown are implemented by the scene context instead of the top-level renderer.
- Animation pose synchronization, transformed bounds, frustum culling, transparent ordering, and indirect batch construction are likewise scene operations. The renderer receives compact draw counts and records the resulting batches.
- The scene context is composed after the allocator/device owners, guaranteeing its Vulkan allocations are destroyed first even during partial construction. A CPU-focused context test covers range allocation/coalescing, capacity growth, transformed bounds, culling, transparent ordering, and pipeline batches.

### Memory and uploads

- Vulkan Memory Allocator 3.3.0 replaces manual memory-type selection, allocation, binding, mapping, and release.
- A pure memory-policy planner selects the highest mutually supported color/depth sample count under an optional renderer cap. `RendererConfig::memory.maxMsaaSamples` can disable MSAA or bound its memory cost without exposing Vulkan flags through the public API.
- Writable MSAA color and depth targets are owned per fenced frame in flight rather than per presentation image. With two frames and three swapchain images on the current 2560x1440 macOS baseline, this removes one color/depth pair and reduces VMA block memory by 114.5 MiB.
- Both discard-only color and depth targets declare transient attachment usage. VMA prefers lazily allocated memory when requested and supported, with ordinary device-local memory remaining the portable fallback.
- `RendererMemoryStats` reports VMA block/allocation counts and bytes, aggregate heap usage/budget, selected samples, and frame-indexed attachment-set count. The animated stress output captures these values alongside timing diagnostics.
- Host-visible uniform, storage, indirect, and staging buffers are persistently mapped; writes are flushed through VMA for non-coherent-memory portability.
- Static vertex and index data are staged into shared device-local buffers.
- `UploadContext` is a focused renderer subsystem in its own translation unit. It owns a dedicated transient command pool, reusable command buffer, and fence, plus one persistently mapped, geometrically grown VMA staging arena. Synchronous fence completion lets every upload safely reuse offset zero; growing payloads replace the arena, while smaller later uploads create no Vulkan buffer or allocation churn.
- Scene/resource code describes buffer or RGBA8 image uploads using destination, payload, and intended usage. Buffer copies, synchronization2 barriers, image layout transitions, queue submission, and fence waits remain private to the subsystem.
- A texture upload records its undefined-to-transfer transition, buffer-to-image copy, and shader-read transition in one synchronized submission. Uploads no longer borrow frame command pools or wait for the entire graphics queue to become idle.
- Sampled textures allocate a complete mip chain when their Vulkan format supports filtered blits. The upload subsystem generates every level on the GPU in the same synchronized submission, and image views and samplers expose the full chain so imported trilinear filtering works as authored.
- Scene plans retain only packed, used vertex and index elements; configured spare GPU capacity no longer pads the CPU vectors. Initial and replacement payloads are released after upload instead of becoming renderer-owned mirrors.
- Device-local geometry buffers include transfer-source usage. Capacity growth allocates replacement buffers and preserves existing ranges with GPU-to-GPU copies, so runtime mesh uploads no longer require full CPU vertex/index ownership.
- Joint-palette, visible-mesh bucket, indirect-command, and visible-draw scratch capacities are reserved from the prepared scene and retained across frames. Animation synchronization and frustum culling clear and reuse this storage rather than rebuilding it each frame.
- Memory telemetry includes renderer-sampled peak VMA block/allocation bytes, counts, and heap usage, plus staging-arena capacity/growth/submission counters, retained CPU geometry bytes, and reserved CPU scratch bytes. Peak sampling occurs while old and replacement scene resources coexist.
- In the 24-character animated stress path, the staging arena grows once to 16 MiB and services nine upload submissions; steady renderer CPU geometry ownership is zero and persistent animation/culling scratch is about 568 KiB. Current VMA allocation bytes are 281.5 MiB and the scene-replacement high-water mark is 313.5 MiB on the current macOS debug baseline. VMA block reservation remains 357.0 MiB because the arena fits in the allocator's existing blocks.

### Asset boundary

- `DanVulkan::Assets` is a Vulkan-free CPU asset library with scene, mesh, material, decoded RGBA8 texture, vertex, and bounds types.
- Generation-tagged `MeshHandle`, `MaterialHandle`, and `TextureHandle` replace ambiguous cross-resource integer references.
- The compatibility OBJ importer owns tinyobjloader and stb_image. It validates attribute indices, splits shapes by material, deduplicates textures, supplies a white fallback texture, and generates missing normals and tangents.
- Texture color space is explicit: albedo payloads upload as sRGB while normal and specular payloads upload as linear UNORM.
- `VulkanRenderer` now consumes a decoded `SceneAsset` through a focused CPU-to-GPU upload function; parsing and image decoding no longer create Vulkan resources.
- A CPU-only importer/handle test runs without a Vulkan device or window.
- fastgltf 0.9.0 is pinned privately behind `DanVulkan::Assets`; no importer-specific types leak into public headers.
- `loadScene` dispatches `.gltf`, `.glb`, and compatibility `.obj` assets. External, embedded, and GLB buffer/image payloads are decoded on the CPU.
- `SceneAsset` preserves node hierarchies, local transforms, mesh instances, and default-scene roots. The renderer uploads shared geometry once and emits per-node transform/draw data.
- A Vulkan-free scene planner validates textures, cross-resource handles, animation samples, skin joints, hierarchy cycles and ownership, and renderer capacities. It deterministically packs geometry and emits transforms, world bounds, draw records, and joint-palette offsets.
- Initial upload and transactional replacement now consume the same scene plan. Animated replacement is supported, and skinned primitives attached to the same node share one planned joint palette.
- Instance bounds are transformed to world space before frustum culling.
- glTF base-color, metallic-roughness, normal, occlusion, emissive, alpha, and double-sided inputs are represented in CPU materials and carried across the renderer upload boundary.
- The renderer's default demo scene is the bundled `models/naruto_hiddenly_village.glb`; the small triangle glTF and OBJ assets remain deterministic importer and runtime-replacement fixtures.

### Validation tests

- Validation-layer tests cover the standalone demo loop, the step-driven application loop, forced window resize/swapchain recreation, and deterministic zero-extent minimize/restore recovery; a validation error makes the executable and test fail.
- Fifteen tests now run: nine CPU-focused asset, animation, device, descriptor, swapchain, graphics-pipeline, memory-policy, scene-planning, and scene-context paths, plus six renderer validation/stress paths.
- The animated stress path instances 24 characters while sharing their uploaded textures, materials, and geometry. One aggregate clip advances all 3,648 animation channels; the test verifies 122 active/visible draws and 9,000 joint matrices without imposing hardware-specific timing thresholds.
- Rolling diagnostics split CPU animation evaluation from renderer pose/palette synchronization and also report culling, mapped-buffer writes, command recording, whole-frame CPU time, and GPU timestamps. On the current macOS debug baseline, animation synchronization is the dominant measured hot zone, followed by clip evaluation; culling, buffer writes, and command recording are negligible by comparison.

### Game integration

- `VulkanRenderer` exposes explicit `initialize`, `beginFrame`, `submitScene`, `endFrame`, and `shutdown` operations so an application can own its main loop.
- `SceneSubmission` makes the per-frame view, projection, camera position, and light position explicit rather than sourcing them unconditionally from the renderer's demo camera.
- `run()` is now a thin convenience loop over the same public frame API instead of a separate rendering path.
- Frame-order validation reports invalid lifecycle use such as submitting outside a frame, submitting twice, or ending a frame without a submission.
- `RendererConfig::platform` accepts a host-supplied `RendererPlatform`. The interface covers close state, event pumping, framebuffer extent/resize notification, required Vulkan instance extensions, and surface creation without exposing GLFW types.
- A null platform retains the convenient renderer-owned GLFW behavior, and `makeGlfwRendererPlatform()` permits explicit injection of that same backend. The renderer owns the returned Vulkan surface; a custom adapter keeps native-window and input ownership in the host.
- Window title and forced-resize hooks are optional conveniences. Host event loops can implement `pollEvents()` as a no-op, while minimized-surface recovery uses the adapter's framebuffer query and wait operation.
- The scene configured by `RendererConfig::modelPath` is loaded during initialization. Runtime texture, material, mesh, and instance composition plus transactional whole-scene replacement are available; simultaneous submission of multiple independent scenes remains future work.

### PBR material pipeline

- The legacy Blinn-Phong shader has been replaced with a Cook-Torrance metallic-roughness path using GGX distribution, Smith geometry, and Schlick Fresnel terms.
- The CPU/GPU material ABI carries base-color, metallic, roughness, normal scale, occlusion strength, emissive, alpha cutoff/mode, double-sided state, texture tiling, and all five core glTF texture inputs.
- The uniform ABI now uses explicitly aligned `vec4` camera/light fields, fixing the legacy `std140` mismatch between GLM `vec3` fields and GLSL uniform alignment.
- Imported tangent handedness is preserved. Generated tangents now calculate handedness, and normals use an inverse-transpose model matrix.
- Each decoded glTF texture retains its minification, magnification, mip-filter selection, and U/V wrap state. The Vulkan descriptor array uses a sampler created from that texture's settings instead of one global fixed sampler, with generated mip levels and anisotropic filtering available for minified surfaces. A device-limit-clamped `RendererConfig::textureMipLodBias` provides explicit sharpness control; the bundled atlas scene uses the renderer's mild `-0.5` default.
- Six graphics pipeline variants cover opaque, masked, and blended materials in single- and double-sided forms. Masked materials discard below their cutoff; blended variants enable source-alpha blending and disable depth writes.
- `KHR_materials_unlit` is preserved across the asset and runtime material APIs. Unlit fragments use their authored base color and alpha without applying the renderer's PBR lighting or tone mapping.
- Visible transparent draws are ordered back-to-front within each culling variant. Fully global ordering across pipeline variants remains future work.
- The bundled glTF fixture exercises a masked, double-sided PBR material, and CPU tests cover sampler import, alpha controls, emissive factors, and tangent handedness.

The current lighting environment is intentionally small: one submitted point light plus a constant ambient term. Image-based lighting, environment maps, multiple lights, shadowing, and `KHR_texture_transform` remain future rendering work.

### Skeletal animation

- The Vulkan-free asset model represents skins, joint-node handles, inverse bind matrices, animation clips, typed TRS channels, interpolation, and four normalized joint influences per vertex.
- fastgltf imports glTF skins plus linear and step translation, rotation, and scale channels. The ninja-run fixture verifies its 75-joint skin, 152 animation channels, 0.52-second duration, vertex weights, and all remapped handles after scene composition.
- `SceneAsset::append` composes independently authored assets while remapping textures, materials, meshes, nodes, skins, clips, and hierarchy roots. A caller-supplied transform places an appended model without modifying its source file.
- `AnimationPlayer` evaluates a selected imported clip on the CPU, uses quaternion slerp for rotations, propagates animated local transforms through the node hierarchy, and produces mesh-relative joint matrices. Animated rigid mesh nodes now receive evaluated transforms as well as skinned draws.
- Per-swapchain-image joint storage buffers keep palette writes synchronized with the existing frame fences. Draw metadata selects rigid or skinned vertex processing; the shader blends positions, normals, and tangents without changing static village rendering.
- The demo composes `ninja_run_free_fire_emote.glb` into the village at an authored scale and loops its clip automatically from `beginFrame`, including host-owned application loops.
- `sceneAnimations()` exposes clip names, durations, and generation-tagged handles. Public controls select/restart a clip, play, pause, resume, stop and rewind, seek, toggle looping, set positive playback speed, and query the current playback state. Mutations use the same between-frame contract as runtime scene updates, and scene replacement makes old clip handles stale.
- A Vulkan-free player test covers deterministic clip selection, pause/resume, speed scaling, seeking, looping policy, terminal state, stop/rewind behavior, and invalid inputs. The step-driven validation test exercises the public controls and stale-handle rejection against the live renderer.

The supplied `naruto.glb` has no skin, joint weights, skeleton, or animations, so it cannot be deformed by the run clip yet. Replacing the temporary character requires a rigged Naruto export with compatible humanoid joint semantics. Cubic-spline channels, morph targets, clip blending, independent player instances for multiple animated actors, general cross-skeleton retargeting, and a gameplay locomotion controller remain future work.

### Runtime scene updates

- Uploaded scene instances and materials now have public generation-tagged handles that remain stable for the lifetime of the current uploaded scene.
- `sceneInstances()` exposes each mesh-bearing node's name, handle, and evaluated model-to-world transform. `updateInstanceTransform()` changes that transform and immediately rebuilds the instance's world-space culling bounds.
- `sceneMaterials()` exposes each material's name, handle, and mutable PBR factors. `updateMaterialProperties()` changes base color, tiling, emissive, metallic, roughness, normal scale, occlusion strength, and alpha cutoff without disturbing its texture bindings or pipeline variant.
- Runtime mutation is explicitly restricted to the interval between frames. GPU transform and material data remain duplicated per swapchain image and are only rewritten after that image's in-flight fence completes, so an update never overwrites data still consumed by the GPU.
- Handles carry the uploaded scene generation, allowing a future scene replacement to reject stale references instead of silently addressing a reused slot.
- Uploaded meshes now have public handles and retain their geometry offsets, material, pipeline variant, name, and local bounds after the initial upload.
- `createMeshInstance()` creates another draw from an uploaded mesh without duplicating its device-local geometry. `destroyInstance()` removes every draw owned by that instance.
- Destroyed transform slots are reused. Their individual generation advances on retirement, so a stale instance handle cannot mutate a replacement that occupies the same slot.
- Transform and draw storage buffers and descriptor ranges reserve the renderer's full declared capacity rather than the initialization-time scene count, making runtime-created instances valid shader inputs.
- `uploadMesh()` accepts Vulkan-free CPU vertices and indices plus an existing material handle after initialization. It validates index bounds, derives local bounds, appends a persistent mesh resource, and makes it immediately available to instance creation.
- Mesh resource slots now track occupancy and individual generations. `destroyMesh()` rejects a mesh while any live instance references it, removes unreferenced metadata from `sceneMeshes()`, and advances the slot generation before reuse.
- `RendererConfig::initialVertexCapacity` and `initialIndexCapacity` reserve geometry storage beyond the imported scene. Runtime meshes allocate independent vertex and index ranges and upload only those bytes through the dedicated upload context.
- Destroyed mesh ranges enter a deferred queue rather than becoming immediately reusable. Each swapchain image advances its range-consumption generation only after its in-flight fence completes; once every image has advanced, the ranges return to sorted free lists and adjacent ranges coalesce.
- A missing contiguous range grows capacity geometrically, copies the CPU-side capacity image once, and creates replacement device-local vertex/index buffers. Ordinary uploads within capacity no longer replace or copy existing GPU geometry.
- Swapchain images track the geometry generation referenced by their vertex descriptor. A descriptor migrates only after that image's in-flight fence completes; replaced vertex/index buffer pairs stay in a deferred-retirement queue until every image has advanced beyond their generation.
- `RendererConfig::maxTextures` reserves a bindless combined-image-sampler array, clamped to `maxPerStageDescriptorSamplers` and `maxDescriptorSetSamplers`. Unused entries are initialized with a valid fallback descriptor, so runtime growth does not depend on partially-bound descriptors.
- Uploaded textures now have public generation-tagged handles and retain their name, extent, color space, and sampler metadata. `uploadTexture()` creates a sampled image, image view, and per-texture sampler from an already-decoded `TextureAsset` after initialization.
- `updateMaterialTextures()` rebinds base-color, normal, metallic-roughness, occlusion, and emissive roles independently while retaining the material's factors and pipeline state.
- Swapchain images also track texture generations. New descriptors are written only after the image's in-flight fence completes, while that image's older material-buffer copy cannot reference the new slot.
- Runtime texture slots now track occupancy and individual generations. `destroyTexture()` rejects the last fallback texture and any texture still referenced by a material, preventing dangling material-to-descriptor references.
- Destroyed texture slots immediately disappear from `sceneTextures()` and advance their generation before reuse, so stale handles cannot address a replacement texture occupying the same descriptor index.
- The old image, view, and sampler move into a deferred-retirement queue. Freed descriptor slots resolve to a live fallback texture; each old Vulkan resource stays alive until every swapchain image has waited for its fence and migrated beyond the retired descriptor generation.
- `RendererConfig::maxMaterials` reserves material storage per swapchain image, clamped to the shader ABI's 2048-entry limit. Material buffer allocations and descriptor ranges no longer stop at the imported scene's initial material count.
- `createMaterial()` appends a complete PBR or unlit material between frames from factors, texture handles, alpha mode, and double-sided state. The returned persistent handle can be supplied immediately to `uploadMesh()`, and `sceneMaterials()` reports its pipeline-affecting state as well as its mutable data.
- Material slots now track occupancy and individual generations. `destroyMaterial()` rejects a material while any live mesh references its GPU index, removes it from `sceneMaterials()`, and returns the stable slot to `createMaterial()` with an advanced generation.
- Material slot reuse needs no device-wide idle or separate Vulkan retirement queue: each swapchain image receives the replacement value only after its own in-flight fence completes, while older submissions retain that image's previous material-buffer copy.
- `replaceScene()` validates a complete decoded `SceneAsset` and creates its texture and device-local geometry owners before committing the replacement, so CPU validation or GPU preparation failure leaves the active scene untouched.
- Replacement uses the same pure plan and animation preparation as initial upload, removing the former static-only replacement restriction.
- A successful replacement advances the scene generation and installs packed instance, mesh, material, and texture slots, making every handle from the previous scene stale even when its numeric slot is reused.
- Previous texture and geometry owners enter their established generation queues. Each swapchain image migrates descriptors and records commands for the replacement only after its fence completes; material, transform, draw, and indirect buffers are likewise rewritten per image without a device-wide idle.
- The step-driven validation test uses deliberately small geometry capacities to exercise in-capacity range uploads, buffer growth, deferred range reclamation, and later reuse. It also verifies referenced mesh/material/texture destruction is rejected, safely reuses their metadata slots, rejects stale handles, verifies transform-slot reuse, rejects invalid replacement scenes transactionally, and renders after whole-scene replacement under validation layers.
- A dedicated validation smoke path explicitly injects the public GLFW adapter, alongside coverage of the implicit default backend and adapter-routed swapchain resize handling.
- CPU asset tests load the bundled village GLB, verify its embedded 2048x2048 atlas and trilinear sampler, and confirm `KHR_materials_unlit` survives the import boundary. They also validate the ninja skin/clip and composed-scene handle remapping.

## Next refactors

1. **Lighting and environment** - add image-based lighting, environment resources, multiple lights, and shadows on top of the PBR material model.
2. **Testing, animation profiling, and draw compaction** - add resource-creation tests that can run without the demo scene, then use the animated stress diagnostics to reduce repeated pose synchronization and joint-palette work. Independently timed actors still require their own evaluated pose and joint palette, but shared meshes can be submitted as instanced indirect draws by compacting visible per-actor transform and palette indices into mesh/pipeline groups. Pose or palette sharing is an optional optimization only for actors that actually resolve to the same animation state; it must not be assumed by the batching design.
3. **Upload batching and asynchronous transfer** - the reusable staging arena currently reflects the upload subsystem's intentionally synchronous fence contract. A later resource-streaming boundary can suballocate multiple payloads in one batch, submit them together, and retire arena ranges by transfer timeline value without changing scene ownership.

The recommended next implementation slice establishes the lighting/environment resource boundary before adding new shading features. Start with a Vulkan-free light/environment description and descriptor planning, then add environment image ownership and shader bindings without coupling asset import to renderer internals.
