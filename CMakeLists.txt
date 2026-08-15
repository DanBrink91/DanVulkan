cmake_minimum_required(VERSION 3.25)

project(DanVulkan VERSION 0.29.0 LANGUAGES C CXX)

include(CTest)
include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYOBJLOADER_BUILD_TEST_LOADER OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_DOCS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "" FORCE)

find_package(Vulkan 1.4 REQUIRED)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG v2.0.0rc13
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    fastgltf
    GIT_REPOSITORY https://github.com/spnda/fastgltf.git
    GIT_TAG v0.9.0
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)
FetchContent_Declare(
    vulkan_memory_allocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.3.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glfw glm tinyobjloader fastgltf stb vulkan_memory_allocator)

# fastgltf adds /Ob3 on top of CMake's /Ob2 release setting, which produces MSVC D9025.
# Keep CMake's release inlining policy while retaining fastgltf's other optimization flags.
if(MSVC AND TARGET fastgltf)
    get_target_property(FASTGLTF_COMPILE_OPTIONS fastgltf COMPILE_OPTIONS)
    list(TRANSFORM FASTGLTF_COMPILE_OPTIONS REPLACE "[ ]*/Ob3" "")
    set_property(TARGET fastgltf PROPERTY COMPILE_OPTIONS "${FASTGLTF_COMPILE_OPTIONS}")
    unset(FASTGLTF_COMPILE_OPTIONS)
endif()

find_program(
    GLSLC_EXECUTABLE
    NAMES glslc glslc.exe
    HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin"
    REQUIRED
)

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SHADER_OUTPUTS)
foreach(SHADER_NAME IN ITEMS vert frag)
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.${SHADER_NAME}")
    set(SHADER_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT "${SHADER_OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.4 "${SHADER_SOURCE}" -o "${SHADER_OUTPUT}"
        DEPENDS "${SHADER_SOURCE}"
        COMMENT "Compiling ${SHADER_NAME} shader for Vulkan 1.4"
        VERBATIM
    )
    list(APPEND SHADER_OUTPUTS "${SHADER_OUTPUT}")
endforeach()
add_custom_target(danvulkan_shaders ALL DEPENDS ${SHADER_OUTPUTS})

add_library(danvulkan_assets STATIC
    src/assets.cpp
    src/assets_internal.hpp
    src/gltf_loader.cpp
    include/danvulkan/assets.hpp
)
add_library(DanVulkan::Assets ALIAS danvulkan_assets)

target_compile_features(danvulkan_assets PUBLIC cxx_std_20)
target_include_directories(
    danvulkan_assets
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    SYSTEM PRIVATE "${stb_SOURCE_DIR}"
)
target_link_libraries(danvulkan_assets PUBLIC glm::glm PRIVATE tinyobjloader fastgltf::fastgltf)

add_library(danvulkan_renderer STATIC
    src/animation_player.cpp
    src/animation_player.hpp
    src/attachment_context.cpp
    src/attachment_context.hpp
    src/descriptor_context.cpp
    src/descriptor_context.hpp
    src/descriptor_planner.cpp
    src/descriptor_planner.hpp
    src/device_context.cpp
    src/device_context.hpp
    src/device_planner.cpp
    src/device_planner.hpp
    src/frame_context.cpp
    src/frame_context.hpp
    src/glfw_platform.cpp
    src/glfw_platform.hpp
    src/memory_planner.cpp
    src/memory_planner.hpp
    src/pipeline_context.cpp
    src/pipeline_context.hpp
    src/pipeline_planner.cpp
    src/pipeline_planner.hpp
    src/presentation_context.cpp
    src/presentation_context.hpp
    src/renderer.cpp
    src/scene_context.cpp
    src/scene_context.hpp
    src/scene_planner.cpp
    src/scene_planner.hpp
    src/swapchain_context.cpp
    src/swapchain_context.hpp
    src/swapchain_planner.cpp
    src/swapchain_planner.hpp
    src/upload_context.cpp
    src/upload_context.hpp
    src/vma.cpp
    src/vulkan_raii.hpp
    src/vulkan_result.hpp
    camera.hpp
    include/danvulkan/platform.hpp
    include/danvulkan/renderer.hpp
)
add_library(DanVulkan::Renderer ALIAS danvulkan_renderer)

target_compile_features(danvulkan_renderer PUBLIC cxx_std_20)
target_include_directories(
    danvulkan_renderer
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(
    danvulkan_renderer
    PUBLIC
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
    PRIVATE
        DANVULKAN_SHADER_DIR="${SHADER_OUTPUT_DIR}/"
        GLFW_INCLUDE_VULKAN
)
if(APPLE)
    # VK_KHR_portability_subset declarations are provisional in the Vulkan headers.
    # MoltenVK exposes this extension and requires applications to enable it.
    target_compile_definitions(danvulkan_renderer PRIVATE VK_ENABLE_BETA_EXTENSIONS)
endif()
target_link_libraries(
    danvulkan_renderer
    PUBLIC DanVulkan::Assets Vulkan::Vulkan glm::glm
    PRIVATE glfw GPUOpen::VulkanMemoryAllocator
)
add_dependencies(danvulkan_renderer danvulkan_shaders)

if(MSVC)
    target_compile_options(danvulkan_assets PRIVATE /W4 /permissive- /Zc:__cplusplus)
    target_compile_options(danvulkan_renderer PRIVATE /W4 /permissive- /Zc:__cplusplus)
    set_source_files_properties(src/vma.cpp PROPERTIES COMPILE_OPTIONS /W0)
else()
    target_compile_options(danvulkan_assets PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(danvulkan_renderer PRIVATE -Wall -Wextra -Wpedantic)
    set_source_files_properties(src/vma.cpp PROPERTIES COMPILE_OPTIONS -w)
endif()

add_executable(DanVulkan src/main.cpp)
target_link_libraries(DanVulkan PRIVATE DanVulkan::Renderer)

set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT DanVulkan)
set_target_properties(DanVulkan PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

if(BUILD_TESTING)
    add_executable(danvulkan_asset_tests tests/assets_tests.cpp)
    target_link_libraries(danvulkan_asset_tests PRIVATE DanVulkan::Assets)
    add_executable(danvulkan_animation_tests
        tests/animation_player_tests.cpp
        src/animation_player.cpp
    )
    target_include_directories(danvulkan_animation_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(danvulkan_animation_tests PRIVATE DanVulkan::Assets glm::glm)
    add_executable(danvulkan_device_planner_tests
        tests/device_planner_tests.cpp
        src/device_planner.cpp
    )
    add_executable(danvulkan_descriptor_planner_tests
        tests/descriptor_planner_tests.cpp
        src/descriptor_planner.cpp
    )
    add_executable(danvulkan_swapchain_planner_tests
        tests/swapchain_planner_tests.cpp
        src/swapchain_planner.cpp
    )
    add_executable(danvulkan_pipeline_planner_tests
        tests/pipeline_planner_tests.cpp
        src/pipeline_planner.cpp
    )
    add_executable(danvulkan_memory_planner_tests
        tests/memory_planner_tests.cpp
        src/memory_planner.cpp
    )
    add_executable(danvulkan_scene_planner_tests
        tests/scene_planner_tests.cpp
        src/scene_planner.cpp
    )
    add_executable(danvulkan_scene_context_tests tests/scene_context_tests.cpp)
    target_include_directories(danvulkan_device_planner_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(danvulkan_device_planner_tests PRIVATE cxx_std_20)
    target_include_directories(danvulkan_descriptor_planner_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(danvulkan_descriptor_planner_tests PRIVATE cxx_std_20)
    target_link_libraries(danvulkan_descriptor_planner_tests PRIVATE Vulkan::Vulkan)
    target_include_directories(danvulkan_swapchain_planner_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(danvulkan_swapchain_planner_tests PRIVATE cxx_std_20)
    target_link_libraries(danvulkan_swapchain_planner_tests PRIVATE Vulkan::Vulkan)
    target_include_directories(danvulkan_pipeline_planner_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(danvulkan_pipeline_planner_tests PRIVATE cxx_std_20)
    target_link_libraries(danvulkan_pipeline_planner_tests PRIVATE Vulkan::Vulkan)
    target_include_directories(danvulkan_memory_planner_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(danvulkan_memory_planner_tests PRIVATE cxx_std_20)
    target_link_libraries(danvulkan_memory_planner_tests PRIVATE Vulkan::Vulkan)
    target_include_directories(danvulkan_scene_planner_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(danvulkan_scene_planner_tests PRIVATE cxx_std_20)
    target_link_libraries(danvulkan_scene_planner_tests PRIVATE DanVulkan::Assets glm::glm)
    target_include_directories(danvulkan_scene_context_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(danvulkan_scene_context_tests PRIVATE cxx_std_20)
    target_link_libraries(danvulkan_scene_context_tests PRIVATE
        DanVulkan::Renderer GPUOpen::VulkanMemoryAllocator)
    if(MSVC)
        target_compile_options(danvulkan_asset_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_animation_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_device_planner_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_descriptor_planner_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_swapchain_planner_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_pipeline_planner_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_memory_planner_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_scene_planner_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
        target_compile_options(danvulkan_scene_context_tests PRIVATE /W4 /permissive- /Zc:__cplusplus)
    else()
        target_compile_options(danvulkan_asset_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_animation_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_device_planner_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_descriptor_planner_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_swapchain_planner_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_pipeline_planner_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_memory_planner_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_scene_planner_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(danvulkan_scene_context_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME danvulkan_assets_import COMMAND danvulkan_asset_tests)
    add_test(NAME danvulkan_animation_playback COMMAND danvulkan_animation_tests)
    add_test(NAME danvulkan_device_planning COMMAND danvulkan_device_planner_tests)
    add_test(NAME danvulkan_descriptor_planning COMMAND danvulkan_descriptor_planner_tests)
    add_test(NAME danvulkan_swapchain_planning COMMAND danvulkan_swapchain_planner_tests)
    add_test(NAME danvulkan_pipeline_planning COMMAND danvulkan_pipeline_planner_tests)
    add_test(NAME danvulkan_memory_planning COMMAND danvulkan_memory_planner_tests)
    add_test(NAME danvulkan_scene_planning COMMAND danvulkan_scene_planner_tests)
    add_test(NAME danvulkan_scene_context COMMAND danvulkan_scene_context_tests)
    add_test(NAME danvulkan_validation_smoke COMMAND DanVulkan --smoke-test)
    add_test(NAME danvulkan_validation_platform COMMAND DanVulkan --platform-smoke-test)
    add_test(NAME danvulkan_validation_step COMMAND DanVulkan --step-smoke-test)
    add_test(NAME danvulkan_validation_resize COMMAND DanVulkan --resize-smoke-test)
    add_test(NAME danvulkan_validation_minimize COMMAND DanVulkan --minimize-smoke-test)
    add_test(NAME danvulkan_animated_stress COMMAND DanVulkan --animated-stress-test)
    set_tests_properties(
        danvulkan_assets_import danvulkan_animation_playback danvulkan_device_planning
        danvulkan_descriptor_planning
        danvulkan_swapchain_planning
        danvulkan_pipeline_planning
        danvulkan_memory_planning
        danvulkan_scene_planning
        danvulkan_scene_context
        danvulkan_validation_smoke danvulkan_validation_platform
        danvulkan_validation_step danvulkan_validation_resize danvulkan_validation_minimize
        danvulkan_animated_stress
        PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(danvulkan_animated_stress PROPERTIES LABELS stress TIMEOUT 180)
endif()
