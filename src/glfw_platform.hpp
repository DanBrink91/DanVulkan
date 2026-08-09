#pragma once

#include <danvulkan/platform.hpp>

#include <cstdint>
#include <memory>
#include <string_view>

struct DemoInputState
{
    bool moveDown = false;
    bool moveUp = false;
    bool moveLeft = false;
    bool moveRight = false;
    float cursorDeltaX = 0.0f;
    float cursorDeltaY = 0.0f;
    float scrollDeltaY = 0.0f;
};

class GlfwRendererPlatform final : public RendererPlatform
{
public:
    GlfwRendererPlatform(std::string_view title, std::uint32_t width, std::uint32_t height);
    ~GlfwRendererPlatform() override;

    GlfwRendererPlatform(const GlfwRendererPlatform&) = delete;
    GlfwRendererPlatform& operator=(const GlfwRendererPlatform&) = delete;

    [[nodiscard]] bool shouldClose() const noexcept override;
    void pollEvents() override;
    void waitEvents() override;
    [[nodiscard]] RendererFramebufferExtent framebufferExtent() const noexcept override;
    [[nodiscard]] bool consumeFramebufferResize() noexcept override;
    [[nodiscard]] std::span<const char* const>
        requiredVulkanInstanceExtensions() const noexcept override;
    [[nodiscard]] VkSurfaceKHR createVulkanSurface(VkInstance instance) override;
    void setWindowTitle(std::string_view title) override;
    [[nodiscard]] bool requestWindowResize(std::uint32_t width,
        std::uint32_t height) override;

    [[nodiscard]] DemoInputState consumeDemoInput();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
