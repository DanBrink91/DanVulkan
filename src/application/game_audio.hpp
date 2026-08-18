#pragma once

#include <filesystem>
#include <memory>

namespace danvulkan::application {

// Application-owned audio playback. Keeping this outside the renderer lets gameplay choose
// what to play without coupling audio resources or device lifetime to Vulkan.
class GameAudio
{
public:
    GameAudio();
    ~GameAudio();

    GameAudio(const GameAudio&) = delete;
    GameAudio& operator=(const GameAudio&) = delete;

    void playBackgroundMusic(const std::filesystem::path& path);
    void stopBackgroundMusic() noexcept;
    void setBackgroundMusicVolume(float volume);
    void setBackgroundMusicMuted(bool muted) noexcept;
    [[nodiscard]] float backgroundMusicVolume() const noexcept;
    [[nodiscard]] bool backgroundMusicMuted() const noexcept;
    [[nodiscard]] bool backgroundMusicActive() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
