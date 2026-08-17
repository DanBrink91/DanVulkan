#include "game_audio.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include <miniaudio.h>

namespace danvulkan::application {

struct GameAudio::Impl
{
    ma_engine engine{};
    ma_sound backgroundMusic{};
    bool engineInitialized = false;
    bool musicInitialized = false;
};

namespace {
std::runtime_error audioError(std::string operation, ma_result result)
{
    return std::runtime_error(std::move(operation) + ": " + ma_result_description(result));
}
}

GameAudio::GameAudio()
    : impl_(std::make_unique<Impl>())
{
}

GameAudio::~GameAudio()
{
    stopBackgroundMusic();
}

void GameAudio::playBackgroundMusic(const std::filesystem::path& path)
{
    stopBackgroundMusic();

    ma_result result = ma_engine_init(nullptr, &impl_->engine);
    if (result != MA_SUCCESS)
    {
        throw audioError("could not initialize the audio device", result);
    }
    impl_->engineInitialized = true;

    const std::string nativePath = path.string();
    result = ma_sound_init_from_file(&impl_->engine, nativePath.c_str(),
        MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_LOOPING, nullptr, nullptr,
        &impl_->backgroundMusic);
    if (result != MA_SUCCESS)
    {
        stopBackgroundMusic();
        throw audioError("could not load background music " + nativePath, result);
    }
    impl_->musicInitialized = true;

    result = ma_sound_start(&impl_->backgroundMusic);
    if (result != MA_SUCCESS)
    {
        stopBackgroundMusic();
        throw audioError("could not start background music", result);
    }
}

void GameAudio::stopBackgroundMusic() noexcept
{
    if (impl_->musicInitialized)
    {
        ma_sound_uninit(&impl_->backgroundMusic);
        impl_->musicInitialized = false;
    }
    if (impl_->engineInitialized)
    {
        ma_engine_uninit(&impl_->engine);
        impl_->engineInitialized = false;
    }
}

}
