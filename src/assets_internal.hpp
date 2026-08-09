#pragma once

#include <danvulkan/assets.hpp>

#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>

namespace danvulkan::assets::detail
{
[[nodiscard]] TextureAsset decodeTextureMemory(std::span<const std::byte> encodedBytes,
                                               std::string_view name,
                                               const std::filesystem::path& sourcePath,
                                               ColorSpace colorSpace);
[[nodiscard]] TextureAsset makeWhiteTexture();
void finishGeometry(MeshAsset& mesh, bool generateNormals);
}
