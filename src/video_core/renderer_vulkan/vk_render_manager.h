// Copyright 2024-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#pragma once

#include <bit>
#include <mutex>

#include "common/math_util.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
enum class PixelFormat : u32;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Framebuffer;

struct RenderPass {
    vk::Framebuffer framebuffer;
    vk::RenderPass render_pass;
    vk::Rect2D render_area;
    std::array<vk::ClearValue, 2> clears;
    u32 do_clear;

    bool operator==(const RenderPass& other) const noexcept {
        return std::tie(framebuffer, render_pass, render_area, do_clear) ==
                   std::tie(other.framebuffer, other.render_pass, other.render_area,
                            other.do_clear) &&
               std::memcmp(&clears, &other.clears, sizeof(clears)) == 0;
    }
};

class RenderManager {
    static constexpr u32 NumColorFormats = static_cast<u32>(VideoCore::PixelFormat::NumColorFormat);
    static constexpr u32 NumDepthFormats = static_cast<u32>(VideoCore::PixelFormat::NumDepthFormat);
    static constexpr size_t MaxSamples = 16;
    static_assert(std::has_single_bit(MaxSamples));

public:
    explicit RenderManager(const Instance& instance, Scheduler& scheduler);
    ~RenderManager();

    /// Renders to the framebuffer, in the pass that is open if it has those attachments and
    /// in a new one otherwise. Returns whether the pass went on with an attachment more
    bool BeginRendering(const Framebuffer* framebuffer);

    /// Whether the pass that is open has the attachments of the framebuffer
    bool Covers(const Framebuffer* framebuffer) const;

    /// The attachments of the pass that is open
    VideoCore::PixelFormat ColorFormat() const noexcept {
        return formats[0];
    }
    VideoCore::PixelFormat DepthFormat() const noexcept {
        return formats[1];
    }
    vk::ImageView ColorView() const noexcept {
        return views[0];
    }

    /// Begins a new renderpass with the provided render state.
    void BeginRendering(const RenderPass& new_pass);

    /// Exits from any currently active renderpass instance
    void EndRendering();

    /// Returns the renderpass associated with the color-depth format pair
    vk::RenderPass GetRenderpass(VideoCore::PixelFormat color, VideoCore::PixelFormat depth,
                                 bool is_clear, u8 sample_count = 1);

private:
    /// Creates a renderpass configured appropriately and stores it in cached_renderpasses
    vk::UniqueRenderPass CreateRenderPass(vk::Format color, vk::Format depth,
                                          vk::AttachmentLoadOp load_op) const;

    /// Creates an MSAA renderpass configured appropriately and stores it in cached_renderpasses
    vk::UniqueRenderPass CreateRenderPassMSAA(vk::Format color, vk::Format depth,
                                              vk::AttachmentLoadOp load_op,
                                              vk::SampleCountFlagBits sample_count) const;

private:
    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueRenderPass cached_renderpasses[NumColorFormats + 1][NumDepthFormats + 1]
                                            [std::bit_width(MaxSamples)][2];
    std::mutex cache_mutex;
    std::array<vk::Image, 4> images;
    std::array<vk::ImageAspectFlags, 4> aspects;
    bool shadow_rendering{};
    std::array<vk::ImageView, 2> views{};
    std::array<VideoCore::PixelFormat, 2> formats{VideoCore::PixelFormat::Invalid,
                                                  VideoCore::PixelFormat::Invalid};
    u8 samples{1};
    RenderPass pass{};
    u32 num_draws{};
};

} // namespace Vulkan
