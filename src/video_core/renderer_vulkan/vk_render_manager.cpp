// Copyright 2024-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <boost/container/static_vector.hpp>
#include "common/assert.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Vulkan {

constexpr u32 MinDrawsToFlush = 20;

using VideoCore::PixelFormat;
using VideoCore::SurfaceType;

namespace {

s32 Right(const vk::Rect2D& rect) {
    return rect.offset.x + static_cast<s32>(rect.extent.width);
}

s32 Top(const vk::Rect2D& rect) {
    return rect.offset.y + static_cast<s32>(rect.extent.height);
}

vk::Rect2D MakeRect(s32 left, s32 bottom, s32 right, s32 top) {
    return vk::Rect2D{
        .offset = {left, bottom},
        .extent = {static_cast<u32>(std::max(right - left, 0)),
                   static_cast<u32>(std::max(top - bottom, 0))},
    };
}

vk::Rect2D MakeRect(const Common::Rectangle<u32>& rect) {
    return MakeRect(static_cast<s32>(rect.left), static_cast<s32>(rect.bottom),
                    static_cast<s32>(rect.right), static_cast<s32>(rect.top));
}

bool IsEmpty(const vk::Rect2D& rect) {
    return rect.extent.width == 0 || rect.extent.height == 0;
}

bool Contains(const vk::Rect2D& outer, const vk::Rect2D& inner) {
    return outer.offset.x <= inner.offset.x && outer.offset.y <= inner.offset.y &&
           Right(inner) <= Right(outer) && Top(inner) <= Top(outer);
}

vk::Rect2D Intersect(const vk::Rect2D& a, const vk::Rect2D& b) {
    return MakeRect(std::max(a.offset.x, b.offset.x), std::max(a.offset.y, b.offset.y),
                    std::min(Right(a), Right(b)), std::min(Top(a), Top(b)));
}

/// The smallest rectangle around all of them
template <typename Rects>
vk::Rect2D Bounds(const Rects& rects) {
    vk::Rect2D bounds = rects.front();
    for (const vk::Rect2D& rect : rects) {
        bounds = MakeRect(std::min(bounds.offset.x, rect.offset.x),
                          std::min(bounds.offset.y, rect.offset.y),
                          std::max(Right(bounds), Right(rect)), std::max(Top(bounds), Top(rect)));
    }
    return bounds;
}

/// Whether the rectangles leave no gap in the box
template <typename Rects>
bool Fill(const Rects& rects, const vk::Rect2D& box) {
    std::vector<s32> xs{box.offset.x, Right(box)};
    std::vector<s32> ys{box.offset.y, Top(box)};
    for (const vk::Rect2D& rect : rects) {
        xs.push_back(std::clamp(rect.offset.x, box.offset.x, Right(box)));
        xs.push_back(std::clamp(Right(rect), box.offset.x, Right(box)));
        ys.push_back(std::clamp(rect.offset.y, box.offset.y, Top(box)));
        ys.push_back(std::clamp(Top(rect), box.offset.y, Top(box)));
    }
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    for (std::size_t i = 0; i + 1 < xs.size(); i++) {
        for (std::size_t j = 0; j + 1 < ys.size(); j++) {
            const vk::Rect2D cell = MakeRect(xs[i], ys[j], xs[i + 1], ys[j + 1]);
            const auto covers = [&cell](const vk::Rect2D& rect) { return Contains(rect, cell); };
            if (!IsEmpty(cell) && std::none_of(rects.begin(), rects.end(), covers)) {
                return false;
            }
        }
    }
    return true;
}

/// The parts of the rectangle that the box leaves out
boost::container::static_vector<vk::Rect2D, 4> Outside(const vk::Rect2D& rect,
                                                       const vk::Rect2D& box) {
    const s32 left = std::clamp(box.offset.x, rect.offset.x, Right(rect));
    const s32 right = std::clamp(Right(box), left, Right(rect));
    const s32 bottom = std::clamp(box.offset.y, rect.offset.y, Top(rect));
    const s32 top = std::clamp(Top(box), bottom, Top(rect));
    boost::container::static_vector<vk::Rect2D, 4> parts;
    for (const vk::Rect2D& part :
         {MakeRect(rect.offset.x, rect.offset.y, left, Top(rect)),
          MakeRect(right, rect.offset.y, Right(rect), Top(rect)),
          MakeRect(left, rect.offset.y, right, bottom), MakeRect(left, top, right, Top(rect))}) {
        if (!IsEmpty(part)) {
            parts.push_back(part);
        }
    }
    return parts;
}

bool SameValue(const vk::ClearValue& a, const vk::ClearValue& b, bool is_color) {
    if (is_color) {
        return std::memcmp(&a.color, &b.color, sizeof(a.color)) == 0;
    }
    // Only a byte of the stencil is given
    return std::memcmp(&a.depthStencil.depth, &b.depthStencil.depth, sizeof(float)) == 0 &&
           static_cast<u8>(a.depthStencil.stencil) == static_cast<u8>(b.depthStencil.stencil);
}

} // Anonymous namespace

RenderManager::RenderManager(const Instance& instance, Scheduler& scheduler)
    : instance{instance}, scheduler{scheduler} {}

RenderManager::~RenderManager() = default;

bool RenderManager::Covers(const Framebuffer* framebuffer, Common::Rectangle<u32> draw_rect) const {
    if (!pass.render_pass || shadow_rendering || framebuffer->shadow_rendering ||
        samples != framebuffer->Samples() || (!views[0] && !views[1]) ||
        !Contains(pass.render_area, MakeRect(draw_rect))) {
        return false;
    }
    const vk::ImageView color = framebuffer->ImageView(VideoCore::SurfaceType::Color);
    const vk::ImageView depth = framebuffer->ImageView(VideoCore::SurfaceType::Depth);
    return (!color || color == views[0]) && (!depth || depth == views[1]);
}

bool RenderManager::BeginRendering(const Framebuffer* framebuffer,
                                   Common::Rectangle<u32> draw_rect) {
    if (Covers(framebuffer, draw_rect)) {
        num_draws++;
        return pass.framebuffer != framebuffer->Handle();
    }

    RenderPass new_pass = {
        .framebuffer = framebuffer->Handle(),
        .render_pass = framebuffer->RenderPass(),
        // The whole target, for a draw with another viewport to stay in the pass
        .render_area = vk::Rect2D{.offset = {0, 0}, .extent = framebuffer->Extent()},
        .clears = {},
        .do_clear = false,
    };
    const std::array attachments = {framebuffer->ImageView(VideoCore::SurfaceType::Color),
                                    framebuffer->ImageView(VideoCore::SurfaceType::Depth)};

    // The clears that wait for these attachments become the load of the pass, in the area that
    // all of them clear, when the draw stays in it: both, or else the color one, or else the
    // depth one
    u32 loads = 0;
    if (!framebuffer->shadow_rendering && framebuffer->Samples() == 1) {
        for (const u32 choice : {3u, 1u, 2u}) {
            vk::Rect2D area = new_pass.render_area;
            bool fits = true;
            for (u32 i = 0; i < 2; i++) {
                if (!(choice & (1u << i))) {
                    continue;
                }
                const PendingClear& pending = pending_clears[i];
                fits = fits && pending.view && pending.view == attachments[i];
                if (fits) {
                    area = Intersect(area, Bounds(pending.areas));
                }
            }
            for (u32 i = 0; i < 2 && fits; i++) {
                fits = !(choice & (1u << i)) || Fill(pending_clears[i].areas, area);
            }
            if (fits && Contains(area, MakeRect(draw_rect))) {
                loads = choice;
                new_pass.render_area = area;
                break;
            }
        }
    }
    std::array<PendingClear, 2> taken{};
    for (u32 i = 0; i < 2; i++) {
        if (loads & (1u << i)) {
            taken[i] = std::exchange(pending_clears[i], {});
        } else if (pending_clears[i].view) {
            FlushClear(pending_clears[i]);
        }
    }
    for (u32 i = 0; i < 2; i++) {
        if (!(loads & (1u << i))) {
            continue;
        }
        // What it clears around the area is cleared in passes of their own
        for (const vk::Rect2D& cleared : taken[i].areas) {
            for (const vk::Rect2D& part : Outside(cleared, new_pass.render_area)) {
                ClearNow(taken[i], part);
            }
        }
        MakeAttachment(taken[i]);
        // Depth is the second attachment when there is a color one
        new_pass.clears[i == 1 && attachments[0] ? 1 : 0] = taken[i].value;
    }
    if (loads) {
        new_pass.render_pass = GetRenderpass(framebuffer->Format(VideoCore::SurfaceType::Color),
                                             framebuffer->Format(VideoCore::SurfaceType::Depth),
                                             (loads & 1) != 0, (loads & 2) != 0, 1);
        new_pass.do_clear = true;
    }
    BeginRendering(new_pass);
    images = framebuffer->Images();
    aspects = framebuffer->Aspects();
    shadow_rendering = framebuffer->shadow_rendering;
    views = {framebuffer->ImageView(VideoCore::SurfaceType::Color),
             framebuffer->ImageView(VideoCore::SurfaceType::Depth)};
    formats = {framebuffer->Format(VideoCore::SurfaceType::Color),
               framebuffer->Format(VideoCore::SurfaceType::Depth)};
    samples = framebuffer->Samples();
    return false;
}

void RenderManager::BeginRendering(const RenderPass& new_pass) {
    if (pass == new_pass) [[likely]] {
        num_draws++;
        return;
    }

    EndRendering();
    BeginPass(new_pass);
}

void RenderManager::BeginPass(const RenderPass& new_pass) {
    scheduler.Record([info = new_pass](vk::CommandBuffer cmdbuf) {
        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = info.render_pass,
            .framebuffer = info.framebuffer,
            .renderArea = info.render_area,
            .clearValueCount = info.do_clear ? 2u : 0u,
            .pClearValues = info.clears.data(),
        };
        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
    });

    pass = new_pass;
}

void RenderManager::Clear(Surface& surface, const VideoCore::TextureClear& clear) {
    static_assert(sizeof(VideoCore::ClearValue) == sizeof(vk::ClearValue));
    EndPass();

    const bool is_color = static_cast<bool>(surface.Aspect() & vk::ImageAspectFlagBits::eColor);
    const vk::ImageView view = surface.FramebufferView();
    const vk::Rect2D whole = MakeRect(surface.GetScaledRect());
    const vk::Rect2D area = MakeRect(clear.texture_rect);
    vk::ClearValue value{};
    std::memcpy(&value, &clear.value, sizeof(value));

    PendingClear& pending = pending_clears[is_color ? 0 : 1];
    if (pending.view == view && area == whole) {
        // What waited is cleared over
        pending = {};
    } else if (pending.view &&
               (pending.view != view || !SameValue(pending.value, value, is_color) ||
                pending.areas.size() == pending.areas.capacity())) {
        FlushClear(pending);
    }
    if (!pending.view) {
        const PixelFormat format = surface.pixel_format;
        pending.view = view;
        pending.image = surface.Image();
        pending.aspect = surface.Aspect();
        pending.access = surface.AccessFlags();
        pending.stages = surface.PipelineStageFlags();
        pending.framebuffer = surface.Framebuffer();
        pending.render_pass = GetRenderpass(is_color ? format : PixelFormat::Invalid,
                                            is_color ? PixelFormat::Invalid : format, true, 1);
        pending.whole = whole;
        pending.value = value;
    }
    pending.areas.push_back(area);
}

void RenderManager::FlushClear(PendingClear& pending) {
    const PendingClear clear = std::exchange(pending, {});
    if (Fill(clear.areas, clear.whole)) {
        ClearNow(clear, clear.whole);
        return;
    }
    for (const vk::Rect2D& area : clear.areas) {
        ClearNow(clear, area);
    }
}

void RenderManager::MakeAttachment(const PendingClear& clear) {
    scheduler.Record([image = clear.image, aspect = clear.aspect, access = clear.access,
                      stages = clear.stages](vk::CommandBuffer cmdbuf) {
        const bool is_color = static_cast<bool>(aspect & vk::ImageAspectFlagBits::eColor);
        const vk::ImageMemoryBarrier barrier = {
            .srcAccessMask = access,
            .dstAccessMask = is_color ? vk::AccessFlagBits::eColorAttachmentRead |
                                            vk::AccessFlagBits::eColorAttachmentWrite
                                      : vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.pipelineBarrier(stages,
                               is_color ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                                        : vk::PipelineStageFlagBits::eEarlyFragmentTests,
                               vk::DependencyFlagBits::eByRegion, {}, {}, barrier);
    });
}

void RenderManager::ClearNow(const PendingClear& clear, const vk::Rect2D& area) {
    MakeAttachment(clear);
    // A pass that draws nothing
    BeginPass(RenderPass{
        .framebuffer = clear.framebuffer,
        .render_pass = clear.render_pass,
        .render_area = area,
        .clears = {clear.value, clear.value},
        .do_clear = true,
    });
    images[0] = clear.image;
    aspects[0] = clear.aspect;
    EndPass();
}

void RenderManager::EndRendering() {
    EndPass();
    for (PendingClear& pending : pending_clears) {
        if (pending.view) {
            FlushClear(pending);
        }
    }
}

void RenderManager::EndPass() {
    if (!pass.render_pass) {
        return;
    }

    scheduler.Record([images = images, aspects = aspects,
                      shadow_rendering = shadow_rendering](vk::CommandBuffer cmdbuf) {
        u32 num_barriers = 0;
        vk::PipelineStageFlags pipeline_flags{};
        vk::AccessFlags src_access_flags{};
        std::array<vk::ImageMemoryBarrier, 4> barriers;
        for (u32 i = 0; i < images.size(); i++) {
            if (!images[i]) {
                continue;
            }
            const bool is_color = static_cast<bool>(aspects[i] & vk::ImageAspectFlagBits::eColor);
            if (is_color) {
                pipeline_flags |= shadow_rendering
                                      ? vk::PipelineStageFlagBits::eFragmentShader
                                      : vk::PipelineStageFlagBits::eColorAttachmentOutput;
                src_access_flags = shadow_rendering ? vk::AccessFlagBits::eShaderWrite
                                                    : vk::AccessFlagBits::eColorAttachmentWrite;
            } else {
                pipeline_flags |= vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                  vk::PipelineStageFlagBits::eLateFragmentTests;
                src_access_flags = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            }
            barriers[num_barriers++] = vk::ImageMemoryBarrier{
                .srcAccessMask = src_access_flags,
                .dstAccessMask =
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = images[i],
                .subresourceRange{
                    .aspectMask = aspects[i],
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
        }
        cmdbuf.endRenderPass();
        if (num_barriers == 0) {
            return;
        }
        cmdbuf.pipelineBarrier(pipeline_flags,
                               vk::PipelineStageFlagBits::eFragmentShader |
                                   vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr,
                               num_barriers, barriers.data());
    });

    // Reset state.
    pass.render_pass = VK_NULL_HANDLE;
    images = {};
    aspects = {};
    shadow_rendering = false;
    views = {};
    formats = {VideoCore::PixelFormat::Invalid, VideoCore::PixelFormat::Invalid};

    // The Mali guide recommends flushing at the end of each major renderpass
    // Testing has shown this has a significant effect on rendering performance
    if (num_draws > MinDrawsToFlush && instance.ShouldFlush()) {
        scheduler.Flush();
        num_draws = 0;
    }
}

vk::RenderPass RenderManager::GetRenderpass(VideoCore::PixelFormat color,
                                            VideoCore::PixelFormat depth, bool is_clear,
                                            u8 sample_count) {
    return GetRenderpass(color, depth, is_clear, is_clear, sample_count);
}

vk::RenderPass RenderManager::GetRenderpass(VideoCore::PixelFormat color,
                                            VideoCore::PixelFormat depth, bool clear_color,
                                            bool clear_depth, u8 sample_count) {
    std::scoped_lock lock{cache_mutex};

    const u32 color_index =
        color == VideoCore::PixelFormat::Invalid ? NumColorFormats : static_cast<u32>(color);
    const u32 depth_index =
        depth == VideoCore::PixelFormat::Invalid
            ? NumDepthFormats
            : (static_cast<u32>(depth - VideoCore::PixelFormat::NumColorFormat));

    ASSERT_MSG(color_index <= NumColorFormats && depth_index <= NumDepthFormats,
               "Invalid color index {} and/or depth_index {}", color_index, depth_index);

    ASSERT_MSG(sample_count && std::has_single_bit(sample_count) && sample_count <= MaxSamples,
               "Invalid sample count {}", static_cast<u32>(sample_count));

    const u32 samples_index = static_cast<u32>(std::bit_width(sample_count) - 1);

    vk::UniqueRenderPass& renderpass =
        cached_renderpasses[color_index][depth_index][samples_index]
                           [(clear_color ? 1 : 0) | (clear_depth ? 2 : 0)];
    if (!renderpass) {
        const vk::Format color_format = instance.GetTraits(color).native;
        const vk::Format depth_format = instance.GetTraits(depth).native;
        const auto load_op = [](bool clear) {
            return clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
        };

        renderpass = (sample_count > 1)
                         ? CreateRenderPassMSAA(color_format, depth_format, load_op(clear_color),
                                                static_cast<vk::SampleCountFlagBits>(sample_count))
                         : CreateRenderPass(color_format, depth_format, load_op(clear_color),
                                            load_op(clear_depth));
    }

    return *renderpass;
}

vk::UniqueRenderPass RenderManager::CreateRenderPass(vk::Format color, vk::Format depth,
                                                     vk::AttachmentLoadOp color_load_op,
                                                     vk::AttachmentLoadOp depth_load_op) const {

    boost::container::static_vector<vk::AttachmentDescription, 2> attachments{};
    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription{
            .format = color,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = color_load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        color_attachment_ref = vk::AttachmentReference{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription{
            .format = depth,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = depth_load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = depth_load_op,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
        };

        use_depth = true;
    }

    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr,
    };

    return instance.GetDevice().createRenderPassUnique(renderpass_info);
}

vk::UniqueRenderPass RenderManager::CreateRenderPassMSAA(
    vk::Format color, vk::Format depth, vk::AttachmentLoadOp load_op,
    vk::SampleCountFlagBits sample_count) const {
    boost::container::static_vector<vk::AttachmentDescription2, 4> attachments{};

    vk::AttachmentReference2 color_resolve_attachment = {.attachment = VK_ATTACHMENT_UNUSED};
    vk::AttachmentReference2 depth_resolve_attachment = {.attachment = VK_ATTACHMENT_UNUSED};

    bool use_color = false;
    vk::AttachmentReference2 color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference2 depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription2{
            .format = color,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        color_attachment_ref = vk::AttachmentReference2{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
            .aspectMask = vk::ImageAspectFlagBits::eColor,
        };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments.emplace_back(vk::AttachmentDescription2{
            .format = depth,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = load_op,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eGeneral,
            .finalLayout = vk::ImageLayout::eGeneral,
        });

        depth_attachment_ref = vk::AttachmentReference2{
            .attachment = static_cast<u32>(attachments.size() - 1),
            .layout = vk::ImageLayout::eGeneral,
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
        };

        use_depth = true;
    }

    // In the case of MSAA, each attachment gets an additional MSAA attachment that now becomes the
    // main attachment and the original attachments now get resolved into
    if (sample_count > vk::SampleCountFlagBits::e1) {
        if (color != vk::Format::eUndefined) {
            attachments.emplace_back(vk::AttachmentDescription2{
                .format = color,
                .samples = sample_count,
                .loadOp = load_op,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                .initialLayout = vk::ImageLayout::eGeneral,
                .finalLayout = vk::ImageLayout::eGeneral,
            });

            color_resolve_attachment = color_attachment_ref;

            color_attachment_ref = vk::AttachmentReference2{
                .attachment = static_cast<u32>(attachments.size() - 1),
                .layout = vk::ImageLayout::eGeneral,
            };
        }

        if (depth != vk::Format::eUndefined) {
            attachments.emplace_back(vk::AttachmentDescription2{
                .format = depth,
                .samples = sample_count,
                .loadOp = load_op,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .stencilLoadOp = load_op,
                .stencilStoreOp = vk::AttachmentStoreOp::eStore,
                .initialLayout = vk::ImageLayout::eGeneral,
                .finalLayout = vk::ImageLayout::eGeneral,
            });

            depth_resolve_attachment = depth_attachment_ref;

            depth_attachment_ref = vk::AttachmentReference2{
                .attachment = static_cast<u32>(attachments.size() - 1),
                .layout = vk::ImageLayout::eGeneral,
            };
        }
    }

    const vk::StructureChain<vk::SubpassDescription2, vk::SubpassDescriptionDepthStencilResolve>
        subpass = {
            vk::SubpassDescription2{
                .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                .inputAttachmentCount = 0,
                .pInputAttachments = nullptr,
                .colorAttachmentCount = use_color ? 1u : 0u,
                .pColorAttachments = &color_attachment_ref,
                .pResolveAttachments = &color_resolve_attachment,
                .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr,
            },
            vk::SubpassDescriptionDepthStencilResolve{
                .depthResolveMode = vk::ResolveModeFlagBits::eSampleZero,
                .stencilResolveMode = vk::ResolveModeFlagBits::eSampleZero,
                .pDepthStencilResolveAttachment = &depth_resolve_attachment},
        };

    const vk::RenderPassCreateInfo2 renderpass_info = {
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass.get(),
        .dependencyCount = 0,
        .pDependencies = nullptr,
    };

    return instance.GetDevice().createRenderPass2Unique(renderpass_info);
}

} // namespace Vulkan
