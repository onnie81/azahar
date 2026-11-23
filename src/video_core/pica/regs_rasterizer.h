// Copyright 2017-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <array>
#include <boost/container/static_vector.hpp>
#include "common/assert.h"
#include "common/bit_field.h"
#include "common/common_funcs.h"
#include "common/math_util.h"
#include "common/vector_math.h"
#include "video_core/pica_types.h"

namespace Pica {

struct RasterizerRegs {
    enum class CullMode : u32 {
        // Select which polygons are considered to be "frontfacing".
        KeepAll = 0,
        KeepClockWise = 1,
        KeepCounterClockWise = 2,
        KeepAll2 = 3,
    };

    union {
        BitField<0, 2, CullMode> cull_mode;
    };

    BitField<0, 24, u32> viewport_size_x;

    BitField<1, 31, u32> viewport_size_x_inv;

    BitField<0, 24, u32> viewport_size_y;

    BitField<1, 31, u32> viewport_size_y_inv;

    INSERT_PADDING_WORDS(0x2);

    BitField<0, 1, u32> clip_enable;
    BitField<0, 24, u32> clip_coef[4]; // float24

    Common::Vec4<f24> GetClipCoef() const {
        return {f24::FromRaw(clip_coef[0]), f24::FromRaw(clip_coef[1]), f24::FromRaw(clip_coef[2]),
                f24::FromRaw(clip_coef[3])};
    }

    /// The corner has ten bits per axis and is added to a pixel's position modulo 1024, so on each
    /// axis the viewport can land at the corner and 1024 below it. Returns the places that reach
    /// the framebuffer, or the corner as written if none does.
    boost::container::static_vector<Common::Rectangle<s32>, 4> GetViewportRects(
        u32 framebuffer_width, u32 framebuffer_height) const {
        // These registers hold half-width and half-height, so must be multiplied by 2
        const s32 width = static_cast<s32>(f24::FromRaw(viewport_size_x).ToFloat32() * 2);
        const s32 height = static_cast<s32>(f24::FromRaw(viewport_size_y).ToFloat32() * 2);
        const s32 corner_x = static_cast<s32>(viewport_corner.x);
        const s32 corner_y = static_cast<s32>(viewport_corner.y);
        const auto rect_at = [width, height](s32 x, s32 y) {
            return Common::Rectangle<s32>{
                x,          // left
                y + height, // top
                x + width,  // right
                y,          // bottom
            };
        };

        boost::container::static_vector<Common::Rectangle<s32>, 4> rects;
        for (const s32 y : {corner_y, corner_y - 1024}) {
            if (y >= static_cast<s32>(framebuffer_height) || y + height <= 0) {
                continue;
            }
            for (const s32 x : {corner_x, corner_x - 1024}) {
                if (x < static_cast<s32>(framebuffer_width) && x + width > 0) {
                    rects.push_back(rect_at(x, y));
                }
            }
        }
        if (rects.empty()) {
            rects.push_back(rect_at(corner_x, corner_y));
        }
        return rects;
    }

    INSERT_PADDING_WORDS(0x1);

    BitField<0, 24, u32> viewport_depth_range;      // float24
    BitField<0, 24, u32> viewport_depth_near_plane; // float24

    BitField<0, 3, u32> vs_output_total;

    union VSOutputAttributes {
        // Maps components of output vertex attributes to semantics
        enum Semantic : u32 {
            POSITION_X = 0,
            POSITION_Y = 1,
            POSITION_Z = 2,
            POSITION_W = 3,

            QUATERNION_X = 4,
            QUATERNION_Y = 5,
            QUATERNION_Z = 6,
            QUATERNION_W = 7,

            COLOR_R = 8,
            COLOR_G = 9,
            COLOR_B = 10,
            COLOR_A = 11,

            TEXCOORD0_U = 12,
            TEXCOORD0_V = 13,
            TEXCOORD1_U = 14,
            TEXCOORD1_V = 15,

            TEXCOORD0_W = 16,

            VIEW_X = 18,
            VIEW_Y = 19,
            VIEW_Z = 20,

            TEXCOORD2_U = 22,
            TEXCOORD2_V = 23,

            INVALID = 31,
        };

        BitField<0, 5, Semantic> map_x;
        BitField<8, 5, Semantic> map_y;
        BitField<16, 5, Semantic> map_z;
        BitField<24, 5, Semantic> map_w;

        u32 raw;
    } vs_output_attributes[7];

    void ValidateSemantics() {
        for (std::size_t attrib = 0; attrib < vs_output_total; ++attrib) {
            const u32 output_register_map = vs_output_attributes[attrib].raw;
            for (std::size_t comp = 0; comp < 4; ++comp) {
                const u32 semantic = (output_register_map >> (8 * comp)) & 0x1F;
                ASSERT_MSG(semantic < 24 || semantic == VSOutputAttributes::INVALID,
                           "Invalid/unknown semantic id: {}", semantic);
            }
        }
    }

    INSERT_PADDING_WORDS(0xe);

    enum class ScissorMode : u32 {
        Disabled = 0,
        Exclude = 1, // Exclude pixels inside the scissor box

        Include = 3 // Exclude pixels outside the scissor box
    };

    struct {
        BitField<0, 2, ScissorMode> mode;

        union {
            BitField<0, 10, u32> x1;
            BitField<16, 10, u32> y1;
        };

        union {
            BitField<0, 10, u32> x2;
            BitField<16, 10, u32> y2;
        };
    } scissor_test;

    union {
        BitField<0, 10, u32> x;
        BitField<16, 10, u32> y;
    } viewport_corner;

    INSERT_PADDING_WORDS(0x1);

    // TODO: early depth
    INSERT_PADDING_WORDS(0x1);

    INSERT_PADDING_WORDS(0x2);

    enum DepthBuffering : u32 {
        WBuffering = 0,
        ZBuffering = 1,
    };
    BitField<0, 1, DepthBuffering> depthmap_enable;

    INSERT_PADDING_WORDS(0x12);
};

static_assert(sizeof(RasterizerRegs) == 0x40 * sizeof(u32),
              "RasterizerRegs struct has incorrect size");

} // namespace Pica
