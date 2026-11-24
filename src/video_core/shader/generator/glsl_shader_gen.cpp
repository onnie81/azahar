// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <string_view>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <nihstro/shader_bytecode.h>
#include "video_core/pica/regs_pipeline.h"
#include "video_core/pica/regs_rasterizer.h"
#include "video_core/shader/generator/glsl_shader_decompiler.h"
#include "video_core/shader/generator/glsl_shader_gen.h"
#include "video_core/shader/generator/shader_gen.h"

using VSOutputAttributes = Pica::RasterizerRegs::VSOutputAttributes;

namespace Pica::Shader::Generator::GLSL {

constexpr std::string_view VSPicaUniformBlockDef = R"(
#ifdef VULKAN
layout (set = 0, binding = 0, std140) uniform vs_pica_data {
#else
layout (binding = 0, std140) uniform vs_pica_data {
#endif
    uint b;
    uvec4 i[4];
    vec4 f[96];
} uniforms;
)";

// clang-format off
constexpr std::string_view VSUniformBlockDef = R"(
#ifdef VULKAN
layout (set = 0, binding = 1, std140) uniform vs_data {
#else
layout (binding = 1, std140) uniform vs_data {
#endif
    bool enable_clip1;
    bool flip_viewport;
    vec4 clip_coef;
)"
#ifdef ENABLE_FREECAM
R"(    mat4 freecam_matrix;
    bool freecam_skip_2d;
)"
#endif
R"(};

const vec2 EPSILON_Z = vec2(0.000001f, -1.00001f);

vec4 SanitizeVertex(vec4 vtx_pos) {
    float ndc_z = vtx_pos.z / vtx_pos.w;
    if (ndc_z > 0.f && ndc_z < EPSILON_Z[0]) {
        vtx_pos.z = 0.f;
    }
    if (ndc_z < -1.f && ndc_z > EPSILON_Z[1]) {
        vtx_pos.z = -vtx_pos.w;
    }
    return vtx_pos;
}
)"
#ifdef ENABLE_FREECAM
R"(
vec4 ApplyFreeCam(vec4 clip_pos) {
    if (freecam_skip_2d && abs(clip_pos.w - 1.0) < 0.0001) {
        return clip_pos;
    }
    return freecam_matrix * clip_pos;
}
)"
#endif
;
// clang-format on

#ifdef ENABLE_FREECAM
constexpr std::string_view PositionOutput =
    "    gl_Position = ApplyFreeCam(vec4(vtx_pos.x, vtx_pos.y, -vtx_pos.z, vtx_pos.w));\n";
#else
constexpr std::string_view PositionOutput =
    "    gl_Position = vec4(vtx_pos.x, vtx_pos.y, -vtx_pos.z, vtx_pos.w);\n";
#endif

static std::string GetVertexInterfaceDeclaration(bool is_output, bool use_clip_planes,
                                                 bool separable_shader,
                                                 bool invariant_position = false) {
    std::string out;

    const auto append_variable = [&](std::string_view var, int location) {
        if (separable_shader) {
            out += fmt::format("layout (location={}) ", location);
        }
        out += fmt::format("{}{};\n", is_output ? "out " : "in ", var);
    };

    append_variable("vec4 primary_color", ATTRIBUTE_COLOR);
    append_variable("vec2 texcoord0", ATTRIBUTE_TEXCOORD0);
    append_variable("vec2 texcoord1", ATTRIBUTE_TEXCOORD1);
    append_variable("vec2 texcoord2", ATTRIBUTE_TEXCOORD2);
    append_variable("float texcoord0_w", ATTRIBUTE_TEXCOORD0_W);
    append_variable("vec4 normquat", ATTRIBUTE_NORMQUAT);
    append_variable("vec3 view", ATTRIBUTE_VIEW);

    if (is_output && separable_shader) {
        // gl_PerVertex redeclaration is required for separate shader object
        out += "out gl_PerVertex {\n";
        // Apple Silicon GPU drivers optimize more aggressively, which can create
        // too much variance and cause visual artifacting in games like Pokemon.
#ifdef __APPLE__
        invariant_position = true;
#endif
        out += invariant_position ? "    invariant vec4 gl_Position;\n" : "    vec4 gl_Position;\n";
        if (use_clip_planes) {
            out += "    float gl_ClipDistance[2];\n";
        }
        out += "};\n";
    }

    return out;
}

std::string GenerateTrivialVertexShader(bool use_clip_planes, bool separable_shader) {
    std::string out;
    if (separable_shader) {
        out += "#extension GL_ARB_separate_shader_objects : enable\n";
    }

    out +=
        fmt::format("layout(location = {}) in vec4 vert_position;\n"
                    "layout(location = {}) in vec4 vert_color;\n"
                    "layout(location = {}) in vec2 vert_texcoord0;\n"
                    "layout(location = {}) in vec2 vert_texcoord1;\n"
                    "layout(location = {}) in vec2 vert_texcoord2;\n"
                    "layout(location = {}) in float vert_texcoord0_w;\n"
                    "layout(location = {}) in vec4 vert_normquat;\n"
                    "layout(location = {}) in vec3 vert_view;\n",
                    ATTRIBUTE_POSITION, ATTRIBUTE_COLOR, ATTRIBUTE_TEXCOORD0, ATTRIBUTE_TEXCOORD1,
                    ATTRIBUTE_TEXCOORD2, ATTRIBUTE_TEXCOORD0_W, ATTRIBUTE_NORMQUAT, ATTRIBUTE_VIEW);

    out += GetVertexInterfaceDeclaration(true, use_clip_planes, separable_shader);
    out += VSUniformBlockDef;

    out += R"(
void main() {
    primary_color = vert_color;
    texcoord0 = vert_texcoord0;
    texcoord1 = vert_texcoord1;
    texcoord2 = vert_texcoord2;
    texcoord0_w = vert_texcoord0_w;
    normquat = vert_normquat;
    view = vert_view;
    vec4 vtx_pos = SanitizeVertex(vert_position);
    if (flip_viewport) {
        vtx_pos.y = -vtx_pos.y;
    }
)";
    out += PositionOutput;
    if (use_clip_planes) {
        out += R"(
        gl_ClipDistance[0] = -vtx_pos.z; // fixed PICA clipping plane z <= 0
        if (enable_clip1) {
            gl_ClipDistance[1] = dot(clip_coef, vtx_pos);
        } else {
            gl_ClipDistance[1] = 0.0;
        }
        )";
    }

    out += "}\n";

    return out;
}

namespace {

// Decoded instruction layout.
// Word 0: op | dest << 5 | dest mask << 11 | address register << 15 | a << 17 | b << 24 |
//         block end << 31
// Word 1: c | swizzle a << 5 | swizzle b << 13 | swizzle c << 21 | negate a, b, c << 29
// Flow control uses word 1 as: target | count << 12 | instruction bits 22-25 << 20
enum class InterpOp : u32 {
    Mov,
    Add,
    Mul,
    Mad,
    Dp3,
    Dp4,
    Dph,
    Max,
    Min,
    Flr,
    Rcp,
    Rsq,
    Sge,
    Slt,
    Ex2,
    Lg2,
    Mova,
    Cmp,
    // Flow control from here on
    End,
    Jmpc,
    Jmpu,
    Call,
    Callc,
    Callu,
    Ifu,
    Ifc,
    Loop,
    Break,
    Breakc,
    Nop,
};

// Register file of the interpreter: inputs, temporaries, outputs and one left at (0, 0, 0, 1)
constexpr u32 INTERP_REG_OUTPUT = 32;
constexpr u32 INTERP_REG_DEFAULT = 48;
constexpr u32 INTERP_REG_COUNT = 49;

} // Anonymous namespace

std::string GenerateVertexInterpreter(bool use_clip_planes, bool sanitize_mul) {
    using CompareOp = nihstro::Instruction::Common::CompareOpType::Op;
    using FlowOp = nihstro::Instruction::FlowControlType::Op;
    using Format = Pica::PipelineRegs::VertexAttributeFormat;

    std::string out = "#extension GL_ARB_separate_shader_objects : enable\n";
    out += fmt::format("#define SANITIZE_MUL {}\n#define USE_CLIP_PLANES {}\n"
                       "#define MAX_INSTRUCTIONS 4096u\n",
                       static_cast<u32>(sanitize_mul), static_cast<u32>(use_clip_planes));
    const auto define = [&out](std::string_view name, auto value) {
        out += fmt::format("#define {} {}u\n", name, static_cast<u32>(value));
    };
    define("OP_ADD", InterpOp::Add);
    define("OP_MUL", InterpOp::Mul);
    define("OP_MAD", InterpOp::Mad);
    define("OP_DP3", InterpOp::Dp3);
    define("OP_DP4", InterpOp::Dp4);
    define("OP_DPH", InterpOp::Dph);
    define("OP_MAX", InterpOp::Max);
    define("OP_MIN", InterpOp::Min);
    define("OP_FLR", InterpOp::Flr);
    define("OP_RCP", InterpOp::Rcp);
    define("OP_RSQ", InterpOp::Rsq);
    define("OP_SGE", InterpOp::Sge);
    define("OP_SLT", InterpOp::Slt);
    define("OP_EX2", InterpOp::Ex2);
    define("OP_LG2", InterpOp::Lg2);
    define("OP_MOVA", InterpOp::Mova);
    define("OP_CMP", InterpOp::Cmp);
    define("OP_END", InterpOp::End);
    define("OP_JMPC", InterpOp::Jmpc);
    define("OP_JMPU", InterpOp::Jmpu);
    define("OP_CALL", InterpOp::Call);
    define("OP_CALLC", InterpOp::Callc);
    define("OP_CALLU", InterpOp::Callu);
    define("OP_IFU", InterpOp::Ifu);
    define("OP_IFC", InterpOp::Ifc);
    define("OP_LOOP", InterpOp::Loop);
    define("OP_BREAK", InterpOp::Break);
    define("OP_BREAKC", InterpOp::Breakc);
    define("REG_COUNT", INTERP_REG_COUNT);
    define("CMP_EQUAL", CompareOp::Equal);
    define("CMP_NOT_EQUAL", CompareOp::NotEqual);
    define("CMP_LESS_THAN", CompareOp::LessThan);
    define("CMP_LESS_EQUAL", CompareOp::LessEqual);
    define("CMP_GREATER_THAN", CompareOp::GreaterThan);
    define("FLOW_OR", FlowOp::Or);
    define("FLOW_AND", FlowOp::And);
    define("FLOW_JUST_X", FlowOp::JustX);
    define("FORMAT_BYTE", Format::BYTE);
    define("FORMAT_UBYTE", Format::UBYTE);
    define("FORMAT_SHORT", Format::SHORT);
    define("POSITION_X", VSOutputAttributes::POSITION_X);
    define("POSITION_Y", VSOutputAttributes::POSITION_Y);
    define("POSITION_Z", VSOutputAttributes::POSITION_Z);
    define("POSITION_W", VSOutputAttributes::POSITION_W);
    define("QUATERNION_X", VSOutputAttributes::QUATERNION_X);
    define("QUATERNION_Y", VSOutputAttributes::QUATERNION_Y);
    define("QUATERNION_Z", VSOutputAttributes::QUATERNION_Z);
    define("QUATERNION_W", VSOutputAttributes::QUATERNION_W);
    define("COLOR_R", VSOutputAttributes::COLOR_R);
    define("COLOR_G", VSOutputAttributes::COLOR_G);
    define("COLOR_B", VSOutputAttributes::COLOR_B);
    define("COLOR_A", VSOutputAttributes::COLOR_A);
    define("TEXCOORD0_U", VSOutputAttributes::TEXCOORD0_U);
    define("TEXCOORD0_V", VSOutputAttributes::TEXCOORD0_V);
    define("TEXCOORD1_U", VSOutputAttributes::TEXCOORD1_U);
    define("TEXCOORD1_V", VSOutputAttributes::TEXCOORD1_V);
    define("TEXCOORD0_W", VSOutputAttributes::TEXCOORD0_W);
    define("VIEW_X", VSOutputAttributes::VIEW_X);
    define("VIEW_Y", VSOutputAttributes::VIEW_Y);
    define("VIEW_Z", VSOutputAttributes::VIEW_Z);
    define("TEXCOORD2_U", VSOutputAttributes::TEXCOORD2_U);
    define("TEXCOORD2_V", VSOutputAttributes::TEXCOORD2_V);

    out += VSPicaUniformBlockDef;
    std::string block{VSUniformBlockDef};
    block.replace(block.find("};"), 2,
                  "    uvec4 interp_config;\n"
                  "    uvec4 interp_semantic_map[6];\n"
                  "    uvec4 interp_attributes[16];\n"
                  "};");
    out += block;
    out += "layout (set = 0, binding = 6, std140) uniform vs_interpreter_program {\n"
           "    uvec4 program_code[2048];\n"
           "};\n"
           "layout (set = 0, binding = 7, std430) readonly buffer vs_vertex_data {\n"
           "    uint vertex_words[];\n"
           "};\n";
    out += GetVertexInterfaceDeclaration(true, use_clip_planes, true, true);

    std::string body = R"glsl(
// inputs 0-15, temporaries 16-31, outputs 32-47, 48 is never written
vec4 regs[REG_COUNT];
bvec2 conditional_code = bvec2(false);
// x stays 0 for operands without an address register, then a0.x, a0.y, aL
ivec4 address_registers = ivec4(0);

#if SANITIZE_MUL
vec4 sanitize_mul(vec4 lhs, vec4 rhs) {
    vec4 product = lhs * rhs;
    return mix(product, mix(mix(vec4(0.0), product, isnan(rhs)), product, isnan(lhs)), isnan(product));
}
#define MUL(a, b) sanitize_mul(a, b)
#else
#define MUL(a, b) ((a) * (b))
#endif

uint bits(uint word, uint shift, uint count) {
    return (word >> shift) & ((1u << count) - 1u);
}

// Reads up to 4 bytes of vertex data, unaligned
uint vertexBytes(uint address, uint count) {
    uint index = address >> 2;
    uint skip = address & 3u;
    uint word = vertex_words[index] >> (skip * 8u);
    if (skip + count > 4u) {
        word |= vertex_words[index + 1u] << (32u - skip * 8u);
    }
    return word;
}

vec4 loadAttribute(uvec4 slot) {
    uint address = slot.x + uint(gl_VertexIndex) * slot.y;
    uint count = slot.w;
    vec4 value = vec4(0.0);
    switch (slot.z) {
    case FORMAT_BYTE:
        value = vec4((ivec4(vertexBytes(address, count)) << ivec4(24, 16, 8, 0)) >> 24);
        break;
    case FORMAT_UBYTE:
        value = vec4((uvec4(vertexBytes(address, count)) >> uvec4(0u, 8u, 16u, 24u)) & 0xFFu);
        break;
    case FORMAT_SHORT: {
        uint low = vertexBytes(address, min(count, 2u) * 2u);
        uint high = count > 2u ? vertexBytes(address + 4u, count * 2u - 4u) : 0u;
        value = vec4((ivec4(low, low, high, high) << ivec4(16, 0, 16, 0)) >> 16);
        break;
    }
    default:
        for (uint i = 0u; i < count; i++) {
            value[i] = uintBitsToFloat(vertexBytes(address + 4u * i, 4u));
        }
        break;
    }
    return mix(vec4(0.0, 0.0, 0.0, 1.0), value, lessThan(uvec4(0u, 1u, 2u, 3u), uvec4(count)));
}

vec4 operand(uint index, uint selector, bool negate, uint address_index) {
    vec4 value;
    if (index < 32u) {
        value = regs[index];
    } else {
        int offset = address_registers[address_index];
        offset = offset >= -128 && offset <= 127 ? offset : 0;
        uint slot = uint(int(index) - 32 + offset) & 0x7Fu;
        value = slot < 96u ? uniforms.f[slot] : vec4(1.0);
    }
    value = vec4(value[selector >> 6], value[(selector >> 4) & 3u], value[(selector >> 2) & 3u],
                 value[selector & 3u]);
    return negate ? -value : value;
}

// flow: bits 22-25 of the instruction
bool evaluateCondition(uint flow) {
    bool result_x = ((flow & 8u) != 0u) == conditional_code.x;
    bool result_y = ((flow & 4u) != 0u) == conditional_code.y;
    switch (flow & 3u) {
    case FLOW_OR:
        return result_x || result_y;
    case FLOW_AND:
        return result_x && result_y;
    case FLOW_JUST_X:
        return result_x;
    default:
        return result_y;
    }
}

bool boolUniform(uint index) {
    return (uniforms.b & (1u << index)) != 0u;
}

bool compare(uint op, float a, float b) {
    switch (op) {
    case CMP_EQUAL:
        return a == b;
    case CMP_NOT_EQUAL:
        return a != b;
    case CMP_LESS_THAN:
        return a < b;
    case CMP_LESS_EQUAL:
        return a <= b;
    case CMP_GREATER_THAN:
        return a > b;
    default:
        return a >= b;
    }
}

// Same if/call/loop stacks as the software interpreter
void runProgram() {
    uint if_else[8];
    uint if_end[8];
    int if_top = 0;
    uint call_end[4];
    uint call_return[4];
    int call_top = 0;
    uint loop_entry[4];
    uint loop_end[4];
    uint loop_count[4];
    uint loop_step[4];
    int loop_previous[4];
    int loop_top = 0;

    uint pc = interp_config.x;
    for (uint executed = 0u; executed < MAX_INSTRUCTIONS; executed++) {
        if (pc >= 4095u) {
            return;
        }
        uvec4 pair = program_code[pc >> 1];
        uvec2 ins = (pc & 1u) == 0u ? pair.xy : pair.zw;
        uint op = ins.x & 31u;
        uint old_pc = pc;
        bool is_break = false;
        pc++;

        if (op < OP_END) {
            uint dest = bits(ins.x, 5u, 6u);
            uint mask = bits(ins.x, 11u, 4u);
            uint address_index = bits(ins.x, 15u, 2u);
            vec4 a = operand(bits(ins.x, 17u, 7u), bits(ins.y, 5u, 8u), (ins.y & 0x20000000u) != 0u,
                             address_index);
            vec4 b = operand(bits(ins.x, 24u, 7u), bits(ins.y, 13u, 8u),
                             (ins.y & 0x40000000u) != 0u, address_index);
            vec4 value = a;
            bool write = true;
            switch (op) {
            case OP_ADD:
                value = a + b;
                break;
            case OP_MUL:
                value = MUL(a, b);
                break;
            case OP_MAD: {
                vec4 c = operand(ins.y & 31u, bits(ins.y, 21u, 8u), (ins.y >> 31) != 0u, 0u);
                value = MUL(c, a) + b;
                break;
            }
            case OP_DP3:
#if SANITIZE_MUL
                value = vec4(dot(vec3(sanitize_mul(a, b)), vec3(1.0)));
#else
                value = vec4(dot(vec3(a), vec3(b)));
#endif
                break;
            case OP_DP4:
#if SANITIZE_MUL
                value = vec4(dot(sanitize_mul(a, b), vec4(1.0)));
#else
                value = vec4(dot(a, b));
#endif
                break;
            case OP_DPH:
#if SANITIZE_MUL
                value = vec4(dot(sanitize_mul(vec4(a.xyz, 1.0), b), vec4(1.0)));
#else
                value = vec4(dot(a, b));
#endif
                break;
            case OP_MAX:
#if SANITIZE_MUL
                value = mix(b, a, greaterThan(a, b));
#else
                value = max(a, b);
#endif
                break;
            case OP_MIN:
#if SANITIZE_MUL
                value = mix(b, a, lessThan(a, b));
#else
                value = min(a, b);
#endif
                break;
            case OP_FLR:
                value = floor(a);
                break;
            case OP_RCP:
#if !SANITIZE_MUL
                write = a.x != 0.0;
#endif
                value = vec4(1.0 / a.x);
                break;
            case OP_RSQ:
#if !SANITIZE_MUL
                write = a.x > 0.0;
#endif
                value = vec4(inversesqrt(a.x));
                break;
            case OP_SGE:
                value = vec4(greaterThanEqual(a, b));
                break;
            case OP_SLT:
                value = vec4(lessThan(a, b));
                break;
            case OP_EX2:
                value = vec4(exp2(a.x));
                break;
            case OP_LG2:
                value = vec4(log2(a.x));
                break;
            case OP_MOVA: {
                ivec2 address = ivec2(a);
                if ((mask & 8u) != 0u) {
                    address_registers.y = address.x;
                }
                if ((mask & 4u) != 0u) {
                    address_registers.z = address.y;
                }
                write = false;
                break;
            }
            case OP_CMP:
                conditional_code = bvec2(compare(dest & 7u, a.x, b.x), compare(dest >> 3, a.y, b.y));
                write = false;
                break;
            default:
                break;
            }
            if (write) {
                bvec4 lanes = notEqual(uvec4(mask) & uvec4(8u, 4u, 2u, 1u), uvec4(0u));
                regs[dest] = mix(regs[dest], value, lanes);
            }
        } else {
            uint target = ins.y & 0xFFFu;
            uint count = bits(ins.y, 12u, 8u);
            uint flow = bits(ins.y, 20u, 4u);
            if (op == OP_END) {
                return;
            } else if (op == OP_JMPC) {
                if (evaluateCondition(flow)) {
                    pc = target;
                }
            } else if (op == OP_JMPU) {
                if (boolUniform(flow) == ((count & 1u) == 0u)) {
                    pc = target;
                }
            } else if (op == OP_CALL || op == OP_CALLC || op == OP_CALLU) {
                bool taken = op == OP_CALL ||
                             (op == OP_CALLC ? evaluateCondition(flow) : boolUniform(flow));
                if (taken) {
                    if (call_top < 4) {
                        call_end[call_top] = target + count;
                        call_return[call_top] = old_pc + 1u;
                        call_top++;
                    }
                    pc = target;
                }
            } else if (op == OP_IFU || op == OP_IFC) {
                bool taken = op == OP_IFC ? evaluateCondition(flow) : boolUniform(flow);
                if (!taken) {
                    pc = target;
                } else if (if_top < 8) {
                    if_else[if_top] = target;
                    if_end[if_top] = target + count;
                    if_top++;
                }
            } else if (op == OP_LOOP) {
                uvec4 loop = uniforms.i[flow & 3u];
                if (loop_top < 4) {
                    loop_entry[loop_top] = old_pc + 1u;
                    loop_end[loop_top] = target + 1u;
                    loop_count[loop_top] = loop.x;
                    loop_step[loop_top] = loop.z;
                    loop_previous[loop_top] = int(loop.y);
                    loop_top++;
                }
                address_registers.w = int(loop.y);
            } else if (op == OP_BREAK) {
                is_break = true;
            } else if (op == OP_BREAKC) {
                is_break = evaluateCondition(flow);
            }
        }

        // The top bit is set on the last instruction of every call, if and loop block
        if ((ins.x >> 31) == 0u && !is_break) {
            continue;
        }
        uint next_pc = old_pc + 1u;
        for (int i = 0; i < 4; i++) {
            if (call_top == 0 || call_end[call_top - 1] != next_pc) {
                break;
            }
            if (i < 3) {
                pc = call_return[call_top - 1];
                next_pc = pc;
            }
            call_top--;
        }
        if (if_top > 0 && if_else[if_top - 1] == old_pc + 1u) {
            pc = if_end[if_top - 1];
            if_top--;
        }
        if (loop_top > 0 && (loop_end[loop_top - 1] == old_pc + 1u || is_break)) {
            int top = loop_top - 1;
            address_registers.w += int(loop_step[top]);
            if (!is_break && loop_count[top] != 0u) {
                loop_count[top]--;
                pc = loop_entry[top];
            } else {
                pc = loop_end[top];
                if (loop_top > 1) {
                    address_registers.w = loop_previous[top];
                }
                loop_top--;
            }
        }
    }
}

float semantic(uint slot) {
    uint map = interp_semantic_map[slot >> 2][slot & 3u];
    return map < 0xFFu ? regs[map >> 2][map & 3u] : 1.0;
}

void main() {
    for (uint i = 0u; i < REG_COUNT; i++) {
        regs[i] = vec4(0.0, 0.0, 0.0, 1.0);
    }
    for (uint pending = interp_config.y; pending != 0u; pending &= pending - 1u) {
        uint i = uint(findLSB(pending));
        regs[i] = loadAttribute(interp_attributes[i]);
    }

    runProgram();

    vec4 vtx_pos = vec4(semantic(POSITION_X), semantic(POSITION_Y), semantic(POSITION_Z),
                        semantic(POSITION_W));
    vtx_pos = SanitizeVertex(vtx_pos);
    if (flip_viewport) {
        vtx_pos.y = -vtx_pos.y;
    }
POSITION_OUTPUT
#if USE_CLIP_PLANES
    gl_ClipDistance[0] = -vtx_pos.z;
    if (enable_clip1) {
        gl_ClipDistance[1] = dot(clip_coef, vtx_pos);
    } else {
        gl_ClipDistance[1] = 0.0;
    }
#endif

    normquat = vec4(semantic(QUATERNION_X), semantic(QUATERNION_Y), semantic(QUATERNION_Z),
                    semantic(QUATERNION_W));
    vec4 vtx_color = vec4(semantic(COLOR_R), semantic(COLOR_G), semantic(COLOR_B),
                          semantic(COLOR_A));
    primary_color = min(abs(vtx_color), vec4(1.0));
    texcoord0 = vec2(semantic(TEXCOORD0_U), semantic(TEXCOORD0_V));
    texcoord1 = vec2(semantic(TEXCOORD1_U), semantic(TEXCOORD1_V));
    texcoord0_w = semantic(TEXCOORD0_W);
    view = vec3(semantic(VIEW_X), semantic(VIEW_Y), semantic(VIEW_Z));
    texcoord2 = vec2(semantic(TEXCOORD2_U), semantic(TEXCOORD2_V));
}
)glsl";
    const std::string_view marker = "POSITION_OUTPUT\n";
    body.replace(body.find(marker), marker.size(), PositionOutput);
    out += body;
    return out;
}

void PackVertexInterpreterConfig(const PicaVSConfig& config, VSUniformData& data) {
    const auto& state = config.state;
    data.interp_config = {state.main_offset, 0u, 0u, 0u};

    // Output attribute -> the register that writes it
    std::array<u32, 16> registers;
    registers.fill(INTERP_REG_DEFAULT);
    for (u32 reg = 0; reg < 16; reg++) {
        if (state.output_map[reg] < state.num_outputs) {
            registers[state.output_map[reg]] = INTERP_REG_OUTPUT + reg;
        }
    }
    const auto semantic_maps = state.gs_state.GetSemanticMaps();
    for (u32 slot = 0; slot < semantic_maps.size(); slot++) {
        const auto& map = semantic_maps[slot];
        const bool mapped = map.attribute_index < state.gs_state.gs_output_attributes_count &&
                            map.attribute_index < registers.size();
        data.interp_semantic_map[slot / 4][slot % 4] =
            mapped ? registers[map.attribute_index] * 4 + map.component_index : 0xFFu;
    }
}

void DecodeVertexProgram(std::span<const u32> code, std::span<const u32> swizzles,
                         VSInterpreterProgram& program) {
    using OpCode = nihstro::OpCode::Id;
    using CompareOp = nihstro::Instruction::Common::CompareOpType::Op;
    const auto bits = [](u32 word, u32 shift, u32 count) {
        return (word >> shift) & ((1u << count) - 1);
    };
    const auto id = [](OpCode opcode) { return static_cast<u32>(opcode); };

    // Addresses right after the end of a call, if or loop block
    std::bitset<4096 + 1> block_ends;
    for (const u32 instr : code) {
        const u32 target = bits(instr, 10, 12);
        u32 end;
        switch (static_cast<OpCode>(instr >> 26)) {
        case OpCode::CALL:
        case OpCode::CALLC:
        case OpCode::CALLU:
            end = target + bits(instr, 0, 8);
            break;
        case OpCode::IFU:
        case OpCode::IFC:
            end = target;
            break;
        case OpCode::LOOP:
            end = target + 1;
            break;
        default:
            continue;
        }
        if (end < block_ends.size()) {
            block_ends.set(end);
        }
    }

    struct Operand {
        u32 reg;
        u32 selector;
        u32 negate;
    };
    // index: 0-2 for src1-src3 of the operand descriptor
    const auto operand = [&](u32 pattern, u32 index, u32 reg) {
        const u32 shift = 4 + index * 9;
        return Operand{reg, bits(pattern, shift + 1, 8), bits(pattern, shift, 1)};
    };
    const auto pack = [](InterpOp op, u32 dest, u32 mask, u32 address, const Operand& a,
                         const Operand& b, const Operand& c) {
        return std::array<u32, 2>{
            static_cast<u32>(op) | dest << 5 | mask << 11 | address << 15 | a.reg << 17 |
                b.reg << 24,
            c.reg | a.selector << 5 | b.selector << 13 | c.selector << 21 | a.negate << 29 |
                b.negate << 30 | c.negate << 31,
        };
    };
    const auto dest_register = [](u32 dest) { return dest < 16 ? INTERP_REG_OUTPUT + dest : dest; };

    const u32 length = static_cast<u32>(std::min(code.size(), program.code.size() / 2));
    for (u32 pc = 0; pc < length; pc++) {
        const u32 instr = code[pc];
        const u32 opcode = instr >> 26;
        std::array<u32, 2> words{static_cast<u32>(InterpOp::Nop), 0};

        if (opcode >= id(OpCode::MADI)) {
            const bool inverted = opcode < id(OpCode::MAD);
            const u32 pattern = swizzles[bits(instr, 0, 5)];
            const u32 src2 = inverted ? bits(instr, 12, 5) : bits(instr, 10, 7);
            const u32 src3 = inverted ? bits(instr, 5, 7) : bits(instr, 5, 5);
            // src1 can't be a uniform, so it takes the narrow slot
            words = pack(InterpOp::Mad, dest_register(bits(instr, 24, 5)), pattern & 0xF,
                         bits(instr, 22, 2), operand(pattern, 1, src2), operand(pattern, 2, src3),
                         operand(pattern, 0, bits(instr, 17, 5)));
        } else if (opcode == id(OpCode::CMP) || opcode == id(OpCode::CMP) + 1) {
            const u32 pattern = swizzles[bits(instr, 0, 7)];
            const u32 op_x = bits(instr, 24, 3);
            const u32 op_y = bits(instr, 21, 3);
            const u32 last = static_cast<u32>(CompareOp::GreaterEqual);
            if (op_x <= last && op_y <= last) {
                // The compare ops go in the dest field
                words = pack(InterpOp::Cmp, op_x | op_y << 3, 0, bits(instr, 19, 2),
                             operand(pattern, 0, bits(instr, 12, 7)),
                             operand(pattern, 1, bits(instr, 7, 5)), {});
            }
        } else if (opcode < id(OpCode::BREAK)) {
            InterpOp op = InterpOp::Nop;
            bool inverted = false;
            switch (static_cast<OpCode>(opcode)) {
            case OpCode::ADD:
                op = InterpOp::Add;
                break;
            case OpCode::DP3:
                op = InterpOp::Dp3;
                break;
            case OpCode::DP4:
                op = InterpOp::Dp4;
                break;
            case OpCode::DPHI:
                inverted = true;
                [[fallthrough]];
            case OpCode::DPH:
                op = InterpOp::Dph;
                break;
            case OpCode::EX2:
                op = InterpOp::Ex2;
                break;
            case OpCode::LG2:
                op = InterpOp::Lg2;
                break;
            case OpCode::MUL:
                op = InterpOp::Mul;
                break;
            case OpCode::SGEI:
                inverted = true;
                [[fallthrough]];
            case OpCode::SGE:
                op = InterpOp::Sge;
                break;
            case OpCode::SLTI:
                inverted = true;
                [[fallthrough]];
            case OpCode::SLT:
                op = InterpOp::Slt;
                break;
            case OpCode::FLR:
                op = InterpOp::Flr;
                break;
            case OpCode::MAX:
                op = InterpOp::Max;
                break;
            case OpCode::MIN:
                op = InterpOp::Min;
                break;
            case OpCode::RCP:
                op = InterpOp::Rcp;
                break;
            case OpCode::RSQ:
                op = InterpOp::Rsq;
                break;
            case OpCode::MOVA:
                op = InterpOp::Mova;
                break;
            case OpCode::MOV:
                op = InterpOp::Mov;
                break;
            default:
                break;
            }
            if (op != InterpOp::Nop) {
                const u32 pattern = swizzles[bits(instr, 0, 7)];
                const u32 src1 = inverted ? bits(instr, 14, 5) : bits(instr, 12, 7);
                const u32 src2 = inverted ? bits(instr, 7, 7) : bits(instr, 7, 5);
                words =
                    pack(op, dest_register(bits(instr, 21, 5)), pattern & 0xF, bits(instr, 19, 2),
                         operand(pattern, 0, src1), operand(pattern, 1, src2), {});
            }
        } else {
            InterpOp op = InterpOp::Nop;
            switch (static_cast<OpCode>(opcode)) {
            case OpCode::END:
                op = InterpOp::End;
                break;
            case OpCode::JMPC:
                op = InterpOp::Jmpc;
                break;
            case OpCode::JMPU:
                op = InterpOp::Jmpu;
                break;
            case OpCode::CALL:
                op = InterpOp::Call;
                break;
            case OpCode::CALLC:
                op = InterpOp::Callc;
                break;
            case OpCode::CALLU:
                op = InterpOp::Callu;
                break;
            case OpCode::IFU:
                op = InterpOp::Ifu;
                break;
            case OpCode::IFC:
                op = InterpOp::Ifc;
                break;
            case OpCode::LOOP:
                op = InterpOp::Loop;
                break;
            case OpCode::BREAK:
                op = InterpOp::Break;
                break;
            case OpCode::BREAKC:
                op = InterpOp::Breakc;
                break;
            default:
                break;
            }
            words = {static_cast<u32>(op),
                     bits(instr, 10, 12) | bits(instr, 0, 8) << 12 | bits(instr, 22, 4) << 20};
        }

        if (block_ends[pc + 1]) {
            words[0] |= 1u << 31;
        }
        program.code[2 * pc] = words[0];
        program.code[2 * pc + 1] = words[1];
    }
}

std::string_view MakeLoadPrefix(AttribLoadFlags flag) {
    if (True(flag & AttribLoadFlags::Float)) {
        return "";
    } else if (True(flag & AttribLoadFlags::Sint)) {
        return "i";
    } else if (True(flag & AttribLoadFlags::Uint)) {
        return "u";
    }
    return "";
}

std::string GenerateVertexShader(const ShaderSetup& setup, const PicaVSConfig& config,
                                 const ExtraVSConfig& extra) {
    const bool separable_shader = extra.separable_shader;

    std::string out;
    if (separable_shader) {
        out += "#extension GL_ARB_separate_shader_objects : enable\n";
    }

    out += VSPicaUniformBlockDef;
    out += VSUniformBlockDef;

    std::array<bool, 16> used_regs{};
    const auto get_input_reg = [&used_regs](u32 reg) {
        ASSERT(reg < 16);
        used_regs[reg] = true;
        return fmt::format("vs_in_reg{}", reg);
    };

    const auto get_output_reg = [&](u32 reg) -> std::string {
        ASSERT(reg < 16);
        if (config.state.output_map[reg] < config.state.num_outputs) {
            return fmt::format("vs_out_attr{}", config.state.output_map[reg]);
        }
        return "";
    };

    auto program_source = DecompileProgram(setup.GetProgramCode(), setup.GetSwizzleData(),
                                           config.state.main_offset, get_input_reg, get_output_reg,
                                           extra.sanitize_mul, config.state.bool_uniforms);

    if (program_source.empty()) {
        return "";
    }

    // input attributes declaration
    for (std::size_t i = 0; i < used_regs.size(); ++i) {
        if (used_regs[i]) {
            const auto flags = extra.load_flags[i];
            const std::string_view prefix = MakeLoadPrefix(flags);
            out +=
                fmt::format("layout(location = {0}) in {1}vec4 vs_in_typed_reg{0};\n", i, prefix);
            out += fmt::format("vec4 vs_in_reg{0};\n", i);
        }
    }
    out += '\n';

    if (extra.use_geometry_shader) {
        // output attributes declaration
        for (u32 i = 0; i < config.state.num_outputs; ++i) {
            if (separable_shader) {
                out += fmt::format("layout(location = {}) ", i);
            }
            out += fmt::format("out vec4 vs_out_attr{};\n", i);
            if (extra.invariant_position) {
                out += fmt::format("invariant vs_out_attr{};\n", i);
            }
        }
        out += "void EmitVtx() {}\n";
    } else {
        out += GetVertexInterfaceDeclaration(true, extra.use_clip_planes, separable_shader,
                                             extra.invariant_position);

        // output attributes declaration
        for (u32 i = 0; i < config.state.num_outputs; ++i) {
            out += fmt::format("vec4 vs_out_attr{};\n", i);
        }

        const auto semantic_maps = config.state.gs_state.GetSemanticMaps();

        const auto semantic = [&state = config.state, &semantic_maps](
                                  VSOutputAttributes::Semantic slot_semantic) -> std::string {
            const u32 slot = static_cast<u32>(slot_semantic);
            const u32 attrib = semantic_maps[slot].attribute_index;
            const u32 comp = semantic_maps[slot].component_index;
            if (attrib < state.gs_state.gs_output_attributes_count) {
                return fmt::format("vs_out_attr{}.{}", attrib, "xyzw"[comp]);
            }
            return "1.0";
        };

        out += "vec4 GetVertexQuaternion() {\n";
        out += "    return vec4(" + semantic(VSOutputAttributes::QUATERNION_X) + ", " +
               semantic(VSOutputAttributes::QUATERNION_Y) + ", " +
               semantic(VSOutputAttributes::QUATERNION_Z) + ", " +
               semantic(VSOutputAttributes::QUATERNION_W) + ");\n";
        out += "}\n\n";

        out += "void EmitVtx() {\n";
        out += "    vec4 vtx_pos = vec4(" + semantic(VSOutputAttributes::POSITION_X) + ", " +
               semantic(VSOutputAttributes::POSITION_Y) + ", " +
               semantic(VSOutputAttributes::POSITION_Z) + ", " +
               semantic(VSOutputAttributes::POSITION_W) + ");\n";
        out += "    vtx_pos = SanitizeVertex(vtx_pos);\n";
        out += "    if (flip_viewport) {\n";
        out += "        vtx_pos.y = -vtx_pos.y;\n";
        out += "    }\n";
        out += PositionOutput;
        if (extra.use_clip_planes) {
            out += "    gl_ClipDistance[0] = -vtx_pos.z;\n"; // fixed PICA clipping plane z <= 0
            out += "    if (enable_clip1) {\n";
            out += "        gl_ClipDistance[1] = dot(clip_coef, vtx_pos);\n";
            out += "    } else {\n";
            out += "        gl_ClipDistance[1] = 0.0;\n";
            out += "    }\n\n";
        }

        out += "    normquat = GetVertexQuaternion();\n";
        out += "    vec4 vtx_color = vec4(" + semantic(VSOutputAttributes::COLOR_R) + ", " +
               semantic(VSOutputAttributes::COLOR_G) + ", " +
               semantic(VSOutputAttributes::COLOR_B) + ", " +
               semantic(VSOutputAttributes::COLOR_A) + ");\n";
        out += "    primary_color = min(abs(vtx_color), vec4(1.0));\n\n";

        out += "    texcoord0 = vec2(" + semantic(VSOutputAttributes::TEXCOORD0_U) + ", " +
               semantic(VSOutputAttributes::TEXCOORD0_V) + ");\n";
        out += "    texcoord1 = vec2(" + semantic(VSOutputAttributes::TEXCOORD1_U) + ", " +
               semantic(VSOutputAttributes::TEXCOORD1_V) + ");\n\n";

        out += "    texcoord0_w = " + semantic(VSOutputAttributes::TEXCOORD0_W) + ";\n";
        out += "    view = vec3(" + semantic(VSOutputAttributes::VIEW_X) + ", " +
               semantic(VSOutputAttributes::VIEW_Y) + ", " + semantic(VSOutputAttributes::VIEW_Z) +
               ");\n\n";

        out += "    texcoord2 = vec2(" + semantic(VSOutputAttributes::TEXCOORD2_U) + ", " +
               semantic(VSOutputAttributes::TEXCOORD2_V) + ");\n\n";
        out += "}\n";
    }

    out += "bool exec_shader();\n\n";

    out += "\nvoid main() {\n";
    for (std::size_t i = 0; i < used_regs.size(); ++i) {
        if (used_regs[i]) {
            out += fmt::format("vs_in_reg{0} = vec4(vs_in_typed_reg{0});\n", i);
            if (True(extra.load_flags[i] & AttribLoadFlags::ZeroW)) {
                out += fmt::format("vs_in_reg{0}.w = 0;\n", i);
            }
        }
    }
    for (u32 i = 0; i < config.state.num_outputs; ++i) {
        out += fmt::format("    vs_out_attr{} = vec4(0.0, 0.0, 0.0, 1.0);\n", i);
    }
    out += "\n    exec_shader();\n    EmitVtx();\n}\n\n";

    out += program_source;

    return out;
}

static std::string GetGSCommonSource(const PicaGSConfigState& state,
                                     const ExtraFixedGSConfig& extra) {
    std::string out =
        GetVertexInterfaceDeclaration(true, extra.use_clip_planes, extra.separable_shader);
    out += VSUniformBlockDef;

    out += '\n';
    for (u32 i = 0; i < state.vs_output_attributes_count; ++i) {
        if (extra.separable_shader) {
            out += fmt::format("layout(location = {}) ", i);
        }
        out += fmt::format("in vec4 vs_out_attr{}[];\n", i);
    }

    out += R"(
struct Vertex {
)";
    out += fmt::format("    vec4 attributes[{}];\n", state.gs_output_attributes_count);
    out += "};\n\n";

    const auto semantic_maps = state.GetSemanticMaps();

    const auto semantic =
        [&state, &semantic_maps](VSOutputAttributes::Semantic slot_semantic) -> std::string {
        const u32 slot = static_cast<u32>(slot_semantic);
        const u32 attrib = semantic_maps[slot].attribute_index;
        const u32 comp = semantic_maps[slot].component_index;
        if (attrib < state.gs_output_attributes_count) {
            return fmt::format("vtx.attributes[{}].{}", attrib, "xyzw"[comp]);
        }
        return "1.0";
    };

    out += "vec4 GetVertexQuaternion(Vertex vtx) {\n";
    out += "    return vec4(" + semantic(VSOutputAttributes::QUATERNION_X) + ", " +
           semantic(VSOutputAttributes::QUATERNION_Y) + ", " +
           semantic(VSOutputAttributes::QUATERNION_Z) + ", " +
           semantic(VSOutputAttributes::QUATERNION_W) + ");\n";
    out += "}\n\n";

    out += "void EmitVtx(Vertex vtx, bool quats_opposite) {\n";
    out += "    vec4 vtx_pos = vec4(" + semantic(VSOutputAttributes::POSITION_X) + ", " +
           semantic(VSOutputAttributes::POSITION_Y) + ", " +
           semantic(VSOutputAttributes::POSITION_Z) + ", " +
           semantic(VSOutputAttributes::POSITION_W) + ");\n";
    out += "    vtx_pos = SanitizeVertex(vtx_pos);\n";
    out += "    if (flip_viewport) {\n";
    out += "        vtx_pos.y = -vtx_pos.y;\n";
    out += "    }\n";
    out += PositionOutput;
    if (extra.use_clip_planes) {
        out += "    gl_ClipDistance[0] = -vtx_pos.z;\n"; // fixed PICA clipping plane z <= 0
        out += "    if (enable_clip1) {\n";
        out += "        gl_ClipDistance[1] = dot(clip_coef, vtx_pos);\n";
        out += "    } else {\n";
        out += "        gl_ClipDistance[1] = 0.0;\n";
        out += "    }\n\n";
    }

    out += "    vec4 vtx_quat = GetVertexQuaternion(vtx);\n";
    out += "    normquat = mix(vtx_quat, -vtx_quat, bvec4(quats_opposite));\n\n";

    out += "    vec4 vtx_color = vec4(" + semantic(VSOutputAttributes::COLOR_R) + ", " +
           semantic(VSOutputAttributes::COLOR_G) + ", " + semantic(VSOutputAttributes::COLOR_B) +
           ", " + semantic(VSOutputAttributes::COLOR_A) + ");\n";
    out += "    primary_color = min(abs(vtx_color), vec4(1.0));\n\n";

    out += "    texcoord0 = vec2(" + semantic(VSOutputAttributes::TEXCOORD0_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD0_V) + ");\n";
    out += "    texcoord1 = vec2(" + semantic(VSOutputAttributes::TEXCOORD1_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD1_V) + ");\n\n";

    out += "    texcoord0_w = " + semantic(VSOutputAttributes::TEXCOORD0_W) + ";\n";
    out += "    view = vec3(" + semantic(VSOutputAttributes::VIEW_X) + ", " +
           semantic(VSOutputAttributes::VIEW_Y) + ", " + semantic(VSOutputAttributes::VIEW_Z) +
           ");\n\n";

    out += "    texcoord2 = vec2(" + semantic(VSOutputAttributes::TEXCOORD2_U) + ", " +
           semantic(VSOutputAttributes::TEXCOORD2_V) + ");\n\n";

    out += "    EmitVertex();\n";
    out += "}\n";

    out += R"(
bool AreQuaternionsOpposite(vec4 qa, vec4 qb) {
    return (dot(qa, qb) < 0.0);
}

void EmitPrim(Vertex vtx0, Vertex vtx1, Vertex vtx2) {
    EmitVtx(vtx0, false);
    EmitVtx(vtx1, AreQuaternionsOpposite(GetVertexQuaternion(vtx0), GetVertexQuaternion(vtx1)));
    EmitVtx(vtx2, AreQuaternionsOpposite(GetVertexQuaternion(vtx0), GetVertexQuaternion(vtx2)));
    EndPrimitive();
}
)";

    return out;
};

std::string GenerateFixedGeometryShader(const PicaFixedGSConfig& config,
                                        const ExtraFixedGSConfig& extra) {
    const bool separable_shader = extra.separable_shader;

    std::string out;
    if (separable_shader) {
        out += "#extension GL_ARB_separate_shader_objects : enable\n";
    }

    out += R"(
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

)";

    out += GetGSCommonSource(config.state, extra);

    out += R"(
void main() {
    Vertex prim_buffer[3];
)";
    for (u32 vtx = 0; vtx < 3; ++vtx) {
        out += fmt::format("    prim_buffer[{}].attributes = vec4[{}](", vtx,
                           config.state.gs_output_attributes_count);
        for (u32 i = 0; i < config.state.vs_output_attributes_count; ++i) {
            out += fmt::format("{}vs_out_attr{}[{}]", i == 0 ? "" : ", ", i, vtx);
        }
        out += ");\n";
    }
    out += "    EmitPrim(prim_buffer[0], prim_buffer[1], prim_buffer[2]);\n";
    out += "}\n";

    return out;
}
} // namespace Pica::Shader::Generator::GLSL
