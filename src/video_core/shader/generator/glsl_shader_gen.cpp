// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#include <string_view>
#include <fmt/format.h>

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
                                                 bool separable_shader) {
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
        out += "    invariant vec4 gl_Position;\n";
#else
        out += "    vec4 gl_Position;\n";
#endif
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

std::string GenerateVertexInterpreter(bool use_clip_planes, bool sanitize_mul) {
    using OpCode = nihstro::OpCode::Id;
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
    define("OP_ADD", OpCode::ADD);
    define("OP_DP3", OpCode::DP3);
    define("OP_DP4", OpCode::DP4);
    define("OP_DPH", OpCode::DPH);
    define("OP_EX2", OpCode::EX2);
    define("OP_LG2", OpCode::LG2);
    define("OP_MUL", OpCode::MUL);
    define("OP_SGE", OpCode::SGE);
    define("OP_SLT", OpCode::SLT);
    define("OP_FLR", OpCode::FLR);
    define("OP_MAX", OpCode::MAX);
    define("OP_MIN", OpCode::MIN);
    define("OP_RCP", OpCode::RCP);
    define("OP_RSQ", OpCode::RSQ);
    define("OP_MOVA", OpCode::MOVA);
    define("OP_MOV", OpCode::MOV);
    define("OP_DPHI", OpCode::DPHI);
    define("OP_DSTI", OpCode::DSTI);
    define("OP_SGEI", OpCode::SGEI);
    define("OP_SLTI", OpCode::SLTI);
    define("OP_BREAK", OpCode::BREAK);
    define("OP_END", OpCode::END);
    define("OP_BREAKC", OpCode::BREAKC);
    define("OP_CALL", OpCode::CALL);
    define("OP_CALLC", OpCode::CALLC);
    define("OP_CALLU", OpCode::CALLU);
    define("OP_IFU", OpCode::IFU);
    define("OP_IFC", OpCode::IFC);
    define("OP_LOOP", OpCode::LOOP);
    define("OP_JMPC", OpCode::JMPC);
    define("OP_JMPU", OpCode::JMPU);
    define("OP_CMP", OpCode::CMP);
    define("OP_MADI", OpCode::MADI);
    define("OP_MAD", OpCode::MAD);
    define("CMP_EQUAL", CompareOp::Equal);
    define("CMP_NOT_EQUAL", CompareOp::NotEqual);
    define("CMP_LESS_THAN", CompareOp::LessThan);
    define("CMP_LESS_EQUAL", CompareOp::LessEqual);
    define("CMP_GREATER_THAN", CompareOp::GreaterThan);
    define("CMP_GREATER_EQUAL", CompareOp::GreaterEqual);
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
    // Regular vs uniform block plus the interpreter fields
    std::string block{VSUniformBlockDef};
    block.replace(block.find("};"), 2,
                  "    uvec4 interp_config;\n"
                  "    uvec4 interp_output_map[4];\n"
                  "    uvec4 interp_semantic_map[6];\n"
                  "    uvec4 interp_attributes[16];\n"
                  "};");
    out += block;
    out += "layout (set = 0, binding = 6, std140) uniform vs_interpreter_program {\n"
           "    uvec4 program_code[1024];\n"
           "    uvec4 swizzle_data[32];\n"
           "};\n"
           "layout (set = 0, binding = 7, std430) readonly buffer vs_vertex_data {\n"
           "    uint vertex_words[];\n"
           "};\n";
    out += GetVertexInterfaceDeclaration(true, use_clip_planes, true);

    std::string body = R"glsl(
vec4 reg_in[16];
vec4 reg_tmp[16];
vec4 reg_out[16];
bvec2 conditional_code = bvec2(false);
ivec3 address_registers = ivec3(0);

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

// Vertex data is not word aligned
uint vertexWord(uint address) {
    uint index = address >> 2;
    uint shift = (address & 3u) * 8u;
    uint low = vertex_words[index];
    if (shift == 0u) {
        return low;
    }
    return (low >> shift) | (vertex_words[index + 1u] << (32u - shift));
}

vec4 loadAttribute(uvec4 slot) {
    vec4 value = vec4(0.0, 0.0, 0.0, 1.0);
    uint address = slot.x + uint(gl_VertexIndex) * slot.y;
    for (uint i = 0u; i < slot.w; i++) {
        switch (slot.z) {
        case FORMAT_BYTE:
            value[i] = float(int(vertexWord(address + i) << 24) >> 24);
            break;
        case FORMAT_UBYTE:
            value[i] = float(vertexWord(address + i) & 0xFFu);
            break;
        case FORMAT_SHORT:
            value[i] = float(int(vertexWord(address + 2u * i) << 16) >> 16);
            break;
        default:
            value[i] = uintBitsToFloat(vertexWord(address + 4u * i));
            break;
        }
    }
    return value;
}

vec4 sourceRegister(uint reg, uint address_index) {
    if (reg < 0x10u) {
        return reg_in[reg];
    }
    if (reg < 0x20u) {
        return reg_tmp[reg - 0x10u];
    }
    uint index = reg - 0x20u;
    if (address_index != 0u) {
        int offset = address_registers[address_index - 1u];
        int fixed_offset = offset >= -128 && offset <= 127 ? offset : 0;
        index = uint((int(index) + fixed_offset) & 0x7F);
        return index < 96u ? uniforms.f[index] : vec4(1.0);
    }
    return uniforms.f[index];
}

vec4 source(uint reg, uint address_index, uint pattern, uint negate_bit, uint selector_shift) {
    vec4 value = sourceRegister(reg, address_index);
    uint selector = bits(pattern, selector_shift, 8u);
    value = vec4(value[bits(selector, 6u, 2u)], value[bits(selector, 4u, 2u)],
                 value[bits(selector, 2u, 2u)], value[bits(selector, 0u, 2u)]);
    return bits(pattern, negate_bit, 1u) != 0u ? -value : value;
}

void setDest(uint dest, uint pattern, vec4 value) {
    bvec4 mask = bvec4((pattern & 8u) != 0u, (pattern & 4u) != 0u, (pattern & 2u) != 0u,
                       (pattern & 1u) != 0u);
    if (dest < 0x10u) {
        uint mapped = interp_output_map[dest >> 2][dest & 3u];
        if (mapped < interp_config.y) {
            reg_out[mapped] = mix(reg_out[mapped], value, mask);
        }
    } else if (dest < 0x20u) {
        reg_tmp[dest - 0x10u] = mix(reg_tmp[dest - 0x10u], value, mask);
    }
}

uint swizzlePattern(uint id) {
    return swizzle_data[id >> 2][id & 3u];
}

bool evaluateCondition(uint instr) {
    bool result_x = (bits(instr, 25u, 1u) != 0u) == conditional_code.x;
    bool result_y = (bits(instr, 24u, 1u) != 0u) == conditional_code.y;
    switch (bits(instr, 22u, 2u)) {
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

void arithmetic(uint opcode, uint instr) {
    uint pattern = swizzlePattern(bits(instr, 0u, 7u));
    bool inverted = opcode == OP_DPHI || opcode == OP_DSTI || opcode == OP_SGEI || opcode == OP_SLTI;
    uint address_index = bits(instr, 19u, 2u);
    uint reg_1 = inverted ? bits(instr, 14u, 5u) : bits(instr, 12u, 7u);
    uint reg_2 = inverted ? bits(instr, 7u, 7u) : bits(instr, 7u, 5u);
    vec4 src1 = source(reg_1, inverted ? 0u : address_index, pattern, 4u, 5u);
    vec4 src2 = source(reg_2, inverted ? address_index : 0u, pattern, 13u, 14u);
    uint dest = bits(instr, 21u, 5u);

    switch (opcode) {
    case OP_ADD:
        setDest(dest, pattern, src1 + src2);
        break;
    case OP_MUL:
        setDest(dest, pattern, MUL(src1, src2));
        break;
    case OP_FLR:
        setDest(dest, pattern, floor(src1));
        break;
    case OP_MAX:
#if SANITIZE_MUL
        setDest(dest, pattern, mix(src2, src1, greaterThan(src1, src2)));
#else
        setDest(dest, pattern, max(src1, src2));
#endif
        break;
    case OP_MIN:
#if SANITIZE_MUL
        setDest(dest, pattern, mix(src2, src1, lessThan(src1, src2)));
#else
        setDest(dest, pattern, min(src1, src2));
#endif
        break;
    case OP_DP3:
#if SANITIZE_MUL
        setDest(dest, pattern, vec4(dot(vec3(sanitize_mul(src1, src2)), vec3(1.0))));
#else
        setDest(dest, pattern, vec4(dot(vec3(src1), vec3(src2))));
#endif
        break;
    case OP_DP4:
#if SANITIZE_MUL
        setDest(dest, pattern, vec4(dot(sanitize_mul(src1, src2), vec4(1.0))));
#else
        setDest(dest, pattern, vec4(dot(src1, src2)));
#endif
        break;
    case OP_DPH:
    case OP_DPHI:
#if SANITIZE_MUL
        setDest(dest, pattern, vec4(dot(sanitize_mul(vec4(src1.xyz, 1.0), src2), vec4(1.0))));
#else
        setDest(dest, pattern, vec4(dot(src1, src2)));
#endif
        break;
    case OP_RCP:
#if !SANITIZE_MUL
        if (src1.x != 0.0)
#endif
        setDest(dest, pattern, vec4(1.0 / src1.x));
        break;
    case OP_RSQ:
#if !SANITIZE_MUL
        if (src1.x > 0.0)
#endif
        setDest(dest, pattern, vec4(inversesqrt(src1.x)));
        break;
    case OP_MOVA: {
        ivec2 address = ivec2(src1);
        if ((pattern & 8u) != 0u) {
            address_registers.x = address.x;
        }
        if ((pattern & 4u) != 0u) {
            address_registers.y = address.y;
        }
        break;
    }
    case OP_MOV:
        setDest(dest, pattern, src1);
        break;
    case OP_SGE:
    case OP_SGEI:
        setDest(dest, pattern, vec4(greaterThanEqual(src1, src2)));
        break;
    case OP_SLT:
    case OP_SLTI:
        setDest(dest, pattern, vec4(lessThan(src1, src2)));
        break;
    case OP_EX2:
        setDest(dest, pattern, vec4(exp2(src1.x)));
        break;
    case OP_LG2:
        setDest(dest, pattern, vec4(log2(src1.x)));
        break;
    default:
        break;
    }
}

void compareInstruction(uint instr) {
    uint pattern = swizzlePattern(bits(instr, 0u, 7u));
    vec4 src1 = source(bits(instr, 12u, 7u), bits(instr, 19u, 2u), pattern, 4u, 5u);
    vec4 src2 = source(bits(instr, 7u, 5u), 0u, pattern, 13u, 14u);
    uint op_x = bits(instr, 24u, 3u);
    uint op_y = bits(instr, 21u, 3u);
    if (op_x <= CMP_GREATER_EQUAL && op_y <= CMP_GREATER_EQUAL) {
        conditional_code = bvec2(compare(op_x, src1.x, src2.x), compare(op_y, src1.y, src2.y));
    }
}

void multiplyAdd(uint opcode, uint instr) {
    bool inverted = opcode < OP_MAD;
    uint pattern = swizzlePattern(bits(instr, 0u, 5u));
    uint address_index = bits(instr, 22u, 2u);
    uint reg_2 = inverted ? bits(instr, 12u, 5u) : bits(instr, 10u, 7u);
    uint reg_3 = inverted ? bits(instr, 5u, 7u) : bits(instr, 5u, 5u);
    vec4 src1 = source(bits(instr, 17u, 5u), 0u, pattern, 4u, 5u);
    vec4 src2 = source(reg_2, inverted ? 0u : address_index, pattern, 13u, 14u);
    vec4 src3 = source(reg_3, inverted ? address_index : 0u, pattern, 22u, 23u);
    setDest(bits(instr, 24u, 5u), pattern, MUL(src1, src2) + src3);
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
        uint old_pc = pc;
        bool is_break = false;
        uint instr = pc < 4095u ? program_code[pc >> 2][pc & 3u] : (OP_END << 26);
        uint opcode = instr >> 26;
        uint target = bits(instr, 10u, 12u);
        uint count = bits(instr, 0u, 8u);

        if (opcode >= OP_MADI) {
            multiplyAdd(opcode, instr);
        } else if (opcode == OP_CMP || opcode == OP_CMP + 1u) {
            compareInstruction(instr);
        } else if (opcode < OP_BREAK) {
            arithmetic(opcode, instr);
        } else if (opcode == OP_END) {
            return;
        } else if (opcode == OP_JMPC) {
            if (evaluateCondition(instr)) {
                pc = target - 1u;
            }
        } else if (opcode == OP_JMPU) {
            if (boolUniform(bits(instr, 22u, 4u)) == ((count & 1u) == 0u)) {
                pc = target - 1u;
            }
        } else if (opcode == OP_CALL || opcode == OP_CALLC || opcode == OP_CALLU) {
            bool taken = opcode == OP_CALL ||
                         (opcode == OP_CALLC ? evaluateCondition(instr)
                                             : boolUniform(bits(instr, 22u, 4u)));
            if (taken) {
                if (call_top < 4) {
                    call_end[call_top] = target + count;
                    call_return[call_top] = pc + 1u;
                    call_top++;
                }
                pc = target - 1u;
            }
        } else if (opcode == OP_IFU || opcode == OP_IFC) {
            bool taken = opcode == OP_IFC ? evaluateCondition(instr)
                                          : boolUniform(bits(instr, 22u, 4u));
            if (!taken) {
                pc = target - 1u;
            } else if (if_top < 8) {
                if_else[if_top] = target;
                if_end[if_top] = target + count;
                if_top++;
            }
        } else if (opcode == OP_LOOP) {
            uvec4 loop = uniforms.i[bits(instr, 22u, 2u)];
            if (loop_top < 4) {
                loop_entry[loop_top] = pc + 1u;
                loop_end[loop_top] = target + 1u;
                loop_count[loop_top] = loop.x;
                loop_step[loop_top] = loop.z;
                loop_previous[loop_top] = int(loop.y);
                loop_top++;
            }
            address_registers.z = int(loop.y);
        } else if (opcode == OP_BREAK) {
            is_break = true;
        } else if (opcode == OP_BREAKC) {
            is_break = evaluateCondition(instr);
        }

        pc++;
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
            address_registers.z += int(loop_step[top]);
            if (!is_break && loop_count[top] != 0u) {
                loop_count[top]--;
                pc = loop_entry[top];
            } else {
                pc = loop_end[top];
                if (loop_top > 1) {
                    address_registers.z = loop_previous[top];
                }
                loop_top--;
            }
        }
    }
}

float semantic(uint slot) {
    uint map = interp_semantic_map[slot >> 2][slot & 3u];
    return map < 64u ? reg_out[map >> 2][map & 3u] : 1.0;
}

void main() {
    for (uint i = 0u; i < 16u; i++) {
        reg_in[i] = loadAttribute(interp_attributes[i]);
        reg_tmp[i] = vec4(0.0, 0.0, 0.0, 1.0);
        reg_out[i] = vec4(0.0, 0.0, 0.0, 1.0);
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
    data.interp_config = {state.main_offset, state.num_outputs, 0u, 0u};
    for (u32 reg = 0; reg < 16; reg++) {
        data.interp_output_map[reg / 4][reg % 4] = state.output_map[reg];
    }
    const auto semantic_maps = state.gs_state.GetSemanticMaps();
    for (u32 slot = 0; slot < semantic_maps.size(); slot++) {
        const auto& map = semantic_maps[slot];
        const bool mapped = map.attribute_index < state.gs_state.gs_output_attributes_count;
        data.interp_semantic_map[slot / 4][slot % 4] =
            mapped ? map.attribute_index * 4 + map.component_index : 0xFFu;
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

    auto program_source =
        DecompileProgram(setup.GetProgramCode(), setup.GetSwizzleData(), config.state.main_offset,
                         get_input_reg, get_output_reg, extra.sanitize_mul);

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
        }
        out += "void EmitVtx() {}\n";
    } else {
        out += GetVertexInterfaceDeclaration(true, extra.use_clip_planes, separable_shader);

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
