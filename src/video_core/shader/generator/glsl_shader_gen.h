// Copyright 2023-2026 Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include "video_core/shader/generator/shader_gen.h"
#include "video_core/shader/generator/shader_uniforms.h"

// High precision may or may not be supported in GLES3. If it isn't, use medium precision instead.
static constexpr char fragment_shader_precision_OES[] = R"(
#if GL_ES
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp int;
precision highp float;
precision highp samplerBuffer;
precision highp uimage2D;
#else
precision mediump int;
precision mediump float;
precision mediump samplerBuffer;
precision mediump uimage2D;
#endif // GL_FRAGMENT_PRECISION_HIGH
#endif
)";

namespace Pica {
struct ShaderSetup;
}

namespace Pica::Shader::Generator {
struct PicaVSConfig;
struct ExtraVSConfig;
struct PicaFixedGSConfig;
struct ExtraFixedGSConfig;
} // namespace Pica::Shader::Generator

namespace Pica::Shader::Generator::GLSL {

/**
 * Generates the GLSL vertex shader program source code that accepts vertices from software shader
 * and directly passes them to the fragment shader.
 * @returns String of the shader source code
 */
std::string GenerateTrivialVertexShader(bool use_clip_planes, bool separable_shader);

/// Generates the vertex shader that runs a PICA vertex program from a uniform block
std::string GenerateVertexInterpreter(bool use_clip_planes, bool sanitize_mul);

/// Fills the vertex interpreter part of the vs uniform block
void PackVertexInterpreterConfig(const PicaVSConfig& config, VSUniformData& data);

/**
 * Generates the GLSL vertex shader program source code for the given VS program
 * @returns String of the shader source code; empty on failure
 */
std::string GenerateVertexShader(const Pica::ShaderSetup& setup, const PicaVSConfig& config,
                                 const ExtraVSConfig& extra);

/**
 * Generates the GLSL fixed geometry shader program source code for non-GS PICA pipeline
 * @returns String of the shader source code
 */
std::string GenerateFixedGeometryShader(const PicaFixedGSConfig& config,
                                        const ExtraFixedGSConfig& extra_config);

} // namespace Pica::Shader::Generator::GLSL
