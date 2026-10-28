cmake_minimum_required(VERSION 3.24)

project(quattro
    VERSION 0.1.0
    DESCRIPTION "Native INT8 inference research on AMD RDNA 3 (gfx1101)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# All executables land in build/bin regardless of configuration or preset.
# (VS generators append a per-config subfolder by default; pin it flat.)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin)
foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_config} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()

# Experiment CLI — the single primary executable (docs/architecture.md §6).
add_executable(rdna3-int8
    src/main.cpp
    src/core/numerics.cpp
    src/core/tensor.cpp
    src/gpu/device.cpp
    src/gpu/compute.cpp
    src/gpu/amdext.cpp
    src/gpu/matmul.cpp
    src/gpu/sequence.cpp
    src/gpu/graph.cpp
    src/gpu/tensor_ops.cpp
    src/reference/reference.cpp
    src/reference/fixed_network.cpp
    src/reference/tensor_ops.cpp
    src/experiments/tensor_ops.cpp
    src/experiments/fixed_network_reference.cpp
    src/experiments/fixed_network_inference.cpp
    src/experiments/fsr4_oracle.cpp
    src/experiments/experiments.cpp)

target_link_libraries(rdna3-int8 PRIVATE d3d12.lib dxgi.lib)

if(MSVC)
    target_compile_options(rdna3-int8 PRIVATE /W4)
endif()
target_include_directories(rdna3-int8 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)

# --- Offline HLSL compilation (DXC from the Windows SDK). ---
# AGENTS.md documents the toolchain; override with -DDXC_EXECUTABLE=... if it moves.
set(DXC_EXECUTABLE
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"
    CACHE FILEPATH "Path to dxc.exe used for offline HLSL compilation")
if(NOT EXISTS "${DXC_EXECUTABLE}")
    message(FATAL_ERROR "dxc.exe not found at DXC_EXECUTABLE='${DXC_EXECUTABLE}'")
endif()

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders)
# Vendored AMD AmdExtD3D headers live here; shaders include them as "AmdExtD3D/...".
set(SHADER_INC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/amd)
set(FSR4_SHADER_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)

add_custom_command(
    OUTPUT ${FSR4_SHADER_DIR}/vector_add.cso
    COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_0 -E main
            -Fo ${FSR4_SHADER_DIR}/vector_add.cso
            ${SHADER_SRC}/vector_add.hlsl
    DEPENDS ${SHADER_SRC}/vector_add.hlsl
    COMMENT "dxc: vector_add.hlsl -> shaders/vector_add.cso")

# M06: INT8 WMMA probe (AMD intrinsics, cs_6_6 + native 16-bit types for the
# packed-type headers) — one CSO per signedness mode — and the DP4A control
# (same flags, so both probes run through the same compiler path).
foreach(_mode i8 u8 iu8 ui8)
    add_custom_command(
        OUTPUT ${FSR4_SHADER_DIR}/int8_wmma_${_mode}.cso
        COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -enable-16bit-types -E main_${_mode} -I "${SHADER_INC}"
                -Fo ${FSR4_SHADER_DIR}/int8_wmma_${_mode}.cso
                ${SHADER_SRC}/int8_wmma.hlsl
        DEPENDS ${SHADER_SRC}/int8_wmma.hlsl
        COMMENT "dxc: int8_wmma.hlsl(${_mode}) -> shaders/int8_wmma_${_mode}.cso")
endforeach()
add_custom_command(
    OUTPUT ${FSR4_SHADER_DIR}/int8_dp4a.cso
    COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -enable-16bit-types -E main -I "${SHADER_INC}"
            -Fo ${FSR4_SHADER_DIR}/int8_dp4a.cso
            ${SHADER_SRC}/int8_dp4a.hlsl
    DEPENDS ${SHADER_SRC}/int8_dp4a.hlsl
    COMMENT "dxc: int8_dp4a.hlsl -> shaders/int8_dp4a.cso")

# M07: FP16 WMMA tile probes and the multi-tile GEMM comparison kernels.
# The GEMM kernels are fixed graphs (FSR4/ML2Code codegen style): K and the
# GEMM count are compile-time -D defines, one CSO per (kernel, K) pair.
foreach(_mode f32acc f16acc)
    add_custom_command(
        OUTPUT ${FSR4_SHADER_DIR}/fp16_wmma_${_mode}.cso
        COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -enable-16bit-types -E main_${_mode} -I "${SHADER_INC}"
                -Fo ${FSR4_SHADER_DIR}/fp16_wmma_${_mode}.cso
                ${SHADER_SRC}/fp16_wmma.hlsl
        DEPENDS ${SHADER_SRC}/fp16_wmma.hlsl
        COMMENT "dxc: fp16_wmma.hlsl(${_mode}) -> shaders/fp16_wmma_${_mode}.cso")
endforeach()
foreach(_k64_256 4 16)
    if(_k64_256 STREQUAL "4")
        foreach(_entry gemm_int8_wmma gemm_fp16_wmma gemm_int8_dp4a)
            add_custom_command(
                OUTPUT ${FSR4_SHADER_DIR}/${_entry}_k64.cso
                COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -enable-16bit-types -E ${_entry}
                        -D GEMM_K_TILES=4 -I "${SHADER_INC}"
                        -Fo ${FSR4_SHADER_DIR}/${_entry}_k64.cso
                        ${SHADER_SRC}/gemm_compare.hlsl
                DEPENDS ${SHADER_SRC}/gemm_compare.hlsl
                COMMENT "dxc: gemm_compare.hlsl(${_entry}, K=64) -> shaders/${_entry}_k64.cso")
        endforeach()
    else()
        foreach(_entry gemm_int8_wmma gemm_fp16_wmma gemm_int8_dp4a)
            add_custom_command(
                OUTPUT ${FSR4_SHADER_DIR}/${_entry}_k256.cso
                COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -enable-16bit-types -E ${_entry}
                        -D GEMM_K_TILES=16 -I "${SHADER_INC}"
                        -Fo ${FSR4_SHADER_DIR}/${_entry}_k256.cso
                        ${SHADER_SRC}/gemm_compare.hlsl
                DEPENDS ${SHADER_SRC}/gemm_compare.hlsl
                COMMENT "dxc: gemm_compare.hlsl(${_entry}, K=256) -> shaders/${_entry}_k256.cso")
        endforeach()
    endif()
endforeach()

# M08: AMD's shipped QuantizeLinear operator, one CSO per baked
# quantizationScale (ML2Code bakes scales into generated passes the same way;
# the FSR4-real scale 0.014884727 comes from the v07 pre.hlsl downscale conv).
# Compile flags match the shipped model header: -O3 -enable-16bit-types -HV 2021.
set(M08_SCALES s1:1.0f s2p:2.0f spow2:0.0078125f sfsr:0.014884727f)
foreach(_pair ${M08_SCALES})
    string(REPLACE ":" ";" _pairList ${_pair})
    list(GET _pairList 0 _suf)
    list(GET _pairList 1 _scale)
    add_custom_command(
        OUTPUT ${FSR4_SHADER_DIR}/quantize_${_suf}.cso
        COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                -E quantize_amd -D "PROBE_SCALE=${_scale}" -I "${SHADER_INC}"
                -Fo ${FSR4_SHADER_DIR}/quantize_${_suf}.cso
                ${SHADER_SRC}/quantize_probe.hlsl
        DEPENDS ${SHADER_SRC}/quantize_probe.hlsl
                ${SHADER_SRC}/../amd/ml2code_runtime/config.hlsli
                ${SHADER_SRC}/../amd/ml2code_runtime/globals.hlsli
                ${SHADER_SRC}/../amd/ml2code_runtime/storage.hlsli
                ${SHADER_SRC}/../amd/ml2code_runtime/tensor_int8.hlsli
                ${SHADER_SRC}/../amd/ml2code_runtime/tensor_float16.hlsli
                ${SHADER_SRC}/../amd/ml2code_runtime/tensor_float32.hlsli
                ${SHADER_SRC}/../amd/ml2code_runtime/operators/int8_NHWC/QuantizeLinear.hlsli
        COMMENT "dxc: quantize_probe.hlsl(scale=${_scale}) -> shaders/quantize_${_suf}.cso")
endforeach()

# M09: generalized INT8 GEMM (per-shape fixed-graph CSOs: K tiles + padded N
# are compile-time because the magic-code immediates bake the strides) and
# the requant pass (runtime, no intrinsics). Blob offsets are baked to match
# the experiment's layouts (A@0 | B@Mp*K | bias | zeroTile / acc@0 | bias).
# raw: KT, N, B_OFF=Mp*K, ZERO_OFF=Mp*K+K*Np(+Np*2 when a bias section exists)
foreach(_pair raw_m16k64n16:4:16:1024:2048 raw_m32k64n32:4:32:2048:4096 raw_m16k256n16:16:16:4096:8192 raw_m64k144n16:9:16:9216:11552 raw_m64k144n32:9:32:9216:13888 raw_m256k64n256:4:256:16384:32768 raw_m256k256n256:16:256:65536:131072)
    string(REPLACE ":" ";" _pairList ${_pair})
    list(GET _pairList 0 _suf)
    list(GET _pairList 1 _tiles)
    list(GET _pairList 2 _n)
    list(GET _pairList 3 _boff)
    list(GET _pairList 4 _zoff)
    add_custom_command(
        OUTPUT ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
        COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                -E gemm_raw -D "GEMM_K_TILES=${_tiles}" -D "GEMM_N=${_n}"
                -D "GEMM_B_OFF=${_boff}" -D "GEMM_ZERO_OFF=${_zoff}" -I "${SHADER_INC}"
                -Fo ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
                ${SHADER_SRC}/gemm_i8.hlsl
        DEPENDS ${SHADER_SRC}/gemm_i8.hlsl
                ${SHADER_SRC}/../amd/AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl
        COMMENT "dxc: gemm_i8.hlsl(${_suf}) -> shaders/gemm_${_suf}.cso")
endforeach()
# requant: N (padded), N_REAL, ACC_OFF=0, BIAS_OFF=Mp*Np*4
foreach(_pair requant_m32n24:32:24:4096 requant_m64n16:16:16:4096 requant_m64n20:32:20:8192)
    string(REPLACE ":" ";" _pairList ${_pair})
    list(GET _pairList 0 _suf)
    list(GET _pairList 1 _n)
    list(GET _pairList 2 _nreal)
    list(GET _pairList 3 _biasoff)
    add_custom_command(
        OUTPUT ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
        COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                -E gemm_requant -D "GEMM_N=${_n}" -D "GEMM_N_REAL=${_nreal}"
                -D "GEMM_ACC_OFF=0" -D "GEMM_BIAS_OFF=${_biasoff}" -I "${SHADER_INC}"
                -Fo ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
                ${SHADER_SRC}/gemm_i8.hlsl
        DEPENDS ${SHADER_SRC}/gemm_i8.hlsl
        COMMENT "dxc: gemm_i8.hlsl(${_suf}) -> shaders/gemm_${_suf}.cso")
endforeach()

add_custom_command(
    OUTPUT ${FSR4_SHADER_DIR}/tensor_ops.cso
    COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -HV 2021 -denorm preserve -E main
            -Fo ${FSR4_SHADER_DIR}/tensor_ops.cso ${SHADER_SRC}/tensor_ops.hlsl
    DEPENDS ${SHADER_SRC}/tensor_ops.hlsl
    COMMENT "dxc: tensor_ops.hlsl -> shaders/tensor_ops.cso")

# M11: fixed-network kernels. One 16x16x16 WMMA GEMM tile serves both layers;
# requant variants differ only in the baked ReLU clamp. Scales are exact
# powers of two (sIn=1, sW=1, sOut=16) so requantization is exact integer
# division by 16 — see EXP-009.
foreach(_pair raw_m16k16n16:1:16:256:512 requant_m16n16:0 requant_m16n16_relu:1)
    string(REPLACE ":" ";" _pairList ${_pair})
    list(GET _pairList 0 _suf)
    list(GET _pairList 1 _tilesOrRelu)
    if(_suf MATCHES "^raw_")
        add_custom_command(
            OUTPUT ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
            COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                    -E gemm_raw -D "GEMM_K_TILES=${_tilesOrRelu}" -D "GEMM_N=16"
                    -D "GEMM_B_OFF=256" -D "GEMM_ZERO_OFF=512" -I "${SHADER_INC}"
                    -Fo ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
                    ${SHADER_SRC}/gemm_i8.hlsl
            DEPENDS ${SHADER_SRC}/gemm_i8.hlsl
                    ${SHADER_SRC}/../amd/AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl
            COMMENT "dxc: gemm_i8.hlsl(${_suf}) -> shaders/gemm_${_suf}.cso")
    else()
        add_custom_command(
            OUTPUT ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
            COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                    -E gemm_requant -D "GEMM_RELU=${_tilesOrRelu}" -D "GEMM_N=16"
                    -D "GEMM_N_REAL=16" -D "GEMM_ACC_OFF=0" -D "GEMM_BIAS_OFF=1024"
                    -D "GEMM_SIN=1.0f" -D "GEMM_SW=1.0f" -D "GEMM_SOUT=16.0f" -I "${SHADER_INC}"
                    -Fo ${FSR4_SHADER_DIR}/gemm_${_suf}.cso
                    ${SHADER_SRC}/gemm_i8.hlsl
            DEPENDS ${SHADER_SRC}/gemm_i8.hlsl
            COMMENT "dxc: gemm_i8.hlsl(${_suf}) -> shaders/gemm_${_suf}.cso")
    endif()
endforeach()

# --- M12 reference oracle: the pinned FSR 4 v07 INT8 model passes. ---
# Compiled from the read-only research tree (never modified) into build/ so the
# reproduction has a same-version numeric reference. Optional: skipped cleanly
# when the reference submodule is not checked out.
set(_M12_PINNED_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/.references/sources/FSR-4.0.2-reference/Kits/FidelityFX/upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality")
if(DEFINED FSR4_REFERENCE_MODEL AND NOT FSR4_REFERENCE_MODEL STREQUAL _M12_PINNED_MODEL)
    message(FATAL_ERROR "M12 supports only the pinned reference path; custom FSR4_REFERENCE_MODEL is unsupported")
endif()
set(FSR4_REFERENCE_MODEL "${_M12_PINNED_MODEL}")
set(FSR4_REFERENCE_PASSES "${FSR4_REFERENCE_MODEL}/passes_1080.hlsl")
set(FSR4_REFERENCE_INCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/.references/sources/FSR-4.0.2-reference/Kits/FidelityFX/upscalers/fsr4/dx12")
set(M12REF_SHADER_DIR ${FSR4_SHADER_DIR}/m12ref)
set(M12_REF_CSOS)
if(EXISTS "${FSR4_REFERENCE_PASSES}")
    file(GLOB_RECURSE M12_REFERENCE_HEADERS CONFIGURE_DEPENDS
        "${FSR4_REFERENCE_INCLUDE}/ml2code_runtime/*.hlsli"
        "${FSR4_REFERENCE_INCLUDE}/AmdExtD3D/*.hlsl")
    foreach(_pass RANGE 0 13)
        add_custom_command(
            OUTPUT ${M12REF_SHADER_DIR}/pass${_pass}.cso
            COMMAND ${CMAKE_COMMAND} -E make_directory ${M12REF_SHADER_DIR}
            COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                    -E fsr4_model_v07_i8_pass${_pass} -D "MLSR_PASS_${_pass}=1"
                    -I "${FSR4_REFERENCE_INCLUDE}"
                    -Fo ${M12REF_SHADER_DIR}/pass${_pass}.cso
                    ${FSR4_REFERENCE_PASSES}
            DEPENDS ${FSR4_REFERENCE_PASSES} ${M12_REFERENCE_HEADERS}
            COMMENT "dxc: reference fsr4_model_v07_i8_pass${_pass} -> shaders/m12ref/pass${_pass}.cso")
        list(APPEND M12_REF_CSOS ${M12REF_SHADER_DIR}/pass${_pass}.cso)
    endforeach()

    # M12 fixed model data generation is offline only. Reuse the existing
    # compile_shaders target rather than introducing another build target.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(M12_DATA_DIR "${PROJECT_SOURCE_DIR}/build/reports/m12")
    set(M12_CONTROL_HEADER "${M12_DATA_DIR}/control/include/ml2code_runtime/operators/int8_NHWC/Fused/FNB_CT2D_ADD.hlsli")
    add_custom_command(
        OUTPUT ${FSR4_SHADER_DIR}/m12control/pass11.cso ${M12_CONTROL_HEADER}
               ${M12_DATA_DIR}/control/manifest.json
        COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/scripts/m12_reference_control.py
        COMMAND ${CMAKE_COMMAND} -E make_directory ${FSR4_SHADER_DIR}/m12control
        COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                -E fsr4_model_v07_i8_pass11 -D MLSR_PASS_11=1
                -I "${M12_DATA_DIR}/control/include" -I "${FSR4_REFERENCE_INCLUDE}"
                -Fo ${FSR4_SHADER_DIR}/m12control/pass11.cso ${FSR4_REFERENCE_PASSES}
        DEPENDS ${PROJECT_SOURCE_DIR}/scripts/m12_reference_control.py
                ${FSR4_REFERENCE_PASSES} ${M12_REFERENCE_HEADERS}
        COMMENT "M12: compile labelled pass11 predication causal control"
        VERBATIM)
    list(APPEND M12_REF_CSOS ${FSR4_SHADER_DIR}/m12control/pass11.cso)
    set(M12_GENERATED
        ${M12_DATA_DIR}/graph.json ${M12_DATA_DIR}/graph.md
        ${M12_DATA_DIR}/weights.bin ${M12_DATA_DIR}/weights.json
        ${M12_DATA_DIR}/weights.generated.h ${M12_DATA_DIR}/weights.generated.hlsli)
    add_custom_command(
        OUTPUT ${M12_GENERATED}
        COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/scripts/m12_extract_graph.py
        COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/scripts/m12_pack_model.py
        DEPENDS ${FSR4_REFERENCE_PASSES} ${FSR4_REFERENCE_MODEL}/initializers.bin
                ${PROJECT_SOURCE_DIR}/scripts/m12_extract_graph.py
                ${PROJECT_SOURCE_DIR}/scripts/m12_pack_model.py
        COMMENT "M12: extract fixed graph and pack checked model constants"
        VERBATIM)
    add_custom_command(
        OUTPUT ${FSR4_SHADER_DIR}/m12/pass0.cso
        COMMAND ${CMAKE_COMMAND} -E make_directory ${FSR4_SHADER_DIR}/m12
        COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -O3 -enable-16bit-types -HV 2021
                -E main -I "${M12_DATA_DIR}"
                -Fo ${FSR4_SHADER_DIR}/m12/pass0.cso
                ${PROJECT_SOURCE_DIR}/src/inference/models/shaders/fsr4_pass0.hlsl
        DEPENDS ${PROJECT_SOURCE_DIR}/src/inference/models/shaders/fsr4_pass0.hlsl
                ${M12_GENERATED}
        COMMENT "dxc: our FSR4 input convolution -> shaders/m12/pass0.cso"
        VERBATIM)
    list(APPEND M12_REF_CSOS ${M12_GENERATED} ${FSR4_SHADER_DIR}/m12/pass0.cso)
    foreach(_conv RANGE 0 2)
        if(_conv EQUAL 0)
            set(_k 144)
            set(_c 16)
            set(_n 16)
        elseif(_conv EQUAL 1)
            set(_k 16)
            set(_c 16)
            set(_n 32)
        else()
            set(_k 32)
            set(_c 32)
            set(_n 16)
        endif()
        math(EXPR _kt "${_k}/16")
        math(EXPR _b "518400*${_k}")
        math(EXPR _zero "${_b}+${_k}*${_n}")
        add_custom_command(
            OUTPUT ${FSR4_SHADER_DIR}/m12/cnb${_conv}raw.cso
            COMMAND ${CMAKE_COMMAND} -E make_directory ${FSR4_SHADER_DIR}/m12
            COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -E gemm_raw -O3
                    -enable-16bit-types -HV 2021 -I "${SHADER_INC}"
                    -D GEMM_K_TILES=${_kt} -D GEMM_M=518400 -D GEMM_N=${_n}
                    -D GEMM_B_OFF=${_b} -D GEMM_ZERO_OFF=${_zero}
                    -Fo ${FSR4_SHADER_DIR}/m12/cnb${_conv}raw.cso ${SHADER_SRC}/gemm_i8.hlsl
            DEPENDS ${SHADER_SRC}/gemm_i8.hlsl
                    ${SHADER_INC}/AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl
            COMMENT "dxc: native WMMA ConvNext convolution ${_conv}"
            VERBATIM)
        list(APPEND M12_REF_CSOS ${FSR4_SHADER_DIR}/m12/cnb${_conv}raw.cso)
        foreach(_model_pass 1 2 12)
            foreach(_step pack post)
                add_custom_command(
                    OUTPUT ${FSR4_SHADER_DIR}/m12/p${_model_pass}c${_conv}${_step}.cso
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${FSR4_SHADER_DIR}/m12
                    COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -E ${_step} -O3
                            -enable-16bit-types -HV 2021 -I "${M12_DATA_DIR}" -I "${SHADER_SRC}"
                            -D CNB_PASS=${_model_pass} -D CNB_STAGE=${_conv}
                            -D CONV_M=518400 -D CONV_K=${_k} -D CONV_C=${_c} -D CONV_N=${_n}
                            -Fo ${FSR4_SHADER_DIR}/m12/p${_model_pass}c${_conv}${_step}.cso
                            ${PROJECT_SOURCE_DIR}/src/inference/models/shaders/fsr4_cnb.hlsl
                    DEPENDS ${PROJECT_SOURCE_DIR}/src/inference/models/shaders/fsr4_cnb.hlsl
                            ${SHADER_SRC}/conv_i8.hlsli ${M12_GENERATED}
                    COMMENT "dxc: our pass${_model_pass} ConvNext conv${_conv} ${_step}"
                    VERBATIM)
                list(APPEND M12_REF_CSOS ${FSR4_SHADER_DIR}/m12/p${_model_pass}c${_conv}${_step}.cso)
            endforeach()
        endforeach()
    endforeach()

    # Passes 3 and 6: quantized stride-2 downscale convolution. Three fixed
    # dispatches per pass (im2col/weight pack -> native WMMA GEMM -> two-group
    # requantization), all shapes baked per CSO like the reference codegen.
    foreach(_ds_pass 3 6)
        if(_ds_pass EQUAL 3)
            set(_ds_pixels 129600)
            set(_ds_features 32)
            set(_ds_reduction 64)
        else()
            set(_ds_pixels 32400)
            set(_ds_features 64)
            set(_ds_reduction 128)
        endif()
        math(EXPR _ds_kt "${_ds_reduction}/16")
        math(EXPR _ds_b "${_ds_pixels}*${_ds_reduction}")
        math(EXPR _ds_zero "${_ds_b}+${_ds_reduction}*${_ds_features}")
        add_custom_command(
            OUTPUT ${FSR4_SHADER_DIR}/m12/ds${_ds_pass}raw.cso
            COMMAND ${CMAKE_COMMAND} -E make_directory ${FSR4_SHADER_DIR}/m12
            COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -E gemm_raw -O3
                    -enable-16bit-types -HV 2021 -I "${SHADER_INC}"
                    -D GEMM_K_TILES=${_ds_kt} -D GEMM_M=${_ds_pixels} -D GEMM_N=${_ds_features}
                    -D GEMM_B_OFF=${_ds_b} -D GEMM_ZERO_OFF=${_ds_zero}
                    -Fo ${FSR4_SHADER_DIR}/m12/ds${_ds_pass}raw.cso ${SHADER_SRC}/gemm_i8.hlsl
            DEPENDS ${SHADER_SRC}/gemm_i8.hlsl
                    ${SHADER_INC}/AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl
            COMMENT "dxc: native WMMA downscale convolution pass ${_ds_pass}"
            VERBATIM)
        list(APPEND M12_REF_CSOS ${FSR4_SHADER_DIR}/m12/ds${_ds_pass}raw.cso)
        foreach(_step pack post)
            add_custom_command(
                OUTPUT ${FSR4_SHADER_DIR}/m12/ds${_ds_pass}${_step}.cso
                COMMAND ${CMAKE_COMMAND} -E make_directory ${FSR4_SHADER_DIR}/m12
                COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_6 -E ${_step} -O3
                        -enable-16bit-types -HV 2021 -I "${M12_DATA_DIR}" -I "${SHADER_SRC}"
                        -D DS_PASS=${_ds_pass}
                        -Fo ${FSR4_SHADER_DIR}/m12/ds${_ds_pass}${_step}.cso
                        ${PROJECT_SOURCE_DIR}/src/inference/models/shaders/fsr4_downscale.hlsl
                DEPENDS ${PROJECT_SOURCE_DIR}/src/inference/models/shaders/fsr4_downscale.hlsl
                        ${SHADER_SRC}/conv_i8.hlsli ${M12_GENERATED}
                COMMENT "dxc: our downscale pass${_ds_pass} ${_step}"
                VERBATIM)
            list(APPEND M12_REF_CSOS ${FSR4_SHADER_DIR}/m12/ds${_ds_pass}${_step}.cso)
        endforeach()
    endforeach()
    target_sources(rdna3-int8 PRIVATE
        src/inference/models/fsr4_weights.cpp
        src/inference/models/fsr4_pass0.cpp
        src/inference/models/fsr4_cnb.cpp
        src/inference/models/fsr4_downscale.cpp
        src/experiments/fsr4_reproduction.cpp
        ${M12_DATA_DIR}/weights.generated.h)
    target_include_directories(rdna3-int8 PRIVATE ${M12_DATA_DIR})
    target_compile_definitions(rdna3-int8 PRIVATE QUATTRO_M12_MODEL_AVAILABLE=1)
else()
    message(STATUS "M12 reference oracle: FSR4_REFERENCE_MODEL not found; run fsr4-oracle will be unavailable")
endif()

add_custom_target(compile_shaders DEPENDS
    ${FSR4_SHADER_DIR}/tensor_ops.cso
    ${FSR4_SHADER_DIR}/vector_add.cso
    ${FSR4_SHADER_DIR}/int8_wmma_i8.cso
    ${FSR4_SHADER_DIR}/int8_wmma_u8.cso
    ${FSR4_SHADER_DIR}/int8_wmma_iu8.cso
    ${FSR4_SHADER_DIR}/int8_wmma_ui8.cso
    ${FSR4_SHADER_DIR}/int8_dp4a.cso
    ${FSR4_SHADER_DIR}/fp16_wmma_f32acc.cso
    ${FSR4_SHADER_DIR}/fp16_wmma_f16acc.cso
    ${FSR4_SHADER_DIR}/gemm_int8_wmma_k64.cso
    ${FSR4_SHADER_DIR}/gemm_fp16_wmma_k64.cso
    ${FSR4_SHADER_DIR}/gemm_int8_dp4a_k64.cso
    ${FSR4_SHADER_DIR}/gemm_int8_wmma_k256.cso
    ${FSR4_SHADER_DIR}/gemm_fp16_wmma_k256.cso
    ${FSR4_SHADER_DIR}/gemm_int8_dp4a_k256.cso
    ${FSR4_SHADER_DIR}/quantize_s1.cso
    ${FSR4_SHADER_DIR}/quantize_s2p.cso
    ${FSR4_SHADER_DIR}/quantize_spow2.cso
    ${FSR4_SHADER_DIR}/quantize_sfsr.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m16k64n16.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m32k64n32.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m16k256n16.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m64k144n16.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m64k144n32.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m256k64n256.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m256k256n256.cso
    ${FSR4_SHADER_DIR}/gemm_requant_m32n24.cso
    ${FSR4_SHADER_DIR}/gemm_requant_m64n16.cso
    ${FSR4_SHADER_DIR}/gemm_requant_m64n20.cso
    ${FSR4_SHADER_DIR}/gemm_raw_m16k16n16.cso
    ${FSR4_SHADER_DIR}/gemm_requant_m16n16.cso
    ${FSR4_SHADER_DIR}/gemm_requant_m16n16_relu.cso
    ${M12_REF_CSOS})
add_dependencies(rdna3-int8 compile_shaders)

enable_testing()

# M01 smoke tests: exercise the CLI's real contract (exit codes) against the
# built binary. No test sources exist yet; add targets when real code needs them.
add_test(NAME cli_list COMMAND rdna3-int8 list)
add_test(NAME cli_help COMMAND rdna3-int8 help)
add_test(NAME cli_run_unregistered COMMAND rdna3-int8 run not-registered)
set_tests_properties(cli_run_unregistered PROPERTIES WILL_FAIL TRUE)

# M05 hardware test: the compute harness experiment must pass on this machine.
add_test(NAME experiment_d3d12_compute COMMAND rdna3-int8 run d3d12-compute)

# M06 hardware test: INT8 WMMA probes must pass on this machine.
add_test(NAME experiment_wmma_int8 COMMAND rdna3-int8 run wmma-int8)

# M07 hardware test: FP16 WMMA + comparison must pass on this machine.
add_test(NAME experiment_wmma_fp16 COMMAND rdna3-int8 run wmma-fp16)

# M08 hardware test: quantization contract (incl. AMD's shipped QuantizeLinear
# operator on this GPU) must pass.
add_test(NAME experiment_quantization COMMAND rdna3-int8 run quantization)

# M09 hardware test: INT8 GEMM/requant/conv kernels must pass on this machine.
add_test(NAME experiment_kernels_int8 COMMAND rdna3-int8 run kernels-int8)
add_test(NAME experiment_tensor_ops COMMAND rdna3-int8 run tensor-ops)

# M11 preliminary CPU oracle validation; this is not the inference gate.
add_test(NAME experiment_m11_reference COMMAND rdna3-int8 run m11-reference)

# M11 hardware gate: the fixed two-layer INT8 network must pass end-to-end.
add_test(NAME experiment_fixed_network COMMAND rdna3-int8 run fixed-network)

# M12 reference oracle needs the pinned research tree; skip cleanly without it.
if(EXISTS "${FSR4_REFERENCE_PASSES}")
    # Reference smoke checks finiteness/nonzero only, not numeric equivalence.
    add_test(NAME experiment_fsr4_oracle COMMAND rdna3-int8 run fsr4-oracle)
    add_test(NAME experiment_fsr4_weights COMMAND rdna3-int8 run fsr4-weights)
    add_test(NAME experiment_fsr4_pass0 COMMAND rdna3-int8 run fsr4-pass0)
    add_test(NAME experiment_fsr4_cnb COMMAND rdna3-int8 run fsr4-cnb)
    add_test(NAME experiment_fsr4_downscale COMMAND rdna3-int8 run fsr4-downscale)
    add_test(NAME m12_model_packing COMMAND ${Python3_EXECUTABLE} -B
        ${PROJECT_SOURCE_DIR}/scripts/m12_test_pack.py)
    add_test(NAME m12_tensor_metadata COMMAND ${Python3_EXECUTABLE} -B
        ${PROJECT_SOURCE_DIR}/scripts/m12_validate_graph.py)
    add_test(NAME m12_capture_integrity COMMAND ${Python3_EXECUTABLE} -B
        ${PROJECT_SOURCE_DIR}/scripts/m12_test_capture.py)
endif()
