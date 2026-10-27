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
    src/gpu/device.cpp
    src/gpu/compute.cpp
    src/gpu/amdext.cpp
    src/gpu/matmul.cpp
    src/reference/reference.cpp
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

add_custom_target(compile_shaders DEPENDS
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
    ${FSR4_SHADER_DIR}/gemm_requant_m64n20.cso)
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
