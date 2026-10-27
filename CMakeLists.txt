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
    src/gpu/device.cpp
    src/gpu/compute.cpp
    src/reference/reference.cpp
    src/experiments/experiments.cpp)

target_link_libraries(rdna3-int8 PRIVATE d3d12.lib dxgi.lib)

if(MSVC)
    target_compile_options(rdna3-int8 PRIVATE /W4)
endif()
target_include_directories(rdna3-int8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Offline HLSL compilation (DXC from the Windows SDK). ---
# AGENTS.md documents the toolchain; override with -DDXC_EXECUTABLE=... if it moves.
set(DXC_EXECUTABLE
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"
    CACHE FILEPATH "Path to dxc.exe used for offline HLSL compilation")
if(NOT EXISTS "${DXC_EXECUTABLE}")
    message(FATAL_ERROR "dxc.exe not found at DXC_EXECUTABLE='${DXC_EXECUTABLE}'")
endif()

set(FSR4_SHADER_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders)
add_custom_command(
    OUTPUT ${FSR4_SHADER_DIR}/vector_add.cso
    COMMAND "${DXC_EXECUTABLE}" -nologo -T cs_6_0 -E main
            -Fo ${FSR4_SHADER_DIR}/vector_add.cso
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/vector_add.hlsl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/vector_add.hlsl
    COMMENT "dxc: vector_add.hlsl -> shaders/vector_add.cso")
add_custom_target(compile_shaders DEPENDS ${FSR4_SHADER_DIR}/vector_add.cso)
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
