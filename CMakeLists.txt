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
    src/main.cpp)

if(MSVC)
    target_compile_options(rdna3-int8 PRIVATE /W4)
endif()

enable_testing()

# M01 smoke tests: exercise the CLI's real contract (exit codes) against the
# built binary. No test sources exist yet; add targets when real code needs them.
add_test(NAME cli_list COMMAND rdna3-int8 list)
add_test(NAME cli_help COMMAND rdna3-int8 help)
add_test(NAME cli_run_unregistered COMMAND rdna3-int8 run not-registered)
set_tests_properties(cli_run_unregistered PROPERTIES WILL_FAIL TRUE)
