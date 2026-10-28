// quattro — experiment CLI.
//
// The single primary executable for all experiments (docs/architecture.md §6).
// Experiments register explicitly in `experiments()` as their milestones
// authorize them; there is no dynamic discovery or plugin mechanism.
//
// Exit codes: 0 success · 1 experiment failure · 2 usage error.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "experiments/experiments.h"

namespace {

struct Experiment {
    const char* name;
    const char* description;
    int (*run)(); // CLI exit code: 0 pass, 1 failure
};

// Explicit registration point (docs/architecture.md §6).
// Registered when their milestone authorizes them.
const std::vector<Experiment>& experiments() {
    static const std::vector<Experiment> kExperiments = {
        {"d3d12-compute", "M05 D3D12 compute harness validation", &experiments::RunComputeHarness},
        {"wmma-int8", "M06 native INT8 WMMA probe (AMD intrinsics + DP4A control)", &experiments::RunWmmaInt8},
        {"wmma-fp16", "M07 FP16 WMMA + INT8/FP16/DP4A comparison", &experiments::RunWmmaFp16},
        {"quantization", "M08 FP8/INT8 quantization contract (incl. AMD QuantizeLinear on hw)", &experiments::RunQuantization},
        {"kernels-int8", "M09 INT8 GEMM/requant/conv kernels on native WMMA", &experiments::RunInt8Kernels},
        {"tensor-ops", "M10 neural tensor operations and precision boundaries", &experiments::RunTensorOps},
        {"m11-reference", "M11 fixed-network CPU reference checks only", &experiments::RunFixedNetworkReference},
        {"fixed-network", "M11 standalone INT8 network inference on native WMMA (end-to-end)", &experiments::RunFixedNetworkGpu},
        {"fsr4-oracle", "M12 reference oracle: pinned FSR 4 v07 INT8 model passes (comparison reference only)", &experiments::RunFsr4Oracle},
#ifdef QUATTRO_M12_MODEL_AVAILABLE
        {"fsr4-weights", "M12 checked model weight loading and corruption rejection", &experiments::RunFsr4Weights},
        {"fsr4-pass0", "M12 our input convolution vs reference (1/14 passes, not the gate)", &experiments::RunFsr4Pass0},
        {"fsr4-cnb", "M12 our native WMMA ConvNextBlocks vs isolated reference passes", &experiments::RunFsr4Cnb},
        {"fsr4-chain", "M12 GPU-resident passes 0 -> 1 vs shipped reference chain (not full graph)", &experiments::RunFsr4Chain},
        {"fsr4-graph", "M12 GPU-resident 14-pass reproduction vs predication control and shipped graph", &experiments::RunFsr4Graph},
        {"fsr4-downscale", "M12 our quantized downscale convolutions (passes 3/6) vs isolated reference passes", &experiments::RunFsr4Downscale},
        {"fsr4-fnb", "M12 our FasterNetBlocks (passes 4/5/7/8/10) vs isolated reference passes", &experiments::RunFsr4Fnb},
#endif
    };
    return kExperiments;
}

constexpr char kUsage[] =
    "quattro experiment CLI\n"
    "\n"
    "usage:\n"
    "  rdna3-int8 list          list registered experiments\n"
    "  rdna3-int8 run <name>    run one experiment\n"
    "  rdna3-int8 help          show this help\n";

int printUsage(std::FILE* out, int exitCode) {
    std::fputs(kUsage, out);
    return exitCode;
}

int listExperiments() {
    std::puts("registered experiments:");
    const std::vector<Experiment>& all = experiments();
    if (all.empty()) {
        std::puts("  (none yet — see docs/MILESTONES.md)");
        return 0;
    }
    for (const Experiment& e : all) {
        std::printf("  %-16s %s\n", e.name, e.description);
    }
    return 0;
}

int runExperiment(const char* name) {
    for (const Experiment& e : experiments()) {
        if (std::strcmp(name, e.name) == 0) {
            std::printf("running %s ...\n", e.name);
            return e.run();
        }
    }
    std::fprintf(stderr,
                 "error: no experiment registered as '%s'\n"
                 "       use 'rdna3-int8 list' to see what is registered\n",
                 name);
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fputs("error: a command is required\n\n", stderr);
        return printUsage(stderr, 2);
    }

    const std::string command = argv[1];

    if (command == "help" || command == "--help" || command == "-h") {
        return printUsage(stdout, 0);
    }
    if (command == "list") {
        return listExperiments();
    }
    if (command == "run") {
        if (argc < 3) {
            std::fputs("error: run requires an experiment name\n", stderr);
            return 2;
        }
        return runExperiment(argv[2]);
    }

    std::fprintf(stderr, "error: unknown command '%s'\n\n", argv[1]);
    return printUsage(stderr, 2);
}
