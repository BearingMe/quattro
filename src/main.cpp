// quattro — experiment CLI.
//
// The single primary executable for all experiments (docs/architecture.md §6).
// Experiments register explicitly in `experiments()` as their milestones
// authorize them; there is no dynamic discovery or plugin mechanism.
//
// Exit codes: 0 success · 1 experiment failure · 2 usage error.

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Experiment {
    const char* name;
    const char* description;
};

// Explicit registration point (docs/architecture.md §6).
// No experiments exist yet — M01 bootstrap only.
const std::vector<Experiment>& experiments() {
    static const std::vector<Experiment> kExperiments = {
        // Register experiments here, e.g.:
        // {"wmma-int8", "Native INT8 WMMA correctness (M06)"},
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
        if (name == e.name) {
            std::printf("running %s ...\n", e.name);
            // Dispatch to the experiment implementation here as it lands.
            return 0;
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
