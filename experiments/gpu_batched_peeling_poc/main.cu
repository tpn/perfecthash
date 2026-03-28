#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr uint32_t MAX_NUMBER_OF_SEEDS = 8;
constexpr uint64_t RNG_DEFAULT_SEED = 0x2019090319811025ull;

enum class StorageMode {
    Auto,
    Bits16,
    Bits32,
};

enum class SolveMode {
    HostRoundTrip,
    DeviceSerial,
};

enum class GraphGeometry {
    Thread,
    Warp,
    Block,
};

enum class AllocationMode {
    ExplicitDevice,
    ManagedDefault,
    ManagedPrefetchGpu,
};

enum class AssignmentBackend {
    Gpu,
    Cpu,
};

enum class OutputFormat {
    Human,
    Json,
};

enum class HashFunctionKind {
    SplitMix,
    MultiplyShiftR,
    MultiplyShiftRX,
    Mulshrolate1RX,
    Mulshrolate2RX,
    Mulshrolate3RX,
    Mulshrolate4RX,
};

struct Options {
    uint32_t Edges = 2048;
    uint32_t Batch = 256;
    uint32_t Threads = 256;
    uint64_t KeySeed = 0x123456789abcdef0ull;
    uint64_t GraphSeed = RNG_DEFAULT_SEED;
    uint64_t RngSubsequence = 0;
    uint64_t RngOffset = 0;
    uint64_t FixedAttempts = 0;
    std::string KeysFile;
    std::string SeedsFile;
    std::string DumpSolvedSeedsFile;
    StorageMode Storage = StorageMode::Auto;
    SolveMode Solve = SolveMode::HostRoundTrip;
    GraphGeometry AssignGeometry = GraphGeometry::Thread;
    GraphGeometry DeviceSerialPeelGeometry = GraphGeometry::Thread;
    HashFunctionKind HashFunction = HashFunctionKind::MultiplyShiftR;
    AllocationMode Allocation = AllocationMode::ExplicitDevice;
    AssignmentBackend AssignmentBackendMode = AssignmentBackend::Gpu;
    OutputFormat Output = OutputFormat::Human;
    double MemoryHeadroomPct = 10.0;
    bool AutoScaleBatchToFit = true;
    bool FirstSolutionWins = false;
    bool Verbose = false;
};

template<typename StorageT>
struct FrontierItemT {
    uint32_t Graph;
    StorageT Vertex;
    StorageT Edge;
};

template<typename StorageT>
struct HashPairT {
    StorageT Vertex1;
    StorageT Vertex2;
};

union ULongBytes {
    uint32_t AsULong;
    struct {
        uint8_t Byte1;
        uint8_t Byte2;
        uint8_t Byte3;
        uint8_t Byte4;
    };
};

struct Philox4x32State {
    uint32_t CounterX = 0;
    uint32_t CounterY = 0;
    uint32_t CounterZ = 0;
    uint32_t CounterW = 0;
    uint32_t OutputX = 0;
    uint32_t OutputY = 0;
    uint32_t OutputZ = 0;
    uint32_t OutputW = 0;
    uint32_t KeyX = 0;
    uint32_t KeyY = 0;
    uint32_t CurrentCount = 0;
};

struct GraphSeeds {
    uint32_t NumberOfSeeds = 0;
    uint32_t Padding = 0;
    uint32_t Seeds[MAX_NUMBER_OF_SEEDS] = {};
};

struct CpuResult {
    bool Success = false;
    bool Verified = false;
    bool Invalid = false;
    uint32_t Peeled = 0;
    double AddBuildMilliseconds = 0.0;
    double PeelMilliseconds = 0.0;
    double AssignMilliseconds = 0.0;
    double VerifyMilliseconds = 0.0;
};

struct CpuStageTimings {
    double AddBuildMilliseconds = 0.0;
    double PeelMilliseconds = 0.0;
    double AssignMilliseconds = 0.0;
    double VerifyMilliseconds = 0.0;
};

struct CpuStageTimingViews {
    CpuStageTimings AllAttempts;
    CpuStageTimings SolvedOnly;
};

struct ExperimentResult {
    std::string DatasetName;
    uint32_t KeyCount = 0;
    uint32_t EdgeCapacity = 0;
    uint32_t Vertices = 0;
    uint32_t Batch = 0;
    uint64_t RequestedFixedAttempts = 0;
    uint64_t ActualAttemptsTried = 0;
    uint64_t BatchesRun = 0;
    bool FirstSolutionWins = false;
    int64_t FirstSolvedAttempt = -1;
    uint32_t Rounds = 0;
    uint32_t GpuSuccess = 0;
    uint32_t CpuSuccess = 0;
    uint32_t Mismatches = 0;
    uint32_t CpuVerifyIssues = 0;
    float GpuMilliseconds = 0.0f;
    double CpuMilliseconds = 0.0;
    CpuStageTimingViews CpuStageTimingsMs;
    double AddBuildMilliseconds = 0.0;
    double PeelMilliseconds = 0.0;
    double AssignMilliseconds = 0.0;
    double VerifyMilliseconds = 0.0;
    uint32_t StorageBits = 0;
    const char *HashFunctionName = nullptr;
    StorageMode SelectedStorage = StorageMode::Auto;
    const char *SolveModeName = nullptr;
    const char *AssignmentBackendName = nullptr;
    const char *AllocationModeName = nullptr;
    size_t EstimatedDeviceBytes = 0;
    size_t DeviceFreeBytes = 0;
    size_t DeviceTotalBytes = 0;
    bool UnifiedLikeDevice = false;
};

struct DeviceMemoryInfo {
    size_t FreeBytes = 0;
    size_t TotalBytes = 0;
    bool Valid = false;
    bool UnifiedLike = false;
};

struct BatchOutcome {
    bool GpuHadSuccess = false;
    uint32_t GpuSuccess = 0;
    uint32_t CpuSuccess = 0;
    uint32_t Mismatches = 0;
    uint32_t CpuVerifyIssues = 0;
    uint32_t Rounds = 0;
    double GpuMilliseconds = 0.0;
    double CpuMilliseconds = 0.0;
    CpuStageTimings CpuAllAttempts;
    CpuStageTimings CpuSolvedOnly;
    double AddBuildMilliseconds = 0.0;
    double PeelMilliseconds = 0.0;
    double AssignMilliseconds = 0.0;
    double VerifyMilliseconds = 0.0;
};

template<typename T>
constexpr T
MaxValue()
{
    return std::numeric_limits<T>::max();
}

inline void
CheckCuda(cudaError_t Error, const char *Message)
{
    if (Error != cudaSuccess) {
        std::cerr << Message << ": " << cudaGetErrorString(Error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

std::string
FormatBytes(size_t Value)
{
    static const char *Suffixes[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double Amount = static_cast<double>(Value);
    size_t Index = 0;

    while (Amount >= 1024.0 && Index + 1 < (sizeof(Suffixes) / sizeof(Suffixes[0]))) {
        Amount /= 1024.0;
        ++Index;
    }

    char Buffer[64];
    std::snprintf(Buffer, sizeof(Buffer), "%.2f %s", Amount, Suffixes[Index]);
    return Buffer;
}

DeviceMemoryInfo
QueryDeviceMemoryInfo()
{
    DeviceMemoryInfo Info = {};
    int Device = 0;
    cudaDeviceProp Props = {};
    int Integrated = 0;
    int ConcurrentManagedAccess = 0;
    int PageableMemoryAccess = 0;

    if (cudaGetDevice(&Device) != cudaSuccess) {
        return Info;
    }

    if (cudaMemGetInfo(&Info.FreeBytes, &Info.TotalBytes) == cudaSuccess) {
        Info.Valid = true;
    }

    if (cudaGetDeviceProperties(&Props, Device) == cudaSuccess) {
        Integrated = Props.integrated;
    }

    if (cudaDeviceGetAttribute(&ConcurrentManagedAccess,
                               cudaDevAttrConcurrentManagedAccess,
                               Device) != cudaSuccess) {
        ConcurrentManagedAccess = 0;
    }

    if (cudaDeviceGetAttribute(&PageableMemoryAccess,
                               cudaDevAttrPageableMemoryAccess,
                               Device) != cudaSuccess) {
        PageableMemoryAccess = 0;
    }

    Info.UnifiedLike = (Integrated != 0 ||
                        ConcurrentManagedAccess != 0 ||
                        PageableMemoryAccess != 0);

    return Info;
}

std::vector<uint32_t>
LoadKeysFromFile(const std::string &Path)
{
    std::ifstream File(Path, std::ios::binary | std::ios::ate);
    if (!File) {
        std::cerr << "Failed to open keys file: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::streamsize Size = File.tellg();
    if (Size <= 0 || (Size % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) {
        std::cerr << "Invalid keys file size: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint32_t> Keys(static_cast<size_t>(Size / sizeof(uint32_t)));
    File.seekg(0, std::ios::beg);
    if (!File.read(reinterpret_cast<char *>(Keys.data()), Size)) {
        std::cerr << "Failed to read keys file: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    return Keys;
}

std::string
Trim(const std::string &Value)
{
    size_t Start = 0;
    size_t End = Value.size();

    while (Start < End && std::isspace(static_cast<unsigned char>(Value[Start]))) {
        ++Start;
    }

    while (End > Start && std::isspace(static_cast<unsigned char>(Value[End - 1]))) {
        --End;
    }

    return Value.substr(Start, End - Start);
}

GraphSeeds
LoadSeedsFromFile(const std::string &Path)
{
    std::ifstream File(Path);
    if (!File) {
        std::cerr << "Failed to open seeds file: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    GraphSeeds Seeds = {};
    std::string Line;

    while (std::getline(File, Line)) {
        Line = Trim(Line);
        if (Line.empty() || Line[0] == '#') {
            continue;
        }

        auto Equals = Line.find('=');
        if (Equals == std::string::npos) {
            continue;
        }

        std::string Name = Trim(Line.substr(0, Equals));
        std::string Value = Trim(Line.substr(Equals + 1));

        if (Name.rfind("Seed", 0) != 0 || Name.find('_') != std::string::npos) {
            continue;
        }

        char *End = nullptr;
        unsigned long Parsed = std::strtoul(Value.c_str(), &End, 0);
        if (!End || *End != '\0') {
            std::cerr << "Invalid seed value in " << Path << ": " << Line << "\n";
            std::exit(EXIT_FAILURE);
        }

        unsigned Index = static_cast<unsigned>(std::strtoul(Name.c_str() + 4, &End, 10));
        if (Index == 0 || Index > MAX_NUMBER_OF_SEEDS) {
            std::cerr << "Invalid seed index in " << Path << ": " << Name << "\n";
            std::exit(EXIT_FAILURE);
        }

        Seeds.Seeds[Index - 1] = static_cast<uint32_t>(Parsed);
        Seeds.NumberOfSeeds = std::max<uint32_t>(Seeds.NumberOfSeeds, Index);
    }

    if (Seeds.NumberOfSeeds == 0) {
        std::cerr << "No seeds found in seeds file: " << Path << "\n";
        std::exit(EXIT_FAILURE);
    }

    return Seeds;
}

std::string
ToLower(std::string Value)
{
    std::transform(Value.begin(),
                   Value.end(),
                   Value.begin(),
                   [](unsigned char Ch) { return static_cast<char>(std::tolower(Ch)); });
    return Value;
}

const char *
StorageModeToString(StorageMode Mode)
{
    switch (Mode) {
        case StorageMode::Auto:
            return "auto";
        case StorageMode::Bits16:
            return "16";
        case StorageMode::Bits32:
            return "32";
        default:
            return "unknown";
    }
}

const char *
SolveModeToString(SolveMode Mode)
{
    switch (Mode) {
        case SolveMode::HostRoundTrip:
            return "host-roundtrip";
        case SolveMode::DeviceSerial:
            return "device-serial";
        default:
            return "unknown";
    }
}

const char *
GraphGeometryToString(GraphGeometry Geometry)
{
    switch (Geometry) {
        case GraphGeometry::Thread:
            return "thread";
        case GraphGeometry::Warp:
            return "warp";
        case GraphGeometry::Block:
            return "block";
        default:
            return "unknown";
    }
}

const char *
AllocationModeToString(AllocationMode Mode)
{
    switch (Mode) {
        case AllocationMode::ExplicitDevice:
            return "explicit-device";
        case AllocationMode::ManagedDefault:
            return "managed-default";
        case AllocationMode::ManagedPrefetchGpu:
            return "managed-prefetch-gpu";
        default:
            return "unknown";
    }
}

const char *
AssignmentBackendToString(AssignmentBackend Mode)
{
    switch (Mode) {
        case AssignmentBackend::Gpu:
            return "gpu";
        case AssignmentBackend::Cpu:
            return "cpu";
        default:
            return "unknown";
    }
}

const char *
OutputFormatToString(OutputFormat Mode)
{
    switch (Mode) {
        case OutputFormat::Human:
            return "human";
        case OutputFormat::Json:
            return "json";
        default:
            return "unknown";
    }
}

HashFunctionKind
ParseHashFunctionKind(const std::string &Value)
{
    const std::string Lower = ToLower(Value);

    if (Lower == "splitmix") {
        return HashFunctionKind::SplitMix;
    } else if (Lower == "multiplyshiftr") {
        return HashFunctionKind::MultiplyShiftR;
    } else if (Lower == "multiplyshiftrx") {
        return HashFunctionKind::MultiplyShiftRX;
    } else if (Lower == "mulshrolate1rx") {
        return HashFunctionKind::Mulshrolate1RX;
    } else if (Lower == "mulshrolate2rx") {
        return HashFunctionKind::Mulshrolate2RX;
    } else if (Lower == "mulshrolate3rx") {
        return HashFunctionKind::Mulshrolate3RX;
    } else if (Lower == "mulshrolate4rx") {
        return HashFunctionKind::Mulshrolate4RX;
    }

    std::cerr << "Invalid --hash-function value: " << Value << "\n";
    std::exit(EXIT_FAILURE);
}

AllocationMode
ParseAllocationMode(const std::string &Value)
{
    const std::string Lower = ToLower(Value);

    if (Lower == "explicit-device") {
        return AllocationMode::ExplicitDevice;
    } else if (Lower == "managed-default") {
        return AllocationMode::ManagedDefault;
    } else if (Lower == "managed-prefetch-gpu") {
        return AllocationMode::ManagedPrefetchGpu;
    }

    std::cerr << "Invalid --allocation-mode value: " << Value << "\n";
    std::exit(EXIT_FAILURE);
}

AssignmentBackend
ParseAssignmentBackend(const std::string &Value)
{
    const std::string Lower = ToLower(Value);

    if (Lower == "gpu") {
        return AssignmentBackend::Gpu;
    } else if (Lower == "cpu") {
        return AssignmentBackend::Cpu;
    }

    std::cerr << "Invalid --assignment-backend value: " << Value << "\n";
    std::exit(EXIT_FAILURE);
}

OutputFormat
ParseOutputFormat(const std::string &Value)
{
    const std::string Lower = ToLower(Value);

    if (Lower == "human") {
        return OutputFormat::Human;
    } else if (Lower == "json") {
        return OutputFormat::Json;
    }

    std::cerr << "Invalid --output-format value: " << Value << "\n";
    std::exit(EXIT_FAILURE);
}

GraphGeometry
ParseGraphGeometry(const std::string &Value, const char *FlagName)
{
    const std::string Lower = ToLower(Value);

    if (Lower == "thread") {
        return GraphGeometry::Thread;
    } else if (Lower == "warp") {
        return GraphGeometry::Warp;
    } else if (Lower == "block") {
        return GraphGeometry::Block;
    }

    std::cerr << "Invalid " << FlagName << " value: " << Value << "\n";
    std::exit(EXIT_FAILURE);
}

std::string
BaseName(const std::string &Path)
{
    const size_t Slash = Path.find_last_of("/\\");
    if (Slash == std::string::npos) {
        return Path;
    }

    return Path.substr(Slash + 1);
}

std::string
JsonEscape(const std::string &Value)
{
    std::string Result;
    Result.reserve(Value.size() + 8);

    for (unsigned char Ch : Value) {
        switch (Ch) {
            case '\\':
                Result += "\\\\";
                break;
            case '"':
                Result += "\\\"";
                break;
            case '\b':
                Result += "\\b";
                break;
            case '\f':
                Result += "\\f";
                break;
            case '\n':
                Result += "\\n";
                break;
            case '\r':
                Result += "\\r";
                break;
            case '\t':
                Result += "\\t";
                break;
            default:
                if (Ch < 0x20) {
                    char Buffer[7];
                    std::snprintf(Buffer, sizeof(Buffer), "\\u%04x", static_cast<unsigned>(Ch));
                    Result += Buffer;
                } else {
                    Result.push_back(static_cast<char>(Ch));
                }
                break;
        }
    }

    return Result;
}

template<typename T>
void
AllocateGpuMemory(T **Ptr, size_t Count, AllocationMode Mode, const char *Name)
{
    if (Mode == AllocationMode::ExplicitDevice) {
        CheckCuda(cudaMalloc(reinterpret_cast<void **>(Ptr), Count * sizeof(T)), Name);
    } else {
        CheckCuda(cudaMallocManaged(reinterpret_cast<void **>(Ptr), Count * sizeof(T)), Name);
    }
}

template<typename T>
void
PrefetchGpuMemory(T *Ptr, size_t Count, int Device, const char *Name)
{
    if (Count == 0) {
        return;
    }

    cudaMemLocation Location = {};
    Location.type = cudaMemLocationTypeDevice;
    Location.id = Device;

    CheckCuda(cudaMemPrefetchAsync(Ptr, Count * sizeof(T), Location, 0), Name);
}

Options
ParseOptions(int argc, char **argv)
{
    Options Opts;

    for (int Index = 1; Index < argc; ++Index) {
        std::string Arg(argv[Index]);

        auto RequireValue = [&](const char *Name) -> std::string {
            if (Index + 1 >= argc) {
                std::cerr << "Missing value for " << Name << "\n";
                std::exit(EXIT_FAILURE);
            }
            return argv[++Index];
        };

        if (Arg == "--edges") {
            Opts.Edges = static_cast<uint32_t>(std::stoul(RequireValue("--edges")));
        } else if (Arg == "--keys-file") {
            Opts.KeysFile = RequireValue("--keys-file");
        } else if (Arg == "--seeds-file") {
            Opts.SeedsFile = RequireValue("--seeds-file");
        } else if (Arg == "--dump-solved-seeds") {
            Opts.DumpSolvedSeedsFile = RequireValue("--dump-solved-seeds");
        } else if (Arg == "--batch") {
            Opts.Batch = static_cast<uint32_t>(std::stoul(RequireValue("--batch")));
        } else if (Arg == "--threads") {
            Opts.Threads = static_cast<uint32_t>(std::stoul(RequireValue("--threads")));
        } else if (Arg == "--key-seed") {
            Opts.KeySeed = std::stoull(RequireValue("--key-seed"), nullptr, 0);
        } else if (Arg == "--graph-seed") {
            Opts.GraphSeed = std::stoull(RequireValue("--graph-seed"), nullptr, 0);
        } else if (Arg == "--rng-subsequence") {
            Opts.RngSubsequence = std::stoull(RequireValue("--rng-subsequence"), nullptr, 0);
        } else if (Arg == "--rng-offset") {
            Opts.RngOffset = std::stoull(RequireValue("--rng-offset"), nullptr, 0);
        } else if (Arg == "--fixed-attempts") {
            Opts.FixedAttempts = std::stoull(RequireValue("--fixed-attempts"), nullptr, 0);
        } else if (Arg == "--storage-bits") {
            const auto Value = RequireValue("--storage-bits");
            if (Value == "auto") {
                Opts.Storage = StorageMode::Auto;
            } else if (Value == "16") {
                Opts.Storage = StorageMode::Bits16;
            } else if (Value == "32") {
                Opts.Storage = StorageMode::Bits32;
            } else {
                std::cerr << "Invalid --storage-bits value: " << Value << "\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (Arg == "--solve-mode") {
            const auto Value = ToLower(RequireValue("--solve-mode"));
            if (Value == "host-roundtrip") {
                Opts.Solve = SolveMode::HostRoundTrip;
            } else if (Value == "device-serial") {
                Opts.Solve = SolveMode::DeviceSerial;
            } else {
                std::cerr << "Invalid --solve-mode value: " << Value << "\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (Arg == "--assign-geometry") {
            Opts.AssignGeometry = ParseGraphGeometry(RequireValue("--assign-geometry"),
                                                     "--assign-geometry");
        } else if (Arg == "--device-serial-peel-geometry") {
            Opts.DeviceSerialPeelGeometry =
                ParseGraphGeometry(RequireValue("--device-serial-peel-geometry"),
                                   "--device-serial-peel-geometry");
        } else if (Arg == "--memory-headroom-pct") {
            Opts.MemoryHeadroomPct = std::stod(RequireValue("--memory-headroom-pct"));
        } else if (Arg == "--disable-auto-batch-scale") {
            Opts.AutoScaleBatchToFit = false;
        } else if (Arg == "--hash-function") {
            Opts.HashFunction = ParseHashFunctionKind(RequireValue("--hash-function"));
        } else if (Arg == "--allocation-mode") {
            Opts.Allocation = ParseAllocationMode(RequireValue("--allocation-mode"));
        } else if (Arg == "--assignment-backend") {
            Opts.AssignmentBackendMode = ParseAssignmentBackend(RequireValue("--assignment-backend"));
        } else if (Arg == "--output-format") {
            Opts.Output = ParseOutputFormat(RequireValue("--output-format"));
        } else if (Arg == "--first-solution-wins") {
            Opts.FirstSolutionWins = true;
        } else if (Arg == "--verbose") {
            Opts.Verbose = true;
        } else if (Arg == "--help" || Arg == "-h") {
            std::cout
                << "Usage: gpu_batched_peeling_poc [options]\n"
                << "  --edges <n>           Number of logical keys for generated input\n"
                << "  --keys-file <p>       Load 32-bit keys from a .keys file\n"
                << "  --seeds-file <p>      Load a fixed seed set from a .seeds file\n"
                << "  --dump-solved-seeds <p>\n"
                << "                        Write successful graph indices and seeds to a CSV\n"
                << "  --batch <n>           Number of graph attempts in the batch\n"
                << "  --threads <n>         Threads per block for build/collect/peel kernels\n"
                << "  --assign-geometry <x> stage-1 config/reporting only; assignment stays thread mode\n"
                << "  --device-serial-peel-geometry <x>\n"
                << "                        device-serial peel geometry: thread, warp, or block\n"
                << "  --storage-bits <x>    auto, 16, or 32\n"
                << "  --solve-mode <x>      host-roundtrip or device-serial\n"
                << "  --memory-headroom-pct <x>\n"
                << "                        required free-memory headroom percentage [default: 10]\n"
                << "  --disable-auto-batch-scale\n"
                << "                        fail instead of shrinking batch to fit available memory\n"
                << "  --hash-function <x>   SplitMix, MultiplyShiftR, MultiplyShiftRX,\n"
                << "                        Mulshrolate1RX, Mulshrolate2RX, Mulshrolate3RX,\n"
                << "                        or Mulshrolate4RX\n"
                << "  --allocation-mode <x> explicit-device, managed-default, or\n"
                << "                        managed-prefetch-gpu\n"
                << "  --assignment-backend <x>\n"
                << "                        gpu or cpu\n"
                << "  --output-format <x>   human or json\n"
                << "  --key-seed <x>        Base seed for generated keys\n"
                << "  --graph-seed <x>      Philox seed for per-graph hash seeds\n"
                << "  --rng-subsequence <x> Philox subsequence base for graph seeds\n"
                << "  --rng-offset <x>      Philox offset base for graph seeds\n"
                << "  --fixed-attempts <n>  Run repeated batches until at least n attempts\n"
                << "  --first-solution-wins stop after the first batch with a solved attempt\n"
                << "  --verbose             Print per-graph mismatch details\n";
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown argument: " << Arg << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    if (Opts.Edges == 0 || Opts.Batch == 0 || Opts.Threads == 0) {
        std::cerr << "Edges, batch, and threads must be non-zero.\n";
        std::exit(EXIT_FAILURE);
    }

    if (Opts.MemoryHeadroomPct < 0.0 || Opts.MemoryHeadroomPct >= 100.0) {
        std::cerr << "--memory-headroom-pct must be in the range [0, 100).\n";
        std::exit(EXIT_FAILURE);
    }

    if (Opts.Solve == SolveMode::DeviceSerial) {
        if (Opts.DeviceSerialPeelGeometry == GraphGeometry::Warp && (Opts.Threads % 32u) != 0u) {
            std::cerr << "--device-serial-peel-geometry warp requires --threads to be a multiple of 32.\n";
            std::exit(EXIT_FAILURE);
        }

        if (Opts.DeviceSerialPeelGeometry == GraphGeometry::Block && Opts.Threads < 32u) {
            std::cerr << "--device-serial-peel-geometry block requires --threads to be at least 32.\n";
            std::exit(EXIT_FAILURE);
        }
    } else if (Opts.DeviceSerialPeelGeometry != GraphGeometry::Thread) {
        std::cerr << "--device-serial-peel-geometry only applies with --solve-mode device-serial.\n";
        std::exit(EXIT_FAILURE);
    }

    return Opts;
}

inline uint32_t
NextPowerOfTwo(uint32_t Value)
{
    if (Value <= 1) {
        return 1;
    }

    --Value;
    Value |= Value >> 1;
    Value |= Value >> 2;
    Value |= Value >> 4;
    Value |= Value >> 8;
    Value |= Value >> 16;
    return Value + 1;
}

__host__ __device__ inline uint64_t
SplitMix64(uint64_t Value)
{
    Value += 0x9e3779b97f4a7c15ull;
    Value = (Value ^ (Value >> 30)) * 0xbf58476d1ce4e5b9ull;
    Value = (Value ^ (Value >> 27)) * 0x94d049bb133111ebull;
    return Value ^ (Value >> 31);
}

inline uint32_t
TrailingZeros32(uint32_t Value)
{
    if (Value == 0) {
        return 32;
    }

#if defined(__GNUG__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctz(Value));
#else
    uint32_t Count = 0;
    while ((Value & 1u) == 0) {
        Value >>= 1;
        ++Count;
    }
    return Count;
#endif
}

template<typename ValueType,
         typename ShiftType>
__host__ __device__ inline ValueType
RotateRight(ValueType Value, ShiftType Shift)
{
    constexpr ShiftType Bits = static_cast<ShiftType>(sizeof(ValueType) * 8);

    if (Shift == 0) {
        return Value;
    }

    Shift %= Bits;
    return (Value >> Shift) | (Value << (Bits - Shift));
}

inline uint32_t
MulHighLow32(uint32_t A, uint32_t B, uint32_t *OutHigh)
{
    uint64_t Product = static_cast<uint64_t>(A) * static_cast<uint64_t>(B);
    *OutHigh = static_cast<uint32_t>(Product >> 32u);
    return static_cast<uint32_t>(Product);
}

inline void
PhiloxStateIncrement(Philox4x32State *State, uint64_t Offset)
{
    uint32_t Low = static_cast<uint32_t>(Offset);
    uint32_t High = static_cast<uint32_t>(Offset >> 32u);

    State->CounterX += Low;

    if (State->CounterX < Low) {
        ++High;
    }

    State->CounterY += High;

    if (High <= State->CounterY) {
        return;
    }

    if (++State->CounterZ) {
        return;
    }

    ++State->CounterW;
}

inline void
PhiloxStateIncrementHigh(Philox4x32State *State, uint64_t Number)
{
    uint32_t Low = static_cast<uint32_t>(Number);
    uint32_t High = static_cast<uint32_t>(Number >> 32u);

    State->CounterZ += Low;
    if (State->CounterZ < Low) {
        ++High;
    }

    State->CounterW += High;
}

inline void
PhiloxStateIncrementNoOffset(Philox4x32State *State)
{
    if (++State->CounterX) {
        return;
    }
    if (++State->CounterY) {
        return;
    }
    if (++State->CounterZ) {
        return;
    }
    ++State->CounterW;
}

inline void
Philox4x32Round(uint32_t CounterX,
                uint32_t CounterY,
                uint32_t CounterZ,
                uint32_t CounterW,
                uint32_t KeyX,
                uint32_t KeyY,
                uint32_t *OutX,
                uint32_t *OutY,
                uint32_t *OutZ,
                uint32_t *OutW)
{
    constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53u;
    constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57u;

    uint32_t High0 = 0;
    uint32_t High1 = 0;
    uint32_t Low0 = MulHighLow32(PHILOX_M4x32_0, CounterX, &High0);
    uint32_t Low1 = MulHighLow32(PHILOX_M4x32_1, CounterZ, &High1);

    *OutX = High1 ^ CounterY ^ KeyX;
    *OutY = Low1;
    *OutZ = High0 ^ CounterW ^ KeyY;
    *OutW = Low0;
}

inline void
Philox4x32_10(Philox4x32State *State)
{
    constexpr uint32_t PHILOX_W32_0 = 0x9E3779B9u;
    constexpr uint32_t PHILOX_W32_1 = 0xBB67AE85u;

    uint32_t Cx = State->CounterX;
    uint32_t Cy = State->CounterY;
    uint32_t Cz = State->CounterZ;
    uint32_t Cw = State->CounterW;
    uint32_t Kx = State->KeyX;
    uint32_t Ky = State->KeyY;

    for (int Round = 0; Round < 9; ++Round) {
        uint32_t Ox = 0;
        uint32_t Oy = 0;
        uint32_t Oz = 0;
        uint32_t Ow = 0;
        Philox4x32Round(Cx, Cy, Cz, Cw, Kx, Ky, &Ox, &Oy, &Oz, &Ow);
        Cx = Ox;
        Cy = Oy;
        Cz = Oz;
        Cw = Ow;
        Kx += PHILOX_W32_0;
        Ky += PHILOX_W32_1;
    }

    Philox4x32Round(Cx, Cy, Cz, Cw, Kx, Ky,
                    &State->OutputX,
                    &State->OutputY,
                    &State->OutputZ,
                    &State->OutputW);
}

inline void
Skipahead(Philox4x32State *State, uint64_t Offset)
{
    State->CurrentCount += static_cast<uint32_t>(Offset & 3u);
    Offset /= 4u;

    if (State->CurrentCount > 3) {
        Offset += 1u;
        State->CurrentCount -= 4u;
    }

    PhiloxStateIncrement(State, Offset);
    Philox4x32_10(State);
}

inline void
SkipaheadSubsequence(Philox4x32State *State, uint64_t Subsequence)
{
    PhiloxStateIncrementHigh(State, Subsequence);
    Philox4x32_10(State);
}

inline Philox4x32State
InitializePhiloxState(uint64_t Seed,
                      uint64_t Subsequence,
                      uint64_t Offset)
{
    Philox4x32State State = {};
    State.KeyX = static_cast<uint32_t>(Seed);
    State.KeyY = static_cast<uint32_t>(Seed >> 32u);
    SkipaheadSubsequence(&State, Subsequence);
    Skipahead(&State, Offset);
    return State;
}

inline uint32_t
GetRandomLong(Philox4x32State *State)
{
    uint32_t Value = 0;

    switch (State->CurrentCount++) {
        case 0:
            Value = State->OutputX;
            break;
        case 1:
            Value = State->OutputY;
            break;
        case 2:
            Value = State->OutputZ;
            break;
        case 3:
            Value = State->OutputW;
            break;
        default:
            std::abort();
    }

    if (State->CurrentCount == 4) {
        PhiloxStateIncrementNoOffset(State);
        Philox4x32_10(State);
        State->CurrentCount = 0;
    }

    return Value;
}

inline uint32_t
GetRandomNonZeroLong(Philox4x32State *State)
{
    uint32_t Random = 0;
    unsigned Attempts = 0;

    do {
        Random = GetRandomLong(State);
        ++Attempts;
        if (Attempts > 16) {
            std::cerr << "Philox produced too many zero outputs in a row.\n";
            std::exit(EXIT_FAILURE);
        }
    } while (Random == 0);

    return Random;
}

template<HashFunctionKind Kind>
struct HashTraits;

template<>
struct HashTraits<HashFunctionKind::SplitMix> {
    static constexpr const char *Name = "SplitMix";
    static constexpr uint32_t NumberOfSeeds = 2;
    static constexpr uint32_t Seed3Mask = 0;
};

template<>
struct HashTraits<HashFunctionKind::MultiplyShiftR> {
    static constexpr const char *Name = "MultiplyShiftR";
    static constexpr uint32_t NumberOfSeeds = 3;
    static constexpr uint32_t Seed3Mask = 0x1f1f;
};

template<>
struct HashTraits<HashFunctionKind::MultiplyShiftRX> {
    static constexpr const char *Name = "MultiplyShiftRX";
    static constexpr uint32_t NumberOfSeeds = 3;
    static constexpr uint32_t Seed3Mask = 0x1f1f;
};

template<>
struct HashTraits<HashFunctionKind::Mulshrolate1RX> {
    static constexpr const char *Name = "Mulshrolate1RX";
    static constexpr uint32_t NumberOfSeeds = 3;
    static constexpr uint32_t Seed3Mask = 0x1f1f1f1f;
};

template<>
struct HashTraits<HashFunctionKind::Mulshrolate2RX> {
    static constexpr const char *Name = "Mulshrolate2RX";
    static constexpr uint32_t NumberOfSeeds = 3;
    static constexpr uint32_t Seed3Mask = 0x1f1f1f1f;
};

template<>
struct HashTraits<HashFunctionKind::Mulshrolate3RX> {
    static constexpr const char *Name = "Mulshrolate3RX";
    static constexpr uint32_t NumberOfSeeds = 4;
    static constexpr uint32_t Seed3Mask = 0x1f1f1f1f;
};

template<>
struct HashTraits<HashFunctionKind::Mulshrolate4RX> {
    static constexpr const char *Name = "Mulshrolate4RX";
    static constexpr uint32_t NumberOfSeeds = 5;
    static constexpr uint32_t Seed3Mask = 0x1f1f1f1f;
};

template<HashFunctionKind Kind>
constexpr bool
RequiresAndMask()
{
    return !(Kind == HashFunctionKind::MultiplyShiftRX ||
             Kind == HashFunctionKind::Mulshrolate1RX ||
             Kind == HashFunctionKind::Mulshrolate2RX ||
             Kind == HashFunctionKind::Mulshrolate3RX ||
             Kind == HashFunctionKind::Mulshrolate4RX);
}

template<HashFunctionKind Kind>
GraphSeeds
MakeGraphSeeds(uint64_t GraphSeed, uint64_t AttemptId)
{
    GraphSeeds Seeds = {};
    constexpr uint32_t NumberOfSeeds = HashTraits<Kind>::NumberOfSeeds;
    Philox4x32State State = InitializePhiloxState(GraphSeed,
                                                  AttemptId,
                                                  0);

    Seeds.NumberOfSeeds = NumberOfSeeds;

    for (uint32_t Index = 0; Index < NumberOfSeeds; ++Index) {
        Seeds.Seeds[Index] = GetRandomNonZeroLong(&State);
    }

    return Seeds;
}

template<HashFunctionKind Kind>
std::vector<GraphSeeds>
MakeGraphSeedsVector(uint32_t Batch,
                     uint64_t GraphSeed,
                     uint64_t RngSubsequence,
                     uint64_t RngOffset,
                     uint32_t Vertices,
                     const std::string &SeedsFile,
                     uint64_t AttemptBase)
{
    if (!SeedsFile.empty()) {
        GraphSeeds Fixed = LoadSeedsFromFile(SeedsFile);
        std::vector<GraphSeeds> Seeds(Batch, Fixed);
        return Seeds;
    }

    std::vector<GraphSeeds> Seeds(Batch);
    constexpr uint32_t Seed3Mask = HashTraits<Kind>::Seed3Mask;
    const uint32_t HashShift = 32u - TrailingZeros32(Vertices);

    for (uint32_t Graph = 0; Graph < Batch; ++Graph) {
        GraphSeeds Generated = {};
        Philox4x32State State = InitializePhiloxState(GraphSeed,
                                                      RngSubsequence + AttemptBase + Graph,
                                                      RngOffset);

        Generated.NumberOfSeeds = HashTraits<Kind>::NumberOfSeeds;

        for (uint32_t Index = 0; Index < Generated.NumberOfSeeds; ++Index) {
            Generated.Seeds[Index] = GetRandomNonZeroLong(&State);
        }

        if constexpr (!RequiresAndMask<Kind>()) {
            ULongBytes Seed3 = {};
            Seed3.AsULong = Generated.Seeds[2];
            Seed3.Byte1 = static_cast<uint8_t>(HashShift);
            Generated.Seeds[2] = Seed3.AsULong;
        }

        if constexpr (Seed3Mask != 0) {
            Generated.Seeds[2] &= Seed3Mask;
            if constexpr (!RequiresAndMask<Kind>()) {
                ULongBytes Seed3 = {};
                Seed3.AsULong = Generated.Seeds[2];
                Seed3.Byte1 = static_cast<uint8_t>(HashShift);
                Generated.Seeds[2] = Seed3.AsULong;
            }
        }

        Seeds[Graph] = Generated;
    }
    return Seeds;
}

template<HashFunctionKind Kind, typename StorageT>
struct HashImpl;

template<typename StorageT>
struct HashImpl<HashFunctionKind::SplitMix, StorageT> {
    __host__ __device__ static HashPairT<StorageT>
    Apply(uint32_t Key, const GraphSeeds &Seeds, uint32_t Mask)
    {
        uint32_t Vertex1 = static_cast<uint32_t>(SplitMix64(static_cast<uint64_t>(Key) ^ Seeds.Seeds[0])) & Mask;
        uint32_t Vertex2 = static_cast<uint32_t>(SplitMix64(static_cast<uint64_t>(Key) ^ Seeds.Seeds[1])) & Mask;

        return {
            static_cast<StorageT>(Vertex1),
            static_cast<StorageT>(Vertex2),
        };
    }
};

template<typename StorageT>
struct HashImpl<HashFunctionKind::MultiplyShiftR, StorageT> {
    __host__ __device__ static HashPairT<StorageT>
    Apply(uint32_t Key, const GraphSeeds &Seeds, uint32_t Mask)
    {
        ULongBytes Seed3 = {};
        Seed3.AsULong = Seeds.Seeds[2];

        uint32_t Vertex1 = (Key * Seeds.Seeds[0]) >> Seed3.Byte1;
        uint32_t Vertex2 = (Key * Seeds.Seeds[1]) >> Seed3.Byte2;

        return {
            static_cast<StorageT>(Vertex1 & Mask),
            static_cast<StorageT>(Vertex2 & Mask),
        };
    }
};

template<typename StorageT>
struct HashImpl<HashFunctionKind::MultiplyShiftRX, StorageT> {
    __host__ __device__ static HashPairT<StorageT>
    Apply(uint32_t Key, const GraphSeeds &Seeds, uint32_t Mask)
    {
        ULongBytes Seed3 = {};
        Seed3.AsULong = Seeds.Seeds[2];
        (void)Mask;

        uint32_t Vertex1 = (Key * Seeds.Seeds[0]) >> Seed3.Byte1;
        uint32_t Vertex2 = (Key * Seeds.Seeds[1]) >> Seed3.Byte1;

        return {
            static_cast<StorageT>(Vertex1),
            static_cast<StorageT>(Vertex2),
        };
    }
};

template<typename StorageT>
struct HashImpl<HashFunctionKind::Mulshrolate1RX, StorageT> {
    __host__ __device__ static HashPairT<StorageT>
    Apply(uint32_t Key, const GraphSeeds &Seeds, uint32_t Mask)
    {
        ULongBytes Seed3 = {};
        Seed3.AsULong = Seeds.Seeds[2];
        (void)Mask;

        uint32_t Vertex1 = Key * Seeds.Seeds[0];
        Vertex1 = RotateRight(Vertex1, Seed3.Byte2);
        Vertex1 = Vertex1 >> Seed3.Byte1;

        uint32_t Vertex2 = Key * Seeds.Seeds[1];
        Vertex2 = Vertex2 >> Seed3.Byte1;

        return {
            static_cast<StorageT>(Vertex1),
            static_cast<StorageT>(Vertex2),
        };
    }
};

template<typename StorageT>
struct HashImpl<HashFunctionKind::Mulshrolate2RX, StorageT> {
    __host__ __device__ static HashPairT<StorageT>
    Apply(uint32_t Key, const GraphSeeds &Seeds, uint32_t Mask)
    {
        ULongBytes Seed3 = {};
        Seed3.AsULong = Seeds.Seeds[2];
        (void)Mask;

        uint32_t Vertex1 = Key * Seeds.Seeds[0];
        Vertex1 = RotateRight(Vertex1, Seed3.Byte2);
        Vertex1 = Vertex1 >> Seed3.Byte1;

        uint32_t Vertex2 = Key * Seeds.Seeds[1];
        Vertex2 = RotateRight(Vertex2, Seed3.Byte3);
        Vertex2 = Vertex2 >> Seed3.Byte1;

        return {
            static_cast<StorageT>(Vertex1),
            static_cast<StorageT>(Vertex2),
        };
    }
};

template<typename StorageT>
struct HashImpl<HashFunctionKind::Mulshrolate3RX, StorageT> {
    __host__ __device__ static HashPairT<StorageT>
    Apply(uint32_t Key, const GraphSeeds &Seeds, uint32_t Mask)
    {
        ULongBytes Seed3 = {};
        Seed3.AsULong = Seeds.Seeds[2];
        (void)Mask;

        uint32_t Vertex1 = Key * Seeds.Seeds[0];
        Vertex1 = RotateRight(Vertex1, Seed3.Byte2);
        Vertex1 = Vertex1 * Seeds.Seeds[3];
        Vertex1 = Vertex1 >> Seed3.Byte1;

        uint32_t Vertex2 = Key * Seeds.Seeds[1];
        Vertex2 = RotateRight(Vertex2, Seed3.Byte3);
        Vertex2 = Vertex2 >> Seed3.Byte1;

        return {
            static_cast<StorageT>(Vertex1),
            static_cast<StorageT>(Vertex2),
        };
    }
};

template<typename StorageT>
struct HashImpl<HashFunctionKind::Mulshrolate4RX, StorageT> {
    __host__ __device__ static HashPairT<StorageT>
    Apply(uint32_t Key, const GraphSeeds &Seeds, uint32_t Mask)
    {
        ULongBytes Seed3 = {};
        Seed3.AsULong = Seeds.Seeds[2];
        (void)Mask;

        uint32_t Vertex1 = Key * Seeds.Seeds[0];
        Vertex1 = RotateRight(Vertex1, Seed3.Byte2);
        Vertex1 = Vertex1 * Seeds.Seeds[3];
        Vertex1 = Vertex1 >> Seed3.Byte1;

        uint32_t Vertex2 = Key * Seeds.Seeds[1];
        Vertex2 = RotateRight(Vertex2, Seed3.Byte3);
        Vertex2 = Vertex2 * Seeds.Seeds[4];
        Vertex2 = Vertex2 >> Seed3.Byte1;

        return {
            static_cast<StorageT>(Vertex1),
            static_cast<StorageT>(Vertex2),
        };
    }
};

template<HashFunctionKind Kind, typename StorageT>
__global__ void
BuildGraphsKernel(uint32_t Edges,
                  uint32_t Vertices,
                  uint32_t Batch,
                  const uint32_t *Keys,
                  const GraphSeeds *GraphSeedsArray,
                  StorageT *EdgeU,
                  StorageT *EdgeV,
                  uint32_t *Degree,
                  uint32_t *XorEdge,
                  uint32_t *InvalidGraphs)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t Total = static_cast<uint64_t>(Edges) * Batch;
    uint32_t VertexMask = Vertices - 1;

    while (Global < Total) {
        uint32_t Graph = static_cast<uint32_t>(Global / Edges);
        uint32_t Edge = static_cast<uint32_t>(Global % Edges);
        uint32_t EdgeIndex = Graph * Edges + Edge;
        uint32_t VertexBase = Graph * Vertices;

        uint32_t Key = Keys[Edge];
        const GraphSeeds Seeds = GraphSeedsArray[Graph];
        const auto Hash = HashImpl<Kind, StorageT>::Apply(Key, Seeds, VertexMask);

        uint32_t U = static_cast<uint32_t>(Hash.Vertex1);
        uint32_t V = static_cast<uint32_t>(Hash.Vertex2);

        EdgeU[EdgeIndex] = Hash.Vertex1;
        EdgeV[EdgeIndex] = Hash.Vertex2;

        if (U == V || U >= Vertices || V >= Vertices) {
            atomicExch(&InvalidGraphs[Graph], 1u);
            Global += blockDim.x * gridDim.x;
            continue;
        }

        atomicAdd(&Degree[VertexBase + U], 1u);
        atomicAdd(&Degree[VertexBase + V], 1u);
        atomicXor(&XorEdge[VertexBase + U], Edge);
        atomicXor(&XorEdge[VertexBase + V], Edge);

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
__global__ void
CollectFrontierKernel(uint32_t Vertices,
                      uint32_t Batch,
                      const uint32_t *Degree,
                      const uint32_t *XorEdge,
                      const uint32_t *InvalidGraphs,
                      FrontierItemT<StorageT> *Frontier,
                      uint32_t *FrontierCount)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t Total = static_cast<uint64_t>(Vertices) * Batch;

    while (Global < Total) {
        uint32_t Graph = static_cast<uint32_t>(Global / Vertices);
        uint32_t Vertex = static_cast<uint32_t>(Global % Vertices);

        if (InvalidGraphs[Graph] == 0 && Degree[Global] == 1) {
            uint32_t Position = atomicAdd(FrontierCount, 1u);
            Frontier[Position].Graph = Graph;
            Frontier[Position].Vertex = static_cast<StorageT>(Vertex);
            Frontier[Position].Edge = static_cast<StorageT>(XorEdge[Global]);
        }

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
__global__ void
PeelFrontierKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t FrontierCount,
                   const FrontierItemT<StorageT> *Frontier,
                   const StorageT *EdgeU,
                   const StorageT *EdgeV,
                   uint32_t *Degree,
                   uint32_t *XorEdge,
                   uint32_t *EdgePeeled,
                   StorageT *OwnerVertex,
                   StorageT *PeelOrder,
                   uint32_t *PeeledCount)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;

    while (Global < FrontierCount) {
        FrontierItemT<StorageT> Item = Frontier[Global];
        uint32_t Edge = static_cast<uint32_t>(Item.Edge);
        uint32_t EdgeIndex = Item.Graph * Edges + Edge;

        if (atomicCAS(&EdgePeeled[EdgeIndex], 0u, 1u) == 0u) {
            uint32_t Order = atomicAdd(&PeeledCount[Item.Graph], 1u);
            uint32_t VertexBase = Item.Graph * Vertices;
            uint32_t U = static_cast<uint32_t>(EdgeU[EdgeIndex]);
            uint32_t V = static_cast<uint32_t>(EdgeV[EdgeIndex]);

            OwnerVertex[EdgeIndex] = Item.Vertex;
            PeelOrder[Item.Graph * Edges + Order] = static_cast<StorageT>(Edge);

            atomicSub(&Degree[VertexBase + U], 1u);
            atomicXor(&XorEdge[VertexBase + U], Edge);
            atomicSub(&Degree[VertexBase + V], 1u);
            atomicXor(&XorEdge[VertexBase + V], Edge);
        }

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
__global__ void
AssignGraphsKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t EdgeMask,
                   const uint32_t *InvalidGraphs,
                   const StorageT *EdgeU,
                   const StorageT *EdgeV,
                   const StorageT *OwnerVertex,
                   const StorageT *PeelOrder,
                   const uint32_t *PeeledCount,
                   StorageT *Assigned)
{
    uint32_t Graph = blockIdx.x;

    if (threadIdx.x != 0) {
        return;
    }

    if (InvalidGraphs[Graph] != 0 || PeeledCount[Graph] != Edges) {
        return;
    }

    uint32_t VertexBase = Graph * Vertices;
    uint32_t EdgeBase = Graph * Edges;

    for (int64_t Index = static_cast<int64_t>(Edges) - 1; Index >= 0; --Index) {
        uint32_t Edge = static_cast<uint32_t>(PeelOrder[EdgeBase + static_cast<uint32_t>(Index)]);
        uint32_t Owner = static_cast<uint32_t>(OwnerVertex[EdgeBase + Edge]);
        uint32_t U = static_cast<uint32_t>(EdgeU[EdgeBase + Edge]);
        uint32_t V = static_cast<uint32_t>(EdgeV[EdgeBase + Edge]);
        uint32_t Other = (Owner == U) ? V : U;
        uint32_t OtherAssigned = static_cast<uint32_t>(Assigned[VertexBase + Other]);
        uint32_t Value = (Edge - OtherAssigned) & EdgeMask;

        Assigned[VertexBase + Owner] = static_cast<StorageT>(Value);
    }
}

template<typename StorageT>
__global__ void
VerifyGraphsKernel(uint32_t Edges,
                   uint32_t Vertices,
                   uint32_t Batch,
                   uint32_t EdgeMask,
                   const uint32_t *InvalidGraphs,
                   const StorageT *EdgeU,
                   const StorageT *EdgeV,
                   const StorageT *Assigned,
                   const uint32_t *PeeledCount,
                   uint32_t *VerifyFailures)
{
    uint64_t Global = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t Total = static_cast<uint64_t>(Edges) * Batch;

    while (Global < Total) {
        uint32_t Graph = static_cast<uint32_t>(Global / Edges);
        uint32_t Edge = static_cast<uint32_t>(Global % Edges);

        if (InvalidGraphs[Graph] != 0 || PeeledCount[Graph] != Edges) {
            atomicAdd(&VerifyFailures[Graph], 1u);
            Global += blockDim.x * gridDim.x;
            continue;
        }

        uint32_t EdgeBase = Graph * Edges;
        uint32_t VertexBase = Graph * Vertices;
        uint32_t U = static_cast<uint32_t>(EdgeU[EdgeBase + Edge]);
        uint32_t V = static_cast<uint32_t>(EdgeV[EdgeBase + Edge]);
        uint32_t Index = (
            static_cast<uint32_t>(Assigned[VertexBase + U]) +
            static_cast<uint32_t>(Assigned[VertexBase + V])
        ) & EdgeMask;

        if (Index != Edge) {
            atomicAdd(&VerifyFailures[Graph], 1u);
        }

        Global += blockDim.x * gridDim.x;
    }
}

template<typename StorageT>
__global__ void
PeelGraphsDeviceSerialThreadKernel(uint32_t Edges,
                                   uint32_t Vertices,
                                   uint32_t Batch,
                                   const uint32_t *InvalidGraphs,
                                   const StorageT *EdgeU,
                                   const StorageT *EdgeV,
                                   uint32_t *Degree,
                                   uint32_t *XorEdge,
                                   uint32_t *EdgePeeled,
                                   StorageT *OwnerVertex,
                                   StorageT *PeelOrder,
                                   uint32_t *PeeledCount,
                                   uint32_t *VerifyFailures,
                                   uint32_t *GraphRounds)
{
    const uint32_t Graph = blockIdx.x;

    if (Graph >= Batch || threadIdx.x != 0) {
        return;
    }

    const uint32_t VertexBase = Graph * Vertices;
    const uint32_t EdgeBase = Graph * Edges;
    uint32_t Rounds = 0;

    if (InvalidGraphs[Graph] != 0) {
        VerifyFailures[Graph] = 1;
        GraphRounds[Graph] = 0;
        return;
    }

    for (;;) {
        bool Progress = false;

        for (uint32_t Vertex = 0; Vertex < Vertices; ++Vertex) {
            if (Degree[VertexBase + Vertex] != 1) {
                continue;
            }

            const uint32_t Edge = XorEdge[VertexBase + Vertex];
            const uint32_t EdgeIndex = EdgeBase + Edge;

            if (EdgePeeled[EdgeIndex] != 0) {
                continue;
            }

            EdgePeeled[EdgeIndex] = 1;

            const uint32_t Order = PeeledCount[Graph]++;
            const uint32_t U = static_cast<uint32_t>(EdgeU[EdgeIndex]);
            const uint32_t V = static_cast<uint32_t>(EdgeV[EdgeIndex]);

            OwnerVertex[EdgeIndex] = static_cast<StorageT>(Vertex);
            PeelOrder[EdgeBase + Order] = static_cast<StorageT>(Edge);

            if (Degree[VertexBase + U] > 0) {
                --Degree[VertexBase + U];
                XorEdge[VertexBase + U] ^= Edge;
            }

            if (Degree[VertexBase + V] > 0) {
                --Degree[VertexBase + V];
                XorEdge[VertexBase + V] ^= Edge;
            }

            Progress = true;
        }

        if (!Progress) {
            break;
        }

        ++Rounds;
    }

    GraphRounds[Graph] = Rounds;

    if (PeeledCount[Graph] != Edges) {
        VerifyFailures[Graph] = 1;
    }
}

template<typename StorageT>
__global__ void
PeelGraphsDeviceSerialWarpKernel(uint32_t Edges,
                                 uint32_t Vertices,
                                 uint32_t Batch,
                                 const uint32_t *InvalidGraphs,
                                 const StorageT *EdgeU,
                                 const StorageT *EdgeV,
                                 uint32_t *Degree,
                                 uint32_t *XorEdge,
                                 uint32_t *EdgePeeled,
                                 StorageT *OwnerVertex,
                                 StorageT *PeelOrder,
                                 uint32_t *PeeledCount,
                                 FrontierItemT<StorageT> *Frontier,
                                 uint32_t *FrontierCount,
                                 uint32_t *VerifyFailures,
                                 uint32_t *GraphRounds)
{
    constexpr uint32_t WarpSize = 32u;
    const uint32_t Thread = threadIdx.x;
    const uint32_t WarpId = Thread / WarpSize;
    const uint32_t Lane = Thread & (WarpSize - 1u);
    const uint32_t WarpsPerBlock = blockDim.x / WarpSize;
    const uint32_t Graph = blockIdx.x * WarpsPerBlock + WarpId;

    if (Graph >= Batch) {
        return;
    }

    const uint32_t VertexBase = Graph * Vertices;
    const uint32_t EdgeBase = Graph * Edges;
    FrontierItemT<StorageT> *GraphFrontier = Frontier + static_cast<size_t>(Graph) * Vertices;
    uint32_t Rounds = 0;

    if (InvalidGraphs[Graph] != 0) {
        if (Lane == 0) {
            VerifyFailures[Graph] = 1;
            GraphRounds[Graph] = 0;
        }
        return;
    }

    for (;;) {
        if (Lane == 0) {
            FrontierCount[Graph] = 0;
        }
        __syncwarp();

        for (uint32_t Vertex = Lane; Vertex < Vertices; Vertex += WarpSize) {
            if (Degree[VertexBase + Vertex] != 1) {
                continue;
            }

            const uint32_t Edge = XorEdge[VertexBase + Vertex];
            const uint32_t EdgeIndex = EdgeBase + Edge;

            if (atomicCAS(&EdgePeeled[EdgeIndex], 0u, 1u) != 0u) {
                continue;
            }

            const uint32_t Slot = atomicAdd(&FrontierCount[Graph], 1u);
            GraphFrontier[Slot].Graph = Graph;
            GraphFrontier[Slot].Vertex = static_cast<StorageT>(Vertex);
            GraphFrontier[Slot].Edge = static_cast<StorageT>(Edge);
        }

        __syncwarp();

        const uint32_t FrontierSize = FrontierCount[Graph];
        if (FrontierSize == 0) {
            break;
        }

        for (uint32_t Index = Lane; Index < FrontierSize; Index += WarpSize) {
            const FrontierItemT<StorageT> Item = GraphFrontier[Index];
            const uint32_t Edge = static_cast<uint32_t>(Item.Edge);
            const uint32_t EdgeIndex = EdgeBase + Edge;
            const uint32_t Order = atomicAdd(&PeeledCount[Graph], 1u);
            const uint32_t U = static_cast<uint32_t>(EdgeU[EdgeIndex]);
            const uint32_t V = static_cast<uint32_t>(EdgeV[EdgeIndex]);

            OwnerVertex[EdgeIndex] = Item.Vertex;
            PeelOrder[EdgeBase + Order] = static_cast<StorageT>(Edge);

            atomicSub(&Degree[VertexBase + U], 1u);
            atomicXor(&XorEdge[VertexBase + U], Edge);
            atomicSub(&Degree[VertexBase + V], 1u);
            atomicXor(&XorEdge[VertexBase + V], Edge);
        }

        __syncwarp();
        ++Rounds;
    }

    if (Lane == 0) {
        GraphRounds[Graph] = Rounds;
        if (PeeledCount[Graph] != Edges) {
            VerifyFailures[Graph] = 1;
        }
    }
}

template<typename StorageT>
__global__ void
PeelGraphsDeviceSerialBlockKernel(uint32_t Edges,
                                  uint32_t Vertices,
                                  uint32_t Batch,
                                  const uint32_t *InvalidGraphs,
                                  const StorageT *EdgeU,
                                  const StorageT *EdgeV,
                                  uint32_t *Degree,
                                  uint32_t *XorEdge,
                                  uint32_t *EdgePeeled,
                                  StorageT *OwnerVertex,
                                  StorageT *PeelOrder,
                                  uint32_t *PeeledCount,
                                  FrontierItemT<StorageT> *Frontier,
                                  uint32_t *FrontierCount,
                                  uint32_t *VerifyFailures,
                                  uint32_t *GraphRounds)
{
    const uint32_t Graph = blockIdx.x;

    if (Graph >= Batch) {
        return;
    }

    const uint32_t Thread = threadIdx.x;
    const uint32_t VertexBase = Graph * Vertices;
    const uint32_t EdgeBase = Graph * Edges;
    FrontierItemT<StorageT> *GraphFrontier = Frontier + static_cast<size_t>(Graph) * Vertices;
    __shared__ uint32_t SharedFrontierCount;
    uint32_t Rounds = 0;

    if (InvalidGraphs[Graph] != 0) {
        if (Thread == 0) {
            VerifyFailures[Graph] = 1;
            GraphRounds[Graph] = 0;
        }
        return;
    }

    for (;;) {
        if (Thread == 0) {
            FrontierCount[Graph] = 0;
        }
        __syncthreads();

        for (uint32_t Vertex = Thread; Vertex < Vertices; Vertex += blockDim.x) {
            if (Degree[VertexBase + Vertex] != 1) {
                continue;
            }

            const uint32_t Edge = XorEdge[VertexBase + Vertex];
            const uint32_t EdgeIndex = EdgeBase + Edge;

            if (atomicCAS(&EdgePeeled[EdgeIndex], 0u, 1u) != 0u) {
                continue;
            }

            const uint32_t Slot = atomicAdd(&FrontierCount[Graph], 1u);
            GraphFrontier[Slot].Graph = Graph;
            GraphFrontier[Slot].Vertex = static_cast<StorageT>(Vertex);
            GraphFrontier[Slot].Edge = static_cast<StorageT>(Edge);
        }

        __syncthreads();

        if (Thread == 0) {
            SharedFrontierCount = FrontierCount[Graph];
        }
        __syncthreads();

        if (SharedFrontierCount == 0) {
            break;
        }

        for (uint32_t Index = Thread; Index < SharedFrontierCount; Index += blockDim.x) {
            const FrontierItemT<StorageT> Item = GraphFrontier[Index];
            const uint32_t Edge = static_cast<uint32_t>(Item.Edge);
            const uint32_t EdgeIndex = EdgeBase + Edge;
            const uint32_t Order = atomicAdd(&PeeledCount[Graph], 1u);
            const uint32_t U = static_cast<uint32_t>(EdgeU[EdgeIndex]);
            const uint32_t V = static_cast<uint32_t>(EdgeV[EdgeIndex]);

            OwnerVertex[EdgeIndex] = Item.Vertex;
            PeelOrder[EdgeBase + Order] = static_cast<StorageT>(Edge);

            atomicSub(&Degree[VertexBase + U], 1u);
            atomicXor(&XorEdge[VertexBase + U], Edge);
            atomicSub(&Degree[VertexBase + V], 1u);
            atomicXor(&XorEdge[VertexBase + V], Edge);
        }

        __syncthreads();

        if (Thread == 0) {
            ++Rounds;
        }
        __syncthreads();
    }

    if (Thread == 0) {
        GraphRounds[Graph] = Rounds;
        if (PeeledCount[Graph] != Edges) {
            VerifyFailures[Graph] = 1;
        }
    }
}

template<HashFunctionKind Kind, typename StorageT>
CpuResult
RunCpuAssignmentAndVerify(uint32_t GraphBase,
                          uint32_t Edges,
                          uint32_t Vertices,
                          uint32_t EdgeMask,
                          const std::vector<StorageT> &EdgeU,
                          const std::vector<StorageT> &EdgeV,
                          const std::vector<StorageT> &Owner,
                          const std::vector<StorageT> &Order)
{
    CpuResult Result;
    std::vector<StorageT> Assigned(Vertices, 0);
    const uint32_t EdgeBase = GraphBase * Edges;

    auto MeasureStage = [&](auto &&Stage) -> double {
        const auto Start = std::chrono::steady_clock::now();
        Stage();
        const auto Stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(Stop - Start).count();
    };

    Result.Success = true;

    Result.AssignMilliseconds = MeasureStage([&]() {
        for (int64_t Index = static_cast<int64_t>(Edges) - 1; Index >= 0; --Index) {
            uint32_t Edge = static_cast<uint32_t>(Order[EdgeBase + static_cast<uint32_t>(Index)]);
            uint32_t OwnerVertex = static_cast<uint32_t>(Owner[EdgeBase + Edge]);
            uint32_t U = static_cast<uint32_t>(EdgeU[EdgeBase + Edge]);
            uint32_t V = static_cast<uint32_t>(EdgeV[EdgeBase + Edge]);
            uint32_t Other = (OwnerVertex == U) ? V : U;
            uint32_t Value = (Edge - static_cast<uint32_t>(Assigned[Other])) & EdgeMask;

            Assigned[OwnerVertex] = static_cast<StorageT>(Value);
        }
    });

    Result.Verified = true;

    Result.VerifyMilliseconds = MeasureStage([&]() {
        for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
            uint32_t Index = (
                static_cast<uint32_t>(Assigned[static_cast<uint32_t>(EdgeU[EdgeBase + Edge])]) +
                static_cast<uint32_t>(Assigned[static_cast<uint32_t>(EdgeV[EdgeBase + Edge])])
            ) & EdgeMask;
            if (Index != Edge) {
                Result.Verified = false;
                break;
            }
        }
    });

    return Result;
}

template<HashFunctionKind Kind, typename StorageT>
CpuResult
RunCpuReference(uint32_t Graph,
                uint32_t Edges,
                uint32_t Vertices,
                uint32_t EdgeMask,
                const std::vector<uint32_t> &Keys,
                const std::vector<GraphSeeds> &GraphSeedsVector)
{
    CpuResult Result;
    std::vector<uint32_t> Degree(Vertices, 0);
    std::vector<uint32_t> XorEdge(Vertices, 0);
    std::vector<StorageT> EdgeU(Edges, 0);
    std::vector<StorageT> EdgeV(Edges, 0);
    std::vector<StorageT> Owner(Edges, MaxValue<StorageT>());
    std::vector<StorageT> Order;
    std::vector<StorageT> Assigned(Vertices, 0);
    std::vector<uint8_t> Peeled(Edges, 0);
    std::vector<uint32_t> Queue;
    const GraphSeeds Seeds = GraphSeedsVector[Graph];

    Order.reserve(Edges);
    Queue.reserve(Vertices);

    auto MeasureStage = [&](auto &&Stage) -> double {
        const auto Start = std::chrono::steady_clock::now();
        Stage();
        const auto Stop = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(Stop - Start).count();
    };

    Result.AddBuildMilliseconds = MeasureStage([&]() {
        for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
            const auto Hash = HashImpl<Kind, StorageT>::Apply(Keys[Edge], Seeds, Vertices - 1);
            const uint32_t U = static_cast<uint32_t>(Hash.Vertex1);
            const uint32_t V = static_cast<uint32_t>(Hash.Vertex2);

            EdgeU[Edge] = Hash.Vertex1;
            EdgeV[Edge] = Hash.Vertex2;

            if (U == V || U >= Vertices || V >= Vertices) {
                Result.Invalid = true;
                break;
            }

            ++Degree[U];
            ++Degree[V];
            XorEdge[U] ^= Edge;
            XorEdge[V] ^= Edge;
        }
    });

    if (Result.Invalid) {
        return Result;
    }

    Result.PeelMilliseconds = MeasureStage([&]() {
        for (uint32_t Vertex = 0; Vertex < Vertices; ++Vertex) {
            if (Degree[Vertex] == 1) {
                Queue.push_back(Vertex);
            }
        }

        for (size_t Head = 0; Head < Queue.size(); ++Head) {
            uint32_t Vertex = Queue[Head];

            if (Degree[Vertex] != 1) {
                continue;
            }

            uint32_t Edge = XorEdge[Vertex];
            if (Peeled[Edge] != 0) {
                continue;
            }

            Peeled[Edge] = 1;
            Owner[Edge] = static_cast<StorageT>(Vertex);
            Order.push_back(static_cast<StorageT>(Edge));

            uint32_t U = static_cast<uint32_t>(EdgeU[Edge]);
            uint32_t V = static_cast<uint32_t>(EdgeV[Edge]);

            if (Degree[U] > 0) {
                --Degree[U];
                XorEdge[U] ^= Edge;
                if (Degree[U] == 1) {
                    Queue.push_back(U);
                }
            }

            if (Degree[V] > 0) {
                --Degree[V];
                XorEdge[V] ^= Edge;
                if (Degree[V] == 1) {
                    Queue.push_back(V);
                }
            }
        }
    });

    Result.Peeled = static_cast<uint32_t>(Order.size());
    Result.Success = (Order.size() == Edges);
    if (!Result.Success) {
        return Result;
    }

    CpuResult AssignmentResult = RunCpuAssignmentAndVerify<Kind, StorageT>(0,
                                                                            Edges,
                                                                            Vertices,
                                                                            EdgeMask,
                                                                            EdgeU,
                                                                            EdgeV,
                                                                            Owner,
                                                                            Order);
    Result.AssignMilliseconds = AssignmentResult.AssignMilliseconds;
    Result.VerifyMilliseconds = AssignmentResult.VerifyMilliseconds;
    Result.Verified = AssignmentResult.Verified;

    return Result;
}

template<HashFunctionKind Kind, typename StorageT>
ExperimentResult
RunExperiment(const Options &Opts,
              const std::vector<uint32_t> &Keys,
              const std::vector<GraphSeeds> &GraphSeedsVector,
              const std::string &RequestedKeys,
              const std::string &DatasetName,
              StorageMode SelectedStorage)
{
    using FrontierItem = FrontierItemT<StorageT>;

    ExperimentResult Result;

    const uint32_t Batch = Opts.Batch;
    const uint32_t KeyCount = static_cast<uint32_t>(Keys.size());
    const uint32_t EdgeCapacity = NextPowerOfTwo(KeyCount);
    const uint32_t Vertices = NextPowerOfTwo(EdgeCapacity + 1);
    const uint32_t EdgeMask = EdgeCapacity - 1;

    Result.KeyCount = KeyCount;
    Result.EdgeCapacity = EdgeCapacity;
    Result.Vertices = Vertices;
    Result.Batch = Batch;
    Result.DatasetName = DatasetName;
    Result.StorageBits = static_cast<uint32_t>(sizeof(StorageT) * 8);
    Result.HashFunctionName = HashTraits<Kind>::Name;
    Result.SelectedStorage = SelectedStorage;
    Result.SolveModeName = SolveModeToString(Opts.Solve);
    Result.AssignmentBackendName = AssignmentBackendToString(Opts.AssignmentBackendMode);
    Result.AllocationModeName = AllocationModeToString(Opts.Allocation);
    Result.RequestedFixedAttempts = Opts.FixedAttempts;
    Result.FirstSolutionWins = Opts.FirstSolutionWins;

    const uint64_t TotalEdges = static_cast<uint64_t>(KeyCount) * Batch;
    const uint64_t TotalVertices = static_cast<uint64_t>(Vertices) * Batch;
    const uint64_t FrontierCapacity = TotalVertices;
    const bool NeedsFrontier = (
        Opts.Solve == SolveMode::HostRoundTrip ||
        (Opts.Solve == SolveMode::DeviceSerial &&
         Opts.DeviceSerialPeelGeometry != GraphGeometry::Thread));

    uint32_t *DKeys = nullptr;
    GraphSeeds *DGraphSeeds = nullptr;
    StorageT *DEdgeU = nullptr;
    StorageT *DEdgeV = nullptr;
    uint32_t *DDegree = nullptr;
    uint32_t *DXorEdge = nullptr;
    uint32_t *DInvalidGraphs = nullptr;
    uint32_t *DEdgePeeled = nullptr;
    StorageT *DOwnerVertex = nullptr;
    StorageT *DPeelOrder = nullptr;
    uint32_t *DPeeledCount = nullptr;
    StorageT *DAssigned = nullptr;
    uint32_t *DVerifyFailures = nullptr;
    FrontierItem *DFrontier = nullptr;
    uint32_t *DFrontierCount = nullptr;
    uint32_t *DGraphRounds = nullptr;

    AllocateGpuMemory(&DKeys, Keys.size(), Opts.Allocation, "cudaMalloc(DKeys)");
    AllocateGpuMemory(&DGraphSeeds,
                      GraphSeedsVector.size(),
                      Opts.Allocation,
                      "cudaMalloc(DGraphSeeds)");
    AllocateGpuMemory(&DEdgeU, TotalEdges, Opts.Allocation, "cudaMalloc(DEdgeU)");
    AllocateGpuMemory(&DEdgeV, TotalEdges, Opts.Allocation, "cudaMalloc(DEdgeV)");
    AllocateGpuMemory(&DDegree, TotalVertices, Opts.Allocation, "cudaMalloc(DDegree)");
    AllocateGpuMemory(&DXorEdge, TotalVertices, Opts.Allocation, "cudaMalloc(DXorEdge)");
    AllocateGpuMemory(&DInvalidGraphs, Batch, Opts.Allocation, "cudaMalloc(DInvalidGraphs)");
    AllocateGpuMemory(&DEdgePeeled, TotalEdges, Opts.Allocation, "cudaMalloc(DEdgePeeled)");
    AllocateGpuMemory(&DOwnerVertex, TotalEdges, Opts.Allocation, "cudaMalloc(DOwnerVertex)");
    AllocateGpuMemory(&DPeelOrder, TotalEdges, Opts.Allocation, "cudaMalloc(DPeelOrder)");
    AllocateGpuMemory(&DPeeledCount, Batch, Opts.Allocation, "cudaMalloc(DPeeledCount)");
    AllocateGpuMemory(&DAssigned, TotalVertices, Opts.Allocation, "cudaMalloc(DAssigned)");
    AllocateGpuMemory(&DVerifyFailures, Batch, Opts.Allocation, "cudaMalloc(DVerifyFailures)");
    AllocateGpuMemory(&DGraphRounds, Batch, Opts.Allocation, "cudaMalloc(DGraphRounds)");

    if (NeedsFrontier) {
        AllocateGpuMemory(&DFrontier, FrontierCapacity, Opts.Allocation, "cudaMalloc(DFrontier)");
        AllocateGpuMemory(&DFrontierCount, Batch, Opts.Allocation, "cudaMalloc(DFrontierCount)");
    }

    CheckCuda(cudaMemcpy(DKeys, Keys.data(), Keys.size() * sizeof(Keys[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DKeys)");
    std::ofstream SolvedSeedsFile;
    if (!Opts.DumpSolvedSeedsFile.empty()) {
        SolvedSeedsFile.open(Opts.DumpSolvedSeedsFile, std::ios::trunc);
        if (!SolvedSeedsFile) {
            std::cerr << "Failed to open solved seeds output file: "
                      << Opts.DumpSolvedSeedsFile
                      << "\n";
            std::exit(EXIT_FAILURE);
        }

        SolvedSeedsFile
            << "GraphIndex,HashFunction,Seed1,Seed2,Seed3,Seed4,Seed5,Seed6,Seed7,Seed8\n";
    }

    cudaEvent_t StageStart = nullptr;
    cudaEvent_t StageStop = nullptr;
    CheckCuda(cudaEventCreate(&StageStart), "cudaEventCreate(StageStart)");
    CheckCuda(cudaEventCreate(&StageStop), "cudaEventCreate(StageStop)");

    const uint32_t BuildBlocks = static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads);
    const uint32_t VertexBlocks = static_cast<uint32_t>((TotalVertices + Opts.Threads - 1) / Opts.Threads);
    const uint32_t VerifyBlocks = static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads);
    const bool UseController = (Opts.FixedAttempts > 0);
    const bool UseCpuAssignmentBackend = (Opts.AssignmentBackendMode == AssignmentBackend::Cpu);

    auto MeasureStage = [&](auto &&Launch) -> double {
        float StageMilliseconds = 0.0f;
        CheckCuda(cudaEventRecord(StageStart), "cudaEventRecord(StageStart)");
        Launch();
        CheckCuda(cudaEventRecord(StageStop), "cudaEventRecord(StageStop)");
        CheckCuda(cudaEventSynchronize(StageStop), "cudaEventSynchronize(StageStop)");
        CheckCuda(cudaEventElapsedTime(&StageMilliseconds, StageStart, StageStop), "cudaEventElapsedTime");
        return static_cast<double>(StageMilliseconds);
    };

    auto RunOneBatch = [&](const std::vector<GraphSeeds> &CurrentGraphSeedsVector,
                           uint64_t AttemptBase) -> BatchOutcome {
        BatchOutcome Outcome = {};

        CheckCuda(cudaMemcpy(DGraphSeeds,
                             CurrentGraphSeedsVector.data(),
                             CurrentGraphSeedsVector.size() * sizeof(CurrentGraphSeedsVector[0]),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(DGraphSeeds)");

        CheckCuda(cudaMemset(DEdgeU, 0, TotalEdges * sizeof(StorageT)), "cudaMemset(DEdgeU)");
        CheckCuda(cudaMemset(DEdgeV, 0, TotalEdges * sizeof(StorageT)), "cudaMemset(DEdgeV)");
        CheckCuda(cudaMemset(DDegree, 0, TotalVertices * sizeof(uint32_t)), "cudaMemset(DDegree)");
        CheckCuda(cudaMemset(DXorEdge, 0, TotalVertices * sizeof(uint32_t)), "cudaMemset(DXorEdge)");
        CheckCuda(cudaMemset(DInvalidGraphs, 0, Batch * sizeof(uint32_t)), "cudaMemset(DInvalidGraphs)");
        CheckCuda(cudaMemset(DEdgePeeled, 0, TotalEdges * sizeof(uint32_t)), "cudaMemset(DEdgePeeled)");
        CheckCuda(cudaMemset(DOwnerVertex, 0xff, TotalEdges * sizeof(StorageT)), "cudaMemset(DOwnerVertex)");
        CheckCuda(cudaMemset(DPeelOrder, 0xff, TotalEdges * sizeof(StorageT)), "cudaMemset(DPeelOrder)");
        CheckCuda(cudaMemset(DPeeledCount, 0, Batch * sizeof(uint32_t)), "cudaMemset(DPeeledCount)");
        CheckCuda(cudaMemset(DAssigned, 0, TotalVertices * sizeof(StorageT)), "cudaMemset(DAssigned)");
        CheckCuda(cudaMemset(DVerifyFailures, 0, Batch * sizeof(uint32_t)), "cudaMemset(DVerifyFailures)");
        CheckCuda(cudaMemset(DGraphRounds, 0, Batch * sizeof(uint32_t)), "cudaMemset(DGraphRounds)");

        if (NeedsFrontier) {
            CheckCuda(cudaMemset(DFrontierCount, 0, Batch * sizeof(uint32_t)), "cudaMemset(DFrontierCount)");
        }

        auto GpuStart = std::chrono::steady_clock::now();

        if (Opts.Allocation == AllocationMode::ManagedPrefetchGpu) {
            int Device = 0;
            CheckCuda(cudaGetDevice(&Device), "cudaGetDevice");

            PrefetchGpuMemory(DKeys, Keys.size(), Device, "cudaMemPrefetchAsync(DKeys)");
            PrefetchGpuMemory(DGraphSeeds, CurrentGraphSeedsVector.size(), Device, "cudaMemPrefetchAsync(DGraphSeeds)");
            PrefetchGpuMemory(DEdgeU, TotalEdges, Device, "cudaMemPrefetchAsync(DEdgeU)");
            PrefetchGpuMemory(DEdgeV, TotalEdges, Device, "cudaMemPrefetchAsync(DEdgeV)");
            PrefetchGpuMemory(DDegree, TotalVertices, Device, "cudaMemPrefetchAsync(DDegree)");
            PrefetchGpuMemory(DXorEdge, TotalVertices, Device, "cudaMemPrefetchAsync(DXorEdge)");
            PrefetchGpuMemory(DInvalidGraphs, Batch, Device, "cudaMemPrefetchAsync(DInvalidGraphs)");
            PrefetchGpuMemory(DEdgePeeled, TotalEdges, Device, "cudaMemPrefetchAsync(DEdgePeeled)");
            PrefetchGpuMemory(DOwnerVertex, TotalEdges, Device, "cudaMemPrefetchAsync(DOwnerVertex)");
            PrefetchGpuMemory(DPeelOrder, TotalEdges, Device, "cudaMemPrefetchAsync(DPeelOrder)");
            PrefetchGpuMemory(DPeeledCount, Batch, Device, "cudaMemPrefetchAsync(DPeeledCount)");
            PrefetchGpuMemory(DAssigned, TotalVertices, Device, "cudaMemPrefetchAsync(DAssigned)");
            PrefetchGpuMemory(DVerifyFailures, Batch, Device, "cudaMemPrefetchAsync(DVerifyFailures)");
            PrefetchGpuMemory(DGraphRounds, Batch, Device, "cudaMemPrefetchAsync(DGraphRounds)");
            if (NeedsFrontier) {
                PrefetchGpuMemory(DFrontier, FrontierCapacity, Device, "cudaMemPrefetchAsync(DFrontier)");
                PrefetchGpuMemory(DFrontierCount, Batch, Device, "cudaMemPrefetchAsync(DFrontierCount)");
            }
            CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(prefetch)");
        }

        const uint32_t BatchSize = Batch;
        std::vector<uint32_t> InvalidGraphs(BatchSize);
        std::vector<uint32_t> PeeledCount(BatchSize);
        std::vector<uint32_t> VerifyFailures(BatchSize);
        std::vector<uint32_t> GraphRounds(BatchSize);
        std::vector<CpuResult> CpuResults(BatchSize);

        Outcome.AddBuildMilliseconds = MeasureStage([&]() {
            BuildGraphsKernel<Kind, StorageT><<<BuildBlocks, Opts.Threads>>>(KeyCount,
                                                                             Vertices,
                                                                             BatchSize,
                                                                             DKeys,
                                                                             DGraphSeeds,
                                                                             DEdgeU,
                                                                             DEdgeV,
                                                                             DDegree,
                                                                             DXorEdge,
                                                                             DInvalidGraphs);
            CheckCuda(cudaGetLastError(), "BuildGraphsKernel launch");
        });

        if (Opts.Solve == SolveMode::HostRoundTrip) {
            uint32_t FrontierCount = 0;

            for (;;) {
                CheckCuda(cudaMemset(DFrontierCount, 0, BatchSize * sizeof(uint32_t)), "cudaMemset(DFrontierCount)");

                Outcome.PeelMilliseconds += MeasureStage([&]() {
                    CollectFrontierKernel<StorageT><<<VertexBlocks, Opts.Threads>>>(Vertices,
                                                                                    BatchSize,
                                                                                    DDegree,
                                                                                    DXorEdge,
                                                                                    DInvalidGraphs,
                                                                                    DFrontier,
                                                                                    DFrontierCount);
                    CheckCuda(cudaGetLastError(), "CollectFrontierKernel launch");
                });

                CheckCuda(cudaMemcpy(&FrontierCount, DFrontierCount, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                          "cudaMemcpy(FrontierCount)");

                if (FrontierCount == 0) {
                    break;
                }

                ++Outcome.Rounds;

                const uint32_t PeelBlocks = static_cast<uint32_t>((FrontierCount + Opts.Threads - 1) / Opts.Threads);
                Outcome.PeelMilliseconds += MeasureStage([&]() {
                    PeelFrontierKernel<StorageT><<<PeelBlocks, Opts.Threads>>>(KeyCount,
                                                                               Vertices,
                                                                               FrontierCount,
                                                                               DFrontier,
                                                                               DEdgeU,
                                                                               DEdgeV,
                                                                               DDegree,
                                                                               DXorEdge,
                                                                               DEdgePeeled,
                                                                               DOwnerVertex,
                                                                               DPeelOrder,
                                                                               DPeeledCount);
                    CheckCuda(cudaGetLastError(), "PeelFrontierKernel launch");
                });
            }

            if (!UseCpuAssignmentBackend) {
                Outcome.AssignMilliseconds = MeasureStage([&]() {
                    AssignGraphsKernel<StorageT><<<BatchSize, 1>>>(KeyCount,
                                                                   Vertices,
                                                                   EdgeMask,
                                                                   DInvalidGraphs,
                                                                   DEdgeU,
                                                                   DEdgeV,
                                                                   DOwnerVertex,
                                                                   DPeelOrder,
                                                                   DPeeledCount,
                                                                   DAssigned);
                    CheckCuda(cudaGetLastError(), "AssignGraphsKernel launch");
                });

                Outcome.VerifyMilliseconds = MeasureStage([&]() {
                    VerifyGraphsKernel<StorageT><<<VerifyBlocks, Opts.Threads>>>(KeyCount,
                                                                                 Vertices,
                                                                                 BatchSize,
                                                                                 EdgeMask,
                                                                                 DInvalidGraphs,
                                                                                 DEdgeU,
                                                                                 DEdgeV,
                                                                                 DAssigned,
                                                                                 DPeeledCount,
                                                                                 DVerifyFailures);
                    CheckCuda(cudaGetLastError(), "VerifyGraphsKernel launch");
                });
            }
        } else {
            if (Opts.DeviceSerialPeelGeometry == GraphGeometry::Thread) {
                Outcome.PeelMilliseconds = MeasureStage([&]() {
                    PeelGraphsDeviceSerialThreadKernel<StorageT><<<BatchSize, 1>>>(KeyCount,
                                                                                   Vertices,
                                                                                   BatchSize,
                                                                                   DInvalidGraphs,
                                                                                   DEdgeU,
                                                                                   DEdgeV,
                                                                                   DDegree,
                                                                                   DXorEdge,
                                                                                   DEdgePeeled,
                                                                                   DOwnerVertex,
                                                                                   DPeelOrder,
                                                                                   DPeeledCount,
                                                                                   DVerifyFailures,
                                                                                   DGraphRounds);
                    CheckCuda(cudaGetLastError(), "PeelGraphsDeviceSerialThreadKernel launch");
                });
            } else if (Opts.DeviceSerialPeelGeometry == GraphGeometry::Warp) {
                const uint32_t WarpsPerBlock = Opts.Threads / 32u;
                const uint32_t PeelBlocks = static_cast<uint32_t>((BatchSize + WarpsPerBlock - 1u) / WarpsPerBlock);
                Outcome.PeelMilliseconds = MeasureStage([&]() {
                    PeelGraphsDeviceSerialWarpKernel<StorageT><<<PeelBlocks, Opts.Threads>>>(KeyCount,
                                                                                            Vertices,
                                                                                            BatchSize,
                                                                                            DInvalidGraphs,
                                                                                            DEdgeU,
                                                                                            DEdgeV,
                                                                                            DDegree,
                                                                                            DXorEdge,
                                                                                            DEdgePeeled,
                                                                                            DOwnerVertex,
                                                                                            DPeelOrder,
                                                                                            DPeeledCount,
                                                                                            DFrontier,
                                                                                            DFrontierCount,
                                                                                            DVerifyFailures,
                                                                                            DGraphRounds);
                    CheckCuda(cudaGetLastError(), "PeelGraphsDeviceSerialWarpKernel launch");
                });
            } else {
                Outcome.PeelMilliseconds = MeasureStage([&]() {
                    PeelGraphsDeviceSerialBlockKernel<StorageT><<<BatchSize, Opts.Threads>>>(KeyCount,
                                                                                             Vertices,
                                                                                             BatchSize,
                                                                                             DInvalidGraphs,
                                                                                             DEdgeU,
                                                                                             DEdgeV,
                                                                                             DDegree,
                                                                                             DXorEdge,
                                                                                             DEdgePeeled,
                                                                                             DOwnerVertex,
                                                                                             DPeelOrder,
                                                                                             DPeeledCount,
                                                                                             DFrontier,
                                                                                             DFrontierCount,
                                                                                             DVerifyFailures,
                                                                                             DGraphRounds);
                    CheckCuda(cudaGetLastError(), "PeelGraphsDeviceSerialBlockKernel launch");
                });
            }

            if (!UseCpuAssignmentBackend) {
                Outcome.AssignMilliseconds = MeasureStage([&]() {
                    AssignGraphsKernel<StorageT><<<BatchSize, 1>>>(KeyCount,
                                                                   Vertices,
                                                                   EdgeMask,
                                                                   DInvalidGraphs,
                                                                   DEdgeU,
                                                                   DEdgeV,
                                                                   DOwnerVertex,
                                                                   DPeelOrder,
                                                                   DPeeledCount,
                                                                   DAssigned);
                    CheckCuda(cudaGetLastError(), "AssignGraphsKernel launch");
                });

                Outcome.VerifyMilliseconds = MeasureStage([&]() {
                    VerifyGraphsKernel<StorageT><<<VerifyBlocks, Opts.Threads>>>(KeyCount,
                                                                                 Vertices,
                                                                                 BatchSize,
                                                                                 EdgeMask,
                                                                                 DInvalidGraphs,
                                                                                 DEdgeU,
                                                                                 DEdgeV,
                                                                                 DAssigned,
                                                                                 DPeeledCount,
                                                                                 DVerifyFailures);
                    CheckCuda(cudaGetLastError(), "VerifyGraphsKernel launch");
                });
            }
        }

        CheckCuda(cudaMemcpy(InvalidGraphs.data(), DInvalidGraphs, BatchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(InvalidGraphs)");
        CheckCuda(cudaMemcpy(PeeledCount.data(), DPeeledCount, BatchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(PeeledCount)");
        CheckCuda(cudaMemcpy(VerifyFailures.data(), DVerifyFailures, BatchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(VerifyFailures)");
        CheckCuda(cudaMemcpy(GraphRounds.data(), DGraphRounds, BatchSize * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(GraphRounds)");

        if (Opts.Solve == SolveMode::DeviceSerial) {
            uint32_t MaxRounds = 0;
            for (uint32_t Round : GraphRounds) {
                MaxRounds = std::max(MaxRounds, Round);
            }
            Outcome.Rounds = MaxRounds;
        }

        auto GpuStop = std::chrono::steady_clock::now();
        Outcome.GpuMilliseconds =
            std::chrono::duration<double, std::milli>(GpuStop - GpuStart).count();

        std::vector<StorageT> HostEdgeU;
        std::vector<StorageT> HostEdgeV;
        std::vector<StorageT> HostOwnerVertex;
        std::vector<StorageT> HostPeelOrder;

        if (UseCpuAssignmentBackend) {
            HostEdgeU.resize(TotalEdges);
            HostEdgeV.resize(TotalEdges);
            HostOwnerVertex.resize(TotalEdges);
            HostPeelOrder.resize(TotalEdges);

            CheckCuda(cudaMemcpy(HostEdgeU.data(),
                                 DEdgeU,
                                 TotalEdges * sizeof(StorageT),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(HostEdgeU)");
            CheckCuda(cudaMemcpy(HostEdgeV.data(),
                                 DEdgeV,
                                 TotalEdges * sizeof(StorageT),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(HostEdgeV)");
            CheckCuda(cudaMemcpy(HostOwnerVertex.data(),
                                 DOwnerVertex,
                                 TotalEdges * sizeof(StorageT),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(HostOwnerVertex)");
            CheckCuda(cudaMemcpy(HostPeelOrder.data(),
                                 DPeelOrder,
                                 TotalEdges * sizeof(StorageT),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(HostPeelOrder)");
        }

        auto CpuStart = std::chrono::steady_clock::now();
        CpuStageTimings CpuAllAttempts = {};
        CpuStageTimings CpuSolvedOnly = {};

        if (UseCpuAssignmentBackend) {
            for (uint32_t Graph = 0; Graph < BatchSize; ++Graph) {
                const bool ThisGpuSuccess = (
                    InvalidGraphs[Graph] == 0 &&
                    PeeledCount[Graph] == KeyCount &&
                    VerifyFailures[Graph] == 0
                );

                CpuResult CpuAssignmentResult = {};
                if (ThisGpuSuccess) {
                    CpuAssignmentResult = RunCpuAssignmentAndVerify<Kind, StorageT>(Graph,
                                                                                     KeyCount,
                                                                                     Vertices,
                                                                                     EdgeMask,
                                                                                     HostEdgeU,
                                                                                     HostEdgeV,
                                                                                     HostOwnerVertex,
                                                                                     HostPeelOrder);
                    CpuAllAttempts.AssignMilliseconds += CpuAssignmentResult.AssignMilliseconds;
                    CpuAllAttempts.VerifyMilliseconds += CpuAssignmentResult.VerifyMilliseconds;
                    if (CpuAssignmentResult.Success && CpuAssignmentResult.Verified) {
                        CpuSolvedOnly.AssignMilliseconds += CpuAssignmentResult.AssignMilliseconds;
                        CpuSolvedOnly.VerifyMilliseconds += CpuAssignmentResult.VerifyMilliseconds;
                    }
                }

                const bool ThisCpuSuccess = (ThisGpuSuccess &&
                                             CpuAssignmentResult.Success &&
                                             CpuAssignmentResult.Verified);

                if (ThisGpuSuccess) {
                    ++Outcome.GpuSuccess;
                    Outcome.GpuHadSuccess = true;
                    if (Result.FirstSolvedAttempt < 0) {
                        Result.FirstSolvedAttempt = static_cast<int64_t>(AttemptBase + Graph);
                    }
                    if (SolvedSeedsFile.is_open()) {
                        const GraphSeeds &Seeds = CurrentGraphSeedsVector[Graph];
                        SolvedSeedsFile
                            << (AttemptBase + Graph)
                            << "," << HashTraits<Kind>::Name
                            << "," << Seeds.Seeds[0]
                            << "," << Seeds.Seeds[1]
                            << "," << Seeds.Seeds[2]
                            << "," << Seeds.Seeds[3]
                            << "," << Seeds.Seeds[4]
                            << "," << Seeds.Seeds[5]
                            << "," << Seeds.Seeds[6]
                            << "," << Seeds.Seeds[7]
                            << "\n";
                    }
                }

                if (ThisCpuSuccess) {
                    ++Outcome.CpuSuccess;
                }

                if (ThisGpuSuccess != ThisCpuSuccess) {
                    ++Outcome.Mismatches;
                    if (Opts.Verbose) {
                        std::ostream &Details = (Opts.Output == OutputFormat::Json) ? std::cerr : std::cout;
                        Details << "Mismatch graph=" << (AttemptBase + Graph)
                                << " gpu_success=" << ThisGpuSuccess
                                << " cpu_success=" << ThisCpuSuccess
                                << " invalid=" << InvalidGraphs[Graph]
                                << " peeled_gpu=" << PeeledCount[Graph]
                                << " peeled_cpu=" << KeyCount
                                << " verify_failures=" << VerifyFailures[Graph]
                                << "\n";
                    }
                }

                if (ThisGpuSuccess && !CpuAssignmentResult.Verified) {
                    ++Outcome.CpuVerifyIssues;
                }
            }
        } else {
            std::vector<CpuResult> CpuResults(BatchSize);

            for (uint32_t Graph = 0; Graph < BatchSize; ++Graph) {
                CpuResults[Graph] = RunCpuReference<Kind, StorageT>(Graph,
                                                                    KeyCount,
                                                                    Vertices,
                                                                    EdgeMask,
                                                                    Keys,
                                                                    CurrentGraphSeedsVector);
                CpuAllAttempts.AddBuildMilliseconds += CpuResults[Graph].AddBuildMilliseconds;
                CpuAllAttempts.PeelMilliseconds += CpuResults[Graph].PeelMilliseconds;
                CpuAllAttempts.AssignMilliseconds += CpuResults[Graph].AssignMilliseconds;
                CpuAllAttempts.VerifyMilliseconds += CpuResults[Graph].VerifyMilliseconds;
            }

            for (uint32_t Graph = 0; Graph < BatchSize; ++Graph) {
                const bool ThisGpuSuccess = (
                    InvalidGraphs[Graph] == 0 &&
                    PeeledCount[Graph] == KeyCount &&
                    VerifyFailures[Graph] == 0
                );
                const bool ThisCpuSuccess = (CpuResults[Graph].Success && CpuResults[Graph].Verified);

                if (ThisGpuSuccess) {
                    ++Outcome.GpuSuccess;
                    Outcome.GpuHadSuccess = true;
                    if (Result.FirstSolvedAttempt < 0) {
                        Result.FirstSolvedAttempt = static_cast<int64_t>(AttemptBase + Graph);
                    }
                }
                if (ThisCpuSuccess) {
                    ++Outcome.CpuSuccess;
                    CpuSolvedOnly.AddBuildMilliseconds += CpuResults[Graph].AddBuildMilliseconds;
                    CpuSolvedOnly.PeelMilliseconds += CpuResults[Graph].PeelMilliseconds;
                    CpuSolvedOnly.AssignMilliseconds += CpuResults[Graph].AssignMilliseconds;
                    CpuSolvedOnly.VerifyMilliseconds += CpuResults[Graph].VerifyMilliseconds;
                }
                if (ThisGpuSuccess && SolvedSeedsFile.is_open()) {
                    const GraphSeeds &Seeds = CurrentGraphSeedsVector[Graph];
                    SolvedSeedsFile
                        << (AttemptBase + Graph)
                        << "," << HashTraits<Kind>::Name
                        << "," << Seeds.Seeds[0]
                        << "," << Seeds.Seeds[1]
                        << "," << Seeds.Seeds[2]
                        << "," << Seeds.Seeds[3]
                        << "," << Seeds.Seeds[4]
                        << "," << Seeds.Seeds[5]
                        << "," << Seeds.Seeds[6]
                        << "," << Seeds.Seeds[7]
                        << "\n";
                }
                if (ThisGpuSuccess != ThisCpuSuccess) {
                    ++Outcome.Mismatches;
                    if (Opts.Verbose) {
                        std::ostream &Details = (Opts.Output == OutputFormat::Json) ? std::cerr : std::cout;
                        Details << "Mismatch graph=" << (AttemptBase + Graph)
                                << " gpu_success=" << ThisGpuSuccess
                                << " cpu_success=" << ThisCpuSuccess
                                << " invalid=" << InvalidGraphs[Graph]
                                << " peeled_gpu=" << PeeledCount[Graph]
                                << " peeled_cpu=" << CpuResults[Graph].Peeled
                                << " verify_failures=" << VerifyFailures[Graph]
                                << "\n";
                    }
                }
                if (CpuResults[Graph].Success && !CpuResults[Graph].Verified) {
                    ++Outcome.CpuVerifyIssues;
                }
            }
        }

        auto CpuStop = std::chrono::steady_clock::now();
        Outcome.CpuMilliseconds =
            std::chrono::duration<double, std::milli>(CpuStop - CpuStart).count();

        Outcome.CpuAllAttempts = CpuAllAttempts;
        Outcome.CpuSolvedOnly = CpuSolvedOnly;
        return Outcome;
    };

    const uint64_t BatchAttempts = static_cast<uint64_t>(Batch);
    uint64_t AttemptBase = 0;

    for (;;) {
        std::vector<GraphSeeds> CurrentGraphSeedsVector = MakeGraphSeedsVector<Kind>(Batch,
                                                                                     Opts.GraphSeed,
                                                                                     Opts.RngSubsequence,
                                                                                     Opts.RngOffset,
                                                                                     Vertices,
                                                                                     Opts.SeedsFile,
                                                                                     AttemptBase);
        BatchOutcome Outcome = RunOneBatch(CurrentGraphSeedsVector, AttemptBase);

        Result.GpuSuccess += Outcome.GpuSuccess;
        Result.CpuSuccess += Outcome.CpuSuccess;
        Result.Mismatches += Outcome.Mismatches;
        Result.CpuVerifyIssues += Outcome.CpuVerifyIssues;
        Result.Rounds += Outcome.Rounds;
        Result.GpuMilliseconds += Outcome.GpuMilliseconds;
        Result.CpuMilliseconds += Outcome.CpuMilliseconds;
        Result.CpuStageTimingsMs.AllAttempts.AddBuildMilliseconds += Outcome.CpuAllAttempts.AddBuildMilliseconds;
        Result.CpuStageTimingsMs.AllAttempts.PeelMilliseconds += Outcome.CpuAllAttempts.PeelMilliseconds;
        Result.CpuStageTimingsMs.AllAttempts.AssignMilliseconds += Outcome.CpuAllAttempts.AssignMilliseconds;
        Result.CpuStageTimingsMs.AllAttempts.VerifyMilliseconds += Outcome.CpuAllAttempts.VerifyMilliseconds;
        Result.CpuStageTimingsMs.SolvedOnly.AddBuildMilliseconds += Outcome.CpuSolvedOnly.AddBuildMilliseconds;
        Result.CpuStageTimingsMs.SolvedOnly.PeelMilliseconds += Outcome.CpuSolvedOnly.PeelMilliseconds;
        Result.CpuStageTimingsMs.SolvedOnly.AssignMilliseconds += Outcome.CpuSolvedOnly.AssignMilliseconds;
        Result.CpuStageTimingsMs.SolvedOnly.VerifyMilliseconds += Outcome.CpuSolvedOnly.VerifyMilliseconds;
        Result.AddBuildMilliseconds += Outcome.AddBuildMilliseconds;
        Result.PeelMilliseconds += Outcome.PeelMilliseconds;
        Result.AssignMilliseconds += Outcome.AssignMilliseconds;
        Result.VerifyMilliseconds += Outcome.VerifyMilliseconds;
        Result.ActualAttemptsTried += BatchAttempts;
        ++Result.BatchesRun;

        if (!UseController) {
            break;
        }

        if (Opts.FirstSolutionWins && Outcome.GpuHadSuccess) {
            break;
        }

        if (Result.ActualAttemptsTried >= Opts.FixedAttempts) {
            break;
        }

        AttemptBase += BatchAttempts;
    }

    auto ControllerStop = std::chrono::steady_clock::now();
    (void)ControllerStop;

    if (Opts.Output == OutputFormat::Json) {
        std::ostringstream Json;
        Json << std::fixed << std::setprecision(3);
        Json << "{"
             << "\"dataset\":\"" << JsonEscape(Result.DatasetName) << "\","
             << "\"batch\":" << Result.Batch << ","
             << "\"requested_fixed_attempts\":" << Result.RequestedFixedAttempts << ","
             << "\"actual_attempts_tried\":" << Result.ActualAttemptsTried << ","
             << "\"batches_run\":" << Result.BatchesRun << ","
             << "\"first_solution_wins\":" << (Result.FirstSolutionWins ? "true" : "false") << ","
             << "\"first_solved_attempt\":" << Result.FirstSolvedAttempt << ","
             << "\"solve_mode\":\"" << JsonEscape(Result.SolveModeName) << "\","
             << "\"threads\":" << Opts.Threads << ","
             << "\"assign_geometry\":\"" << JsonEscape(GraphGeometryToString(Opts.AssignGeometry)) << "\","
             << "\"device_serial_peel_geometry\":\""
             << JsonEscape(GraphGeometryToString(Opts.DeviceSerialPeelGeometry)) << "\","
             << "\"storage_bits\":" << Result.StorageBits << ","
             << "\"storage_mode\":\"" << JsonEscape(StorageModeToString(Result.SelectedStorage)) << "\","
             << "\"hash_function\":\"" << JsonEscape(Result.HashFunctionName) << "\","
             << "\"assignment_backend\":\"" << JsonEscape(Result.AssignmentBackendName) << "\","
             << "\"allocation_mode\":\"" << JsonEscape(Result.AllocationModeName) << "\","
             << "\"output_format\":\"" << JsonEscape(OutputFormatToString(Opts.Output)) << "\","
             << "\"edge_capacity\":" << Result.EdgeCapacity << ","
             << "\"vertices\":" << Result.Vertices << ","
             << "\"gpu_ms\":" << Result.GpuMilliseconds << ","
             << "\"cpu_ms\":" << Result.CpuMilliseconds << ","
             << "\"cpu_stage_timings_ms_all_attempts\":{"
             << "\"add_build\":" << Result.CpuStageTimingsMs.AllAttempts.AddBuildMilliseconds << ","
             << "\"peel\":" << Result.CpuStageTimingsMs.AllAttempts.PeelMilliseconds << ","
             << "\"assign\":" << Result.CpuStageTimingsMs.AllAttempts.AssignMilliseconds << ","
             << "\"verify\":" << Result.CpuStageTimingsMs.AllAttempts.VerifyMilliseconds
             << "},"
             << "\"cpu_stage_timings_ms_solved_only\":{"
             << "\"add_build\":" << Result.CpuStageTimingsMs.SolvedOnly.AddBuildMilliseconds << ","
             << "\"peel\":" << Result.CpuStageTimingsMs.SolvedOnly.PeelMilliseconds << ","
             << "\"assign\":" << Result.CpuStageTimingsMs.SolvedOnly.AssignMilliseconds << ","
             << "\"verify\":" << Result.CpuStageTimingsMs.SolvedOnly.VerifyMilliseconds
             << "},"
             << "\"solved\":" << Result.GpuSuccess << ","
             << "\"cpu_success\":" << Result.CpuSuccess << ","
             << "\"mismatches\":" << Result.Mismatches << ","
             << "\"cpu_verify_issues\":" << Result.CpuVerifyIssues << ","
             << "\"peel_rounds\":" << Result.Rounds << ","
             << "\"stage_timings_ms\":{"
             << "\"add_build\":" << Result.AddBuildMilliseconds << ","
             << "\"peel\":" << Result.PeelMilliseconds << ","
             << "\"assign\":" << Result.AssignMilliseconds << ","
             << "\"verify\":" << Result.VerifyMilliseconds
             << "}"
             << "}";
        std::cout << Json.str() << "\n";
    } else {
        std::cout
            << "GPU Batched Peeling POC\n"
            << "  Dataset:           " << Result.DatasetName << "\n"
            << "  Keys file:         "
            << (Opts.KeysFile.empty() ? "<generated>" : Opts.KeysFile) << "\n"
            << "  Requested keys:    " << RequestedKeys << "\n"
            << "  Actual keys:       " << KeyCount << "\n"
            << "  Edge capacity:     " << EdgeCapacity << "\n"
            << "  Vertices:          " << Vertices << "\n"
            << "  Batch size:        " << Batch << "\n"
            << "  Requested fixed attempts: " << Result.RequestedFixedAttempts << "\n"
            << "  Actual attempts tried:    " << Result.ActualAttemptsTried << "\n"
            << "  Batches run:              " << Result.BatchesRun << "\n"
            << "  First solution wins:      " << (Result.FirstSolutionWins ? "Y" : "N") << "\n"
            << "  First solved attempt:     "
            << (Result.FirstSolvedAttempt >= 0 ? std::to_string(Result.FirstSolvedAttempt) : std::string("<none>"))
            << "\n"
            << "  Hash function:     " << Result.HashFunctionName << "\n"
            << "  Solve mode:        " << Result.SolveModeName << "\n"
            << "  Assignment backend: " << Result.AssignmentBackendName << "\n"
            << "  Allocation mode:   " << Result.AllocationModeName << "\n"
            << "  Assign geometry:   " << GraphGeometryToString(Opts.AssignGeometry) << "\n"
            << "  Device-serial peel geometry: "
            << GraphGeometryToString(Opts.DeviceSerialPeelGeometry) << "\n"
            << "  Storage bits:      " << Result.StorageBits << "\n"
            << "  Storage mode:      " << StorageModeToString(Result.SelectedStorage) << "\n"
            << "  Peel rounds:       " << Result.Rounds << "\n"
            << "  GPU success:       " << Result.GpuSuccess << "/" << Result.ActualAttemptsTried << "\n"
            << "  CPU success:       " << Result.CpuSuccess << "/" << Result.ActualAttemptsTried << "\n"
            << "  Success mismatches: " << Result.Mismatches << "\n"
            << "  CPU verify issues: " << Result.CpuVerifyIssues << "\n"
            << std::fixed << std::setprecision(3)
            << "  GPU time (ms):     " << Result.GpuMilliseconds << "\n"
            << "  CPU time (ms):     " << Result.CpuMilliseconds << "\n"
            << "  CPU stage timings (all attempts):\n"
            << "    Add/build (ms):  " << Result.CpuStageTimingsMs.AllAttempts.AddBuildMilliseconds << "\n"
            << "    Peel (ms):       " << Result.CpuStageTimingsMs.AllAttempts.PeelMilliseconds << "\n"
            << "    Assign (ms):     " << Result.CpuStageTimingsMs.AllAttempts.AssignMilliseconds << "\n"
            << "    Verify (ms):     " << Result.CpuStageTimingsMs.AllAttempts.VerifyMilliseconds << "\n"
            << "  CPU stage timings (solved only):\n"
            << "    Add/build (ms):  " << Result.CpuStageTimingsMs.SolvedOnly.AddBuildMilliseconds << "\n"
            << "    Peel (ms):       " << Result.CpuStageTimingsMs.SolvedOnly.PeelMilliseconds << "\n"
            << "    Assign (ms):     " << Result.CpuStageTimingsMs.SolvedOnly.AssignMilliseconds << "\n"
            << "    Verify (ms):     " << Result.CpuStageTimingsMs.SolvedOnly.VerifyMilliseconds << "\n"
            << "  Add/build (ms):    " << Result.AddBuildMilliseconds << "\n"
            << "  Peel (ms):         " << Result.PeelMilliseconds << "\n"
            << "  Assign (ms):       " << Result.AssignMilliseconds << "\n"
            << "  Verify (ms):       " << Result.VerifyMilliseconds << "\n";
    }

    if (DFrontierCount) {
        CheckCuda(cudaFree(DFrontierCount), "cudaFree(DFrontierCount)");
    }
    if (DFrontier) {
        CheckCuda(cudaFree(DFrontier), "cudaFree(DFrontier)");
    }
    CheckCuda(cudaFree(DVerifyFailures), "cudaFree(DVerifyFailures)");
    CheckCuda(cudaFree(DGraphRounds), "cudaFree(DGraphRounds)");
    CheckCuda(cudaFree(DAssigned), "cudaFree(DAssigned)");
    CheckCuda(cudaFree(DPeeledCount), "cudaFree(DPeeledCount)");
    CheckCuda(cudaFree(DPeelOrder), "cudaFree(DPeelOrder)");
    CheckCuda(cudaFree(DOwnerVertex), "cudaFree(DOwnerVertex)");
    CheckCuda(cudaFree(DEdgePeeled), "cudaFree(DEdgePeeled)");
    CheckCuda(cudaFree(DInvalidGraphs), "cudaFree(DInvalidGraphs)");
    CheckCuda(cudaFree(DXorEdge), "cudaFree(DXorEdge)");
    CheckCuda(cudaFree(DDegree), "cudaFree(DDegree)");
    CheckCuda(cudaFree(DEdgeV), "cudaFree(DEdgeV)");
    CheckCuda(cudaFree(DEdgeU), "cudaFree(DEdgeU)");
    CheckCuda(cudaFree(DGraphSeeds), "cudaFree(DGraphSeeds)");
    CheckCuda(cudaFree(DKeys), "cudaFree(DKeys)");
    CheckCuda(cudaEventDestroy(StageStop), "cudaEventDestroy(StageStop)");
    CheckCuda(cudaEventDestroy(StageStart), "cudaEventDestroy(StageStart)");

    return Result;
}

template<typename StorageT>
bool
SupportsStorage(uint32_t EdgeCapacity, uint32_t Vertices)
{
    constexpr uint64_t Max = static_cast<uint64_t>(std::numeric_limits<StorageT>::max());
    return (static_cast<uint64_t>(EdgeCapacity - 1) <= Max &&
            static_cast<uint64_t>(Vertices - 1) <= Max);
}

template<typename StorageT>
size_t
EstimateDeviceBytes(uint32_t KeyCount,
                    uint32_t Vertices,
                    uint32_t Batch,
                    SolveMode Mode,
                    GraphGeometry DeviceSerialPeelGeometry)
{
    using FrontierItem = FrontierItemT<StorageT>;

    const size_t TotalEdges = static_cast<size_t>(KeyCount) * Batch;
    const size_t TotalVertices = static_cast<size_t>(Vertices) * Batch;

    size_t Bytes = 0;

    Bytes += static_cast<size_t>(KeyCount) * sizeof(uint32_t);      // DKeys
    Bytes += static_cast<size_t>(Batch) * sizeof(GraphSeeds);       // DGraphSeeds
    Bytes += TotalEdges * sizeof(StorageT);                         // DEdgeU
    Bytes += TotalEdges * sizeof(StorageT);                         // DEdgeV
    Bytes += TotalVertices * sizeof(uint32_t);                      // DDegree
    Bytes += TotalVertices * sizeof(uint32_t);                      // DXorEdge
    Bytes += static_cast<size_t>(Batch) * sizeof(uint32_t);         // DInvalidGraphs
    Bytes += TotalEdges * sizeof(uint32_t);                         // DEdgePeeled
    Bytes += TotalEdges * sizeof(StorageT);                         // DOwnerVertex
    Bytes += TotalEdges * sizeof(StorageT);                         // DPeelOrder
    Bytes += static_cast<size_t>(Batch) * sizeof(uint32_t);         // DPeeledCount
    Bytes += TotalVertices * sizeof(StorageT);                      // DAssigned
    Bytes += static_cast<size_t>(Batch) * sizeof(uint32_t);         // DVerifyFailures
    Bytes += static_cast<size_t>(Batch) * sizeof(uint32_t);         // DGraphRounds

    const bool UseFrontier = (
        Mode == SolveMode::HostRoundTrip ||
        (Mode == SolveMode::DeviceSerial &&
         DeviceSerialPeelGeometry != GraphGeometry::Thread));

    if (UseFrontier) {
        Bytes += TotalVertices * sizeof(FrontierItem);              // DFrontier
        Bytes += static_cast<size_t>(Batch) * sizeof(uint32_t);     // DFrontierCount
    }

    return Bytes;
}

template<HashFunctionKind Kind>
ExperimentResult
RunWithSelectedStorage(const Options &Opts,
                       const std::vector<uint32_t> &Keys,
                       const std::string &RequestedKeys,
                       const std::string &DatasetName,
                       StorageMode SelectedStorage)
{
    auto GraphSeedsVector = MakeGraphSeedsVector<Kind>(Opts.Batch,
                                                       Opts.GraphSeed,
                                                       Opts.RngSubsequence,
                                                       Opts.RngOffset,
                                                       NextPowerOfTwo(NextPowerOfTwo(
                                                           static_cast<uint32_t>(Keys.size())
                                                       ) + 1),
                                                       Opts.SeedsFile,
                                                       0);

    if (SelectedStorage == StorageMode::Bits16) {
        return RunExperiment<Kind, uint16_t>(Opts,
                                             Keys,
                                             GraphSeedsVector,
                                             RequestedKeys,
                                             DatasetName,
                                             SelectedStorage);
    } else {
        return RunExperiment<Kind, uint32_t>(Opts,
                                             Keys,
                                             GraphSeedsVector,
                                             RequestedKeys,
                                             DatasetName,
                                             SelectedStorage);
    }
}

} // namespace

int
main(int argc, char **argv)
{
    Options Opts = ParseOptions(argc, argv);

    std::vector<uint32_t> Keys;
    std::string RequestedKeys = std::to_string(Opts.Edges);
    std::string DatasetName = "generated-" + std::to_string(Opts.Edges);

    if (!Opts.KeysFile.empty()) {
        Keys = LoadKeysFromFile(Opts.KeysFile);
        RequestedKeys = "<n/a>";
        DatasetName = BaseName(Opts.KeysFile);
    } else {
        Keys.resize(Opts.Edges);
        for (uint32_t Edge = 0; Edge < Opts.Edges; ++Edge) {
            Keys[Edge] = static_cast<uint32_t>(SplitMix64(Opts.KeySeed + Edge));
        }
    }

    if (Keys.empty()) {
        std::cerr << "No keys available.\n";
        return EXIT_FAILURE;
    }

    const uint32_t KeyCount = static_cast<uint32_t>(Keys.size());
    const uint32_t EdgeCapacity = NextPowerOfTwo(KeyCount);
    const uint32_t Vertices = NextPowerOfTwo(EdgeCapacity + 1);

    StorageMode SelectedStorage = Opts.Storage;
    if (SelectedStorage == StorageMode::Auto) {
        if (SupportsStorage<uint16_t>(EdgeCapacity, Vertices)) {
            SelectedStorage = StorageMode::Bits16;
        } else {
            SelectedStorage = StorageMode::Bits32;
        }
    }

    if (SelectedStorage == StorageMode::Bits16 &&
        !SupportsStorage<uint16_t>(EdgeCapacity, Vertices)) {
        std::cerr << "16-bit storage does not support edge capacity "
                  << EdgeCapacity
                  << " and vertex count "
                  << Vertices
                  << ".\n";
        return EXIT_FAILURE;
    }

    const DeviceMemoryInfo MemoryInfo = QueryDeviceMemoryInfo();
    const size_t RequestedBytes = (
        SelectedStorage == StorageMode::Bits16 ?
            EstimateDeviceBytes<uint16_t>(KeyCount, Vertices, Opts.Batch, Opts.Solve, Opts.DeviceSerialPeelGeometry) :
            EstimateDeviceBytes<uint32_t>(KeyCount, Vertices, Opts.Batch, Opts.Solve, Opts.DeviceSerialPeelGeometry)
    );

    Options EffectiveOpts = Opts;

    if (MemoryInfo.Valid) {
        const double HeadroomScale = (100.0 - Opts.MemoryHeadroomPct) / 100.0;
        const size_t AllowedBytes = static_cast<size_t>(
            static_cast<double>(MemoryInfo.FreeBytes) * HeadroomScale
        );

        if (RequestedBytes > AllowedBytes) {
            size_t FixedBytes = (
                SelectedStorage == StorageMode::Bits16 ?
                    EstimateDeviceBytes<uint16_t>(KeyCount, Vertices, 0, Opts.Solve, Opts.DeviceSerialPeelGeometry) :
                    EstimateDeviceBytes<uint32_t>(KeyCount, Vertices, 0, Opts.Solve, Opts.DeviceSerialPeelGeometry)
            );
            size_t OneGraphBytes = (
                SelectedStorage == StorageMode::Bits16 ?
                    EstimateDeviceBytes<uint16_t>(KeyCount, Vertices, 1, Opts.Solve, Opts.DeviceSerialPeelGeometry) :
                    EstimateDeviceBytes<uint32_t>(KeyCount, Vertices, 1, Opts.Solve, Opts.DeviceSerialPeelGeometry)
            );

            if (OneGraphBytes <= FixedBytes || AllowedBytes <= FixedBytes) {
                std::cerr
                    << "Estimated device memory requirement "
                    << FormatBytes(RequestedBytes)
                    << " exceeds allowed headroom-adjusted free memory "
                    << FormatBytes(AllowedBytes)
                    << ".\n";
                return EXIT_FAILURE;
            }

            const size_t PerBatchBytes = OneGraphBytes - FixedBytes;
            const size_t MaxBatch = (AllowedBytes - FixedBytes) / PerBatchBytes;

            if (!Opts.AutoScaleBatchToFit) {
                std::cerr
                    << "Requested batch " << Opts.Batch
                    << " requires " << FormatBytes(RequestedBytes)
                    << ", but only " << FormatBytes(AllowedBytes)
                    << " is allowed after leaving "
                    << Opts.MemoryHeadroomPct << "% headroom.\n";
                return EXIT_FAILURE;
            }

            if (MaxBatch < 1) {
                std::cerr
                    << "Unable to fit even batch size 1 within available device memory headroom.\n";
                return EXIT_FAILURE;
            }

            EffectiveOpts.Batch = static_cast<uint32_t>(MaxBatch);

            if (Opts.Output == OutputFormat::Human) {
                std::cout
                    << "Adjusting batch from " << Opts.Batch
                    << " to " << EffectiveOpts.Batch
                    << " to fit available device memory (estimated "
                    << FormatBytes(RequestedBytes)
                    << ", allowed "
                    << FormatBytes(AllowedBytes)
                    << ").\n";
            }

            if (Opts.Solve == SolveMode::HostRoundTrip) {
                const size_t DeviceSerialBytes = (
                    SelectedStorage == StorageMode::Bits16 ?
                        EstimateDeviceBytes<uint16_t>(KeyCount, Vertices, Opts.Batch, SolveMode::DeviceSerial, Opts.DeviceSerialPeelGeometry) :
                        EstimateDeviceBytes<uint32_t>(KeyCount, Vertices, Opts.Batch, SolveMode::DeviceSerial, Opts.DeviceSerialPeelGeometry)
                );
                if (DeviceSerialBytes <= AllowedBytes) {
                    if (Opts.Output == OutputFormat::Human) {
                        std::cout
                            << "Hint: requested batch would fit without scaling under --solve-mode device-serial.\n";
                    }
                }
            }
        }

        const size_t EffectiveBytes = (
            SelectedStorage == StorageMode::Bits16 ?
                EstimateDeviceBytes<uint16_t>(KeyCount, Vertices, EffectiveOpts.Batch, EffectiveOpts.Solve, EffectiveOpts.DeviceSerialPeelGeometry) :
                EstimateDeviceBytes<uint32_t>(KeyCount, Vertices, EffectiveOpts.Batch, EffectiveOpts.Solve, EffectiveOpts.DeviceSerialPeelGeometry)
        );

        if (Opts.Output == OutputFormat::Human) {
            std::cout
                << "CUDA memory summary\n"
                << "  Free:               " << FormatBytes(MemoryInfo.FreeBytes) << "\n"
                << "  Total:              " << FormatBytes(MemoryInfo.TotalBytes) << "\n"
                << "  Unified-like:       " << (MemoryInfo.UnifiedLike ? "Y" : "N") << "\n"
                << "  Headroom target:    " << Opts.MemoryHeadroomPct << "%\n"
                << "  Estimated bytes:    " << FormatBytes(EffectiveBytes) << "\n"
                << "  Effective batch:    " << EffectiveOpts.Batch << "\n";
        }
    }

    ExperimentResult Result = {};

    switch (Opts.HashFunction) {
        case HashFunctionKind::SplitMix:
            Result = RunWithSelectedStorage<HashFunctionKind::SplitMix>(
                EffectiveOpts,
                Keys,
                RequestedKeys,
                DatasetName,
                SelectedStorage
            );
            break;
        case HashFunctionKind::MultiplyShiftR:
            Result = RunWithSelectedStorage<HashFunctionKind::MultiplyShiftR>(
                EffectiveOpts,
                Keys,
                RequestedKeys,
                DatasetName,
                SelectedStorage
            );
            break;
        case HashFunctionKind::MultiplyShiftRX:
            Result = RunWithSelectedStorage<HashFunctionKind::MultiplyShiftRX>(
                EffectiveOpts,
                Keys,
                RequestedKeys,
                DatasetName,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate1RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate1RX>(
                EffectiveOpts,
                Keys,
                RequestedKeys,
                DatasetName,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate2RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate2RX>(
                EffectiveOpts,
                Keys,
                RequestedKeys,
                DatasetName,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate3RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate3RX>(
                EffectiveOpts,
                Keys,
                RequestedKeys,
                DatasetName,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate4RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate4RX>(
                EffectiveOpts,
                Keys,
                RequestedKeys,
                DatasetName,
                SelectedStorage
            );
            break;
    }

    return (Result.Mismatches == 0 && Result.CpuVerifyIssues == 0) ?
        EXIT_SUCCESS :
        EXIT_FAILURE;
}
