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
    std::string KeysFile;
    std::string SeedsFile;
    std::string DumpSolvedSeedsFile;
    StorageMode Storage = StorageMode::Auto;
    HashFunctionKind HashFunction = HashFunctionKind::MultiplyShiftR;
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
};

struct ExperimentResult {
    uint32_t KeyCount = 0;
    uint32_t EdgeCapacity = 0;
    uint32_t Vertices = 0;
    uint32_t Batch = 0;
    uint32_t Rounds = 0;
    uint32_t GpuSuccess = 0;
    uint32_t CpuSuccess = 0;
    uint32_t Mismatches = 0;
    uint32_t CpuVerifyIssues = 0;
    float GpuMilliseconds = 0.0f;
    double CpuMilliseconds = 0.0;
    uint32_t StorageBits = 0;
    const char *HashFunctionName = nullptr;
    StorageMode SelectedStorage = StorageMode::Auto;
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
        } else if (Arg == "--hash-function") {
            Opts.HashFunction = ParseHashFunctionKind(RequireValue("--hash-function"));
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
                << "  --storage-bits <x>    auto, 16, or 32\n"
                << "  --hash-function <x>   SplitMix, MultiplyShiftR, MultiplyShiftRX,\n"
                << "                        Mulshrolate1RX, Mulshrolate2RX, Mulshrolate3RX,\n"
                << "                        or Mulshrolate4RX\n"
                << "  --key-seed <x>        Base seed for generated keys\n"
                << "  --graph-seed <x>      Philox seed for per-graph hash seeds\n"
                << "  --rng-subsequence <x> Philox subsequence base for graph seeds\n"
                << "  --rng-offset <x>      Philox offset base for graph seeds\n"
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
MakeGraphSeeds(uint64_t GraphSeed, uint32_t GraphIndex)
{
    GraphSeeds Seeds = {};
    constexpr uint32_t NumberOfSeeds = HashTraits<Kind>::NumberOfSeeds;
    Philox4x32State State = InitializePhiloxState(GraphSeed,
                                                  static_cast<uint64_t>(GraphIndex),
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
                     const std::string &SeedsFile)
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
                                                      RngSubsequence + Graph,
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

    for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
        const auto Hash = HashImpl<Kind, StorageT>::Apply(Keys[Edge], Seeds, Vertices - 1);
        uint32_t U = static_cast<uint32_t>(Hash.Vertex1);
        uint32_t V = static_cast<uint32_t>(Hash.Vertex2);

        EdgeU[Edge] = Hash.Vertex1;
        EdgeV[Edge] = Hash.Vertex2;

        if (U == V || U >= Vertices || V >= Vertices) {
            Result.Invalid = true;
            return Result;
        }

        ++Degree[U];
        ++Degree[V];
        XorEdge[U] ^= Edge;
        XorEdge[V] ^= Edge;
    }

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

    Result.Peeled = static_cast<uint32_t>(Order.size());
    Result.Success = (Order.size() == Edges);
    if (!Result.Success) {
        return Result;
    }

    for (int64_t Index = static_cast<int64_t>(Order.size()) - 1; Index >= 0; --Index) {
        uint32_t Edge = static_cast<uint32_t>(Order[static_cast<size_t>(Index)]);
        uint32_t OwnerVertex = static_cast<uint32_t>(Owner[Edge]);
        uint32_t U = static_cast<uint32_t>(EdgeU[Edge]);
        uint32_t V = static_cast<uint32_t>(EdgeV[Edge]);
        uint32_t Other = (OwnerVertex == U) ? V : U;
        uint32_t Value = (Edge - static_cast<uint32_t>(Assigned[Other])) & EdgeMask;

        Assigned[OwnerVertex] = static_cast<StorageT>(Value);
    }

    Result.Verified = true;

    for (uint32_t Edge = 0; Edge < Edges; ++Edge) {
        uint32_t Index = (
            static_cast<uint32_t>(Assigned[static_cast<uint32_t>(EdgeU[Edge])]) +
            static_cast<uint32_t>(Assigned[static_cast<uint32_t>(EdgeV[Edge])])
        ) & EdgeMask;
        if (Index != Edge) {
            Result.Verified = false;
            break;
        }
    }

    return Result;
}

template<HashFunctionKind Kind, typename StorageT>
ExperimentResult
RunExperiment(const Options &Opts,
              const std::vector<uint32_t> &Keys,
              const std::vector<GraphSeeds> &GraphSeedsVector,
              const std::string &RequestedKeys,
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
    Result.StorageBits = static_cast<uint32_t>(sizeof(StorageT) * 8);
    Result.HashFunctionName = HashTraits<Kind>::Name;
    Result.SelectedStorage = SelectedStorage;

    const uint64_t TotalEdges = static_cast<uint64_t>(KeyCount) * Batch;
    const uint64_t TotalVertices = static_cast<uint64_t>(Vertices) * Batch;
    const uint64_t FrontierCapacity = TotalVertices;

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

    CheckCuda(cudaMalloc(&DKeys, Keys.size() * sizeof(Keys[0])), "cudaMalloc(DKeys)");
    CheckCuda(cudaMalloc(&DGraphSeeds,
                         GraphSeedsVector.size() * sizeof(GraphSeedsVector[0])),
              "cudaMalloc(DGraphSeeds)");
    CheckCuda(cudaMalloc(&DEdgeU, TotalEdges * sizeof(StorageT)), "cudaMalloc(DEdgeU)");
    CheckCuda(cudaMalloc(&DEdgeV, TotalEdges * sizeof(StorageT)), "cudaMalloc(DEdgeV)");
    CheckCuda(cudaMalloc(&DDegree, TotalVertices * sizeof(uint32_t)), "cudaMalloc(DDegree)");
    CheckCuda(cudaMalloc(&DXorEdge, TotalVertices * sizeof(uint32_t)), "cudaMalloc(DXorEdge)");
    CheckCuda(cudaMalloc(&DInvalidGraphs, Batch * sizeof(uint32_t)), "cudaMalloc(DInvalidGraphs)");
    CheckCuda(cudaMalloc(&DEdgePeeled, TotalEdges * sizeof(uint32_t)), "cudaMalloc(DEdgePeeled)");
    CheckCuda(cudaMalloc(&DOwnerVertex, TotalEdges * sizeof(StorageT)), "cudaMalloc(DOwnerVertex)");
    CheckCuda(cudaMalloc(&DPeelOrder, TotalEdges * sizeof(StorageT)), "cudaMalloc(DPeelOrder)");
    CheckCuda(cudaMalloc(&DPeeledCount, Batch * sizeof(uint32_t)), "cudaMalloc(DPeeledCount)");
    CheckCuda(cudaMalloc(&DAssigned, TotalVertices * sizeof(StorageT)), "cudaMalloc(DAssigned)");
    CheckCuda(cudaMalloc(&DVerifyFailures, Batch * sizeof(uint32_t)), "cudaMalloc(DVerifyFailures)");
    CheckCuda(cudaMalloc(&DFrontier, FrontierCapacity * sizeof(FrontierItem)), "cudaMalloc(DFrontier)");
    CheckCuda(cudaMalloc(&DFrontierCount, sizeof(uint32_t)), "cudaMalloc(DFrontierCount)");

    CheckCuda(cudaMemcpy(DKeys, Keys.data(), Keys.size() * sizeof(Keys[0]), cudaMemcpyHostToDevice),
              "cudaMemcpy(DKeys)");
    CheckCuda(cudaMemcpy(DGraphSeeds,
                         GraphSeedsVector.data(),
                         GraphSeedsVector.size() * sizeof(GraphSeedsVector[0]),
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

    cudaEvent_t Start = nullptr;
    cudaEvent_t Stop = nullptr;
    CheckCuda(cudaEventCreate(&Start), "cudaEventCreate(Start)");
    CheckCuda(cudaEventCreate(&Stop), "cudaEventCreate(Stop)");

    const uint32_t BuildBlocks = static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads);
    const uint32_t VertexBlocks = static_cast<uint32_t>((TotalVertices + Opts.Threads - 1) / Opts.Threads);

    CheckCuda(cudaEventRecord(Start), "cudaEventRecord(Start)");

    BuildGraphsKernel<Kind, StorageT><<<BuildBlocks, Opts.Threads>>>(KeyCount,
                                                                     Vertices,
                                                                     Batch,
                                                                     DKeys,
                                                                     DGraphSeeds,
                                                                     DEdgeU,
                                                                     DEdgeV,
                                                                     DDegree,
                                                                     DXorEdge,
                                                                     DInvalidGraphs);
    CheckCuda(cudaGetLastError(), "BuildGraphsKernel launch");

    uint32_t FrontierCount = 0;

    for (;;) {
        CheckCuda(cudaMemset(DFrontierCount, 0, sizeof(uint32_t)), "cudaMemset(DFrontierCount)");

        CollectFrontierKernel<StorageT><<<VertexBlocks, Opts.Threads>>>(Vertices,
                                                                        Batch,
                                                                        DDegree,
                                                                        DXorEdge,
                                                                        DInvalidGraphs,
                                                                        DFrontier,
                                                                        DFrontierCount);
        CheckCuda(cudaGetLastError(), "CollectFrontierKernel launch");
        CheckCuda(cudaMemcpy(&FrontierCount, DFrontierCount, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(FrontierCount)");

        if (FrontierCount == 0) {
            break;
        }

        ++Result.Rounds;

        const uint32_t PeelBlocks = static_cast<uint32_t>((FrontierCount + Opts.Threads - 1) / Opts.Threads);
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
    }

    AssignGraphsKernel<StorageT><<<Batch, 1>>>(KeyCount,
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

    const uint32_t VerifyBlocks = static_cast<uint32_t>((TotalEdges + Opts.Threads - 1) / Opts.Threads);
    VerifyGraphsKernel<StorageT><<<VerifyBlocks, Opts.Threads>>>(KeyCount,
                                                                 Vertices,
                                                                 Batch,
                                                                 EdgeMask,
                                                                 DInvalidGraphs,
                                                                 DEdgeU,
                                                                 DEdgeV,
                                                                 DAssigned,
                                                                 DPeeledCount,
                                                                 DVerifyFailures);
    CheckCuda(cudaGetLastError(), "VerifyGraphsKernel launch");

    CheckCuda(cudaEventRecord(Stop), "cudaEventRecord(Stop)");
    CheckCuda(cudaEventSynchronize(Stop), "cudaEventSynchronize(Stop)");
    CheckCuda(cudaEventElapsedTime(&Result.GpuMilliseconds, Start, Stop), "cudaEventElapsedTime");

    std::vector<uint32_t> InvalidGraphs(Batch);
    std::vector<uint32_t> PeeledCount(Batch);
    std::vector<uint32_t> VerifyFailures(Batch);

    CheckCuda(cudaMemcpy(InvalidGraphs.data(), DInvalidGraphs, Batch * sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(InvalidGraphs)");
    CheckCuda(cudaMemcpy(PeeledCount.data(), DPeeledCount, Batch * sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(PeeledCount)");
    CheckCuda(cudaMemcpy(VerifyFailures.data(), DVerifyFailures, Batch * sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy(VerifyFailures)");

    auto CpuStart = std::chrono::steady_clock::now();
    std::vector<CpuResult> CpuResults(Batch);

    for (uint32_t Graph = 0; Graph < Batch; ++Graph) {
        CpuResults[Graph] = RunCpuReference<Kind, StorageT>(Graph,
                                                            KeyCount,
                                                            Vertices,
                                                            EdgeMask,
                                                            Keys,
                                                            GraphSeedsVector);
    }

    auto CpuStop = std::chrono::steady_clock::now();
    Result.CpuMilliseconds =
        std::chrono::duration<double, std::milli>(CpuStop - CpuStart).count();

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

    for (uint32_t Graph = 0; Graph < Batch; ++Graph) {
        const bool ThisGpuSuccess = (
            InvalidGraphs[Graph] == 0 &&
            PeeledCount[Graph] == KeyCount &&
            VerifyFailures[Graph] == 0
        );
        const bool ThisCpuSuccess = (CpuResults[Graph].Success && CpuResults[Graph].Verified);

        if (ThisGpuSuccess) {
            ++Result.GpuSuccess;
        }
        if (ThisCpuSuccess) {
            ++Result.CpuSuccess;
        }
        if (ThisGpuSuccess && SolvedSeedsFile.is_open()) {
            const GraphSeeds &Seeds = GraphSeedsVector[Graph];
            SolvedSeedsFile
                << Graph
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
            ++Result.Mismatches;
            if (Opts.Verbose) {
                std::cout << "Mismatch graph=" << Graph
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
            ++Result.CpuVerifyIssues;
        }
    }

    std::cout
        << "GPU Batched Peeling POC\n"
        << "  Keys file:          "
        << (Opts.KeysFile.empty() ? "<generated>" : Opts.KeysFile) << "\n"
        << "  Requested keys:     " << RequestedKeys << "\n"
        << "  Actual keys:        " << KeyCount << "\n"
        << "  Edge capacity:      " << EdgeCapacity << "\n"
        << "  Vertices:           " << Vertices << "\n"
        << "  Batch size:         " << Batch << "\n"
        << "  Hash function:      " << Result.HashFunctionName << "\n"
        << "  Storage bits:       " << Result.StorageBits << "\n"
        << "  Storage mode:       " << StorageModeToString(Result.SelectedStorage) << "\n"
        << "  Peel rounds:        " << Result.Rounds << "\n"
        << "  GPU success:        " << Result.GpuSuccess << "/" << Batch << "\n"
        << "  CPU success:        " << Result.CpuSuccess << "/" << Batch << "\n"
        << "  Success mismatches: " << Result.Mismatches << "\n"
        << "  CPU verify issues:  " << Result.CpuVerifyIssues << "\n"
        << std::fixed << std::setprecision(3)
        << "  GPU time (ms):      " << Result.GpuMilliseconds << "\n"
        << "  CPU time (ms):      " << Result.CpuMilliseconds << "\n";

    CheckCuda(cudaFree(DFrontierCount), "cudaFree(DFrontierCount)");
    CheckCuda(cudaFree(DFrontier), "cudaFree(DFrontier)");
    CheckCuda(cudaFree(DVerifyFailures), "cudaFree(DVerifyFailures)");
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
    CheckCuda(cudaEventDestroy(Stop), "cudaEventDestroy(Stop)");
    CheckCuda(cudaEventDestroy(Start), "cudaEventDestroy(Start)");

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

template<HashFunctionKind Kind>
ExperimentResult
RunWithSelectedStorage(const Options &Opts,
                       const std::vector<uint32_t> &Keys,
                       const std::string &RequestedKeys,
                       StorageMode SelectedStorage)
{
    auto GraphSeedsVector = MakeGraphSeedsVector<Kind>(Opts.Batch,
                                                       Opts.GraphSeed,
                                                       Opts.RngSubsequence,
                                                       Opts.RngOffset,
                                                       NextPowerOfTwo(NextPowerOfTwo(
                                                           static_cast<uint32_t>(Keys.size())
                                                       ) + 1),
                                                       Opts.SeedsFile);

    if (SelectedStorage == StorageMode::Bits16) {
        return RunExperiment<Kind, uint16_t>(Opts,
                                             Keys,
                                             GraphSeedsVector,
                                             RequestedKeys,
                                             SelectedStorage);
    } else {
        return RunExperiment<Kind, uint32_t>(Opts,
                                             Keys,
                                             GraphSeedsVector,
                                             RequestedKeys,
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

    if (!Opts.KeysFile.empty()) {
        Keys = LoadKeysFromFile(Opts.KeysFile);
        RequestedKeys = "<n/a>";
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

    ExperimentResult Result = {};

    switch (Opts.HashFunction) {
        case HashFunctionKind::SplitMix:
            Result = RunWithSelectedStorage<HashFunctionKind::SplitMix>(
                Opts,
                Keys,
                RequestedKeys,
                SelectedStorage
            );
            break;
        case HashFunctionKind::MultiplyShiftR:
            Result = RunWithSelectedStorage<HashFunctionKind::MultiplyShiftR>(
                Opts,
                Keys,
                RequestedKeys,
                SelectedStorage
            );
            break;
        case HashFunctionKind::MultiplyShiftRX:
            Result = RunWithSelectedStorage<HashFunctionKind::MultiplyShiftRX>(
                Opts,
                Keys,
                RequestedKeys,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate1RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate1RX>(
                Opts,
                Keys,
                RequestedKeys,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate2RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate2RX>(
                Opts,
                Keys,
                RequestedKeys,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate3RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate3RX>(
                Opts,
                Keys,
                RequestedKeys,
                SelectedStorage
            );
            break;
        case HashFunctionKind::Mulshrolate4RX:
            Result = RunWithSelectedStorage<HashFunctionKind::Mulshrolate4RX>(
                Opts,
                Keys,
                RequestedKeys,
                SelectedStorage
            );
            break;
    }

    return (Result.Mismatches == 0 && Result.CpuVerifyIssues == 0) ?
        EXIT_SUCCESS :
        EXIT_FAILURE;
}
