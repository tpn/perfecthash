#include <gtest/gtest.h>

#include <PerfectHash/PerfectHash.h>

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>

namespace {

struct TableShim { PPERFECT_HASH_TABLE_VTBL Vtbl; };

class PerfectHashArgvTests : public ::testing::Test {
protected:
  void SetUp() override {
    HRESULT result = PerfectHashBootstrap(
        &classFactory_,
        &printError_,
        &printMessage_,
        &module_);
    ASSERT_GE(result, 0);

    auto createInstance = classFactory_->Vtbl->CreateInstance;
    result = createInstance(
        classFactory_,
        nullptr,
#ifdef PH_WINDOWS
        IID_PERFECT_HASH_CONTEXT,
#else
        &IID_PERFECT_HASH_CONTEXT,
#endif
        reinterpret_cast<void **>(&context_));
    ASSERT_GE(result, 0);
  }

  void TearDown() override {
    if (retryTable_) {
      reinterpret_cast<TableShim *>(retryTable_)->Vtbl->Release(retryTable_);
    }
    if (retryKeys_) {
      retryKeys_->Vtbl->Release(retryKeys_);
    }
    if (context_) {
      context_->Vtbl->Release(context_);
      context_ = nullptr;
    }
    if (classFactory_) {
      classFactory_->Vtbl->Release(classFactory_);
      classFactory_ = nullptr;
    }
#ifdef PH_WINDOWS
    if (module_) {
      FreeLibrary(module_);
      module_ = nullptr;
    }
#endif
    if (!retryDirectory_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(retryDirectory_, ignored);
    }
  }

  HRESULT RunTableCreate(const std::vector<std::string> &args) {
#ifdef PH_WINDOWS
    std::vector<std::wstring> wide_args;
    wide_args.reserve(args.size());
    for (const auto &arg : args) {
      wide_args.emplace_back(arg.begin(), arg.end());
    }

    std::wstring command_line;
    for (size_t i = 0; i < wide_args.size(); ++i) {
      if (i > 0) {
        command_line.push_back(L' ');
      }
      command_line.append(wide_args[i]);
    }

    std::vector<PWSTR> argvw;
    argvw.reserve(wide_args.size());
    for (auto &arg : wide_args) {
      argvw.push_back(const_cast<PWSTR>(arg.c_str()));
    }

    return context_->Vtbl->TableCreateArgvW(
        context_,
        static_cast<ULONG>(argvw.size()),
        argvw.data(),
        const_cast<PWSTR>(command_line.c_str()));
#else
    std::vector<char *> argva;
    argva.reserve(args.size());
    for (const auto &arg : args) {
      argva.push_back(const_cast<char *>(arg.c_str()));
    }

    return context_->Vtbl->TableCreateArgvA(
        context_,
        static_cast<ULONG>(argva.size()),
        argva.data());
#endif
  }

  std::filesystem::path retryDirectory_;

  PPERFECT_HASH_TABLE retryTable_ = nullptr;
  PPERFECT_HASH_KEYS retryKeys_ = nullptr;
  PICLASSFACTORY classFactory_ = nullptr;
  PPERFECT_HASH_CONTEXT context_ = nullptr;
  PPERFECT_HASH_PRINT_ERROR printError_ = nullptr;
  PPERFECT_HASH_PRINT_MESSAGE printMessage_ = nullptr;
  HMODULE module_ = nullptr;
};

TEST_F(PerfectHashArgvTests, InvalidNumArgs) {
  const std::vector<std::string> args = {
      "PerfectHashCreate",
      "keys",
      "out",
      "Chm01",
  };

  EXPECT_EQ(PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS, RunTableCreate(args));
}

TEST_F(PerfectHashArgvTests, FailedFileCreateRegeneratesSharedFiles) {
  std::random_device entropy;
  const auto candidate = std::filesystem::temp_directory_path() /
      ("perfecthash-retry-" + std::to_string(entropy()) + "-" +
       std::to_string(entropy()));
  ASSERT_TRUE(std::filesystem::create_directory(candidate));
  retryDirectory_ = candidate;
  const auto keysPath = retryDirectory_ / "Retry.keys";
  const auto outputPath = retryDirectory_ / "out";
  ASSERT_TRUE(std::filesystem::create_directory(outputPath));
  {
    const ULONG keys[] = {1, 3, 5, 7};
    std::ofstream file(keysPath, std::ios::binary);
    file.write(reinterpret_cast<const char *>(keys), sizeof(keys));
    ASSERT_TRUE(file.good());
  }
  // The command entry point initializes context RNG/TLS configuration. Its
  // return code describes command processing; the failed solve deletes props.
  ASSERT_EQ(S_OK, RunTableCreate({
      "PerfectHashCreate", keysPath.string(), outputPath.string(), "Chm01",
      "MultiplyShiftR", "And", "1", "--Quiet", "--DoNotTryUseHash16Impl",
      "--Seeds=1,1,0x1010", "--FixedAttempts=1"}));
  EXPECT_FALSE(std::filesystem::exists(outputPath / "CompiledPerfectHash.props"));
  auto keysName = keysPath.wstring();
  UNICODE_STRING keysString = {};
  keysString.Length = static_cast<USHORT>(keysName.size() * sizeof(WCHAR));
  keysString.MaximumLength = keysString.Length + sizeof(WCHAR);
  keysString.Buffer = keysName.data();
#ifdef PH_WINDOWS
#define PH_TEST_IID(Name) Name
#else
#define PH_TEST_IID(Name) &Name
#endif
  ASSERT_EQ(S_OK, classFactory_->Vtbl->CreateInstance(
      classFactory_, nullptr, PH_TEST_IID(IID_PERFECT_HASH_KEYS),
      reinterpret_cast<void **>(&retryKeys_)));
  ASSERT_EQ(S_OK, retryKeys_->Vtbl->Load(retryKeys_, nullptr, &keysString, sizeof(ULONG)));
  ASSERT_EQ(S_OK, classFactory_->Vtbl->CreateInstance(
      classFactory_, nullptr, PH_TEST_IID(IID_PERFECT_HASH_TABLE),
      reinterpret_cast<void **>(&retryTable_)));
  auto tableVtbl = reinterpret_cast<TableShim *>(retryTable_)->Vtbl;
  PERFECT_HASH_TABLE_CREATE_FLAGS flags = {};
  flags.Quiet = TRUE;
  flags.DoNotTryUseHash16Impl = TRUE;
  ULONG seeds[] = {1, 1, 0x1010};
  PERFECT_HASH_TABLE_CREATE_PARAMETER params[2] = {};
  params[0].Id = TableCreateParameterSeedsId;
  params[0].AsValueArray.Values = seeds;
  params[0].AsValueArray.NumberOfValues = 3;
  params[0].AsValueArray.ValueSizeInBytes = sizeof(ULONG);
  params[1].Id = TableCreateParameterFixedAttemptsId;
  params[1].AsULong = 1;
  PERFECT_HASH_TABLE_CREATE_PARAMETERS parameters = {};
  parameters.SizeOfStruct = sizeof(parameters);
  parameters.NumberOfElements = 2;
  parameters.Params = params;
  auto create = [&]() {
    return tableVtbl->Create(retryTable_, context_, PerfectHashChm01AlgorithmId,
        PerfectHashHashMultiplyShiftRFunctionId, PerfectHashAndMaskFunctionId,
        retryKeys_, &flags, &parameters);
  };
  EXPECT_EQ(PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION,
            create());
  EXPECT_FALSE(std::filesystem::exists(outputPath / "CompiledPerfectHash.props"));

  params[0] = {};
  params[0].Id = TableCreateParameterMaxSolveTimeInSecondsId;
  params[0].AsULong = 10;
  parameters.NumberOfElements = 1;
  ASSERT_EQ(S_OK, create());
  EXPECT_GT(std::filesystem::file_size(outputPath / "CompiledPerfectHash.props"), 0u);
  EXPECT_GT(std::filesystem::file_size(outputPath / "CompiledPerfectHash.h"), 0u);

  // A later failed table must retain shared files committed by this success.
  tableVtbl->Release(retryTable_);
  retryTable_ = nullptr;
  ASSERT_EQ(S_OK, classFactory_->Vtbl->CreateInstance(
      classFactory_, nullptr, PH_TEST_IID(IID_PERFECT_HASH_TABLE),
      reinterpret_cast<void **>(&retryTable_)));
  tableVtbl = reinterpret_cast<TableShim *>(retryTable_)->Vtbl;
  params[0] = {};
  params[0].Id = TableCreateParameterSeedsId;
  params[0].AsValueArray.Values = seeds;
  params[0].AsValueArray.NumberOfValues = 3;
  params[0].AsValueArray.ValueSizeInBytes = sizeof(ULONG);
  parameters.NumberOfElements = 2;
  EXPECT_EQ(PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION, create());
  EXPECT_GT(std::filesystem::file_size(outputPath / "CompiledPerfectHash.props"), 0u);
  EXPECT_GT(std::filesystem::file_size(outputPath / "CompiledPerfectHash.h"), 0u);
#undef PH_TEST_IID
}

TEST_F(PerfectHashArgvTests, InvalidAlgorithmId) {
  const std::vector<std::string> args = {
      "PerfectHashCreate",
      "keys",
      "out",
      "NotAnAlgorithm",
      "MultiplyShiftR",
      "And",
      "1",
  };

  EXPECT_EQ(PH_E_INVALID_ALGORITHM_ID, RunTableCreate(args));
}

TEST_F(PerfectHashArgvTests, InvalidHashFunctionId) {
  const std::vector<std::string> args = {
      "PerfectHashCreate",
      "keys",
      "out",
      "Chm01",
      "NotAHash",
      "And",
      "1",
  };

  EXPECT_EQ(PH_E_INVALID_HASH_FUNCTION_ID, RunTableCreate(args));
}

TEST_F(PerfectHashArgvTests, InvalidMaskFunctionId) {
  const std::vector<std::string> args = {
      "PerfectHashCreate",
      "keys",
      "out",
      "Chm01",
      "MultiplyShiftR",
      "NotAMask",
      "1",
  };

  EXPECT_EQ(PH_E_INVALID_MASK_FUNCTION_ID, RunTableCreate(args));
}

TEST_F(PerfectHashArgvTests, MissingTableCreateParamValue) {
  const std::vector<std::string> args = {
      "PerfectHashCreate",
      "keys",
      "out",
      "Chm01",
      "MultiplyShiftR",
      "And",
      "1",
      "--GraphImpl",
  };

  EXPECT_EQ(PH_E_COMMANDLINE_ARG_MISSING_VALUE, RunTableCreate(args));
}

TEST_F(PerfectHashArgvTests, InvalidCommandLineArg) {
  const std::vector<std::string> args = {
      "PerfectHashCreate",
      "keys",
      "out",
      "Chm01",
      "MultiplyShiftR",
      "And",
      "1",
      "--ThisFlagDoesNotExist",
  };

  EXPECT_EQ(PH_E_INVALID_COMMANDLINE_ARG, RunTableCreate(args));
}

} // namespace
