#include <gtest/gtest.h>

#include <PerfectHash.h>

#include <string>
#include <vector>

namespace {

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
        IID_PERFECT_HASH_CONTEXT,
        reinterpret_cast<void **>(&context_));
    ASSERT_GE(result, 0);
  }

  void TearDown() override {
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

private:
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
