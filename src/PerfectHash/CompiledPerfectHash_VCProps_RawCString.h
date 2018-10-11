//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashVCPropsRawCStr[] =
    "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
    "<Project ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n"
    "\n"
    "  <PropertyGroup Label=\"Configuration\">\n"
    "    <Platform Condition=\"'$(Platform)' == ''\">x64</Platform>\n"
    "    <Configuration Condition=\"'$(Configuration)' == ''\">Release</Configuration>\n"
    "    <LinkIncremental>false</LinkIncremental>\n"
    "    <LinkIncremental Condition=\"'$(Configuration)'=='Debug'\">true</LinkIncremental>\n"
    "    <CharacterSet>Unicode</CharacterSet>\n"
    "    <PlatformToolset>v141</PlatformToolset>\n"
    "    <DefaultPlatformToolset>v141</DefaultPlatformToolset>\n"
    "  </PropertyGroup>\n"
    "\n"
    "  <PropertyGroup>\n"
    "    <PGDDirectory>$(SolutionDir)$(Platform)\\</PGDDirectory>\n"
    "    <CodeAnalysisRuleSet>AllRules.ruleset</CodeAnalysisRuleSet>\n"
    "    <RunCodeAnalysis>true</RunCodeAnalysis>\n"
    "  </PropertyGroup>\n"
    "\n"
    "  <ItemDefinitionGroup>\n"
    "    <ClCompile>\n"
    "      <PrecompiledHeader>Use</PrecompiledHeader>\n"
    "      <Optimization>MaxSpeed</Optimization>\n"
    "      <IntrinsicFunctions>true</IntrinsicFunctions>\n"
    "      <StringPooling>true</StringPooling>\n"
    "      <ExceptionHandling></ExceptionHandling>\n"
    "      <FunctionLevelLinking>true</FunctionLevelLinking>\n"
    "      <TreatWarningAsError>true</TreatWarningAsError>\n"
    "      <WarningLevel>EnableAllWarnings</WarningLevel>\n"
    "      <GenerateDebugInformation>true</GenerateDebugInformation>\n"
    "      <DebugInformationFormat>ProgramDatabase</DebugInformationFormat>\n"
    "      <CompileAs>CompileAsC</CompileAs>\n"
    "      <SuppressStartupBanner>true</SuppressStartupBanner>\n"
    "      <BrowseInformation>true</BrowseInformation>\n"
    "      <ExpandAttributedSource>true</ExpandAttributedSource>\n"
    "      <AssemblerOutput>All</AssemblerOutput>\n"
    "      <UseDebugLibraries>false</UseDebugLibraries>\n"
    "      <EnableCOMDATFolding>true</EnableCOMDATFolding>\n"
    "      <OptimizeReferences>true</OptimizeReferences>\n"
    "      <CompileAsManaged>false</CompileAsManaged>\n"
    "      <!--\n"
    "      <SDLCheck>false</SDLCheck>\n"
    "      -->\n"
    "      <BufferSecurityCheck>false</BufferSecurityCheck>\n"
    "      <WholeProgramOptimization>true</WholeProgramOptimization>\n"
    "      <CallingConvention>StdCall</CallingConvention>\n"
    "      <AdditionalIncludeDirectories>$(ProjectDir)\\..</AdditionalIncludeDirectories>\n"
    "      <SubSystem>Console</SubSystem>\n"
    "      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>\n"
    "      <WholeProgramOptimization>true</WholeProgramOptimization>\n"
    "      <PreprocessorDefinitions>_WINDOWS;_USRDLL;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "      <CompileAsManaged>false</CompileAsManaged>\n"
    "      <!--\n"
    "      <BrowseInformationFile>$(IntDir)$(ProjectName).bsc</BrowseInformationFile>\n"
    "      -->\n"
    "    </ClCompile>\n"
    "\n"
    "    <ClCompile Condition=\"$(Platform) == 'x64'\">\n"
    "      <EnableEnhancedInstructionSet>AdvancedVectorExtensions2</EnableEnhancedInstructionSet>\n"
    "    </ClCompile>\n"
    "\n"
    "    <ClCompile Condition=\"$(Configuration) == 'Debug'\">\n"
    "      <Optimization>Disabled</Optimization>\n"
    "      <WholeProgramOptimization>false</WholeProgramOptimization>\n"
    "      <UseDebugLibraries>true</UseDebugLibraries>\n"
    "      <EnableCOMDATFolding>false</EnableCOMDATFolding>\n"
    "      <OptimizeReferences>false</OptimizeReferences>\n"
    "      <DebugInformationFormat>EditAndContinue</DebugInformationFormat>\n"
    "      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>\n"
    "    </ClCompile>\n"
    "\n"
    "    <ClCompile Condition=\"$(Configuration) == 'PGOptimize'\">\n"
    "      <TreatWarningAsError>false</TreatWarningAsError>\n"
    "    </ClCompile>\n"
    "\n"
    "    <!--\n"
    "    <Bscmake>\n"
    "      <PreserveSbr>true</PreserveSbr>\n"
    "    </Bscmake>\n"
    "    -->\n"
    "\n"
    "    <Link>\n"
    "      <IgnoreAllDefaultLibraries>true</IgnoreAllDefaultLibraries>\n"
    "      <SubSystem>Console</SubSystem>\n"
    "      <AdditionalDependencies>chkstk.obj;bufferoverflowU.lib;%(AdditionalDependencies)</AdditionalDependencies>\n"
    "      <RandomizedBaseAddress>true</RandomizedBaseAddress>\n"
    "      <DataExecutionPrevention>true</DataExecutionPrevention>\n"
    "      <SuppressStartupBanner>true</SuppressStartupBanner>\n"
    "      <!--\n"
    "      <TargetMachine>MachineX64</TargetMachine>\n"
    "      <BrowseInformation>true</BrowseInformation>\n"
    "      -->\n"
    "      <AssemblerOutput>All</AssemblerOutput>\n"
    "      <GenerateDebugInformation>true</GenerateDebugInformation>\n"
    "      <OptimizeReferences>true</OptimizeReferences>\n"
    "      <CreateHotPatchableImage></CreateHotPatchableImage>\n"
    "      <SetChecksum>true</SetChecksum>\n"
    "      <!--\n"
    "      <AdditionalOptions>/HIGHENTROPYVA %(AdditionalOptions)</AdditionalOptions>\n"
    "      -->\n"
    "      <ProfileGuidedDatabase>$(PGDDirectory)$(ProjectName).pgd</ProfileGuidedDatabase>\n"
    "      <LinkTimeCodeGeneration Condition=\"$(Configuration) == 'Release'\">UseLinkTimeCodeGeneration</LinkTimeCodeGeneration>\n"
    "      <LinkTimeCodeGeneration Condition=\"$(Configuration) == 'PGInstrument'\">PGInstrument</LinkTimeCodeGeneration>\n"
    "      <LinkTimeCodeGeneration Condition=\"$(Configuration) == 'PGUpdate'\">PGUpdate</LinkTimeCodeGeneration>\n"
    "      <LinkTimeCodeGeneration Condition=\"$(Configuration) == 'PGOptimize'\">PGOptimization</LinkTimeCodeGeneration>\n"
    "    </Link>\n"
    "\n"
    "    <Link Condition=\"$(Configuration) == 'PGInstrument' or $(Configuration) == 'PGUpdate'\">\n"
    "      <AdditionalDependencies>pgobootrun.lib;pgort.lib;libucrt.lib;libcmt.lib;libvcruntime.lib;%(AdditionalDependencies)</AdditionalDependencies>\n"
    "      <AdditionalLibraryDirectories>$(VC_LibraryPath_x64);%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>\n"
    "      <AdditionalOptions>/GENPROFILE:EXACT %(AdditionalOptions)</AdditionalOptions>\n"
    "    </Link>\n"
    "\n"
    "    <Link Condition=\"$(Configuration) == 'Debug'\">\n"
    "      <SetChecksum>false</SetChecksum>\n"
    "      <EnableCOMDATFolding>false</EnableCOMDATFolding>\n"
    "      <OptimizeReferences>false</OptimizeReferences>\n"
    "    </Link>\n"
    "\n"
    "    <Link Condition=\"$(Configuration) != 'Debug'\">\n"
    "      <EnableCOMDATFolding>true</EnableCOMDATFolding>\n"
    "      <Profile>true</Profile>\n"
    "    </Link>\n"
    "\n"
    "    <Link Condition=\"$(Configuration) == 'PGOptimize'\">\n"
    "      <AdditionalOptions>/USEPROFILE %(AdditionalOptions)</AdditionalOptions>\n"
    "    </Link>\n"
    "\n"
    "    <Midl>\n"
    "      <MkTypLibCompatible>true</MkTypLibCompatible>\n"
    "      <SuppressStartupBanner>true</SuppressStartupBanner>\n"
    "      <TargetEnvironment>Win32</TargetEnvironment>\n"
    "      <TargetEnvironment Condition=\"'$(Platform)' == 'x64'\">X64</TargetEnvironment>\n"
    "    </Midl>\n"
    "\n"
    "    <Lib Condition=\"$(Configuration) != 'Debug'\">\n"
    "      <LinkTimeCodeGeneration>true</LinkTimeCodeGeneration>\n"
    "    </Lib>\n"
    "\n"
    "    <ResourceCompile>\n"
    "      <Culture>0x0409</Culture>\n"
    "    </ResourceCompile>\n"
    "\n"
    "  </ItemDefinitionGroup>\n"
    "\n"
    "  <ImportGroup Label=\"ExtensionSettings\">\n"
    "    <Import Project=\"$(VCTargetsPath)\\BuildCustomizations\\masm.props\" />\n"
    "  </ImportGroup>\n"
    "\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='Debug|Win32'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>WIN32;_DEBUG;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='Debug|x64'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>_DEBUG;_WINDOWS;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|Win32'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>WIN32;NDEBUG;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup> <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|x64'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>NDEBUG;_WINDOWS;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='PGInstrument|Win32'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>WIN32;NDEBUG;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='PGInstrument|x64'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>NDEBUG;_WINDOWS;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='PGUpdate|Win32'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>WIN32;NDEBUG;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='PGUpdate|x64'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>NDEBUG;_WINDOWS;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)'=='PGInstrument'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>_BUILD_CONFIG_PGI;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)'=='PGUpdate'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>_BUILD_CONFIG_PGU;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)'=='PGOptimize'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>_BUILD_CONFIG_PGO;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)'=='Release'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>_BUILD_CONFIG_RELEASE;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "  <ItemDefinitionGroup Condition=\"'$(Configuration)'=='Debug'\">\n"
    "    <ClCompile>\n"
    "      <PreprocessorDefinitions>_BUILD_CONFIG_DEBUG;%(PreprocessorDefinitions)</PreprocessorDefinitions>\n"
    "    </ClCompile>\n"
    "  </ItemDefinitionGroup>\n"
    "</Project>\n"
;

const STRING CompiledPerfectHashVCPropsRawCString = {
    sizeof(CompiledPerfectHashVCPropsRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashVCPropsRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashVCPropsRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashVCPropsRawCString)
#endif