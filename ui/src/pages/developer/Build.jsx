import { useEffect, useMemo, useRef, useState } from "react";
import TerminalSession from "../../components/TerminalSession.jsx";

const buildTypes = ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"];

const BUILD_DIR_KEY = "perfecthash.buildDir";
const BUILD_CONFIG_KEY = "perfecthash.buildConfig";
const INSTALL_PREFIX_KEY = "perfecthash.installPrefix";
const CARGO_EXECUTABLE_KEY = "perfecthash.cargoExecutable";

const readStorageValue = (key, fallback) => {
  if (typeof window === "undefined") return fallback;
  try {
    const value = window.localStorage.getItem(key);
    return value || fallback;
  } catch (error) {
    return fallback;
  }
};

const generatorOptions = [
  { value: "Ninja", label: "Ninja" },
  { value: "Ninja Multi-Config", label: "Ninja Multi-Config" },
  { value: "Unix Makefiles", label: "Unix Makefiles" },
  { value: "Visual Studio 17 2022", label: "Visual Studio 17 2022" }
];

const multiConfigGenerators = new Set([
  "Ninja Multi-Config",
  "Visual Studio 17 2022"
]);

const quoteIfNeeded = (value) => {
  if (!value) return "";
  if (/\s/.test(value)) return `"${value}"`;
  return value;
};

const quotePosix = (value) => `'${String(value).replace(/'/g, `'\\''`)}'`;

const quotePowerShell = (value) =>
  `'${String(value).replace(/'/g, "''")}'`;

const formatCommand = (command, args) =>
  [quoteIfNeeded(command), ...(args || []).map(quoteIfNeeded)]
    .filter(Boolean)
    .join(" ");

export default function DeveloperBuild() {
  const [platformInfo, setPlatformInfo] = useState(null);
  const [platformError, setPlatformError] = useState("");
  const [generator, setGenerator] = useState("Ninja");
  const [sourceDir, setSourceDir] = useState(".");
  const [buildDir, setBuildDir] = useState(
    readStorageValue(BUILD_DIR_KEY, "build")
  );
  const [buildType, setBuildType] = useState("Release");
  const [installPrefix, setInstallPrefix] = useState(
    readStorageValue(INSTALL_PREFIX_KEY, "install")
  );
  const [cargoExecutable, setCargoExecutable] = useState(
    readStorageValue(CARGO_EXECUTABLE_KEY, "")
  );
  const [exportCompileCommands, setExportCompileCommands] = useState(true);
  const [enableNativeArch, setEnableNativeArch] = useState(true);
  const [enableTests, setEnableTests] = useState(true);
  const [enableInstall, setEnableInstall] = useState(true);
  const [enableCuda, setEnableCuda] = useState(false);
  const [enablePenter, setEnablePenter] = useState(false);
  const [cudaArch, setCudaArch] = useState("");
  const [buildConfig, setBuildConfig] = useState(
    readStorageValue(BUILD_CONFIG_KEY, "Release")
  );
  const [buildTarget, setBuildTarget] = useState("");
  const [buildParallel, setBuildParallel] = useState("");
  const [cleanFirst, setCleanFirst] = useState(false);
  const [installComponent, setInstallComponent] = useState("");
  const [resetBeforeConfigure, setResetBeforeConfigure] = useState(false);
  const [session, setSession] = useState(null);
  const [sudoPrompt, setSudoPrompt] = useState(null);
  const [sudoPassword, setSudoPassword] = useState("");
  const [actionStatus, setActionStatus] = useState("");
  const completionRef = useRef(new Map());
  const startInFlightRef = useRef(false);

  useEffect(() => {
    const fetchPlatform = async () => {
      try {
        const response = await fetch("/api/platform");
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Unable to detect platform.");
        }
        setPlatformInfo(payload);
        if (payload.platform === "win32") {
          setGenerator("Visual Studio 17 2022");
        }
      } catch (error) {
        setPlatformError(error.message);
      }
    };

    fetchPlatform();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(BUILD_DIR_KEY, buildDir);
    } catch (error) {
      // Ignore storage errors.
    }
  }, [buildDir]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(BUILD_CONFIG_KEY, buildConfig);
    } catch (error) {
      // Ignore storage errors.
    }
  }, [buildConfig]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(INSTALL_PREFIX_KEY, installPrefix);
    } catch (error) {
      // Ignore storage errors.
    }
  }, [installPrefix]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(CARGO_EXECUTABLE_KEY, cargoExecutable);
    } catch (error) {
      // Ignore storage errors.
    }
  }, [cargoExecutable]);

  const isMultiConfig = useMemo(
    () => multiConfigGenerators.has(generator),
    [generator]
  );

  const canResetBuildDir = useMemo(() => {
    const trimmed = buildDir.trim();
    if (!trimmed) return false;
    const normalized = trimmed.replace(/\\/g, "/");
    if (normalized === "." || normalized === "/" || normalized === "./") {
      return false;
    }
    if (normalized.includes("..")) {
      return false;
    }
    return true;
  }, [buildDir]);

  const configureArgs = useMemo(() => {
    const args = ["-S", sourceDir, "-B", buildDir];
    if (generator) {
      args.push("-G", generator);
    }
    if (!isMultiConfig && buildType) {
      args.push(`-DCMAKE_BUILD_TYPE=${buildType}`);
    }
    if (installPrefix) {
      args.push(`-DCMAKE_INSTALL_PREFIX=${installPrefix}`);
    }
    if (cargoExecutable) {
      args.push(`-DCARGO_EXECUTABLE=${cargoExecutable}`);
    }
    args.push(`-DPERFECTHASH_ENABLE_NATIVE_ARCH=${enableNativeArch ? "ON" : "OFF"}`);
    args.push(`-DPERFECTHASH_ENABLE_TESTS=${enableTests ? "ON" : "OFF"}`);
    args.push(`-DPERFECTHASH_ENABLE_INSTALL=${enableInstall ? "ON" : "OFF"}`);
    args.push(`-DPERFECTHASH_ENABLE_PENTER=${enablePenter ? "ON" : "OFF"}`);
    args.push(`-DPERFECTHASH_USE_CUDA=${enableCuda ? "ON" : "OFF"}`);
    if (enableCuda && cudaArch) {
      args.push(`-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}`);
    }
    if (exportCompileCommands) {
      args.push("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");
    }
    return args;
  }, [
    sourceDir,
    buildDir,
    generator,
    isMultiConfig,
    buildType,
    installPrefix,
    cargoExecutable,
    enableNativeArch,
    enableTests,
    enableInstall,
    enableCuda,
    enablePenter,
    cudaArch,
    exportCompileCommands
  ]);

  const configureExecution = useMemo(() => {
    const cmakePreview = formatCommand("cmake", configureArgs);
    if (!resetBeforeConfigure || !canResetBuildDir) {
      return {
        command: "cmake",
        args: configureArgs,
        preview: cmakePreview
      };
    }

    if (platformInfo?.platform === "win32") {
      const resetCommand = `if (Test-Path -LiteralPath ${quotePowerShell(buildDir)}) { Remove-Item -Recurse -Force -LiteralPath ${quotePowerShell(buildDir)} }`;
      return {
        command: "powershell",
        args: ["-NoProfile", "-Command", `${resetCommand}; ${cmakePreview}`],
        preview: `${resetCommand}\n${cmakePreview}`
      };
    }

    const resetCommand = `rm -rf -- ${quotePosix(buildDir)}`;
    return {
      command: "bash",
      args: ["-lc", `${resetCommand} && ${cmakePreview}`],
      preview: `${resetCommand}\n${cmakePreview}`
    };
  }, [
    configureArgs,
    resetBeforeConfigure,
    canResetBuildDir,
    buildDir,
    platformInfo?.platform
  ]);

  const buildArgs = useMemo(() => {
    const args = ["--build", buildDir];
    if (buildConfig) {
      args.push("--config", buildConfig);
    }
    if (buildTarget) {
      args.push("--target", buildTarget);
    }
    if (buildParallel) {
      args.push("--parallel", buildParallel);
    }
    if (cleanFirst) {
      args.push("--clean-first");
    }
    return args;
  }, [buildDir, buildConfig, buildTarget, buildParallel, cleanFirst]);

  const installArgs = useMemo(() => {
    const args = ["--install", buildDir];
    if (buildConfig) {
      args.push("--config", buildConfig);
    }
    if (installPrefix) {
      args.push("--prefix", installPrefix);
    }
    if (installComponent) {
      args.push("--component", installComponent);
    }
    return args;
  }, [buildDir, buildConfig, installPrefix, installComponent]);

  const startSession = async ({
    title,
    command,
    args,
    allowSudo,
    section,
    trackCompletion
  }) => {
    if (startInFlightRef.current) {
      return { id: session?.id ?? null, completion: null };
    }
    startInFlightRef.current = true;

    if (session && session.status === "running") {
      await fetch("/api/terminal/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: session.id })
      });
    }

    try {
      const response = await fetch("/api/terminal/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          command,
          args,
          allowSudo,
          cwd: platformInfo?.rootDir
        })
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Unable to start terminal session.");
      }

      const nextSession = {
        id: payload.id,
        title,
        status: "running",
        minimized: false,
        commandPreview: formatCommand(command, args),
        section
      };

      setSession(nextSession);

      if (trackCompletion) {
        const completion = new Promise((resolve) => {
          completionRef.current.set(payload.id, resolve);
        });
        return { id: payload.id, completion };
      }

      return { id: payload.id, completion: null };
    } finally {
      startInFlightRef.current = false;
    }
  };

  const handleSessionToggle = (id) => {
    setSession((current) => {
      if (!current || current.id !== id) return current;
      return { ...current, minimized: !current.minimized };
    });
  };

  const handleSessionStop = async (id) => {
    setSession((current) => {
      if (!current || current.id !== id) return current;
      if (current.status !== "running") return current;
      return { ...current, status: "stopping" };
    });
    await fetch("/api/terminal/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id })
    });
  };

  const handleSessionStatus = (id, payload) => {
    setSession((current) => {
      if (!current || current.id !== id) return current;
      if (payload.state === "exit") {
        const status = payload.ok ? "success" : "error";
        return {
          ...current,
          status,
          exitCode: payload.exitCode
        };
      }
      return current;
    });

    if (payload.state === "exit") {
      const resolver = completionRef.current.get(id);
      if (resolver) {
        resolver(Boolean(payload.ok));
        completionRef.current.delete(id);
      }
    }
  };

  const handleSudoPrompt = (id, message) => {
    setSudoPrompt({ id, message });
  };

  const handleSubmitPassword = async () => {
    if (!sudoPrompt) return;
    const password = sudoPassword;
    setSudoPassword("");
    setSudoPrompt(null);
    await fetch("/api/terminal/input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: sudoPrompt.id,
        input: password
      })
    });
  };

  const handleCancelPassword = () => {
    setSudoPassword("");
    setSudoPrompt(null);
  };

  const handleConfigure = async () => {
    setActionStatus("");
    try {
      if (resetBeforeConfigure && !canResetBuildDir) {
        setActionStatus("Reset build directory is disabled for unsafe paths.");
        return;
      }
      await startSession({
        title: "CMake configure",
        command: configureExecution.command,
        args: configureExecution.args,
        section: "configure"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleBuild = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "CMake build",
        command: "cmake",
        args: buildArgs,
        section: "build"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleInstall = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "CMake install",
        command: "cmake",
        args: installArgs,
        allowSudo: true,
        section: "install"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  return (
    <div className="stack">
      <section className="panel" style={{ "--stagger": 1 }}>
        <p className="eyebrow">Developer / Build</p>
        <h2>Build the core libraries</h2>
        <p>
          Configure and build with CMake. Pick a generator, build type, and
          common PerfectHash options before compiling or installing.
        </p>
        {platformError && <p className="status">{platformError}</p>}
        {actionStatus && <p className="status">{actionStatus}</p>}
      </section>

      <section className="panel env-row" style={{ "--stagger": 2 }}>
        <div className="env-row-header">
          <div>
            <h3>CMake configure</h3>
            <p>Generate the build directory with the selected options.</p>
          </div>
          <button type="button" className="button" onClick={handleConfigure}>
            Configure
          </button>
        </div>
        <div className="form-grid">
          <label>
            Source directory
            <input
              type="text"
              value={sourceDir}
              onChange={(event) => setSourceDir(event.target.value)}
            />
          </label>
          <label>
            Build directory
            <input
              type="text"
              value={buildDir}
              onChange={(event) => setBuildDir(event.target.value)}
            />
          </label>
          <label>
            Generator
            <select
              value={generator}
              onChange={(event) => setGenerator(event.target.value)}
            >
              {generatorOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Build type
            <select
              value={buildType}
              onChange={(event) => setBuildType(event.target.value)}
              disabled={isMultiConfig}
            >
              {buildTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>
          <label>
            Reset build directory
            <div className="mode-toggle">
              <button
                type="button"
                className={resetBeforeConfigure ? "toggle active" : "toggle"}
                onClick={() => setResetBeforeConfigure(true)}
                disabled={!canResetBuildDir}
              >
                Yes
              </button>
              <button
                type="button"
                className={!resetBeforeConfigure ? "toggle active" : "toggle"}
                onClick={() => setResetBeforeConfigure(false)}
              >
                No
              </button>
            </div>
          </label>
          <label>
            Install prefix preset
            <select
              value=""
              onChange={(event) => {
                if (event.target.value) {
                  setInstallPrefix(event.target.value);
                }
              }}
            >
              <option value="">Select preset</option>
              <option value="./install">./install</option>
              <option value="$HOME">$HOME</option>
              <option value="/usr/local">/usr/local</option>
              <option value="/opt">/opt</option>
              <option value="/usr">/usr</option>
            </select>
          </label>
          <label>
            Install prefix
            <input
              type="text"
              placeholder="/usr/local"
              value={installPrefix}
              onChange={(event) => setInstallPrefix(event.target.value)}
            />
          </label>
          <label>
            Cargo executable
            <input
              type="text"
              placeholder="cargo"
              value={cargoExecutable}
              onChange={(event) => setCargoExecutable(event.target.value)}
            />
          </label>
          <label>
            CUDA architectures
            <input
              type="text"
              placeholder="89"
              value={cudaArch}
              onChange={(event) => setCudaArch(event.target.value)}
              disabled={!enableCuda}
            />
          </label>
        </div>

        {isMultiConfig && (
          <p className="status">
            Multi-config generator selected. Build type is chosen at build time.
          </p>
        )}

        <div className="flag-grid">
          <label className="flag-item">
            <input
              type="checkbox"
              checked={exportCompileCommands}
              onChange={(event) => setExportCompileCommands(event.target.checked)}
            />
            <span className="flag-label">
              <span className="flag-title">Export compile commands</span>
              <span className="flag-description">Generate compile_commands.json.</span>
            </span>
          </label>
          <label className="flag-item">
            <input
              type="checkbox"
              checked={enableNativeArch}
              onChange={(event) => setEnableNativeArch(event.target.checked)}
            />
            <span className="flag-label">
              <span className="flag-title">Enable native arch</span>
              <span className="flag-description">Adds -march=native when supported.</span>
            </span>
          </label>
          <label className="flag-item">
            <input
              type="checkbox"
              checked={enableTests}
              onChange={(event) => setEnableTests(event.target.checked)}
            />
            <span className="flag-label">
              <span className="flag-title">Enable tests</span>
              <span className="flag-description">Build the test targets.</span>
            </span>
          </label>
          <label className="flag-item">
            <input
              type="checkbox"
              checked={enableInstall}
              onChange={(event) => setEnableInstall(event.target.checked)}
            />
            <span className="flag-label">
              <span className="flag-title">Enable install rules</span>
              <span className="flag-description">Include install targets.</span>
            </span>
          </label>
          <label className="flag-item">
            <input
              type="checkbox"
              checked={enableCuda}
              onChange={(event) => setEnableCuda(event.target.checked)}
            />
            <span className="flag-label">
              <span className="flag-title">Enable CUDA</span>
              <span className="flag-description">Build CUDA support when available.</span>
            </span>
          </label>
          <label className="flag-item">
            <input
              type="checkbox"
              checked={enablePenter}
              onChange={(event) => setEnablePenter(event.target.checked)}
            />
            <span className="flag-label">
              <span className="flag-title">Enable Penter</span>
              <span className="flag-description">Include FunctionHook support.</span>
            </span>
          </label>
        </div>

        <pre className="command-preview">
          {configureExecution.preview}
        </pre>

        {session && session.section === "configure" && (
          <TerminalSession
            session={session}
            onToggle={handleSessionToggle}
            onStatusChange={handleSessionStatus}
            onSudoPrompt={handleSudoPrompt}
            onStop={handleSessionStop}
          />
        )}
      </section>

      <section className="panel env-row" style={{ "--stagger": 3 }}>
        <div className="env-row-header">
          <div>
            <h3>CMake build</h3>
            <p>Build the selected configuration.</p>
          </div>
          <button type="button" className="button" onClick={handleBuild}>
            Build
          </button>
        </div>
        <div className="form-grid">
          <label>
            Build directory
            <input
              type="text"
              value={buildDir}
              onChange={(event) => setBuildDir(event.target.value)}
            />
          </label>
          <label>
            Build config
            <select
              value={buildConfig}
              onChange={(event) => setBuildConfig(event.target.value)}
            >
              {buildTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>
          <label>
            Target (optional)
            <input
              type="text"
              placeholder="all"
              value={buildTarget}
              onChange={(event) => setBuildTarget(event.target.value)}
            />
          </label>
          <label>
            Parallel jobs
            <input
              type="number"
              min="1"
              placeholder="8"
              value={buildParallel}
              onChange={(event) => setBuildParallel(event.target.value)}
            />
          </label>
          <label>
            Clean first
            <div className="mode-toggle">
              <button
                type="button"
                className={cleanFirst ? "toggle active" : "toggle"}
                onClick={() => setCleanFirst(true)}
              >
                Yes
              </button>
              <button
                type="button"
                className={!cleanFirst ? "toggle active" : "toggle"}
                onClick={() => setCleanFirst(false)}
              >
                No
              </button>
            </div>
          </label>
        </div>
        <pre className="command-preview">
          {formatCommand("cmake", buildArgs)}
        </pre>

        {session && session.section === "build" && (
          <TerminalSession
            session={session}
            onToggle={handleSessionToggle}
            onStatusChange={handleSessionStatus}
            onSudoPrompt={handleSudoPrompt}
            onStop={handleSessionStop}
          />
        )}
      </section>

      <section className="panel env-row" style={{ "--stagger": 4 }}>
        <div className="env-row-header">
          <div>
            <h3>CMake install</h3>
            <p>Install artifacts to the configured prefix.</p>
          </div>
          <button type="button" className="button" onClick={handleInstall}>
            Install
          </button>
        </div>
        <div className="form-grid">
          <label>
            Build directory
            <input
              type="text"
              value={buildDir}
              onChange={(event) => setBuildDir(event.target.value)}
            />
          </label>
          <label>
            Build config
            <select
              value={buildConfig}
              onChange={(event) => setBuildConfig(event.target.value)}
            >
              {buildTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>
          <label>
            Install prefix override
            <input
              type="text"
              placeholder="/usr/local"
              value={installPrefix}
              onChange={(event) => setInstallPrefix(event.target.value)}
            />
          </label>
          <label>
            Component
            <input
              type="text"
              placeholder=""
              value={installComponent}
              onChange={(event) => setInstallComponent(event.target.value)}
            />
          </label>
        </div>
        <pre className="command-preview">
          {formatCommand("cmake", installArgs)}
        </pre>

        {session && session.section === "install" && (
          <TerminalSession
            session={session}
            onToggle={handleSessionToggle}
            onStatusChange={handleSessionStatus}
            onSudoPrompt={handleSudoPrompt}
            onStop={handleSessionStop}
          />
        )}
      </section>

      {sudoPrompt && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal-card">
            <h3>Sudo password required</h3>
            <p className="env-muted">{sudoPrompt.message}</p>
            <label>
              Password
              <input
                type="password"
                value={sudoPassword}
                onChange={(event) => setSudoPassword(event.target.value)}
              />
            </label>
            <div className="modal-actions">
              <button type="button" className="button" onClick={handleSubmitPassword}>
                Submit
              </button>
              <button
                type="button"
                className="button ghost"
                onClick={handleCancelPassword}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <section className="panel" style={{ "--stagger": 5 }}>
        <h3>Agent companion</h3>
        <p>
          Build guidance for automation lives in
          <code>skills/improved-ui/SKILL.md</code>.
        </p>
      </section>
    </div>
  );
}
