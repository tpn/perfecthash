import { useEffect, useMemo, useRef, useState } from "react";
import TerminalSession from "../../components/TerminalSession.jsx";

const buildTypes = ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"];

const BUILD_DIR_KEY = "perfecthash.buildDir";
const BUILD_CONFIG_KEY = "perfecthash.buildConfig";

const readStorageValue = (key, fallback) => {
  if (typeof window === "undefined") return fallback;
  try {
    const value = window.localStorage.getItem(key);
    return value || fallback;
  } catch (error) {
    return fallback;
  }
};

const quoteIfNeeded = (value) => {
  if (!value) return "";
  if (/\s/.test(value)) return `"${value}"`;
  return value;
};

const formatCommand = (command, args) =>
  [quoteIfNeeded(command), ...(args || []).map(quoteIfNeeded)]
    .filter(Boolean)
    .join(" ");

export default function DeveloperTest() {
  const initialBuildDir = readStorageValue(BUILD_DIR_KEY, "build");
  const initialBuildConfig = readStorageValue(BUILD_CONFIG_KEY, "Release");
  const [platformInfo, setPlatformInfo] = useState(null);
  const [platformError, setPlatformError] = useState("");
  const [buildDir, setBuildDir] = useState(initialBuildDir);
  const [buildConfig, setBuildConfig] = useState(initialBuildConfig);
  const [ctestParallel, setCtestParallel] = useState("");
  const [outputOnFailure, setOutputOnFailure] = useState(true);
  const [ctestRegex, setCtestRegex] = useState("");
  const [testExe, setTestExe] = useState(
    `${initialBuildDir}/bin/PerfectHashCreate`
  );
  const [testKeys, setTestKeys] = useState("data/mshtml-37209.keys");
  const [testOutput, setTestOutput] = useState("output/cli-test");
  const [testArgs, setTestArgs] = useState("");
  const [testFlags, setTestFlags] = useState("");
  const [codegenOutput, setCodegenOutput] = useState("output/cli-codegen");
  const [pythonPath, setPythonPath] = useState("python");
  const [cargoPath, setCargoPath] = useState("cargo");
  const [session, setSession] = useState(null);
  const [sudoPrompt, setSudoPrompt] = useState(null);
  const [sudoPassword, setSudoPassword] = useState("");
  const [actionStatus, setActionStatus] = useState("");
  const completionRef = useRef(new Map());
  const previousDefaultExeRef = useRef(null);
  const startInFlightRef = useRef(false);

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
    const fetchPlatform = async () => {
      try {
        const response = await fetch("/api/platform");
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Unable to detect platform.");
        }
        setPlatformInfo(payload);
      } catch (error) {
        setPlatformError(error.message);
      }
    };

    fetchPlatform();
  }, []);

  const defaultTestExe = useMemo(() => {
    const suffix = platformInfo?.platform === "win32" ? ".exe" : "";
    return `${buildDir}/bin/PerfectHashCreate${suffix}`;
  }, [buildDir, platformInfo?.platform]);

  useEffect(() => {
    if (!previousDefaultExeRef.current) {
      previousDefaultExeRef.current = defaultTestExe;
      return;
    }
    const previousDefault = previousDefaultExeRef.current;
    previousDefaultExeRef.current = defaultTestExe;
    setTestExe((current) =>
      current === previousDefault ? defaultTestExe : current
    );
  }, [defaultTestExe]);

  const ctestArgs = useMemo(() => {
    const args = ["--test-dir", buildDir];
    if (buildConfig) {
      args.push("-C", buildConfig);
    }
    if (outputOnFailure) {
      args.push("--output-on-failure");
    }
    if (ctestParallel) {
      args.push("--parallel", ctestParallel);
    }
    if (ctestRegex) {
      args.push("-R", ctestRegex);
    }
    return args;
  }, [buildDir, buildConfig, outputOnFailure, ctestParallel, ctestRegex]);

  const cliTestArgs = useMemo(() => {
    const args = [
      "-P",
      "tests/run_cli_test.cmake",
      `-DTEST_EXE=${testExe}`,
      `-DTEST_KEYS=${testKeys}`,
      `-DTEST_OUTPUT=${testOutput}`
    ];
    if (testArgs) {
      args.push(`-DTEST_ARGS=${testArgs}`);
    }
    if (testFlags) {
      args.push(`-DTEST_FLAGS=${testFlags}`);
    }
    return args;
  }, [testExe, testKeys, testOutput, testArgs, testFlags]);

  const codegenArgs = useMemo(() => {
    const args = [
      "-P",
      "tests/run_cli_codegen_test.cmake",
      `-DTEST_EXE=${testExe}`,
      `-DTEST_KEYS=${testKeys}`,
      `-DTEST_OUTPUT=${codegenOutput}`,
      `-DTEST_PYTHON=${pythonPath}`
    ];
    if (cargoPath) {
      args.push(`-DTEST_CARGO=${cargoPath}`);
    }
    if (buildConfig) {
      args.push(`-DTEST_BUILD_CONFIG=${buildConfig}`);
    }
    if (testArgs) {
      args.push(`-DTEST_ARGS=${testArgs}`);
    }
    if (testFlags) {
      args.push(`-DTEST_FLAGS=${testFlags}`);
    }
    return args;
  }, [
    testExe,
    testKeys,
    codegenOutput,
    pythonPath,
    cargoPath,
    buildConfig,
    testArgs,
    testFlags
  ]);

  const uiUnitArgs = ["--prefix", "ui", "test"];
  const uiE2EArgs = ["--prefix", "ui", "run", "test:e2e"];

  const startSession = async ({ title, command, args, allowSudo, section }) => {
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

  const handleRunCtest = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "CTest",
        command: "ctest",
        args: ctestArgs,
        section: "ctest"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleRunCliTest = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "CLI smoke test",
        command: "cmake",
        args: cliTestArgs,
        section: "cli"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleRunCodegen = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "CLI codegen test",
        command: "cmake",
        args: codegenArgs,
        section: "codegen"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleRunUiUnit = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "UI unit tests",
        command: "npm",
        args: uiUnitArgs,
        section: "ui"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleRunUiE2e = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "UI e2e tests",
        command: "npm",
        args: uiE2EArgs,
        section: "ui"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  return (
    <div className="stack">
      <section className="panel" style={{ "--stagger": 1 }}>
        <p className="eyebrow">Developer / Test</p>
        <h2>Verify builds and UI flows</h2>
        <p>
          Run CTest, CLI regression checks, and UI test suites. Defaults track
          the build directory and config selected on the Build page.
        </p>
        {platformError && <p className="status">{platformError}</p>}
        {actionStatus && <p className="status">{actionStatus}</p>}
      </section>

      <section className="panel env-row" style={{ "--stagger": 2 }}>
        <div className="env-row-header">
          <div>
            <h3>CTest</h3>
            <p>Run the configured CMake test suite.</p>
          </div>
          <button type="button" className="button" onClick={handleRunCtest}>
            Run
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
            Parallel jobs
            <input
              type="number"
              min="1"
              placeholder="8"
              value={ctestParallel}
              onChange={(event) => setCtestParallel(event.target.value)}
            />
          </label>
          <label>
            Test regex
            <input
              type="text"
              placeholder="PerfectHash"
              value={ctestRegex}
              onChange={(event) => setCtestRegex(event.target.value)}
            />
          </label>
          <label>
            Output on failure
            <div className="mode-toggle">
              <button
                type="button"
                className={outputOnFailure ? "toggle active" : "toggle"}
                onClick={() => setOutputOnFailure(true)}
              >
                Yes
              </button>
              <button
                type="button"
                className={!outputOnFailure ? "toggle active" : "toggle"}
                onClick={() => setOutputOnFailure(false)}
              >
                No
              </button>
            </div>
          </label>
        </div>
        <pre className="command-preview">
          {formatCommand("ctest", ctestArgs)}
        </pre>
        {session && session.section === "ctest" && (
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
            <h3>CLI smoke test</h3>
            <p>Run a single PerfectHashCreate invocation via CMake script.</p>
          </div>
          <button type="button" className="button" onClick={handleRunCliTest}>
            Run
          </button>
        </div>
        <div className="form-grid">
          <label>
            PerfectHashCreate executable
            <input
              type="text"
              value={testExe}
              onChange={(event) => setTestExe(event.target.value)}
            />
            <p className="env-muted">
              Default is `build/bin/PerfectHashCreate` (multi-config builds may
              place the binary under `bin/&lt;Config&gt;`).
            </p>
          </label>
          <label>
            Keys file
            <input
              type="text"
              value={testKeys}
              onChange={(event) => setTestKeys(event.target.value)}
            />
          </label>
          <label>
            Output directory
            <input
              type="text"
              value={testOutput}
              onChange={(event) => setTestOutput(event.target.value)}
            />
          </label>
          <label>
            Extra args (use | to separate)
            <input
              type="text"
              placeholder="--BestCoverageAttempts=8|--BestCoverageType=MemoryCoverage"
              value={testArgs}
              onChange={(event) => setTestArgs(event.target.value)}
            />
          </label>
          <label>
            Extra flags (use | to separate)
            <input
              type="text"
              placeholder="--Compile"
              value={testFlags}
              onChange={(event) => setTestFlags(event.target.value)}
            />
          </label>
        </div>
        <pre className="command-preview">
          {formatCommand("cmake", cliTestArgs)}
        </pre>
        {session && session.section === "cli" && (
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
            <h3>CLI codegen test</h3>
            <p>Generate tables and validate codegen outputs.</p>
          </div>
          <button type="button" className="button" onClick={handleRunCodegen}>
            Run
          </button>
        </div>
        <div className="form-grid">
          <label>
            PerfectHashCreate executable
            <input
              type="text"
              value={testExe}
              onChange={(event) => setTestExe(event.target.value)}
            />
          </label>
          <label>
            Keys file
            <input
              type="text"
              value={testKeys}
              onChange={(event) => setTestKeys(event.target.value)}
            />
          </label>
          <label>
            Output directory
            <input
              type="text"
              value={codegenOutput}
              onChange={(event) => setCodegenOutput(event.target.value)}
            />
          </label>
          <label>
            Python executable
            <input
              type="text"
              value={pythonPath}
              onChange={(event) => setPythonPath(event.target.value)}
            />
          </label>
          <label>
            Cargo (optional)
            <input
              type="text"
              value={cargoPath}
              onChange={(event) => setCargoPath(event.target.value)}
            />
            <p className="env-muted">
              Required when generated code includes Cargo.toml.
            </p>
          </label>
        </div>
        <pre className="command-preview">
          {formatCommand("cmake", codegenArgs)}
        </pre>
        {session && session.section === "codegen" && (
          <TerminalSession
            session={session}
            onToggle={handleSessionToggle}
            onStatusChange={handleSessionStatus}
            onSudoPrompt={handleSudoPrompt}
            onStop={handleSessionStop}
          />
        )}
      </section>

      <section className="panel env-row" style={{ "--stagger": 5 }}>
        <div className="env-row-header">
          <div>
            <h3>UI tests</h3>
            <p>Run unit tests or Playwright e2e from the UI workspace.</p>
          </div>
          <div className="run-controls">
            <button type="button" className="button" onClick={handleRunUiUnit}>
              Unit
            </button>
            <button type="button" className="button" onClick={handleRunUiE2e}>
              E2E
            </button>
          </div>
        </div>
        <pre className="command-preview">
          {formatCommand("npm", uiUnitArgs)}
        </pre>
        <pre className="command-preview">
          {formatCommand("npm", uiE2EArgs)}
        </pre>
        {session && session.section === "ui" && (
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

      <section className="panel" style={{ "--stagger": 6 }}>
        <h3>Agent companion</h3>
        <p>
          Test guidance for automation lives in
          <code>skills/improved-ui/SKILL.md</code>.
        </p>
      </section>
    </div>
  );
}
