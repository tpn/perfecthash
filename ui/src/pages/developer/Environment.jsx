import { useEffect, useMemo, useRef, useState } from "react";
import TerminalSession from "../../components/TerminalSession.jsx";

const pythonOptions = [
  { value: "3.10", label: "Python 3.10" },
  { value: "3.11", label: "Python 3.11" },
  { value: "3.12", label: "Python 3.12" },
  { value: "3.13", label: "Python 3.13" },
  { value: "3.13t", label: "Python 3.13t (free-threaded)" },
  { value: "3.14t", label: "Python 3.14t (free-threaded)" }
];

const cudaOptions = [
  { value: "none", label: "CUDA none" },
  { value: "12.9", label: "CUDA 12.9" },
  { value: "13.0", label: "CUDA 13.0" },
  { value: "13.1", label: "CUDA 13.1" }
];

const gccVersions = ["12", "13", "14"];
const llvmVersions = ["16", "17", "18"];

const basePackages = [
  "rust",
  "cmake",
  "ninja",
  "make",
  "pkg-config",
  "pytest"
];

const quoteIfNeeded = (value) => {
  if (!value) return "";
  if (/\s/.test(value)) return `"${value}"`;
  return value;
};

const normalizePython = (value) => {
  if (!value) return { version: "3.13", freeThreaded: false };
  const trimmed = value.trim();
  if (trimmed.endsWith("t")) {
    return { version: trimmed.slice(0, -1), freeThreaded: true };
  }
  return { version: trimmed, freeThreaded: false };
};

const buildCondaPackages = ({
  pythonChoice,
  compilerFamily,
  compilerVersion,
  cuda
}) => {
  const { version, freeThreaded } = normalizePython(pythonChoice);
  const packages = [`python=${version}`, ...basePackages];

  if (freeThreaded) {
    packages.push(`python-freethreaded=${version}`);
  }

  if (cuda && cuda !== "none") {
    packages.push(`cuda-toolkit=${cuda}`);
  }

  if (compilerFamily === "gcc") {
    packages.push(`gcc=${compilerVersion}`, `gxx=${compilerVersion}`);
  } else {
    packages.push(
      `clang=${compilerVersion}`,
      `clangxx=${compilerVersion}`,
      `lld=${compilerVersion}`,
      `llvmdev=${compilerVersion}`
    );
  }

  return packages;
};

const formatCommand = (command, args) =>
  [quoteIfNeeded(command), ...(args || []).map(quoteIfNeeded)]
    .filter(Boolean)
    .join(" ");

export default function DeveloperEnvironment() {
  const [platformInfo, setPlatformInfo] = useState(null);
  const [platformError, setPlatformError] = useState("");
  const [condaTool, setCondaTool] = useState("mamba");
  const [pythonChoice, setPythonChoice] = useState("3.13");
  const [cudaChoice, setCudaChoice] = useState("none");
  const [compilerFamily, setCompilerFamily] = useState("llvm");
  const [compilerVersion, setCompilerVersion] = useState("18");
  const [envName, setEnvName] = useState("perfecthash-dev");
  const [includeRust, setIncludeRust] = useState(false);
  const [session, setSession] = useState(null);
  const [sudoPrompt, setSudoPrompt] = useState(null);
  const [sudoPassword, setSudoPassword] = useState("");
  const [actionStatus, setActionStatus] = useState("");
  const [bootstrapStatus, setBootstrapStatus] = useState("");
  const [isBootstrapping, setIsBootstrapping] = useState(false);
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
      } catch (error) {
        setPlatformError(error.message);
      }
    };

    fetchPlatform();
  }, []);

  const compilerOptions = useMemo(() => {
    return compilerFamily === "gcc" ? gccVersions : llvmVersions;
  }, [compilerFamily]);

  const condaPackages = useMemo(
    () =>
      buildCondaPackages({
        pythonChoice,
        compilerFamily,
        compilerVersion,
        cuda: cudaChoice
      }),
    [pythonChoice, compilerFamily, compilerVersion, cudaChoice]
  );

  const condaCreateCommand = useMemo(() => {
    return {
      command: condaTool,
      args: ["create", "-y", "-n", envName, "-c", "conda-forge", ...condaPackages]
    };
  }, [condaTool, envName, condaPackages]);

  const condaRunCommand = useMemo(() => {
    return {
      command: condaTool,
      args: ["run", "-n", envName, "python", "--version"]
    };
  }, [condaTool, envName]);

  const systemCommand = useMemo(() => {
    if (!platformInfo?.install) return null;
    return platformInfo.install;
  }, [platformInfo]);

  const systemEnv = useMemo(() => {
    if (!includeRust) return null;
    return { WITH_RUST: "1" };
  }, [includeRust]);

  const startSession = async ({
    title,
    command,
    args,
    env,
    allowSudo,
    autoMinimizeOnSuccess,
    trackCompletion,
    section
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
          env,
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
        autoMinimizeOnSuccess: Boolean(autoMinimizeOnSuccess),
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
        const minimized = payload.ok && current.autoMinimizeOnSuccess
          ? true
          : current.minimized;
        return {
          ...current,
          status,
          minimized,
          exitCode: payload.exitCode
        };
      }
      if (payload.state === "stream-error") {
        return current;
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

  const handleInstallPrereqs = async () => {
    if (!systemCommand) return;
    setActionStatus("");
    try {
      await startSession({
        title: "System prerequisites",
        command: systemCommand.command,
        args: systemCommand.args,
        allowSudo: true,
        env: systemEnv,
        section: "system"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleCreateConda = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "Conda environment (create)",
        command: condaCreateCommand.command,
        args: condaCreateCommand.args,
        section: "conda"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleRunConda = async () => {
    setActionStatus("");
    try {
      await startSession({
        title: "Conda environment (run)",
        command: condaRunCommand.command,
        args: condaRunCommand.args,
        section: "conda"
      });
    } catch (error) {
      setActionStatus(error.message);
    }
  };

  const handleBootstrap = async () => {
    if (!systemCommand || isBootstrapping) return;
    setIsBootstrapping(true);
    setBootstrapStatus("Installing system prerequisites...");
    setActionStatus("");

    try {
      const install = await startSession({
        title: "System prerequisites",
        command: systemCommand.command,
        args: systemCommand.args,
        allowSudo: true,
        autoMinimizeOnSuccess: true,
        trackCompletion: true,
        env: systemEnv,
        section: "system"
      });

      const installOk = install.completion
        ? await install.completion
        : false;

      if (!installOk) {
        setBootstrapStatus("System prerequisites failed.");
        setIsBootstrapping(false);
        return;
      }

      setBootstrapStatus("Creating conda environment...");
      const conda = await startSession({
        title: "Conda environment (create)",
        command: condaCreateCommand.command,
        args: condaCreateCommand.args,
        autoMinimizeOnSuccess: true,
        trackCompletion: true,
        section: "conda"
      });

      const condaOk = conda.completion ? await conda.completion : false;
      if (!condaOk) {
        setBootstrapStatus("Conda environment creation failed.");
      } else {
        setBootstrapStatus("Bootstrap complete.");
      }
    } catch (error) {
      setBootstrapStatus(error.message);
    } finally {
      setIsBootstrapping(false);
    }
  };

  return (
    <div className="stack">
      <section className="panel" style={{ "--stagger": 1 }}>
        <div className="panel-header">
          <div>
            <p className="eyebrow">Developer / Environment</p>
            <h2>Bootstrap a fresh clone</h2>
            <p>
              Pick system prerequisites and craft a Conda environment from
              composable parts. Each action opens a live terminal transcript.
            </p>
          </div>
          <div className="run-controls">
            <button
              type="button"
              className="button"
              onClick={handleBootstrap}
              disabled={!systemCommand || isBootstrapping}
            >
              Bootstrap
            </button>
          </div>
        </div>
        {bootstrapStatus && <p className="status">{bootstrapStatus}</p>}
        {actionStatus && <p className="status">{actionStatus}</p>}
      </section>

      <section className="panel env-row" style={{ "--stagger": 2 }}>
        <div className="env-row-header">
          <div>
            <h3>System prerequisites</h3>
            <p>
              Detects your host OS and runs the matching dependency install
              script.
            </p>
          </div>
          <button
            type="button"
            className="button"
            onClick={handleInstallPrereqs}
            disabled={!systemCommand}
          >
            Install
          </button>
        </div>
        {platformError && <p className="status">{platformError}</p>}
        {platformInfo && (
          <div className="env-meta">
            <div>
              <p className="eyebrow">Detected</p>
              <p className="env-value">
                {platformInfo.platformLabel} / {platformInfo.arch}
              </p>
              {platformInfo.distro && (
                <p className="env-muted">{platformInfo.distro}</p>
              )}
            </div>
            <div>
              <p className="eyebrow">Script</p>
              <p className="env-value">
                {platformInfo.install
                  ? platformInfo.install.label
                  : "No matching script"}
              </p>
              {platformInfo.requiresSudo && (
                <p className="env-muted">sudo password may be required</p>
              )}
            </div>
          </div>
        )}
        <div className="form-grid">
          <label>
            Include Rust/Cargo
            <div className="mode-toggle">
              <button
                type="button"
                className={includeRust ? "toggle active" : "toggle"}
                onClick={() => setIncludeRust(true)}
              >
                Yes
              </button>
              <button
                type="button"
                className={!includeRust ? "toggle active" : "toggle"}
                onClick={() => setIncludeRust(false)}
              >
                No
              </button>
            </div>
          </label>
        </div>
        {includeRust && (
          <p className="env-muted">Adds Rust/Cargo to the install script.</p>
        )}
        {systemCommand && (
          <pre className="command-preview">
            {formatCommand(systemCommand.command, systemCommand.args)}
          </pre>
        )}
        {session && session.section === "system" && (
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
            <h3>Conda environment</h3>
            <p>
              Compose Python, CUDA, and compiler versions, then create or run
              the environment with conda/mamba.
            </p>
          </div>
          <div className="run-controls">
            <button type="button" className="button" onClick={handleCreateConda}>
              Create
            </button>
            <button type="button" className="button" onClick={handleRunConda}>
              Run
            </button>
          </div>
        </div>
        <div className="form-grid">
          <label>
            Tooling
            <div className="mode-toggle">
              <button
                type="button"
                className={condaTool === "mamba" ? "toggle active" : "toggle"}
                onClick={() => setCondaTool("mamba")}
              >
                Mamba
              </button>
              <button
                type="button"
                className={condaTool === "conda" ? "toggle active" : "toggle"}
                onClick={() => setCondaTool("conda")}
              >
                Conda
              </button>
            </div>
          </label>
          <label>
            Python
            <select
              value={pythonChoice}
              onChange={(event) => setPythonChoice(event.target.value)}
            >
              {pythonOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            CUDA
            <select
              value={cudaChoice}
              onChange={(event) => setCudaChoice(event.target.value)}
            >
              {cudaOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Compiler family
            <div className="mode-toggle">
              <button
                type="button"
                className={compilerFamily === "llvm" ? "toggle active" : "toggle"}
                onClick={() => {
                  setCompilerFamily("llvm");
                  setCompilerVersion("18");
                }}
              >
                LLVM
              </button>
              <button
                type="button"
                className={compilerFamily === "gcc" ? "toggle active" : "toggle"}
                onClick={() => {
                  setCompilerFamily("gcc");
                  setCompilerVersion("14");
                }}
              >
                GCC
              </button>
            </div>
          </label>
          <label>
            Compiler version
            <select
              value={compilerVersion}
              onChange={(event) => setCompilerVersion(event.target.value)}
            >
              {compilerOptions.map((version) => (
                <option key={version} value={version}>
                  {version}
                </option>
              ))}
            </select>
          </label>
          <label>
            Environment name
            <input
              type="text"
              value={envName}
              onChange={(event) => setEnvName(event.target.value)}
            />
          </label>
        </div>
        <pre className="command-preview">
          {formatCommand(condaCreateCommand.command, condaCreateCommand.args)}
        </pre>
        <pre className="command-preview">
          {formatCommand(condaRunCommand.command, condaRunCommand.args)}
        </pre>
        {session && session.section === "conda" && (
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
    </div>
  );
}
