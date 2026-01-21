import { useMemo, useState } from "react";

const algorithmOptions = [
  { value: "Chm01", label: "Chm01 (fast, primary)" },
  { value: "Chm02", label: "Chm02 (alternate)" }
];

const hashOptions = [
  "MultiplyShiftR",
  "MultiplyShiftLR",
  "MultiplyShiftRMultiply",
  "MultiplyShiftR2",
  "MultiplyShiftRX",
  "Mulshrolate1RX",
  "Mulshrolate2RX",
  "Mulshrolate3RX",
  "Mulshrolate4RX"
];

const maskOptions = [
  { value: "And", label: "And (power-of-two, fast)" },
  { value: "Modulus", label: "Modulus (experimental)" }
];

const createFlags = [
  {
    id: "skipTest",
    flag: "--SkipTestAfterCreate",
    label: "Skip post-create test",
    description: "Generate without running verification benchmarks."
  },
  {
    id: "compile",
    flag: "--Compile",
    label: "Compile outputs",
    description: "Build the generated projects (msbuild required)."
  }
];

const keysFlags = [
  {
    id: "largePages",
    flag: "--TryLargePagesForKeysData",
    label: "Try large pages",
    description: "Attempt large page allocation for key buffers."
  },
  {
    id: "skipVerify",
    flag: "--SkipKeysVerification",
    label: "Skip key verification",
    description: "Assume keys are sorted and skip bitmap checks."
  },
  {
    id: "disableDownsize",
    flag: "--DisableImplicitKeyDownsizing",
    label: "Disable implicit downsizing",
    description: "Keep 64-bit keys when 32-bit compression is possible."
  },
  {
    id: "inferKeySize",
    flag: "--TryInferKeySizeFromKeysFilename",
    label: "Infer key size",
    description: "Detect 64-bit key files with a 64.keys suffix."
  }
];

const tableFlags = [
  {
    id: "silent",
    flag: "--Silent",
    label: "Silent output",
    description: "Disable console status output (implies --Quiet)."
  },
  {
    id: "quiet",
    flag: "--Quiet",
    label: "Quiet output",
    description: "Suppress best-graph details in console output."
  },
  {
    id: "noFileIo",
    flag: "--NoFileIo",
    label: "No file I/O",
    description: "Run searches without writing any output files."
  },
  {
    id: "findBest",
    flag: "--FindBestGraph",
    label: "Find best graph",
    description: "Search for best coverage (requires coverage args)."
  },
  {
    id: "indexOnly",
    flag: "--IndexOnly",
    label: "Index only",
    description: "Generate only Index() without table values array."
  }
];

const defaultFlags = {
  skipTest: false,
  compile: false,
  largePages: false,
  skipVerify: false,
  disableDownsize: false,
  inferKeySize: true,
  silent: false,
  quiet: false,
  noFileIo: false,
  findBest: false,
  indexOnly: false
};

const quoteIfNeeded = (value) => {
  if (!value) return "";
  if (/\s/.test(value)) return `"${value}"`;
  return value;
};

const parseExtraArgs = (value) => {
  if (!value) return [];
  const tokens = [];
  const regex = /"([^"]*)"|'([^']*)'|\S+/g;
  for (const match of value.matchAll(regex)) {
    tokens.push(match[1] ?? match[2] ?? match[0]);
  }
  return tokens;
};

const formatDuration = (durationMs) => {
  if (typeof durationMs !== "number") return "";
  if (durationMs < 1000) return `${durationMs}ms`;
  const seconds = durationMs / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
};

function FlagGroup({ title, hint, flags, values, onToggle }) {
  return (
    <fieldset className="flag-group">
      <legend>{title}</legend>
      {hint && <p className="flag-hint">{hint}</p>}
      <div className="flag-grid">
        {flags.map((flag) => (
          <label key={flag.id} className="flag-item">
            <input
              type="checkbox"
              checked={values[flag.id]}
              onChange={(event) => onToggle(flag.id, event.target.checked)}
            />
            <span className="flag-label">
              <span className="flag-name">{flag.flag}</span>
              <span className="flag-title">{flag.label}</span>
              <span className="flag-description">{flag.description}</span>
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export default function UserCreate() {
  const [mode, setMode] = useState("create");
  const [keysPath, setKeysPath] = useState("");
  const [outputDir, setOutputDir] = useState("./out");
  const [algorithm, setAlgorithm] = useState("Chm01");
  const [hashFunction, setHashFunction] = useState("MultiplyShiftRX");
  const [maskFunction, setMaskFunction] = useState("And");
  const [concurrency, setConcurrency] = useState("8");
  const [executablePath, setExecutablePath] = useState("");
  const [flags, setFlags] = useState(defaultFlags);
  const [extraArgs, setExtraArgs] = useState("");
  const [copyStatus, setCopyStatus] = useState("");
  const [runStatus, setRunStatus] = useState("");
  const [runOutput, setRunOutput] = useState("");
  const [runError, setRunError] = useState("");
  const [runMeta, setRunMeta] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const commandParts = useMemo(() => {
    const defaultExecutable =
      mode === "bulk" ? "PerfectHashBulkCreate.exe" : "PerfectHashCreate.exe";
    const executable = executablePath.trim() || defaultExecutable;
    const previewArgs = [
      quoteIfNeeded(keysPath || "<keys-path>"),
      quoteIfNeeded(outputDir || "<output-dir>"),
      algorithm,
      hashFunction,
      maskFunction,
      concurrency || "<max-concurrency>"
    ];
    const runtimeArgs = [
      keysPath,
      outputDir,
      algorithm,
      hashFunction,
      maskFunction,
      concurrency
    ].filter(Boolean);
    const activeFlags = [
      ...createFlags,
      ...keysFlags,
      ...tableFlags
    ]
      .filter((flag) => flags[flag.id])
      .map((flag) => flag.flag);
    const extraTokens = parseExtraArgs(extraArgs.trim());
    const preview = [
      quoteIfNeeded(executable),
      ...previewArgs,
      ...activeFlags,
      ...(extraArgs.trim() ? [extraArgs.trim()] : [])
    ]
      .filter(Boolean)
      .join(" ");
    return {
      executable,
      runtimeArgs: [...runtimeArgs, ...activeFlags, ...extraTokens],
      preview
    };
  }, [
    mode,
    keysPath,
    outputDir,
    algorithm,
    hashFunction,
    maskFunction,
    concurrency,
    executablePath,
    flags,
    extraArgs
  ]);

  const commandPreview = commandParts.preview;
  const canRun = Boolean(keysPath && outputDir && concurrency);

  const handleToggle = (id, value) => {
    setFlags((current) => ({ ...current, [id]: value }));
  };

  const handleReset = () => {
    setMode("create");
    setKeysPath("");
    setOutputDir("./out");
    setAlgorithm("Chm01");
    setHashFunction("MultiplyShiftRX");
    setMaskFunction("And");
    setConcurrency("8");
    setExecutablePath("");
    setFlags(defaultFlags);
    setExtraArgs("");
    setCopyStatus("");
    setRunStatus("");
    setRunOutput("");
    setRunError("");
    setRunMeta(null);
  };

  const handleCopy = async () => {
    if (!navigator.clipboard) {
      setCopyStatus("Clipboard not available in this browser.");
      return;
    }
    try {
      await navigator.clipboard.writeText(commandPreview);
      setCopyStatus("Command copied to clipboard.");
      setTimeout(() => setCopyStatus(""), 2000);
    } catch (error) {
      setCopyStatus("Unable to copy command.");
    }
  };

  const handleClearOutput = () => {
    setRunStatus("");
    setRunOutput("");
    setRunError("");
    setRunMeta(null);
  };

  const handleRun = async () => {
    if (!canRun || isRunning) {
      setRunStatus("Enter keys path, output directory, and concurrency first.");
      return;
    }
    setIsRunning(true);
    setRunStatus("Running command...");
    setRunOutput("");
    setRunError("");
    setRunMeta(null);
    try {
      const response = await fetch(
        import.meta.env.VITE_RUNNER_URL || "/api/run",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            executable: commandParts.executable,
            args: commandParts.runtimeArgs
          })
        }
      );
      const payload = await response.json();
      if (!response.ok) {
        setRunStatus(payload.error || "Run failed.");
      } else {
        setRunStatus(payload.ok ? "Run completed." : "Run finished with errors.");
      }
      setRunOutput(payload.stdout || "");
      setRunError(payload.stderr || "");
      setRunMeta({
        exitCode: payload.exitCode,
        durationMs: payload.durationMs
      });
    } catch (error) {
      setRunStatus(`Run failed: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="page-grid">
      <section className="panel" style={{ "--stagger": 1 }}>
        <div className="panel-header">
          <div>
            <p className="eyebrow">User / Create</p>
            <h2>Create perfect hash command</h2>
            <p>
              Assemble a command line for PerfectHashCreate or
              PerfectHashBulkCreate. Defaults are tuned for fast, safe runs.
            </p>
          </div>
          <div className="mode-toggle">
            <button
              type="button"
              className={mode === "create" ? "toggle active" : "toggle"}
              onClick={() => setMode("create")}
            >
              Create
            </button>
            <button
              type="button"
              className={mode === "bulk" ? "toggle active" : "toggle"}
              onClick={() => setMode("bulk")}
            >
              Bulk Create
            </button>
          </div>
        </div>

        <div className="form-grid">
          <label>
            Keys path
            <input
              type="text"
              placeholder="data/keys/HologramWorld-31016.keys"
              value={keysPath}
              onChange={(event) => setKeysPath(event.target.value)}
            />
          </label>
          <label>
            Output directory
            <input
              type="text"
              placeholder="./out"
              value={outputDir}
              onChange={(event) => setOutputDir(event.target.value)}
            />
          </label>
          <label className="full">
            Executable path (optional)
            <input
              type="text"
              placeholder="PerfectHashCreate.exe"
              value={executablePath}
              onChange={(event) => setExecutablePath(event.target.value)}
            />
          </label>
          <label>
            Algorithm
            <select
              value={algorithm}
              onChange={(event) => setAlgorithm(event.target.value)}
            >
              {algorithmOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Hash function
            <select
              value={hashFunction}
              onChange={(event) => setHashFunction(event.target.value)}
            >
              {hashOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label>
            Mask function
            <select
              value={maskFunction}
              onChange={(event) => setMaskFunction(event.target.value)}
            >
              {maskOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Max concurrency
            <input
              type="number"
              min="1"
              value={concurrency}
              onChange={(event) => setConcurrency(event.target.value)}
            />
          </label>
          <label className="full">
            Extra arguments
            <input
              type="text"
              placeholder="--BestCoverageAttempts=8 --BestCoverageType=MemoryCoverage"
              value={extraArgs}
              onChange={(event) => setExtraArgs(event.target.value)}
            />
          </label>
        </div>

        <FlagGroup
          title="Create flags"
          flags={createFlags}
          values={flags}
          onToggle={handleToggle}
        />
        <FlagGroup
          title="Keys load flags"
          hint="Useful for large or pre-sorted key sets."
          flags={keysFlags}
          values={flags}
          onToggle={handleToggle}
        />
        <FlagGroup
          title="Table create flags"
          hint="Options that affect table search strategy and output artifacts."
          flags={tableFlags}
          values={flags}
          onToggle={handleToggle}
        />

        <div className="form-actions">
          <button type="button" className="button ghost" onClick={handleReset}>
            Reset defaults
          </button>
        </div>
      </section>

      <aside className="panel" style={{ "--stagger": 2 }}>
        <div className="panel-header">
          <div>
            <p className="eyebrow">Command output</p>
            <h2>Preview &amp; run</h2>
            <p>
              Copy this command into a terminal or drop it into your automation
              scripts.
            </p>
          </div>
          <div className="run-controls">
            <button
              type="button"
              className="button"
              onClick={handleCopy}
              disabled={isRunning}
            >
              Copy command
            </button>
            <button
              type="button"
              className="button"
              onClick={handleRun}
              disabled={!canRun || isRunning}
            >
              Run locally
            </button>
          </div>
        </div>
        <pre className="command-preview" data-testid="command-preview">
          {commandPreview}
        </pre>
        {copyStatus && <p className="status">{copyStatus}</p>}
        {!canRun && (
          <p className="status">
            Add keys path, output directory, and concurrency to enable running.
          </p>
        )}
        {runStatus && <p className="status">{runStatus}</p>}
        {runMeta && (
          <p className="status">
            Exit code: {runMeta.exitCode ?? "unknown"}{" "}
            {runMeta.durationMs != null && (
              <span>({formatDuration(runMeta.durationMs)})</span>
            )}
          </p>
        )}
        {(runOutput || runError) && (
          <div className="output-stack">
            {runOutput && (
              <div>
                <p className="output-title">Stdout</p>
                <pre className="command-output">{runOutput}</pre>
              </div>
            )}
            {runError && (
              <div>
                <p className="output-title">Stderr</p>
                <pre className="command-output error">{runError}</pre>
              </div>
            )}
            <button
              type="button"
              className="button ghost"
              onClick={handleClearOutput}
            >
              Clear output
            </button>
          </div>
        )}

        <div className="callout">
          <h3>Next steps</h3>
          <ul>
            <li>Use "Developer" to bootstrap build and test environments.</li>
            <li>Save output CSVs for analysis and benchmarking reports.</li>
            <li>Switch to Bulk Create to process a keys directory.</li>
            <li>Start the runner with "npm run server" or "npm run dev:full".</li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
