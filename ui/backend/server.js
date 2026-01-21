import express from "express";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import pty from "node-pty";

const app = express();
const port = Number.parseInt(
  process.env.PERFECTHASH_UI_SERVER_PORT || "7071",
  10
);

const rootDir = path.resolve(process.cwd(), "..");

const allowedExecutables = new Set([
  "PerfectHashCreate.exe",
  "PerfectHashBulkCreate.exe",
  "PerfectHashCreate",
  "PerfectHashBulkCreate"
]);

const allowedCommands = new Set([
  "bash",
  "sh",
  "zsh",
  "powershell",
  "powershell.exe",
  "pwsh",
  "cmd",
  "cmd.exe",
  "cmake",
  "cmake.exe",
  "ctest",
  "ctest.exe",
  "conda",
  "conda.exe",
  "mamba",
  "mamba.exe",
  "micromamba",
  "micromamba.exe",
  "npm",
  "npm.cmd",
  "npm.exe"
]);

const MAX_OUTPUT_BYTES = 1024 * 1024;
const SUDO_PROMPT_REGEX = /(\[sudo\].*password|password for .*:|sudo password|\bPassword:)/i;

app.use(express.json({ limit: "1mb" }));

const appendOutput = (target, chunk) => {
  if (target.truncated) {
    return target;
  }
  const text =
    typeof chunk === "string" ? chunk : chunk.toString("utf8");
  const nextSize = target.size + Buffer.byteLength(text);
  if (nextSize <= MAX_OUTPUT_BYTES) {
    return {
      ...target,
      data: target.data + text,
      size: nextSize
    };
  }
  const remaining = Math.max(0, MAX_OUTPUT_BYTES - target.size);
  const sliced = text.slice(0, remaining);
  return {
    data: target.data + sliced,
    size: MAX_OUTPUT_BYTES,
    truncated: true
  };
};

const finalizeOutput = (target) => {
  if (!target.truncated) return target.data;
  return `${target.data}\n[output truncated]`;
};

const sanitizeExecutable = (executable) => {
  if (typeof executable !== "string") return null;
  const trimmed = executable.trim();
  if (!trimmed) return null;
  const base = path.basename(trimmed);
  if (!allowedExecutables.has(base)) return null;
  return trimmed;
};

const isAllowedCommand = (command) => {
  if (typeof command !== "string") return false;
  const base = path.basename(command);
  if (allowedCommands.has(base)) return true;
  if (allowedExecutables.has(base)) return true;

  const resolved = path.resolve(rootDir, command);
  const scriptsDir = path.join(rootDir, "scripts", "install-deps");
  if (resolved.startsWith(scriptsDir)) {
    return fs.existsSync(resolved);
  }

  return false;
};

const resolveWorkingDir = (cwd) => {
  if (!cwd) return rootDir;
  const resolved = path.isAbsolute(cwd)
    ? cwd
    : path.resolve(rootDir, cwd);
  if (resolved.startsWith(rootDir)) {
    return resolved;
  }
  return rootDir;
};

const sessions = new Map();

const sendEvent = (res, event, payload) => {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
};

const broadcast = (session, event, payload) => {
  session.clients.forEach((client) => sendEvent(client, event, payload));
};

const createSession = ({ command, args, cwd, env, allowSudo }) => {
  const id = randomUUID();
  const workingDir = resolveWorkingDir(cwd);
  const session = {
    id,
    command,
    args,
    cwd: workingDir,
    allowSudo: Boolean(allowSudo),
    awaitingSudo: false,
    output: { data: "", size: 0, truncated: false },
    snapshotSent: false,
    status: "running",
    exitInfo: null,
    clients: new Set()
  };

  const ptyProcess = pty.spawn(command, args, {
    name: "xterm-256color",
    cols: 120,
    rows: 32,
    cwd: workingDir,
    env: {
      ...process.env,
      TERM: "xterm-256color",
      ...(env || {})
    }
  });

  session.pty = ptyProcess;

  ptyProcess.onData((data) => {
    session.output = appendOutput(session.output, data);
    broadcast(session, "output", { chunk: data });

    if (session.allowSudo && !session.awaitingSudo) {
      if (SUDO_PROMPT_REGEX.test(data)) {
        session.awaitingSudo = true;
        broadcast(session, "sudo", { message: data.trim() || "sudo password" });
      }
    }
  });

  ptyProcess.onExit(({ exitCode, signal }) => {
    session.status = "exited";
    session.exitInfo = {
      state: "exit",
      exitCode,
      signal,
      ok: exitCode === 0
    };
    broadcast(session, "status", session.exitInfo);

    setTimeout(() => {
      sessions.delete(id);
    }, 5 * 60 * 1000);
  });

  sessions.set(id, session);
  return session;
};

const parseOsRelease = () => {
  try {
    const content = fs.readFileSync("/etc/os-release", "utf8");
    const map = {};
    for (const line of content.split("\n")) {
      const match = line.match(/^([A-Z_]+)=(.*)$/);
      if (!match) continue;
      const key = match[1];
      const value = match[2].replace(/^"|"$/g, "");
      map[key] = value;
    }
    return map;
  } catch (error) {
    return {};
  }
};

const detectInstallScript = () => {
  const platform = os.platform();
  if (platform === "win32") {
    return {
      label: "Windows (PowerShell)",
      command: "powershell",
      args: [
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        path.join(rootDir, "scripts", "install-deps", "windows.ps1")
      ]
    };
  }

  if (platform === "darwin") {
    return null;
  }

  if (platform === "linux") {
    const info = parseOsRelease();
    const id = (info.ID || "").toLowerCase();
    const like = (info.ID_LIKE || "").toLowerCase();
    const candidates = new Set([
      id,
      ...like.split(/\s+/).filter(Boolean)
    ]);

    const scriptMap = {
      ubuntu: { label: "Ubuntu/Debian", script: "ubuntu.sh" },
      debian: { label: "Ubuntu/Debian", script: "ubuntu.sh" },
      fedora: { label: "Fedora", script: "fedora.sh" },
      arch: { label: "Arch", script: "arch.sh" }
    };

    for (const key of candidates) {
      if (scriptMap[key]) {
        return {
          label: scriptMap[key].label,
          command: path.join(
            rootDir,
            "scripts",
            "install-deps",
            scriptMap[key].script
          ),
          args: []
        };
      }
    }
  }

  return null;
};

app.get("/api/health", (_req, res) => {
  res.json({ ok: true });
});

app.get("/api/platform", (_req, res) => {
  const platform = os.platform();
  const platformLabel =
    platform === "win32"
      ? "Windows"
      : platform === "darwin"
        ? "macOS"
        : "Linux";
  const install = detectInstallScript();
  const distro = platform === "linux" ? parseOsRelease().NAME : null;
  const requiresSudo =
    platform === "linux" || platform === "darwin"
      ? typeof process.getuid === "function" && process.getuid() !== 0
      : false;

  res.json({
    ok: true,
    platform,
    platformLabel,
    arch: os.arch(),
    distro,
    install,
    requiresSudo,
    rootDir
  });
});

app.post("/api/run", (req, res) => {
  const { executable, args } = req.body || {};
  const sanitized = sanitizeExecutable(executable);
  if (!sanitized) {
    res.status(400).json({
      ok: false,
      error:
        "Executable must be PerfectHashCreate(.exe) or PerfectHashBulkCreate(.exe)."
    });
    return;
  }
  if (!Array.isArray(args) || args.some((arg) => typeof arg !== "string")) {
    res.status(400).json({
      ok: false,
      error: "Args must be an array of strings."
    });
    return;
  }

  const start = Date.now();
  let stdout = { data: "", size: 0, truncated: false };
  let stderr = { data: "", size: 0, truncated: false };
  let responded = false;

  const child = spawn(sanitized, args, {
    cwd: process.cwd(),
    windowsHide: true,
    shell: false
  });

  child.stdout.on("data", (chunk) => {
    stdout = appendOutput(stdout, chunk);
  });

  child.stderr.on("data", (chunk) => {
    stderr = appendOutput(stderr, chunk);
  });

  child.on("error", (error) => {
    if (responded) return;
    responded = true;
    res.status(500).json({ ok: false, error: error.message });
  });

  child.on("close", (code, signal) => {
    if (responded) return;
    responded = true;
    res.json({
      ok: code === 0,
      exitCode: code,
      signal,
      durationMs: Date.now() - start,
      stdout: finalizeOutput(stdout),
      stderr: finalizeOutput(stderr)
    });
  });
});

app.post("/api/terminal/start", (req, res) => {
  const { command, args, cwd, env, allowSudo } = req.body || {};
  if (!command || !isAllowedCommand(command)) {
    res.status(400).json({
      error: "Command is not allowed."
    });
    return;
  }
  if (args && (!Array.isArray(args) || args.some((arg) => typeof arg !== "string"))) {
    res.status(400).json({
      error: "Args must be an array of strings."
    });
    return;
  }

  const session = createSession({
    command,
    args: args || [],
    cwd,
    env,
    allowSudo
  });

  res.json({ id: session.id });
});

app.get("/api/terminal/stream/:id", (req, res) => {
  const session = sessions.get(req.params.id);
  if (!session) {
    res.status(404).end();
    return;
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive"
  });

  session.clients.add(res);

  if (session.output.data && !session.snapshotSent) {
    sendEvent(res, "output", { chunk: session.output.data });
    session.snapshotSent = true;
  }
  if (session.exitInfo) {
    sendEvent(res, "status", session.exitInfo);
  }

  req.on("close", () => {
    session.clients.delete(res);
  });
});

app.post("/api/terminal/input", (req, res) => {
  const { id, input } = req.body || {};
  const session = sessions.get(id);
  if (!session) {
    res.status(404).json({ error: "Session not found." });
    return;
  }
  if (typeof input !== "string") {
    res.status(400).json({ error: "Input must be a string." });
    return;
  }

  const payload = input.endsWith("\n") ? input : `${input}\n`;
  session.pty.write(payload);
  session.awaitingSudo = false;

  res.json({ ok: true });
});

app.post("/api/terminal/stop", (req, res) => {
  const { id } = req.body || {};
  const session = sessions.get(id);
  if (!session) {
    res.status(404).json({ error: "Session not found." });
    return;
  }
  if (session.status !== "running") {
    res.json({ ok: true });
    return;
  }

  try {
    if (session.pty && typeof session.pty.write === "function") {
      session.pty.write("\u0003");
    }
    if (session.pty && typeof session.pty.kill === "function") {
      session.pty.kill("SIGINT");
    }
    if (process.platform !== "win32" && session.pty?.pid) {
      try {
        process.kill(-session.pty.pid, "SIGINT");
      } catch (error) {
        // Ignore process group errors.
      }
    }

    setTimeout(() => {
      if (session.status !== "running") return;
      try {
        session.pty.kill("SIGTERM");
      } catch (error) {
        // Ignore termination errors.
      }
      if (process.platform !== "win32" && session.pty?.pid) {
        try {
          process.kill(-session.pty.pid, "SIGTERM");
        } catch (error) {
          // Ignore process group errors.
        }
      }
    }, 800);

    setTimeout(() => {
      if (session.status !== "running") return;
      try {
        session.pty.kill("SIGKILL");
      } catch (error) {
        // Ignore kill errors.
      }
      if (process.platform !== "win32" && session.pty?.pid) {
        try {
          process.kill(-session.pty.pid, "SIGKILL");
        } catch (error) {
          // Ignore process group errors.
        }
      }
    }, 2000);

    res.json({ ok: true });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`PerfectHash UI runner listening on http://127.0.0.1:${port}`);
});
