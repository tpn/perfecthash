import { useEffect, useRef } from "react";
import "@xterm/xterm/css/xterm.css";

const terminalTheme = {
  background: "#0b1520",
  foreground: "#e5edf5",
  cursor: "#9ab4d1",
  selectionBackground: "rgba(79, 105, 164, 0.35)",
  black: "#0b1520",
  red: "#f97c7c",
  green: "#7ddc9b",
  yellow: "#f5d48b",
  blue: "#7aa9ff",
  magenta: "#b18bff",
  cyan: "#7bdff5",
  white: "#e5edf5"
};

export default function TerminalSession({
  session,
  onToggle,
  onStatusChange,
  onSudoPrompt,
  onStop
}) {
  const terminalRef = useRef(null);
  const xtermRef = useRef(null);
  const fitRef = useRef(null);
  const pendingOutput = useRef([]);
  const cardRef = useRef(null);

  useEffect(() => {
    let observer = null;
    let terminal = null;
    let fitAddon = null;
    let cancelled = false;

    const setupTerminal = async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import("@xterm/xterm"),
        import("@xterm/addon-fit")
      ]);
      if (cancelled) return;

      terminal = new Terminal({
        theme: terminalTheme,
        fontFamily: "Comic Mono NF, monospace",
        fontSize: 12,
        lineHeight: 1.3,
        disableStdin: true
      });
      fitAddon = new FitAddon();
      terminal.loadAddon(fitAddon);
      terminal.open(terminalRef.current);
      fitAddon.fit();

      observer = new ResizeObserver(() => {
        fitAddon.fit();
      });
      observer.observe(terminalRef.current);

      xtermRef.current = terminal;
      fitRef.current = fitAddon;

      if (pendingOutput.current.length) {
        pendingOutput.current.forEach((chunk) => terminal.write(chunk));
        pendingOutput.current = [];
      }
    };

    setupTerminal();

    return () => {
      cancelled = true;
      if (observer) observer.disconnect();
      if (terminal) terminal.dispose();
    };
  }, []);

  useEffect(() => {
    const stream = new EventSource(`/api/terminal/stream/${session.id}`);

    stream.addEventListener("output", (event) => {
      const payload = JSON.parse(event.data);
      if (payload.chunk) {
        if (xtermRef.current) {
          xtermRef.current.write(payload.chunk);
        } else {
          pendingOutput.current.push(payload.chunk);
        }
      }
    });

    stream.addEventListener("status", (event) => {
      const payload = JSON.parse(event.data);
      onStatusChange(session.id, payload);
      if (payload.state === "exit") {
        stream.close();
      }
    });

    stream.addEventListener("sudo", (event) => {
      const payload = JSON.parse(event.data);
      onSudoPrompt(session.id, payload.message || "sudo password required");
    });

    stream.onerror = () => {
      onStatusChange(session.id, {
        state: "stream-error",
        message: "Terminal connection lost."
      });
    };

    return () => {
      stream.close();
    };
  }, [session.id, onStatusChange, onSudoPrompt]);

  useEffect(() => {
    if (xtermRef.current) {
      xtermRef.current.reset();
    } else {
      pendingOutput.current = [];
    }
  }, [session.id]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!cardRef.current) return;
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")
      ?.matches;
    const behavior = reduceMotion ? "auto" : "smooth";
    requestAnimationFrame(() => {
      cardRef.current.scrollIntoView({ behavior, block: "start" });
    });
  }, [session.id]);

  const renderStatusIcon = (status) => {
    if (status === "success") {
      return "✅";
    }
    if (status === "error") {
      return "❌";
    }
    if (status === "stopping") {
      return <span className="terminal-spinner" aria-hidden="true" />;
    }
    return <span className="terminal-spinner" aria-hidden="true" />;
  };

  return (
    <section
      className={`terminal-card${session.minimized ? " minimized" : ""}`}
      ref={cardRef}
    >
      <header className="terminal-header">
        <div>
          <p className="terminal-title">{session.title}</p>
          {session.commandPreview && (
            <p className="terminal-command">{session.commandPreview}</p>
          )}
        </div>
        <div className="terminal-controls">
          <span
            className={`terminal-status terminal-status--${session.status}`}
            aria-label={session.status}
            title={session.status}
          >
            {renderStatusIcon(session.status)}
            <span className="sr-only">{session.status}</span>
          </span>
          <button
            type="button"
            className="terminal-stop"
            onClick={() => onStop(session.id)}
            aria-label="Stop session"
            title="Stop session"
            disabled={session.status !== "running"}
          >
            ■
          </button>
          <button
            type="button"
            className="terminal-toggle"
            onClick={() => onToggle(session.id)}
            aria-label={session.minimized ? "Expand terminal" : "Minimize terminal"}
          >
            {session.minimized ? "^" : "v"}
          </button>
        </div>
      </header>
      <div className="terminal-body" aria-hidden={session.minimized}>
        <div className="terminal-surface" ref={terminalRef} />
      </div>
    </section>
  );
}
