import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const parseAllowedHosts = (value) => {
  if (!value) return undefined;
  if (value === "all") return "all";
  const hosts = value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  return hosts.length ? hosts : undefined;
};

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const allowedHosts =
    parseAllowedHosts(env.PERFECTHASH_UI_ALLOWED_HOSTS) ||
    parseAllowedHosts(env.VITE_ALLOWED_HOSTS);

  return {
    plugins: [react()],
    server: {
      allowedHosts,
      proxy: {
        "/api": "http://127.0.0.1:7071"
      }
    },
    test: {
      globals: true,
      environment: "jsdom",
      setupFiles: "./src/setupTests.js",
      css: true,
      exclude: ["tests/**", "**/node_modules/**", "**/dist/**"]
    }
  };
});
