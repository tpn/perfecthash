import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: "http://127.0.0.1:4173"
  },
  webServer: [
    {
      command: "npm run server",
      url: "http://127.0.0.1:7071/api/health",
      reuseExistingServer: !process.env.CI
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4173",
      url: "http://127.0.0.1:4173",
      reuseExistingServer: !process.env.CI
    }
  ]
});
