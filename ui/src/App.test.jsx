import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";

const renderApp = (initialEntry) =>
  render(
    <MemoryRouter
      initialEntries={[initialEntry]}
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <App />
    </MemoryRouter>
  );

afterEach(() => {
  vi.restoreAllMocks();
});

describe("PerfectHash UI", () => {
  it("routes to the create page by default", () => {
    renderApp("/");
    expect(
      screen.getByRole("heading", { name: /Create perfect hash command/i })
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "User" })).toBeInTheDocument();
  });

  it("renders developer sub-navigation", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              platform: "linux",
              platformLabel: "Linux",
              arch: "x86_64",
              rootDir: "/tmp",
              install: null
            })
        })
      )
    );
    renderApp("/developer/build");
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });
    expect(
      screen.getByRole("heading", { name: /Build the core libraries/i })
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Environment" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Build" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Test" })).toBeInTheDocument();
  });

  it("updates the command preview", async () => {
    renderApp("/user/create");
    const user = userEvent.setup();
    const keysInput = screen.getByLabelText(/Keys path/i);
    await user.clear(keysInput);
    await user.type(keysInput, "/tmp/example.keys");
    await user.click(screen.getByRole("button", { name: "Bulk Create" }));

    const preview = screen.getByTestId("command-preview");
    expect(preview).toHaveTextContent("/tmp/example.keys");
    expect(preview).toHaveTextContent("PerfectHashBulkCreate.exe");
  });

  it("disables run locally when required fields are missing", () => {
    renderApp("/user/create");
    const runButton = screen.getByRole("button", { name: "Run locally" });
    expect(runButton).toBeDisabled();
  });
});
