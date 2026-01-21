import { expect, test } from "@playwright/test";

test("navigate across top-level sections", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "PerfectHash UI" })).toBeVisible();

  await page.getByRole("link", { name: "Developer" }).click();
  await expect(
    page.getByRole("heading", { name: "Bootstrap a fresh clone" })
  ).toBeVisible();

  await page.getByRole("link", { name: "Analysis" }).click();
  await expect(
    page.getByRole("heading", { name: "Collect and interpret results" })
  ).toBeVisible();
});

test("command builder updates preview", async ({ page }) => {
  await page.goto("/user/create");
  await page.getByLabel("Keys path").fill("/tmp/keys/example.keys");
  await page.getByRole("button", { name: "Bulk Create" }).click();

  const preview = page.getByTestId("command-preview");
  await expect(preview).toContainText("PerfectHashBulkCreate.exe");
  await expect(preview).toContainText("/tmp/keys/example.keys");
});

test("cmake configure preview updates", async ({ page }) => {
  await page.goto("/developer/build");
  await expect(
    page.getByRole("heading", { name: "CMake configure" })
  ).toBeVisible();

  const buildDirInput = page.getByLabel("Build directory").first();
  await buildDirInput.fill("build.dev");

  const generatorSelect = page.getByLabel("Generator");
  await generatorSelect.selectOption("Ninja Multi-Config");

  const buildTypeSelect = page.getByLabel("Build type");
  await expect(buildTypeSelect).toBeDisabled();

  const configureSection = page.locator("section", {
    has: page.getByRole("heading", { name: "CMake configure" })
  });
  const preview = configureSection.locator("pre.command-preview");
  await expect(preview).toContainText("build.dev");
  await expect(preview).toContainText("Ninja Multi-Config");
});

test("test page honors build defaults", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("perfecthash.buildDir", "build.dev");
    window.localStorage.setItem("perfecthash.buildConfig", "Debug");
  });

  await page.goto("/developer/test");
  await expect(page.getByRole("heading", { name: "CTest" })).toBeVisible();

  const ctestSection = page.locator("section", {
    has: page.getByRole("heading", { name: "CTest" })
  });
  const preview = ctestSection.locator("pre.command-preview");
  await expect(preview).toContainText("build.dev");
  await expect(preview).toContainText("-C Debug");

  const exeInput = page.getByLabel("PerfectHashCreate executable").first();
  await expect(exeInput).toHaveValue(/build\.dev\/bin\/PerfectHashCreate(\.exe)?/);
});
