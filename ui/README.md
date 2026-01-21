# PerfectHash UI

React UI for building PerfectHash command lines and reviewing developer workflows.

## Getting started

```bash
cd ui
npm install
npm run dev
```

To enable the "Run locally" button on the Create page, start the runner too:

```bash
cd ui
npm run dev:full
```

Or run the backend separately:

```bash
cd ui
npm run server
```

To point the UI at a custom runner URL, set `VITE_RUNNER_URL` when starting
Vite (for example, `VITE_RUNNER_URL=http://localhost:7071/api/run npm run dev`).

If you need to allow a specific host for the dev server, set
`PERFECTHASH_UI_ALLOWED_HOSTS` (comma-separated) or `VITE_ALLOWED_HOSTS`.
Use `PERFECTHASH_UI_ALLOWED_HOSTS=all` to allow any host.

## Tests

```bash
cd ui
npm test
npm run test:e2e
```

The Playwright suite expects the UI to be launched on `http://127.0.0.1:4173`.
Use `npm run test:e2e:install` to install browser dependencies if needed.
