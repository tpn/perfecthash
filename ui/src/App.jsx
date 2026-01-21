import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import UserCreate from "./pages/user/Create.jsx";
import DeveloperEnvironment from "./pages/developer/Environment.jsx";
import DeveloperBuild from "./pages/developer/Build.jsx";
import DeveloperTest from "./pages/developer/Test.jsx";
import AnalysisOverview from "./pages/analysis/Overview.jsx";
import AnalysisBenchmarks from "./pages/analysis/Benchmarks.jsx";
import NotFound from "./pages/NotFound.jsx";

const topTabs = [
  { path: "/user/create", label: "User" },
  { path: "/developer/environment", label: "Developer" },
  { path: "/analysis/overview", label: "Analysis" }
];

const subTabs = {
  user: [{ path: "/user/create", label: "Create" }],
  developer: [
    { path: "/developer/environment", label: "Environment" },
    { path: "/developer/build", label: "Build" },
    { path: "/developer/test", label: "Test" }
  ],
  analysis: [
    { path: "/analysis/overview", label: "Overview" },
    { path: "/analysis/benchmarks", label: "Benchmarks" }
  ]
};

function useSection() {
  const location = useLocation();
  const [, section] = location.pathname.split("/");
  return section || "user";
}

function AppShell() {
  const section = useSection();
  const subtabs = subTabs[section] || [];

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <p className="brand-eyebrow">PerfectHash Toolkit</p>
          <h1 className="hero-title10">PerfectHash UI</h1>
          <p className="brand-subtitle">
            A guided command builder and bootstrap hub for PerfectHash workflows.
          </p>
        </div>
        <nav className="tab-group" aria-label="Primary">
          {topTabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              className={({ isActive }) =>
                `tab${isActive ? " tab--active" : ""}`
              }
            >
              {tab.label}
            </NavLink>
          ))}
        </nav>
      </header>

      {subtabs.length > 0 && (
        <nav className="subtab-group" aria-label={`${section} sections`}>
          {subtabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              className={({ isActive }) =>
                `subtab${isActive ? " subtab--active" : ""}`
              }
            >
              {tab.label}
            </NavLink>
          ))}
        </nav>
      )}

      <main className="app-content">
        <Routes>
          <Route path="/" element={<Navigate to="/user/create" replace />} />
          <Route path="/user" element={<Navigate to="/user/create" replace />} />
          <Route path="/user/create" element={<UserCreate />} />
          <Route
            path="/developer"
            element={<Navigate to="/developer/environment" replace />}
          />
          <Route path="/developer/environment" element={<DeveloperEnvironment />} />
          <Route path="/developer/build" element={<DeveloperBuild />} />
          <Route path="/developer/test" element={<DeveloperTest />} />
          <Route
            path="/analysis"
            element={<Navigate to="/analysis/overview" replace />}
          />
          <Route path="/analysis/overview" element={<AnalysisOverview />} />
          <Route path="/analysis/benchmarks" element={<AnalysisBenchmarks />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return <AppShell />;
}
