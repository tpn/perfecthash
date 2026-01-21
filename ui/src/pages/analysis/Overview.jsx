const overviewCards = [
  {
    title: "CSV outputs",
    description:
      "Each run emits CSV rows with coverage, timing, and solution metadata.",
    steps: ["Collect *.csv files from the output directory", "Track coverage deltas across runs"]
  },
  {
    title: "Artifacts",
    description:
      "Compiled tables, headers, and solution logs land alongside the CSVs.",
    steps: ["Inspect generated .c/.h outputs", "Review best graph logs for anomalies"]
  },
  {
    title: "Key metrics",
    description:
      "Focus on solution rate, coverage, and compile size per hash family.",
    steps: ["Compare hash function families", "Flag regressions in coverage"]
  }
];

export default function AnalysisOverview() {
  return (
    <div className="stack">
      <section className="panel" style={{ "--stagger": 1 }}>
        <p className="eyebrow">Analysis / Overview</p>
        <h2>Collect and interpret results</h2>
        <p>
          Start here to review CSV outputs and compiled artifacts. These
          checkpoints keep performance tracking consistent across runs.
        </p>
      </section>

      <div className="card-grid">
        {overviewCards.map((card, index) => (
          <section
            key={card.title}
            className="panel"
            style={{ "--stagger": index + 2 }}
          >
            <h3>{card.title}</h3>
            <p>{card.description}</p>
            <ul className="list">
              {card.steps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ul>
          </section>
        ))}
      </div>

      <section className="panel" style={{ "--stagger": 6 }}>
        <h3>Agent companion</h3>
        <p>
          For automated analysis workflows, see
          <code>skills/improved-ui/SKILL.md</code>.
        </p>
      </section>
    </div>
  );
}
