const benchmarkCards = [
  {
    title: "Baseline runs",
    description:
      "Capture baseline timings with a known key set and hash function.",
    steps: ["Use MultiplyShiftRX with And masking", "Record CSV and DLL size"]
  },
  {
    title: "Hash family sweep",
    description:
      "Run the curated good hash set and compare coverage and speed.",
    steps: [
      "MultiplyShiftR",
      "MultiplyShiftLR",
      "MultiplyShiftRMultiply",
      "MultiplyShiftR2",
      "MultiplyShiftRX",
      "Mulshrolate1RX",
      "Mulshrolate2RX",
      "Mulshrolate3RX",
      "Mulshrolate4RX"
    ]
  },
  {
    title: "Report highlights",
    description:
      "Summarize best coverage, fastest index, and smallest binary outputs.",
    steps: ["Capture top three solutions", "Track changes between commits"]
  }
];

export default function AnalysisBenchmarks() {
  return (
    <div className="stack">
      <section className="panel" style={{ "--stagger": 1 }}>
        <p className="eyebrow">Analysis / Benchmarks</p>
        <h2>Benchmark the good hash set</h2>
        <p>
          Use the curated hash functions as the baseline for performance
          comparisons. Store results alongside CSV artifacts.
        </p>
      </section>

      <div className="card-grid">
        {benchmarkCards.map((card, index) => (
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
          Automated benchmarking notes are in
          <code>skills/improved-ui/SKILL.md</code>.
        </p>
      </section>
    </div>
  );
}
