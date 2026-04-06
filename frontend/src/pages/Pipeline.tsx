import { useState } from "react";
import DatasetTab from "../components/pipeline/DatasetTab";
import TrainingTab from "../components/pipeline/TrainingTab";
import EvaluationTab from "../components/pipeline/EvaluationTab";

const TABS = ["Dataset", "Training", "Evaluation"] as const;
type Tab = typeof TABS[number];

export default function Pipeline() {
  const [tab, setTab] = useState<Tab>("Dataset");

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">ML Pipeline</h1>
      <div className="flex gap-1 mb-6 border-b border-gray-800">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              tab === t
                ? "border-emerald-400 text-emerald-400"
                : "border-transparent text-gray-400 hover:text-gray-200"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Dataset" && <DatasetTab />}
      {tab === "Training" && <TrainingTab />}
      {tab === "Evaluation" && <EvaluationTab />}
    </div>
  );
}
