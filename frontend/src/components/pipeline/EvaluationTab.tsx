import { useEffect, useState } from "react";
import { listRuns, type Run } from "../../lib/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line,
} from "recharts";

export default function EvaluationTab() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [compareRun, setCompareRun] = useState<Run | null>(null);

  useEffect(() => {
    listRuns().then((r) => {
      setRuns(r);
      if (r.length > 0) setSelectedRun(r[0]);
    });
  }, []);

  const cm = selectedRun?.metrics.confusion_matrix;
  const featureImportance = selectedRun?.metrics.feature_importance || [];

  return (
    <div className="space-y-6">
      <div className="flex gap-4">
        <label className="block flex-1">
          <span className="text-xs text-gray-400">Select Run</span>
          <select
            value={selectedRun?.run_id || ""}
            onChange={(e) => {
              const r = runs.find((run) => run.run_id === e.target.value);
              setSelectedRun(r || null);
            }}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} — F1: {(r.metrics.f1 * 100).toFixed(1)}%
              </option>
            ))}
          </select>
        </label>
        <label className="block flex-1">
          <span className="text-xs text-gray-400">Compare With</span>
          <select
            value={compareRun?.run_id || ""}
            onChange={(e) => {
              const r = runs.find((run) => run.run_id === e.target.value);
              setCompareRun(r || null);
            }}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            <option value="">None</option>
            {runs
              .filter((r) => r.run_id !== selectedRun?.run_id)
              .map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} — F1: {(r.metrics.f1 * 100).toFixed(1)}%
                </option>
              ))}
          </select>
        </label>
      </div>

      {selectedRun && (
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <h3 className="text-sm font-medium text-gray-300 mb-3">
              Metrics {compareRun ? "(comparison)" : ""}
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {(["accuracy", "precision", "recall", "f1"] as const).map((m) => (
                <div key={m} className="bg-gray-800 rounded p-3">
                  <div className="text-xs text-gray-400 capitalize">{m}</div>
                  <div className="text-lg font-bold">
                    {(selectedRun.metrics[m] * 100).toFixed(1)}%
                  </div>
                  {compareRun && (
                    <div
                      className={`text-xs ${
                        compareRun.metrics[m] < selectedRun.metrics[m]
                          ? "text-emerald-400"
                          : compareRun.metrics[m] > selectedRun.metrics[m]
                          ? "text-red-400"
                          : "text-gray-500"
                      }`}
                    >
                      vs {(compareRun.metrics[m] * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Confusion Matrix</h3>
            {cm && (
              <div className="grid grid-cols-2 gap-2 max-w-xs">
                <div className="bg-emerald-900/30 border border-emerald-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">True Neg</div>
                  <div className="text-xl font-bold text-emerald-400">{cm[0][0]}</div>
                </div>
                <div className="bg-red-900/30 border border-red-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">False Pos</div>
                  <div className="text-xl font-bold text-red-400">{cm[0][1]}</div>
                </div>
                <div className="bg-red-900/30 border border-red-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">False Neg</div>
                  <div className="text-xl font-bold text-red-400">{cm[1][0]}</div>
                </div>
                <div className="bg-emerald-900/30 border border-emerald-800/50 rounded p-3 text-center">
                  <div className="text-xs text-gray-400">True Pos</div>
                  <div className="text-xl font-bold text-emerald-400">{cm[1][1]}</div>
                </div>
              </div>
            )}
          </div>

          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 col-span-2">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Feature Importance</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={featureImportance} layout="vertical">
                <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={220}
                  tick={{ fill: "#9ca3af", fontSize: 11 }}
                  tickFormatter={(v: string) => v.replace(/_/g, " ")}
                />
                <Tooltip contentStyle={{ background: "#1f2937", border: "1px solid #374151" }} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {featureImportance.map((_: any, i: number) => (
                    <Cell key={i} fill={i === 0 ? "#10b981" : i < 3 ? "#34d399" : "#6ee7b7"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Threshold Analysis */}
          {selectedRun && (selectedRun as any).params?.threshold_analysis && (
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 col-span-2">
              <h3 className="text-sm font-medium text-gray-300 mb-3">
                Threshold Tuning — Precision/Recall Tradeoff
              </h3>
              <div className="text-xs text-gray-400 mb-2">
                Optimal threshold: <span className="text-emerald-400 font-mono">
                  {(selectedRun as any).params.threshold_analysis.optimal_threshold}
                </span> (F1: {(selectedRun as any).params.threshold_analysis.optimal_f1})
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={(selectedRun as any).params.threshold_analysis.threshold_curve}>
                  <XAxis dataKey="threshold" tick={{ fill: "#9ca3af", fontSize: 11 }} label={{ value: "Threshold", position: "insideBottom", offset: -5, fill: "#6b7280" }} />
                  <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "#1f2937", border: "1px solid #374151" }} />
                  <Line type="monotone" dataKey="precision" stroke="#f59e0b" dot={false} strokeWidth={2} name="Precision" />
                  <Line type="monotone" dataKey="recall" stroke="#3b82f6" dot={false} strokeWidth={2} name="Recall" />
                  <Line type="monotone" dataKey="f1" stroke="#10b981" dot={false} strokeWidth={2} name="F1" />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex gap-4 text-[10px] text-gray-500 mt-1 justify-center">
                <span><span className="text-amber-400">--</span> Precision</span>
                <span><span className="text-blue-400">--</span> Recall</span>
                <span><span className="text-emerald-400">--</span> F1</span>
              </div>
            </div>
          )}

          {/* Model Comparison */}
          {selectedRun && (selectedRun as any).params?.model_comparison && (
            <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 col-span-2">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Model Comparison</h3>
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div className="bg-gray-800 rounded p-3">
                  <div className="text-xs text-gray-400">Primary ({(selectedRun as any).params.model_comparison.primary_model})</div>
                  <div className="text-lg font-bold">{((selectedRun as any).params.model_comparison.primary_f1 * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-800 rounded p-3">
                  <div className="text-xs text-gray-400">Alternative ({(selectedRun as any).params.model_comparison.alternative_model})</div>
                  <div className="text-lg font-bold">{((selectedRun as any).params.model_comparison.alternative_f1 * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-gray-800 rounded p-3">
                  <div className="text-xs text-gray-400">Recommended</div>
                  <div className="text-lg font-bold text-emerald-400">{(selectedRun as any).params.model_comparison.recommendation}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
