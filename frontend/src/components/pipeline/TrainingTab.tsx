import { useState, useEffect } from "react";
import { trainModel, retrainWithAnnotations, listRuns, type TrainParams, type Run } from "../../lib/api";

export default function TrainingTab({ onRunComplete }: { onRunComplete?: (runId: string) => void }) {
  const [params, setParams] = useState<TrainParams>({
    n_estimators: 100,
    max_depth: 10,
    min_samples_split: 5,
    min_samples_leaf: 2,
    test_size: 0.2,
    model_type: "random_forest",
    spatial_split: true,
  });
  const [training, setTraining] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    listRuns().then(setRuns).catch(() => {});
  }, []);

  const handleTrain = async () => {
    setTraining(true);
    setError("");
    try {
      const result = await trainModel(params);
      onRunComplete?.(result.run_id);
      const updated = await listRuns();
      setRuns(updated);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
        <h3 className="text-sm font-medium text-gray-300 mb-4">Hyperparameters</h3>
        <div className="grid grid-cols-2 gap-4">
          <label className="block">
            <span className="text-xs text-gray-400">n_estimators</span>
            <input
              type="number"
              value={params.n_estimators}
              onChange={(e) => setParams({ ...params, n_estimators: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">max_depth</span>
            <input
              type="number"
              value={params.max_depth ?? ""}
              onChange={(e) => setParams({ ...params, max_depth: e.target.value ? +e.target.value : null })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">min_samples_split</span>
            <input
              type="number"
              value={params.min_samples_split}
              onChange={(e) => setParams({ ...params, min_samples_split: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">min_samples_leaf</span>
            <input
              type="number"
              value={params.min_samples_leaf}
              onChange={(e) => setParams({ ...params, min_samples_leaf: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">test_size</span>
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.5"
              value={params.test_size}
              onChange={(e) => setParams({ ...params, test_size: +e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
          </label>
          <label className="block col-span-2">
            <span className="text-xs text-gray-400">Model</span>
            <select
              value={params.model_type || "random_forest"}
              onChange={(e) => setParams({ ...params, model_type: e.target.value })}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            >
              <option value="random_forest">Random Forest</option>
              <option value="xgboost">XGBoost</option>
            </select>
          </label>
          <label className="flex items-center gap-2 col-span-2">
            <input
              type="checkbox"
              checked={params.spatial_split || false}
              onChange={(e) => setParams({ ...params, spatial_split: e.target.checked })}
              className="rounded bg-gray-800 border-gray-700"
            />
            <span className="text-xs text-gray-400">Spatial cross-validation (prevents geographic leakage)</span>
          </label>
        </div>

        <div className="mt-4 flex gap-2">
          <button
            onClick={handleTrain}
            disabled={training}
            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm font-medium transition-colors"
          >
            {training ? "Training..." : "Train Model"}
          </button>
          <button
            onClick={async () => {
              setTraining(true);
              setError("");
              try {
                const result = await retrainWithAnnotations(params);
                onRunComplete?.(result.run_id);
                const updated = await listRuns();
                setRuns(updated);
                alert(`Retrained with ${result.annotation_overrides} label overrides from ${result.total_annotations} annotations`);
              } catch (e: any) {
                setError(e.message);
              } finally {
                setTraining(false);
              }
            }}
            disabled={training}
            className="px-4 py-2 bg-amber-600 hover:bg-amber-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm font-medium transition-colors"
            title="Retrain using human accept/reject annotations to override labels"
          >
            {training ? "..." : "Retrain with Annotations"}
          </button>
        </div>
        {error && <div className="mt-2 text-red-400 text-sm">{error}</div>}
      </div>

      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Run History</h3>
        {runs.length === 0 ? (
          <div className="text-gray-500 text-sm">No runs yet. Train a model to get started.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                <th className="text-left py-2">Run ID</th>
                <th className="text-left py-2">Date</th>
                <th className="text-right py-2">Trees</th>
                <th className="text-right py-2">Depth</th>
                <th className="text-right py-2">Accuracy</th>
                <th className="text-right py-2">F1</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="border-b border-gray-800/50 hover:bg-gray-800/50">
                  <td className="py-2 font-mono text-emerald-400">{r.run_id}</td>
                  <td className="py-2 text-gray-400">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="py-2 text-right">{r.params.n_estimators}</td>
                  <td className="py-2 text-right">{r.params.max_depth ?? "None"}</td>
                  <td className="py-2 text-right">{(r.metrics.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-2 text-right">{(r.metrics.f1 * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
