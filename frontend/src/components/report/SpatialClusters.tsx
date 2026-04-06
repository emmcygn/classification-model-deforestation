import { useState } from "react";
import { getSpatialClusters, type ClusterResult } from "../../lib/api";

interface Props {
  runId: string;
  region?: string;
  onNavigate?: (lat: number, lon: number) => void;
}

export default function SpatialClusters({ runId, region, onNavigate }: Props) {
  const [result, setResult] = useState<ClusterResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await getSpatialClusters(runId, region);
      setResult(r);
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to detect clusters");
    } finally {
      setLoading(false);
    }
  };

  const severityColor = (s: string) =>
    s === "critical" ? "text-red-400" : s === "elevated" ? "text-amber-400" : "text-emerald-400";
  const severityBg = (s: string) =>
    s === "critical" ? "bg-red-900/30" : s === "elevated" ? "bg-amber-900/30" : "bg-emerald-900/30";

  return (
    <div className="space-y-2">
      <div className="text-xs text-gray-400 uppercase tracking-wider">Deforestation Fronts</div>
      <button onClick={handleAnalyze} disabled={loading || !runId}
        className="w-full px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 rounded text-xs font-medium">
        {loading ? "Clustering..." : "Detect Spatial Clusters"}
      </button>

      {error && (
        <div className="text-xs text-red-400 bg-red-900/20 rounded px-2 py-1.5">{error}</div>
      )}

      {result && (
        <div className="space-y-2">
          <div className="flex gap-2 text-xs">
            <span className="text-gray-400">{result.n_clusters} clusters</span>
            <span className="text-gray-500">|</span>
            <span className="text-gray-400">{result.n_noise} isolated</span>
            <span className="text-gray-500">|</span>
            <span className="text-gray-400">{result.total_high_risk} high-risk cells</span>
          </div>

          {result.clusters.length === 0 && (
            <div className="text-xs text-gray-500 bg-gray-800/60 rounded px-2 py-2">
              No clusters detected — high-risk cells may be too sparse or isolated to form fronts.
            </div>
          )}

          {result.clusters.map((c) => (
            <button
              key={c.cluster_id}
              onClick={() => onNavigate?.(c.centroid.lat, c.centroid.lon)}
              className={`w-full text-left rounded-lg p-2 ${severityBg(c.severity)} border border-gray-700/30 hover:border-gray-500/50 transition-colors cursor-pointer`}
            >
              <div className="flex justify-between items-center">
                <span className="text-xs font-medium text-gray-200">Front #{c.cluster_id + 1}</span>
                <span className={`text-[10px] uppercase font-bold ${severityColor(c.severity)}`}>{c.severity}</span>
              </div>
              <div className="grid grid-cols-3 gap-1 mt-1 text-[10px] text-gray-400">
                <span>{c.n_cells} cells</span>
                <span>{c.approx_area_km2} km²</span>
                <span>Risk: {(c.mean_risk * 100).toFixed(0)}%</span>
              </div>
              <div className="text-[10px] text-gray-500 mt-0.5 font-mono">
                Center: {c.centroid.lat.toFixed(2)}°N, {c.centroid.lon.toFixed(2)}°E
              </div>
            </button>
          ))}

          <div className="text-[9px] text-gray-600">{result.method}</div>
        </div>
      )}
    </div>
  );
}
