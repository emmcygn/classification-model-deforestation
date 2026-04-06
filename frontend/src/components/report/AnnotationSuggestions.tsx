import { useState } from "react";
import type { GridCell } from "../../lib/api";

const BASE = "/api";

interface Suggestion {
  lat: number;
  lon: number;
  risk_probability: number;
  uncertainty: number;
  prediction: number;
}

interface SuggestionsResponse {
  suggestions: Suggestion[];
  total_unannotated: number;
  total_annotated: number;
}

interface Props {
  runId: string;
  region?: string;
  onCellSelect?: (lat: number, lon: number) => void;
}

export default function AnnotationSuggestions({ runId, region, onCellSelect }: Props) {
  const [data, setData] = useState<SuggestionsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSuggestions = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ run_id: runId, n: "10" });
      if (region) params.set("region", region);
      const resp = await fetch(`${BASE}/explorer/suggest-annotations?${params}`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || resp.statusText);
      }
      const json: SuggestionsResponse = await resp.json();
      setData(json);
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to load suggestions");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="text-xs text-gray-400 uppercase tracking-wider">Active Learning</div>
      <button onClick={fetchSuggestions} disabled={loading || !runId}
        className="w-full px-3 py-1.5 bg-amber-600 hover:bg-amber-700 disabled:bg-gray-800 rounded text-xs font-medium">
        {loading ? "Finding..." : "Suggest Cells to Review"}
      </button>

      {error && (
        <div className="text-xs text-red-400 bg-red-900/20 rounded px-2 py-1.5">{error}</div>
      )}

      {data && (
        <div className="space-y-1.5">
          <div className="text-[10px] text-gray-500">
            {data.total_annotated} reviewed / {data.total_unannotated} remaining
          </div>
          <div className="text-[10px] text-gray-400 mb-1">
            Most uncertain cells — your review here maximizes model improvement:
          </div>
          {data.suggestions.map((s, i) => (
            <button
              key={i}
              onClick={() => onCellSelect?.(s.lat, s.lon)}
              className="w-full flex justify-between items-center bg-gray-800/60 hover:bg-gray-700/60 rounded px-2 py-1.5 text-xs transition-colors"
            >
              <span className="font-mono text-gray-300">{s.lat.toFixed(3)}, {s.lon.toFixed(3)}</span>
              <span className={`font-mono ${s.prediction === 1 ? "text-red-400" : "text-emerald-400"}`}>
                {(s.risk_probability * 100).toFixed(0)}% risk
              </span>
              <span className="text-amber-400 text-[10px]">{"\u00B1"}{(s.uncertainty * 100).toFixed(0)}%</span>
            </button>
          ))}
          <div className="text-[9px] text-gray-600">Method: uncertainty sampling — cells closest to 50% decision boundary</div>
        </div>
      )}
    </div>
  );
}
