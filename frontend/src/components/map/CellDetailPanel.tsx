import { useState, useEffect } from "react";
import { type CellDetail, type Annotation, annotateCell, getCellAnnotation } from "../../lib/api";

interface Props {
  detail: CellDetail | null;
  loading: boolean;
  runId: string;
}

export default function CellDetailPanel({ detail, loading, runId }: Props) {
  const [annotation, setAnnotation] = useState<Annotation | null>(null);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [fetchingAnnotation, setFetchingAnnotation] = useState(false);

  // Fetch existing annotation when cell detail changes
  useEffect(() => {
    if (!detail || !runId) {
      setAnnotation(null);
      setNote("");
      return;
    }
    let cancelled = false;
    setFetchingAnnotation(true);
    getCellAnnotation(detail.lat, detail.lon, runId)
      .then((result) => {
        if (cancelled) return;
        if (result && "id" in result) {
          setAnnotation(result as Annotation);
          setNote((result as Annotation).note || "");
        } else {
          setAnnotation(null);
          setNote("");
        }
      })
      .catch(() => {
        if (!cancelled) setAnnotation(null);
      })
      .finally(() => {
        if (!cancelled) setFetchingAnnotation(false);
      });
    return () => { cancelled = true; };
  }, [detail?.lat, detail?.lon, runId]);

  const handleVerdict = async (verdict: "accept" | "reject") => {
    if (!detail || !runId) return;
    setSaving(true);
    try {
      const result = await annotateCell({
        lat: detail.lat,
        lon: detail.lon,
        run_id: runId,
        prediction: detail.explanation.prediction,
        risk_probability: detail.risk_probability,
        verdict,
        note,
      });
      setAnnotation({
        id: result.id,
        lat: detail.lat,
        lon: detail.lon,
        run_id: runId,
        prediction: detail.explanation.prediction,
        risk_probability: 0,
        verdict,
        note,
        created_at: new Date().toISOString(),
      });
    } catch (err) {
      console.error("Failed to save annotation:", err);
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div className="text-gray-400 text-sm p-4">Loading cell details...</div>;
  if (!detail) return <div className="text-gray-500 text-sm p-4">Click a cell on the map to see details.</div>;

  return (
    <div className="space-y-4">
      <div>
        <div className="text-xs text-gray-400">Location</div>
        <div className="font-mono text-sm">{detail.lat.toFixed(4)}, {detail.lon.toFixed(4)}</div>
      </div>

      <div>
        <div className={`inline-block px-2 py-1 rounded text-sm font-medium ${
          detail.explanation.prediction === 1
            ? "bg-red-900/50 text-red-300"
            : "bg-emerald-900/50 text-emerald-300"
        }`}>
          {detail.explanation.prediction_label}
        </div>
        {detail.loss_year && (
          <div className="text-xs text-amber-400 mt-1">
            Forest loss detected: {detail.loss_year}
          </div>
        )}
      </div>

      <div>
        <div className="text-xs text-gray-400 mb-1">Explainability</div>
        <div className="text-sm text-gray-200 bg-gray-800 rounded p-3 font-mono">
          {detail.summary_text}
        </div>
      </div>

      <div>
        <div className="text-xs text-gray-400 mb-2">Features</div>
        <table className="w-full text-sm">
          <tbody>
            {Object.entries(detail.features).map(([key, entry]) => (
              <tr key={key} className="border-b border-gray-800/50">
                <td className="py-1 text-gray-400">{key.replace(/_/g, " ")}</td>
                <td className="py-1 text-right">
                  <span className="font-mono">{entry.value}</span>
                  {entry.source && <span className="text-[9px] text-gray-500 ml-1">[{entry.source.split(' ')[0]}]</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <div className="text-xs text-gray-400 mb-2">SHAP Contributions</div>
        {detail.explanation.shap_values.map((sv) => (
          <div key={sv.feature} className="flex items-center gap-2 text-sm mb-1">
            <span className="text-gray-400 flex-1 truncate">{sv.feature.replace(/_/g, " ")}</span>
            <div className="w-20 h-3 bg-gray-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${sv.shap_value >= 0 ? "bg-red-500" : "bg-emerald-500"}`}
                style={{ width: `${Math.min(Math.abs(sv.shap_value) * 200, 100)}%` }}
              />
            </div>
            <span className={`font-mono text-xs w-14 text-right ${
              sv.shap_value >= 0 ? "text-red-400" : "text-emerald-400"
            }`}>
              {sv.shap_value >= 0 ? "+" : ""}{sv.shap_value.toFixed(3)}
            </span>
          </div>
        ))}
      </div>

      {/* Human-in-the-loop review section */}
      <div className="border-t border-gray-700 pt-4">
        <div className="text-xs text-gray-400 mb-3 uppercase tracking-wider">Review</div>

        {annotation ? (
          <div className="space-y-2">
            <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium ${
              annotation.verdict === "accept"
                ? "bg-emerald-900/60 text-emerald-300 border border-emerald-700/50"
                : "bg-orange-900/60 text-orange-300 border border-orange-700/50"
            }`}>
              {annotation.verdict === "accept" ? "\u2713 Accepted" : "\u2717 Rejected"}
            </div>
            {annotation.note && (
              <div className="text-sm text-gray-400 italic">"{annotation.note}"</div>
            )}
            <button
              className="text-xs text-gray-500 hover:text-gray-300 underline"
              onClick={() => setAnnotation(null)}
            >
              Change verdict
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex gap-2">
              <button
                disabled={saving}
                onClick={() => handleVerdict("accept")}
                className="flex-1 px-3 py-2 rounded text-sm font-medium bg-emerald-900/40 text-emerald-300 border border-emerald-700/50 hover:bg-emerald-800/60 disabled:opacity-50 transition-colors"
              >
                {saving ? "..." : "\u2713 Accept"}
              </button>
              <button
                disabled={saving}
                onClick={() => handleVerdict("reject")}
                className="flex-1 px-3 py-2 rounded text-sm font-medium bg-orange-900/40 text-orange-300 border border-orange-700/50 hover:bg-orange-800/60 disabled:opacity-50 transition-colors"
              >
                {saving ? "..." : "\u2717 Reject"}
              </button>
            </div>
            <input
              type="text"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Optional note..."
              className="w-full px-3 py-1.5 rounded bg-gray-800 border border-gray-700 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-gray-500"
            />
          </div>
        )}
      </div>
    </div>
  );
}
