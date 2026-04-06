import { type ReviewSummary as ReviewSummaryType } from "../../lib/api";
import ExportButton from "./ExportButton";

interface Props {
  summary: ReviewSummaryType;
  onClose: () => void;
}

export default function ReviewSummary({ summary, onClose }: Props) {
  return (
    <div className="fixed inset-0 z-[2000] bg-gray-950/90 overflow-y-auto">
      <div className="max-w-3xl mx-auto my-8 bg-gray-900 rounded-xl border border-gray-700 shadow-2xl">
        <div id="review-summary-content" className="p-8 space-y-6">
          {/* Header */}
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-xl font-bold text-gray-100">Deforestation Risk Review Summary</h1>
              <p className="text-sm text-gray-400 mt-1">
                {summary.region?.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase())} — {summary.total_sites} sites analyzed
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                Generated {new Date(summary.generated_at).toLocaleString()} | Model: {summary.model_run_id}
              </p>
            </div>
            <button onClick={onClose} className="text-gray-400 hover:text-white text-sm px-3 py-1 border border-gray-700 rounded">
              Close
            </button>
          </div>

          {/* Aggregate stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800 rounded-lg p-4 text-center">
              <div className="text-xs text-gray-400">Total Cells</div>
              <div className="text-2xl font-bold">{summary.total_cells_analyzed}</div>
            </div>
            <div className="bg-red-900/30 rounded-lg p-4 text-center border border-red-800/30">
              <div className="text-xs text-gray-400">High Risk</div>
              <div className="text-2xl font-bold text-red-400">{summary.total_high_risk}</div>
              <div className="text-xs text-gray-500">{summary.overall_risk_pct}%</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 text-center">
              <div className="text-xs text-gray-400">Sites</div>
              <div className="text-2xl font-bold">{summary.total_sites}</div>
            </div>
          </div>

          {/* Priority ranking */}
          {summary.priority_sites.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">Priority Ranking</h2>
              <div className="text-xs text-gray-500 mb-2">Sites ranked by deforestation risk percentage — highest priority first</div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-800">
                    <th className="text-left py-2">Priority</th>
                    <th className="text-left py-2">Site</th>
                    <th className="text-right py-2">Cells</th>
                    <th className="text-right py-2">High Risk</th>
                    <th className="text-right py-2">Risk %</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.priority_sites.map((site, i) => (
                    <tr key={site.name} className="border-b border-gray-800/50">
                      <td className="py-2 font-bold text-amber-400">#{i + 1}</td>
                      <td className="py-2">{site.name}</td>
                      <td className="py-2 text-right text-gray-400">{site.cell_count}</td>
                      <td className="py-2 text-right text-red-400 font-mono">{site.high_risk}</td>
                      <td className="py-2 text-right font-mono font-bold text-red-400">{site.high_risk_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Per-site detail */}
          <div>
            <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">Site Details</h2>
            <div className="space-y-4">
              {summary.sites.map((site) => (
                <div key={site.name} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                  <div className="flex justify-between items-center mb-2">
                    <h3 className="font-medium text-gray-200">{site.name}</h3>
                    {site.high_risk_pct > 30 ? (
                      <span className="text-[10px] bg-red-900/50 text-red-300 px-2 py-0.5 rounded uppercase">Critical</span>
                    ) : site.high_risk_pct > 10 ? (
                      <span className="text-[10px] bg-amber-900/50 text-amber-300 px-2 py-0.5 rounded uppercase">Elevated</span>
                    ) : (
                      <span className="text-[10px] bg-emerald-900/50 text-emerald-300 px-2 py-0.5 rounded uppercase">Low</span>
                    )}
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div>
                      <div className="text-xs text-gray-500">Cells analyzed</div>
                      <div className="font-mono">{site.cell_count}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">High risk</div>
                      <div className="font-mono text-red-400">{site.high_risk} ({site.high_risk_pct}%)</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Primary driver</div>
                      <div className="text-gray-300">{site.primary_driver?.replace(/_/g, " ") || "—"}</div>
                    </div>
                  </div>
                  {site.top_hotspot && (
                    <div className="mt-2 text-xs text-gray-400">
                      Top hotspot: <span className="font-mono text-red-400">{site.top_hotspot.lat.toFixed(2)}°N, {site.top_hotspot.lon.toFixed(2)}°E</span>
                      {" "}({(site.top_hotspot.risk_probability * 100).toFixed(0)}% risk)
                    </div>
                  )}
                  <div className="mt-1 text-[10px] text-gray-500 font-mono">
                    Bounds: {site.bounds.lat_min.toFixed(3)}–{site.bounds.lat_max.toFixed(3)}°N, {site.bounds.lon_min.toFixed(3)}–{site.bounds.lon_max.toFixed(3)}°E
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Footer */}
          <div className="border-t border-gray-800 pt-4 text-xs text-gray-500">
            <p>This review summary was generated by DeforestAI using Hansen/UMD GFW v1.7 data, SRTM elevation, and OpenStreetMap features.</p>
            <p className="mt-1">Risk predictions are from a Random Forest classifier with SHAP explainability. All findings require field verification before enforcement action.</p>
            <p className="mt-1 italic">For official use by DENR/PENRO/CENRO — recommended for forwarding to municipal LGUs for ground-truth verification.</p>
          </div>
        </div>

        {/* Export button outside the capture area */}
        <div className="px-8 pb-6">
          <ExportButton targetId="review-summary-content" filename="deforestai-review-summary" />
        </div>
      </div>
    </div>
  );
}
