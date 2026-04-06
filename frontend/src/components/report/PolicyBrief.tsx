import { type PolicyBrief as PolicyBriefType } from "../../lib/api";

interface Props {
  brief: PolicyBriefType;
  onClose: () => void;
}

export default function PolicyBrief({ brief, onClose }: Props) {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="font-bold text-sm">Policy Brief</h3>
          {brief._cached && (
            <span className="text-[10px] text-yellow-400 bg-yellow-900/30 px-1.5 py-0.5 rounded">
              cached example
            </span>
          )}
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-white text-sm">Close</button>
      </div>

      {brief.executive_summary && (
        <div>
          <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Executive Summary</div>
          <div className="text-sm text-gray-200 leading-relaxed">{brief.executive_summary}</div>
        </div>
      )}

      <div>
        <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Risk Assessment</div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-red-900/20 rounded p-2">
            <div className="text-xs text-gray-400">High Risk</div>
            <div className="font-bold text-red-400">{brief.risk_assessment.high_risk} cells</div>
            <div className="text-[10px] text-gray-500">{brief.risk_assessment.high_risk_pct}%</div>
          </div>
          <div className="bg-emerald-900/20 rounded p-2">
            <div className="text-xs text-gray-400">Low Risk</div>
            <div className="font-bold text-emerald-400">{brief.risk_assessment.low_risk} cells</div>
          </div>
        </div>
      </div>

      {brief.hotspots && brief.hotspots.length > 0 && (
        <div>
          <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Hotspots</div>
          {brief.hotspots.map((h, i) => (
            <div key={i} className="text-sm flex justify-between border-b border-gray-800/50 py-1">
              <span className="font-mono text-gray-300">{h.lat.toFixed(2)}, {h.lon.toFixed(2)}</span>
              <span className="text-red-400 font-mono">{(h.risk_probability * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}

      <div>
        <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Top Risk Drivers</div>
        {brief.top_drivers.slice(0, 5).map((f) => (
          <div key={f.feature} className="text-sm flex justify-between">
            <span className="text-gray-300">{f.feature.replace(/_/g, " ")}</span>
            <span className="font-mono text-emerald-400">{(f.importance * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {brief.recommendations && brief.recommendations.length > 0 && (
        <div>
          <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Recommendations</div>
          <ul className="text-sm space-y-1">
            {brief.recommendations.map((r, i) => (
              <li key={i} className="text-gray-200 pl-3 relative">
                <span className="absolute left-0 text-emerald-400">•</span>
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}

      {brief.notable_findings && brief.notable_findings.length > 0 && (
        <div>
          <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Notable Findings</div>
          <ul className="text-sm space-y-1">
            {brief.notable_findings.map((p, i) => (
              <li key={i} className="text-yellow-300">• {p}</li>
            ))}
          </ul>
        </div>
      )}

      {brief.data_provenance && brief.data_provenance.length > 0 && (
        <div>
          <div className="text-xs text-gray-400 mb-1 uppercase tracking-wider">Data Sources</div>
          {brief.data_provenance.map((p) => (
            <div key={p.feature} className="text-[11px] flex justify-between border-b border-gray-800/30 py-0.5">
              <span className="text-gray-400">{p.feature.replace(/_/g, " ")}</span>
              <span className="text-gray-500">{p.source}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
