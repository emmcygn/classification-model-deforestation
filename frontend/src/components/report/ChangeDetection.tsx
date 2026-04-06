import { useState } from "react";
import { runChangeDetection, validateCV, type ChangeDetectionResult, type CVValidation, type Bounds } from "../../lib/api";

interface Props {
  bounds: Bounds;
}

export default function ChangeDetection({ bounds }: Props) {
  const [yearBefore, setYearBefore] = useState(2018);
  const [yearAfter, setYearAfter] = useState(2023);
  const [result, setResult] = useState<ChangeDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [validation, setValidation] = useState<CVValidation | null>(null);
  const [validating, setValidating] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    setError("");
    try {
      const r = await runChangeDetection(bounds, yearBefore, yearAfter);
      setResult(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="text-xs text-gray-400 uppercase tracking-wider">CV Change Detection</div>
      <div className="flex gap-2 items-end">
        <label className="flex-1">
          <span className="text-[10px] text-gray-500">From</span>
          <select value={yearBefore} onChange={(e) => {
            const v = +e.target.value;
            setYearBefore(v);
            if (yearAfter <= v) setYearAfter(v + 1);
          }}
            className="block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs mt-0.5">
            {[2018, 2019, 2020, 2021, 2022].map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </label>
        <label className="flex-1">
          <span className="text-[10px] text-gray-500">To</span>
          <select value={yearAfter} onChange={(e) => setYearAfter(+e.target.value)}
            className="block w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs mt-0.5">
            {[2019, 2020, 2021, 2022, 2023].filter((y) => y > yearBefore).map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </label>
        <button onClick={handleRun} disabled={loading || yearAfter <= yearBefore}
          className="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 rounded text-xs font-medium whitespace-nowrap">
          {loading ? "Analyzing..." : "Detect Changes"}
        </button>
      </div>

      {error && <div className="text-red-400 text-xs">{error}</div>}

      {result && (
        <div className="space-y-2 bg-gray-800/50 rounded-lg p-3">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[10px] text-gray-500">Tiles Analyzed</div>
              <div className="text-sm font-bold">{result.tiles_analyzed}</div>
            </div>
            <div>
              <div className="text-[10px] text-gray-500">Vegetation Loss</div>
              <div className="text-sm font-bold text-red-400">{result.loss_pct}%</div>
            </div>
            <div>
              <div className="text-[10px] text-gray-500">Vegetation Gain</div>
              <div className="text-sm font-bold text-emerald-400">{result.gain_pct}%</div>
            </div>
          </div>

          <div className="text-xs text-gray-400">
            ~{result.estimated_loss_hectares} ha estimated loss ({result.year_before}&rarr;{result.year_after})
            <span className="block text-[9px] text-gray-600 mt-0.5">Approximate — derived from visualization tiles, not analysis-ready data</span>
          </div>

          {result.hotspot_tiles.length > 0 && (
            <div>
              <div className="text-[10px] text-gray-500 mb-1">Highest Loss Tiles</div>
              {result.hotspot_tiles.slice(0, 3).map((t, i) => (
                <div key={i} className="text-xs flex justify-between text-gray-300 py-0.5">
                  <span className="font-mono">{t.lat.toFixed(2)}&deg;, {t.lon.toFixed(2)}&deg;</span>
                  <span className="text-red-400">{t.loss_pct}% loss</span>
                </div>
              ))}
            </div>
          )}

          <div className="text-[9px] text-gray-600 border-t border-gray-700 pt-1">
            Method: {result.method}
          </div>

          <button
            onClick={async () => {
              setValidating(true);
              setError("");
              try {
                const v = await validateCV(bounds, yearBefore, yearAfter);
                setValidation(v);
              } catch (e: any) {
                setError(e.message);
              } finally {
                setValidating(false);
              }
            }}
            disabled={validating}
            className="w-full px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-700 rounded text-xs mt-2"
          >
            {validating ? "Validating..." : "Validate Against Ground Truth"}
          </button>

          {validation && validation.valid && (
            <div className="bg-gray-800/50 rounded p-2 text-xs space-y-1 mt-2">
              <div className="text-[10px] text-gray-500 uppercase">CV Validation (vs Hansen GFW)</div>
              <div className="grid grid-cols-3 gap-1 text-center">
                <div>
                  <div className="text-[9px] text-gray-500">Correlation</div>
                  <div className="font-mono">{validation.correlation}</div>
                </div>
                <div>
                  <div className="text-[9px] text-gray-500">Precision</div>
                  <div className="font-mono">{(validation.precision * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <div className="text-[9px] text-gray-500">F1</div>
                  <div className="font-mono">{(validation.f1 * 100).toFixed(0)}%</div>
                </div>
              </div>
              <div className="text-[10px] text-gray-400">{validation.interpretation}</div>
            </div>
          )}

          {validation && !validation.valid && (
            <div className="text-yellow-400 text-xs mt-2">
              Insufficient data points for validation ({validation.valid_points} found).
            </div>
          )}
        </div>
      )}
    </div>
  );
}
