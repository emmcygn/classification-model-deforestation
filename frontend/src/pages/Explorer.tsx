import { useState, useEffect, useCallback } from "react";
import { listRuns, getGrid, getCellDetail, generateReport, geocodeSearch, getRegions, getCalibration, generateReviewSummary, getAnnotationStats } from "../lib/api";
import type { Run, GridCell, CellDetail, PolicyBrief as PolicyBriefType, Region, CalibrationCell, Bounds, ReviewSummary as ReviewSummaryType, AnnotationStats } from "../lib/api";
import RiskMap from "../components/map/RiskMap";
import CellDetailPanel from "../components/map/CellDetailPanel";
import TemporalPanel from "../components/map/TemporalPanel";
import PolicyBriefComponent from "../components/report/PolicyBrief";
import ReviewSummaryComponent from "../components/report/ReviewSummary";
import ExportButton from "../components/report/ExportButton";
import ChangeDetection from "../components/report/ChangeDetection";
import SpatialClusters from "../components/report/SpatialClusters";
import AnnotationSuggestions from "../components/report/AnnotationSuggestions";

export default function Explorer() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [regions, setRegions] = useState<Region[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>("");
  const [cells, setCells] = useState<GridCell[]>([]);
  const [cellDetail, setCellDetail] = useState<CellDetail | null>(null);
  const [cellLoading, setCellLoading] = useState(false);
  const [brief, setBrief] = useState<PolicyBriefType | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [mapCenter, setMapCenter] = useState<[number, number] | undefined>([12, 121]);
  const [showCalibration, setShowCalibration] = useState(false);
  const [calibrationCells, setCalibrationCells] = useState<CalibrationCell[]>([]);
  const [calibrationCounts, setCalibrationCounts] = useState<{ tp: number; tn: number; fp: number; fn: number } | null>(null);
  const [calibrationLoading, setCalibrationLoading] = useState(false);
  const [sites, setSites] = useState<{ name: string; bounds: Bounds }[]>([]);
  const [reviewSummary, setReviewSummary] = useState<ReviewSummaryType | null>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [annotationStats, setAnnotationStats] = useState<AnnotationStats | null>(null);
  const [error, setError] = useState("");

  const handleSiteAdd = useCallback((bounds: Bounds) => {
    setSites((prev) => [...prev, { name: `Site ${prev.length + 1}`, bounds }]);
  }, []);

  const handleClearSites = useCallback(() => {
    setSites([]);
  }, []);

  const handleGenerateReviewSummary = async () => {
    if (sites.length === 0 || !selectedRunId) return;
    setReviewLoading(true);
    try {
      const siteBounds = sites.map((s) => ({
        name: s.name,
        ...s.bounds,
      }));
      const summary = await generateReviewSummary(selectedRunId, siteBounds, selectedRegion || undefined);
      setReviewSummary(summary);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setReviewLoading(false);
    }
  };

  // Fetch regions on mount
  useEffect(() => {
    getRegions()
      .then((r) => {
        setRegions(r);
        if (r.length > 0) setSelectedRegion(r[0].name);
      })
      .catch(() => {});
  }, []);

  // Fetch runs on mount
  useEffect(() => {
    listRuns().then((r) => {
      setRuns(r);
      if (r.length > 0) setSelectedRunId(r[0].run_id);
    });
  }, []);

  // Recenter map when region changes
  useEffect(() => {
    if (!selectedRegion || regions.length === 0) return;
    const region = regions.find((r) => r.name === selectedRegion);
    if (region) {
      const lat = (region.bounds.lat_min + region.bounds.lat_max) / 2;
      const lon = (region.bounds.lon_min + region.bounds.lon_max) / 2;
      setMapCenter([lat, lon]);
    }
  }, [selectedRegion, regions]);

  // Fetch annotation stats when run changes
  useEffect(() => {
    if (!selectedRunId) return;
    getAnnotationStats(selectedRunId).then(setAnnotationStats).catch(() => setAnnotationStats(null));
  }, [selectedRunId]);

  // Reload grid when run or region changes
  useEffect(() => {
    if (!selectedRunId) return;
    setShowCalibration(false);
    setCalibrationCells([]);
    setCalibrationCounts(null);
    getGrid(selectedRunId, selectedRegion || undefined)
      .then((g) => setCells(g.cells))
      .catch((e) => setError(e.message));
  }, [selectedRunId, selectedRegion]);

  const handleCellClick = async (cell: GridCell) => {
    setCellLoading(true);
    try {
      const detail = await getCellDetail(cell.lat, cell.lon, selectedRunId, selectedRegion || undefined);
      setCellDetail(detail);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setCellLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    setReportLoading(true);
    try {
      const r = await generateReport(selectedRunId, selectedRegion || undefined);
      setBrief(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setReportLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      const result = await geocodeSearch(searchQuery);
      setMapCenter([result.lat, result.lon]);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleToggleCalibration = async () => {
    if (showCalibration) {
      setShowCalibration(false);
      return;
    }
    if (!selectedRunId) return;
    setCalibrationLoading(true);
    try {
      const data = await getCalibration(selectedRunId, selectedRegion || undefined);
      setCalibrationCells(data.cells);
      setCalibrationCounts(data.counts);
      setShowCalibration(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setCalibrationLoading(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-57px)] relative">
      {/* Sidebar */}
      <div className="w-80 border-r border-gray-800 overflow-y-auto p-4 space-y-4">
        <div>
          <label className="text-xs text-gray-400">Region</label>
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            {regions.map((r) => (
              <option key={r.name} value={r.name}>
                {r.display_name} ({r.cell_count} cells)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-400">Model</label>
          <select
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} (F1: {(r.metrics.f1 * 100).toFixed(1)}%)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-400">Search Location</label>
          <div className="flex gap-2 mt-1">
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="e.g. Palawan"
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm"
            />
            <button
              onClick={handleSearch}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              Go
            </button>
          </div>
        </div>

        <div className="border-t border-gray-800 pt-4">
          <button
            onClick={handleToggleCalibration}
            disabled={calibrationLoading || !selectedRunId}
            className={`w-full px-3 py-2 rounded text-sm font-medium transition-colors ${
              showCalibration
                ? "bg-amber-600 hover:bg-amber-700 text-white"
                : "bg-gray-700 hover:bg-gray-600 text-gray-200"
            } disabled:bg-gray-800 disabled:text-gray-500`}
          >
            {calibrationLoading
              ? "Loading..."
              : showCalibration
              ? "Show Risk Predictions"
              : "Show Model Calibration"}
          </button>
        </div>

        {showCalibration && calibrationCounts && (
          <div className="border-t border-gray-800 pt-4">
            <div className="text-xs text-gray-400 mb-2">Held-out Test Set Performance</div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-emerald-900/30 rounded p-2 text-center">
                <div className="text-[10px] text-gray-400">True Pos</div>
                <div className="font-bold text-emerald-400">{calibrationCounts.tp}</div>
              </div>
              <div className="bg-red-900/30 rounded p-2 text-center">
                <div className="text-[10px] text-gray-400">False Neg</div>
                <div className="font-bold text-red-400">{calibrationCounts.fn}</div>
              </div>
              <div className="bg-amber-900/30 rounded p-2 text-center">
                <div className="text-[10px] text-gray-400">False Pos</div>
                <div className="font-bold text-amber-400">{calibrationCounts.fp}</div>
              </div>
              <div className="bg-gray-800 rounded p-2 text-center">
                <div className="text-[10px] text-gray-400">True Neg</div>
                <div className="font-bold text-gray-400">{calibrationCounts.tn}</div>
              </div>
            </div>
            <div className="mt-2 text-[10px] text-gray-500">
              Accuracy: {((calibrationCounts.tp + calibrationCounts.tn) / (calibrationCounts.tp + calibrationCounts.tn + calibrationCounts.fp + calibrationCounts.fn) * 100).toFixed(1)}%
              <span className="block mt-0.5 text-gray-600">Out-of-sample (held-out test split)</span>
            </div>
          </div>
        )}

        <div className="border-t border-gray-800 pt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-400">Region Stats</span>
            <span className="text-xs text-gray-500">{cells.length} cells</span>
          </div>
          {cells.length > 0 && (
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-emerald-900/30 rounded p-2 text-center">
                <div className="text-xs text-gray-400">Low Risk</div>
                <div className="font-bold text-emerald-400">
                  {cells.filter((c) => c.prediction === 0).length}
                </div>
              </div>
              <div className="bg-red-900/30 rounded p-2 text-center">
                <div className="text-xs text-gray-400">High Risk</div>
                <div className="font-bold text-red-400">
                  {cells.filter((c) => c.prediction === 1).length}
                </div>
              </div>
            </div>
          )}
        </div>

        {selectedRegion && (
          <div className="border-t border-gray-800 pt-4">
            <TemporalPanel region={selectedRegion} />
          </div>
        )}

        {selectedRunId && selectedRegion && (
          <div className="border-t border-gray-800 pt-4">
            <SpatialClusters runId={selectedRunId} region={selectedRegion} onNavigate={(lat, lon) => {
              setMapCenter([lat, lon]);
            }} />
          </div>
        )}

        {selectedRunId && (
          <div className="border-t border-gray-800 pt-4">
            <AnnotationSuggestions
              runId={selectedRunId}
              region={selectedRegion || undefined}
              onCellSelect={(lat, lon) => {
                setMapCenter([lat, lon]);
                handleCellClick({ lat, lon, prediction: 0, risk_probability: 0.5 });
              }}
            />
          </div>
        )}

        <button
          onClick={handleGenerateReport}
          disabled={reportLoading || !selectedRunId}
          className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-sm font-medium"
        >
          {reportLoading ? "Generating..." : "Generate Report"}
        </button>

        {annotationStats && annotationStats.total > 0 && (
          <div className="flex gap-2 text-xs">
            <span className="text-gray-400">Reviews:</span>
            <span className="text-emerald-400">{annotationStats.accepted} accepted</span>
            <span className="text-red-400">{annotationStats.rejected} rejected</span>
          </div>
        )}

        {/* Multi-site Assessment */}
        <div className="border-t border-gray-800 pt-4">
          <div className="text-xs text-gray-400 mb-2 flex justify-between items-center">
            <span>Multi-site Assessment{sites.length > 0 ? ` (${sites.length})` : ""}</span>
            {sites.length > 0 && (
              <button onClick={handleClearSites} className="text-gray-500 hover:text-gray-300 text-[10px] uppercase">Clear All</button>
            )}
          </div>
          {sites.length === 0 ? (
            <div className="text-[11px] text-gray-500 bg-gray-800/40 rounded px-2 py-2">
              Use the <span className="text-gray-300 font-medium">Draw Area</span> button on the map toolbar to select sites for comparative analysis and review summaries.
            </div>
          ) : (
            <>
              <div className="space-y-1.5 mb-3">
                {sites.map((site) => {
                  const inBounds = cells.filter(
                    (c) =>
                      c.lat >= site.bounds.lat_min && c.lat <= site.bounds.lat_max &&
                      c.lon >= site.bounds.lon_min && c.lon <= site.bounds.lon_max
                  );
                  const highRisk = inBounds.filter((c) => c.prediction === 1);
                  const pct = inBounds.length > 0 ? (highRisk.length / inBounds.length * 100).toFixed(0) : "0";
                  return (
                    <div key={site.name} className="bg-gray-800/60 rounded px-2 py-1.5 text-xs flex justify-between items-center">
                      <span className="text-gray-300">{site.name}</span>
                      <span className="text-gray-400">
                        {inBounds.length} cells, <span className={highRisk.length > 0 ? "text-red-400" : "text-gray-500"}>{pct}% risk</span>
                      </span>
                    </div>
                  );
                })}
              </div>
              <button
                onClick={handleGenerateReviewSummary}
                disabled={reviewLoading || !selectedRunId || sites.length === 0}
                className="w-full px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700 disabled:text-gray-500 rounded text-xs font-medium"
              >
                {reviewLoading ? "Generating..." : "Generate Review Summary"}
              </button>
            </>
          )}
        </div>

        {sites.length > 0 && sites[sites.length - 1] && (
          <div className="border-t border-gray-800 pt-4">
            <ChangeDetection bounds={sites[sites.length - 1].bounds} />
          </div>
        )}

        <div className="border-t border-gray-800 pt-4">
          <CellDetailPanel detail={cellDetail} loading={cellLoading} runId={selectedRunId} />
        </div>

        {error && <div className="text-red-400 text-sm">{error}</div>}
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        <RiskMap
          cells={cells}
          onCellClick={handleCellClick}
          center={mapCenter}
          calibrationCells={calibrationCells}
          showCalibration={showCalibration}
          onSiteAdd={handleSiteAdd}
          sites={sites}
        />
      </div>

      {/* Policy Brief slide-over */}
      {brief && (
        <div className="absolute top-0 right-0 z-[1000] w-96 h-full overflow-y-auto bg-gray-900 border-l border-gray-700 shadow-xl p-4">
          <div id="policy-brief-content">
            <PolicyBriefComponent brief={brief} onClose={() => setBrief(null)} />
          </div>
          <div className="mt-4">
            <ExportButton targetId="policy-brief-content" />
          </div>
        </div>
      )}

      {/* Review Summary full-screen overlay */}
      {reviewSummary && (
        <ReviewSummaryComponent summary={reviewSummary} onClose={() => setReviewSummary(null)} />
      )}
    </div>
  );
}
