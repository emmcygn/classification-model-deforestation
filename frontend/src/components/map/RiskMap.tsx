import { useEffect, useState, useRef } from "react";
import {
  MapContainer, TileLayer, CircleMarker, Rectangle, useMap,
} from "react-leaflet";
import { type GridCell, type CalibrationCell } from "../../lib/api";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

function riskColor(prob: number): string {
  if (prob > 0.7) return "#ef4444";
  if (prob > 0.4) return "#f59e0b";
  return "#22c55e";
}

const CALIBRATION_COLORS: Record<string, string> = {
  tp: "#22c55e", tn: "#6b7280", fp: "#f59e0b", fn: "#ef4444",
};
const CALIBRATION_LABELS: Record<string, string> = {
  tp: "True Positive", tn: "True Negative", fp: "False Positive", fn: "False Negative",
};

export interface AreaBounds {
  lat_min: number; lat_max: number; lon_min: number; lon_max: number;
}

interface SiteEntry { name: string; bounds: AreaBounds; }

interface Props {
  cells: GridCell[];
  onCellClick: (cell: GridCell) => void;
  center?: [number, number];
  calibrationCells?: CalibrationCell[];
  showCalibration?: boolean;
  onSiteAdd?: (bounds: AreaBounds) => void;
  sites?: SiteEntry[];
}

function RecenterMap({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => { map.setView(center, map.getZoom()); }, [map, center]);
  return null;
}

function DrawControl({ onSiteAdd, drawing, setDrawing }: {
  onSiteAdd: (bounds: AreaBounds) => void;
  drawing: boolean;
  setDrawing: (d: boolean) => void;
}) {
  const map = useMap();
  const startRef = useRef<L.LatLng | null>(null);
  const rectRef = useRef<L.Rectangle | null>(null);

  useEffect(() => {
    if (!drawing) return;
    const onMouseDown = (e: L.LeafletMouseEvent) => {
      startRef.current = e.latlng;
      map.dragging.disable();
    };
    const onMouseMove = (e: L.LeafletMouseEvent) => {
      if (!startRef.current) return;
      const bounds = L.latLngBounds(startRef.current, e.latlng);
      if (rectRef.current) rectRef.current.setBounds(bounds);
      else rectRef.current = L.rectangle(bounds, { color: "#10b981", weight: 2, fillOpacity: 0.1, fillColor: "#10b981", dashArray: "6 4" }).addTo(map);
    };
    // Use window-level mouseup so releasing outside the map still completes the draw
    const onMouseUp = () => {
      map.dragging.enable();
      if (startRef.current && rectRef.current) {
        const bounds = rectRef.current.getBounds();
        if (bounds) onSiteAdd({ lat_min: bounds.getSouth(), lat_max: bounds.getNorth(), lon_min: bounds.getWest(), lon_max: bounds.getEast() });
        rectRef.current.remove();
        rectRef.current = null;
      } else if (rectRef.current) {
        rectRef.current.remove();
        rectRef.current = null;
      }
      startRef.current = null;
      setDrawing(false);
    };
    map.on("mousedown", onMouseDown);
    map.on("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    map.getContainer().style.cursor = "crosshair";
    return () => {
      map.off("mousedown", onMouseDown);
      map.off("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      map.getContainer().style.cursor = "";
      map.dragging.enable();
    };
  }, [drawing, map, onSiteAdd, setDrawing]);

  return null;
}

const GFW_LOSS = "https://tiles.globalforestwatch.org/umd_tree_cover_loss/v1.11/tcd_30/{z}/{x}/{y}.png";
const S2_YEARLY_URL = (year: number) =>
  `https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-${year}_3857/default/g/{z}/{y}/{x}.jpg`;
const AVAILABLE_YEARS = [2018, 2019, 2020, 2021, 2022, 2023];

type Basemap = "street" | "satellite";
type Overlay = "none" | "loss";
type DisplayMode = "dots" | "grid";
type SatYear = number | "live";

export default function RiskMap({ cells, onCellClick, center, calibrationCells, showCalibration, onSiteAdd, sites }: Props) {
  const defaultCenter: [number, number] = center || [10, 119];
  const [basemap, setBasemap] = useState<Basemap>("street");
  const [overlay, setOverlay] = useState<Overlay>("none");
  const [displayMode, setDisplayMode] = useState<DisplayMode>("dots");
  const [satYear, setSatYear] = useState<SatYear>("live");
  const [drawing, setDrawing] = useState(false);

  const step = 0.02;
  const hasOverlay = overlay !== "none" || satYear !== "live";
  const dotRadius = 8;
  const dotFillOpacity = hasOverlay ? 0.5 : 0.75;
  const gridFillOpacity = hasOverlay ? 0.3 : 0.5;

  return (
    <div className="h-full w-full relative">
      <MapContainer center={defaultCenter} zoom={8} className="h-full w-full rounded-lg" style={{ minHeight: "500px" }} zoomControl={false}>

        {/* Base layer — controlled by state, not LayersControl */}
        {basemap === "street" ? (
          <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        ) : (
          <TileLayer attribution="Tiles &copy; Esri"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" maxZoom={18} />
        )}

        {center && <RecenterMap center={center} />}

        {/* Sentinel-2 yearly overlay */}
        {typeof satYear === "number" && (
          <TileLayer key={`s2-${satYear}`} url={S2_YEARLY_URL(satYear)}
            attribution={`Sentinel-2 ${satYear} EOX`} maxZoom={14} maxNativeZoom={14} zIndex={300} />
        )}

        {/* GFW forest loss overlay */}
        {overlay === "loss" && (
          <TileLayer key="gfw-loss" url={GFW_LOSS} attribution="Hansen/UMD/Google/USGS/NASA"
            opacity={0.9} maxNativeZoom={12} maxZoom={18} minZoom={9} zIndex={400} className="gfw-loss-layer" />
        )}

        {/* Cells */}
        {showCalibration && calibrationCells
          ? calibrationCells.map((cell, i) => {
              const color = CALIBRATION_COLORS[cell.label];
              const isSubtle = cell.label === "tn";
              return displayMode === "grid" ? (
                <Rectangle key={`c-${i}`} bounds={[[cell.lat-step/2,cell.lon-step/2],[cell.lat+step/2,cell.lon+step/2]]}
                  pathOptions={{ fillColor: color, fillOpacity: isSubtle ? 0.12 : 0.5, color, weight: 1 }} />
              ) : (
                <CircleMarker key={`c-${i}`} center={[cell.lat,cell.lon]} radius={isSubtle ? 3 : 5}
                  pathOptions={{ fillColor: color, fillOpacity: isSubtle ? 0.2 : 0.65, color: isSubtle ? color : "#fff", weight: isSubtle ? 0 : 1.5 }} />
              );
            })
          : cells.map((cell, i) => {
              const color = riskColor(cell.risk_probability);
              return displayMode === "grid" ? (
                <Rectangle key={i} bounds={[[cell.lat-step/2,cell.lon-step/2],[cell.lat+step/2,cell.lon+step/2]]}
                  pathOptions={{ fillColor: color, fillOpacity: gridFillOpacity, color: "#1f2937", weight: 1.5, opacity: 0.8 }}
                  eventHandlers={{ click: () => onCellClick(cell) }} />
              ) : (
                <CircleMarker key={i} center={[cell.lat,cell.lon]} radius={dotRadius}
                  pathOptions={{ fillColor: color, fillOpacity: dotFillOpacity, color: "#1f2937", weight: 2, opacity: 0.9 }}
                  eventHandlers={{ click: () => onCellClick(cell) }} />
              );
            })}

        {/* Site rectangles */}
        {sites?.map((site, i) => (
          <Rectangle key={`site-${i}`}
            bounds={[[site.bounds.lat_min,site.bounds.lon_min],[site.bounds.lat_max,site.bounds.lon_max]]}
            pathOptions={{ color: "#10b981", weight: 2, fillOpacity: 0.08, dashArray: "6 4" }} />
        ))}

        {onSiteAdd && <DrawControl onSiteAdd={onSiteAdd} drawing={drawing} setDrawing={setDrawing} />}
      </MapContainer>

      {/* ─── Unified toolbar ─── */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-[1000] bg-gray-900/95 border border-gray-700 rounded-xl shadow-2xl backdrop-blur-sm">
        <div className="flex items-stretch divide-x divide-gray-700/50">

          {/* Basemap */}
          <div className="px-3 py-2">
            <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Basemap</div>
            <div className="flex gap-1">
              <button onClick={() => setBasemap("street")}
                className={`px-2 py-0.5 rounded text-[11px] ${basemap === "street" ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                Map
              </button>
              <button onClick={() => setBasemap("satellite")}
                className={`px-2 py-0.5 rounded text-[11px] ${basemap === "satellite" ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                Satellite
              </button>
            </div>
          </div>

          {/* Data layer */}
          <div className="px-3 py-2">
            <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Data Layer</div>
            <div className="flex gap-1">
              <button onClick={() => setOverlay("none")}
                className={`px-2 py-0.5 rounded text-[11px] ${overlay === "none" ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                Risk
              </button>
              <button onClick={() => setOverlay("loss")}
                className={`px-2 py-0.5 rounded text-[11px] ${overlay === "loss" ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                GFW Loss
              </button>
            </div>
          </div>

          {/* Display */}
          <div className="px-3 py-2">
            <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Display</div>
            <div className="flex gap-1">
              <button onClick={() => setDisplayMode("dots")}
                className={`px-2 py-0.5 rounded text-[11px] ${displayMode === "dots" ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                Dots
              </button>
              <button onClick={() => setDisplayMode("grid")}
                className={`px-2 py-0.5 rounded text-[11px] ${displayMode === "grid" ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                Grid
              </button>
            </div>
          </div>

          {/* Site selection */}
          <div className="px-3 py-2">
            <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Site</div>
            <button onClick={() => setDrawing(!drawing)}
              className={`px-2 py-0.5 rounded text-[11px] ${drawing ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
              {drawing ? "Drawing..." : "Draw Area"}
            </button>
          </div>

          {/* Satellite year */}
          <div className="px-3 py-2">
            <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Imagery Year</div>
            <div className="flex gap-0.5">
              <button onClick={() => setSatYear("live")}
                className={`px-1.5 py-0.5 rounded text-[10px] ${satYear === "live" ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}
                title="ESRI World Imagery — typically 1-3 years old depending on area">
                Live*
              </button>
              {AVAILABLE_YEARS.map((y) => (
                <button key={y} onClick={() => setSatYear(y)}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${satYear === y ? "bg-emerald-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                  {String(y).slice(2)}
                </button>
              ))}
            </div>
          </div>

        </div>
        {/* Info line */}
        <div className="text-[9px] text-gray-600 text-center pb-1 px-3">
          {satYear === "live"
            ? "* Live imagery is ESRI World Imagery — typically 1-3 years old depending on area"
            : `Sentinel-2 cloudless composite ${satYear} — EOX`}
          {overlay === "loss" && " | GFW loss overlay active — zoom in for 30m pixel detail"}
        </div>
      </div>

      {/* Calibration legend — only when active */}
      {showCalibration && calibrationCells && (
        <div className="absolute top-4 left-4 z-[1000] bg-gray-900/95 border border-gray-700 rounded-lg shadow-lg p-3">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">Model Calibration</div>
          <div className="space-y-1">
            {(["tp", "fn", "fp", "tn"] as const).map((key) => (
              <div key={key} className="flex items-center gap-2 text-xs text-gray-300">
                <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: CALIBRATION_COLORS[key], opacity: key === "tn" ? 0.4 : 1 }} />
                <span>{CALIBRATION_LABELS[key]}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
