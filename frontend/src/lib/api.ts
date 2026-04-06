const BASE = "/api";

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// Pipeline
export const getDatasetInfo = () => fetchJSON<DatasetInfo>("/pipeline/dataset");
export const getDatasetSample = (n = 100) => fetchJSON<Record<string, number>[]>(`/pipeline/dataset/sample?n=${n}`);
export const trainModel = (params: TrainParams) =>
  fetchJSON<TrainResult>("/pipeline/train", { method: "POST", body: JSON.stringify(params) });
export const retrainWithAnnotations = (params: TrainParams) =>
  fetchJSON<TrainResult & { annotation_overrides: number; total_annotations: number }>(
    "/pipeline/retrain", { method: "POST", body: JSON.stringify(params) }
  );
export const listRuns = () => fetchJSON<Run[]>("/pipeline/runs");
export const getRun = (id: string) => fetchJSON<Run>(`/pipeline/runs/${id}`);

// Explorer
export const geocodeSearch = (q: string) => fetchJSON<GeocodeResult>(`/explorer/geocode?q=${encodeURIComponent(q)}`);
export const getRegions = () => fetchJSON<Region[]>("/explorer/regions");
export const getGrid = (runId: string, region?: string) => {
  const params = new URLSearchParams({ run_id: runId });
  if (region) params.set("region", region);
  return fetchJSON<GridResponse>(`/explorer/grid?${params}`);
};
export const getCellDetail = (lat: number, lon: number, runId: string, region?: string) => {
  const params = new URLSearchParams({ lat: String(lat), lon: String(lon), run_id: runId });
  if (region) params.set("region", region);
  return fetchJSON<CellDetail>(`/explorer/cell?${params}`);
};
export const getTemporalData = (region: string, bounds?: Bounds) => {
  const params = new URLSearchParams({ region });
  if (bounds) {
    params.set("lat_min", String(bounds.lat_min));
    params.set("lat_max", String(bounds.lat_max));
    params.set("lon_min", String(bounds.lon_min));
    params.set("lon_max", String(bounds.lon_max));
  }
  return fetchJSON<TemporalData>(`/explorer/temporal?${params}`);
};
export const getCalibration = (runId: string, region?: string) => {
  const params = new URLSearchParams({ run_id: runId });
  if (region) params.set("region", region);
  return fetchJSON<CalibrationResponse>(`/explorer/calibration?${params}`);
};
export const generateReport = (runId: string, region?: string, bounds?: Bounds) => {
  const params = new URLSearchParams({ run_id: runId });
  if (region) params.set("region", region);
  if (bounds) {
    params.set("lat_min", String(bounds.lat_min));
    params.set("lat_max", String(bounds.lat_max));
    params.set("lon_min", String(bounds.lon_min));
    params.set("lon_max", String(bounds.lon_max));
  }
  return fetchJSON<PolicyBrief>(`/explorer/report?${params}`, { method: "POST" });
};

// Types
export interface DatasetInfo {
  row_count: number;
  feature_columns: string[];
  feature_stats: Record<string, { min: number; max: number; mean: number; std: number }>;
  class_distribution: { low_risk: number; high_risk: number };
  geo_bounds: Bounds;
}

export interface Bounds {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface TrainParams {
  n_estimators?: number;
  max_depth?: number | null;
  min_samples_split?: number;
  min_samples_leaf?: number;
  test_size?: number;
  feature_columns?: string[] | null;
  region?: string | null;
  spatial_split?: boolean;
  model_type?: string;
}

export interface TrainResult {
  run_id: string;
  metrics: Metrics;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: number[][];
  feature_importance: { feature: string; importance: number }[];
}

export interface Run {
  run_id: string;
  created_at: string;
  params: TrainParams;
  metrics: Metrics;
  feature_names: string[];
  model_path: string;
}

export interface GeocodeResult {
  lat: number;
  lon: number;
  display_name: string;
}

export interface Region {
  name: string;
  display_name: string;
  cell_count: number;
  bounds: Bounds;
}

export interface GridResponse {
  cells: GridCell[];
  run_id: string;
}

export interface CalibrationCell {
  lat: number;
  lon: number;
  prediction: number;
  actual: number;
  risk_probability: number;
  label: "tp" | "tn" | "fp" | "fn";
}

export interface CalibrationResponse {
  cells: CalibrationCell[];
  counts: { tp: number; tn: number; fp: number; fn: number };
  run_id: string;
  region: string;
}

export interface GridCell {
  lat: number;
  lon: number;
  prediction: number;
  risk_probability: number;
}

export interface CellDetail {
  lat: number;
  lon: number;
  features: Record<string, { value: number; source: string }>;
  explanation: {
    prediction: number;
    prediction_label: string;
    base_value: number;
    shap_values: { feature: string; value: number; shap_value: number }[];
  };
  summary_text: string;
  loss_year?: number | null;
  risk_probability: number;
}

export interface TemporalData {
  years: number[];
  loss_counts: number[];
  total_cells: number;
  high_risk_cells: number;
  avg_loss_rate_pct: number;
  region: string;
}

export interface PolicyBrief {
  executive_summary: string;
  site_overview: { name: string; total_cells: number; area_km2: number; bounds: Bounds };
  risk_assessment: { high_risk: number; low_risk: number; high_risk_pct: number; high_risk_hectares: number };
  hotspots: { lat: number; lon: number; risk_probability: number }[];
  top_drivers: { feature: string; importance: number }[];
  notable_findings: string[];
  recommendations: string[];
  data_provenance: { feature: string; source: string; url: string }[];
  _cached?: boolean;
}

// Spatial Clustering
export interface SpatialCluster {
  cluster_id: number;
  n_cells: number;
  centroid: { lat: number; lon: number };
  bounds: Bounds;
  mean_risk: number;
  max_risk: number;
  approx_area_km2: number;
  severity: "critical" | "elevated" | "moderate";
}

export interface ClusterResult {
  total_high_risk: number;
  n_clusters: number;
  n_noise: number;
  clusters: SpatialCluster[];
  noise_points: { lat: number; lon: number; risk: number }[];
  method: string;
}

export const getSpatialClusters = (runId: string, region?: string) => {
  const params = new URLSearchParams({ run_id: runId });
  if (region) params.set("region", region);
  return fetchJSON<ClusterResult>(`/explorer/clusters?${params}`);
};

// Change Detection (CV)
export interface ChangeDetectionResult {
  year_before: number;
  year_after: number;
  bounds: Bounds;
  tiles_analyzed: number;
  total_pixels: number;
  loss_pixels: number;
  gain_pixels: number;
  loss_pct: number;
  gain_pct: number;
  estimated_loss_hectares: number;
  hotspot_tiles: {
    lat: number;
    lon: number;
    loss_pct: number;
    gain_pct: number;
    mean_change: number;
  }[];
  method: string;
}

export const runChangeDetection = (
  bounds: Bounds,
  yearBefore: number = 2018,
  yearAfter: number = 2023,
) => {
  const params = new URLSearchParams({
    lat_min: String(bounds.lat_min),
    lat_max: String(bounds.lat_max),
    lon_min: String(bounds.lon_min),
    lon_max: String(bounds.lon_max),
    year_before: String(yearBefore),
    year_after: String(yearAfter),
  });
  return fetchJSON<ChangeDetectionResult>(`/explorer/change-detection?${params}`, { method: "POST" });
};

// CV Validation
export interface CVValidation {
  valid: boolean;
  valid_points: number;
  correlation: number;
  signal_separation: number;
  confusion_matrix: { tp: number; fp: number; fn: number; tn: number };
  precision: number;
  recall: number;
  f1: number;
  interpretation: string;
  mean_exg_change_loss_cells: number;
  mean_exg_change_no_loss_cells: number;
}

export const validateCV = (bounds: Bounds, yearBefore: number, yearAfter: number) => {
  const params = new URLSearchParams({
    lat_min: String(bounds.lat_min),
    lat_max: String(bounds.lat_max),
    lon_min: String(bounds.lon_min),
    lon_max: String(bounds.lon_max),
    year_before: String(yearBefore),
    year_after: String(yearAfter),
    sample_points: "50",
  });
  return fetchJSON<CVValidation>(`/explorer/validate-cv?${params}`, { method: "POST" });
};

// Multi-site review summary
export interface SiteBounds {
  name: string;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface SiteAnalysis {
  name: string;
  bounds: Bounds;
  cell_count: number;
  high_risk: number;
  high_risk_pct: number;
  top_hotspot: { lat: number; lon: number; risk_probability: number } | null;
  primary_driver: string | null;
}

export interface ReviewSummary {
  generated_at: string;
  model_run_id: string;
  region: string;
  total_sites: number;
  total_cells_analyzed: number;
  total_high_risk: number;
  overall_risk_pct: number;
  sites: SiteAnalysis[];
  priority_sites: SiteAnalysis[];
}

export const generateReviewSummary = (runId: string, sites: SiteBounds[], region?: string) =>
  fetchJSON<ReviewSummary>("/explorer/review-summary", {
    method: "POST",
    body: JSON.stringify({ run_id: runId, sites, region }),
  });

// Annotations (human-in-the-loop)
export interface Annotation {
  id: string;
  lat: number;
  lon: number;
  run_id: string;
  prediction: number;
  risk_probability: number;
  verdict: "accept" | "reject";
  note: string;
  created_at: string;
}

export interface AnnotationStats {
  accepted: number;
  rejected: number;
  total: number;
}

export const annotateCell = (params: {
  lat: number;
  lon: number;
  run_id: string;
  prediction: number;
  risk_probability: number;
  verdict: "accept" | "reject";
  note?: string;
}) => {
  const searchParams = new URLSearchParams({
    lat: String(params.lat),
    lon: String(params.lon),
    run_id: params.run_id,
    prediction: String(params.prediction),
    risk_probability: String(params.risk_probability),
    verdict: params.verdict,
    note: params.note || "",
  });
  return fetchJSON<{ id: string; verdict: string }>(`/explorer/annotate?${searchParams}`, { method: "POST" });
};

export const getAnnotations = (runId?: string) => {
  const params = runId ? `?run_id=${runId}` : "";
  return fetchJSON<Annotation[]>(`/explorer/annotations${params}`);
};

export const getAnnotationStats = (runId: string) =>
  fetchJSON<AnnotationStats>(`/explorer/annotations/stats?run_id=${runId}`);

export const getCellAnnotation = (lat: number, lon: number, runId: string) =>
  fetchJSON<Annotation | Record<string, never>>(`/explorer/annotations/cell?lat=${lat}&lon=${lon}&run_id=${runId}`);
