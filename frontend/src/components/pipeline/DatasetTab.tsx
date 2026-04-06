import { useEffect, useState } from "react";
import { getDatasetInfo, type DatasetInfo } from "../../lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

export default function DatasetTab() {
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    getDatasetInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="text-red-400 p-4">{error}</div>;
  if (!info) return <div className="text-gray-400 p-4">Loading dataset...</div>;

  const distData = [
    { name: "Low Risk", count: info.class_distribution.low_risk, fill: "#22c55e" },
    { name: "High Risk", count: info.class_distribution.high_risk, fill: "#ef4444" },
  ];

  const featureData = Object.entries(info.feature_stats).map(([name, stats]) => ({
    name: name.replace(/_/g, " "),
    mean: stats.mean,
    min: stats.min,
    max: stats.max,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="text-sm text-gray-400">Total Cells</div>
          <div className="text-2xl font-bold">{info.row_count.toLocaleString()}</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="text-sm text-gray-400">Features</div>
          <div className="text-2xl font-bold">{info.feature_columns.length}</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="text-sm text-gray-400">Region</div>
          <div className="text-sm font-mono mt-1">
            {info.geo_bounds.lat_min.toFixed(1)}° to {info.geo_bounds.lat_max.toFixed(1)}°N
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Class Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={distData}>
              <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} />
              <Tooltip contentStyle={{ background: "#1f2937", border: "1px solid #374151" }} />
              <Bar dataKey="count">
                {distData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Feature Statistics</h3>
          <div className="overflow-y-auto max-h-52">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 border-b border-gray-800">
                  <th className="text-left py-1">Feature</th>
                  <th className="text-right py-1">Min</th>
                  <th className="text-right py-1">Mean</th>
                  <th className="text-right py-1">Max</th>
                </tr>
              </thead>
              <tbody>
                {featureData.map((f) => (
                  <tr key={f.name} className="border-b border-gray-800/50">
                    <td className="py-1 text-gray-300">{f.name}</td>
                    <td className="py-1 text-right text-gray-400">{f.min}</td>
                    <td className="py-1 text-right text-gray-300">{f.mean}</td>
                    <td className="py-1 text-right text-gray-400">{f.max}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
