import { useEffect, useState } from "react";
import { getTemporalData, type TemporalData } from "../../lib/api";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface Props {
  region: string;
}

export default function TemporalPanel({ region }: Props) {
  const [data, setData] = useState<TemporalData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!region) return;
    setData(null);
    getTemporalData(region).then(setData).catch((e) => setError(e.message));
  }, [region]);

  if (error) return <div className="text-red-400 text-xs">{error}</div>;
  if (!data) return <div className="text-gray-500 text-xs">Loading temporal data...</div>;

  const chartData = data.years.map((year, i) => ({
    year,
    loss: data.loss_counts[i],
  }));

  return (
    <div>
      <div className="text-xs text-gray-400 mb-2">Forest Loss Over Time</div>
      <ResponsiveContainer width="100%" height={120}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="year"
            tick={{ fill: "#6b7280", fontSize: 10 }}
            tickFormatter={(v) => (v % 5 === 0 ? String(v) : "")}
          />
          <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} width={30} />
          <Tooltip
            contentStyle={{ background: "#1f2937", border: "1px solid #374151", fontSize: 12 }}
            labelFormatter={(v) => `Year ${v}`}
            formatter={(v) => [`${v} cells`, "Loss"]}
          />
          <Area
            type="monotone"
            dataKey="loss"
            stroke="#ef4444"
            fill="url(#lossGradient)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex justify-between text-[10px] text-gray-500 mt-1">
        <span>{data.total_cells} total cells</span>
        <span>{data.high_risk_cells} high risk ({data.avg_loss_rate_pct}%/yr)</span>
      </div>
    </div>
  );
}
