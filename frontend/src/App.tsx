import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import ExplorerPage from "./pages/Explorer";
import PipelinePage from "./pages/Pipeline";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100">
        <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-6">
          <span className="text-lg font-bold text-emerald-400">DeforestAI</span>
          <NavLink to="/" end className={({ isActive }) => isActive ? "text-emerald-400" : "text-gray-400 hover:text-gray-200"}>Explorer</NavLink>
          <NavLink to="/pipeline" className={({ isActive }) => isActive ? "text-emerald-400" : "text-gray-400 hover:text-gray-200"}>Pipeline</NavLink>
        </nav>
        <Routes>
          <Route path="/" element={<ExplorerPage />} />
          <Route path="/pipeline" element={<PipelinePage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
