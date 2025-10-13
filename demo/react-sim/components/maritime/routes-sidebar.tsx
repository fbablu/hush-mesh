"use client";

import type { MaritimeSimulation } from "@/types/maritime";
import { Map, Target, Zap, Shield, Radio, CornerUpLeft } from "lucide-react";

interface RoutesSidebarProps {
  simulation: MaritimeSimulation;
}

export function RoutesSidebar({ simulation }: RoutesSidebarProps) {
  const { routes, selectedRoute, setSelectedRoute, routeAnalysis, missionLog } =
    simulation;

  return (
    <div className="bg-white/5 p-3 border-l border-white/10 overflow-y-auto">
      <h3 className="text-emerald-400 mb-2 text-[10px] font-bold flex items-center gap-1.5">
        <Map className="w-3 h-3" />
        Route Options
      </h3>

      <div className="space-y-1.5 mb-4">
        {Object.entries(routes).map(([type, route]) => {
          const isSelected = type === selectedRoute;
          const borderColors = {
            optimal: "border-emerald-400",
            fastest: "border-red-400",
            safest: "border-cyan-400",
            return: "border-amber-400",
          };
          const icons = {
            optimal: Target,
            fastest: Zap,
            safest: Shield,
            return: CornerUpLeft,
          };

          const Icon = icons[type as keyof typeof icons];

          return (
            <div
              key={type}
              onClick={() => setSelectedRoute(type)}
              className={`bg-white/8 rounded-lg p-2 border-l-4 cursor-pointer transition-all hover:bg-white/12 ${
                borderColors[type as keyof typeof borderColors]
              } ${isSelected ? "bg-emerald-500/10" : ""}`}
            >
              <strong className="text-xs capitalize flex items-center gap-1.5">
                <Icon className="w-3 h-3" />
                {type} Route
              </strong>
              <div className="text-[9px] mt-1 space-x-2">
                <span>Distance: {route.metrics.distance}km</span>
                <span>Risk: {route.metrics.threatExposure}</span>
                <span>Score: {route.metrics.score}</span>
              </div>
            </div>
          );
        })}
      </div>

      <h3 className="text-emerald-400 mb-2 text-[10px] font-bold flex items-center gap-1.5">
        <Radio className="w-3 h-3" />
        Path Analysis
      </h3>
      <div className="bg-black/30 p-2 mb-3 rounded-lg text-[10px] space-y-1.5">
        <div className="p-1.5 bg-white/5 rounded">{routeAnalysis.summary}</div>
        <div className="p-1.5 bg-white/5 rounded">
          {routeAnalysis.mlAnalysis}
        </div>
        <div className="p-1.5 bg-white/5 rounded">
          {routeAnalysis.recommendation}
        </div>
      </div>

      <h3 className="text-emerald-400 mb-2 text-[10px] font-bold flex items-center gap-1.5">
        <Target className="w-3 h-3" />
        Mission Log
      </h3>
      <div className="bg-black/30 p-2 rounded-lg h-32 overflow-y-auto text-[9px] space-y-0.5">
        {missionLog.map((log, i) => (
          <div key={i}>{log}</div>
        ))}
      </div>
    </div>
  );
}
