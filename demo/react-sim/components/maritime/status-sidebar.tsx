"use client";

import type { MaritimeSimulation } from "@/types/maritime";
import {
  AlertTriangle,
  Fuel,
  Shield,
  BarChart3,
  Radar,
  CheckCircle2,
  XCircle,
  Radio,
} from "lucide-react";

interface StatusSidebarProps {
  simulation: MaritimeSimulation;
}

export function StatusSidebar({ simulation }: StatusSidebarProps) {
  const { ship, drones, threats, mlDetections, fuelDecision, emergencyBeacon } =
    simulation;

  const activeDrones = drones.filter((d) => d.active).length;
  const detectedThreats = drones.filter(
    (d) => d.active && d.detectedThreat,
  ).length;

  const getFuelColor = () => {
    if (ship.fuel > 60) return "bg-emerald-500";
    if (ship.fuel > 30) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="bg-white/5 p-3 border-r border-white/10 overflow-y-auto">
      <div className="mb-4">
        <h3 className="text-emerald-400 mb-2 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5">
          <Shield className="w-2.5 h-2.5" />
          Mesh Status
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {/* Active Drones */}
          <div className="bg-black/30 p-2.5 rounded-lg border border-emerald-500/30">
            <Radio className="w-4 h-4 text-emerald-400 mb-1.5" />
            <div className="text-3xl font-bold text-emerald-400 leading-none mb-1">
              {activeDrones}
              <span className="text-base text-gray-400">/{drones.length}</span>
            </div>
            <div className="text-[9px] text-gray-400 uppercase tracking-wide">
              Active
            </div>
          </div>

          {/* Threats Detected */}
          <div
            className={`p-2.5 rounded-lg border ${
              detectedThreats > 0
                ? "bg-red-500/20 border-red-500/50"
                : "bg-black/30 border-white/10"
            }`}
          >
            <AlertTriangle
              className={`w-4 h-4 mb-1.5 ${
                detectedThreats > 0 ? "text-red-400" : "text-gray-500"
              }`}
            />
            <div
              className={`text-3xl font-bold leading-none mb-1 ${
                detectedThreats > 0 ? "text-red-400" : "text-gray-500"
              }`}
            >
              {detectedThreats}
            </div>
            <div className="text-[9px] text-gray-400 uppercase tracking-wide">
              Threats
            </div>
          </div>
        </div>
      </div>

      {emergencyBeacon && (
        <div className="mb-3 p-2 bg-red-500/30 border-2 border-red-500 rounded-lg animate-pulse">
          <div className="flex items-center gap-1.5 text-red-400 font-bold text-xs mb-1">
            <AlertTriangle className="w-4 h-4" />
            EMERGENCY BEACON ACTIVE
          </div>
          <p className="text-[10px] text-red-300">
            Homebase has been notified. Ship is in critical condition.
          </p>
        </div>
      )}

      <h3 className="text-emerald-400 mb-2 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5">
        <Fuel className="w-2.5 h-2.5" />
        Fuel Status
      </h3>
      <div className="bg-black/30 p-2.5 mb-3 rounded-lg">
        <div className="mb-2">
          <div className="text-4xl font-bold text-center mb-1.5">
            <span
              className={
                ship.fuel > 60
                  ? "text-emerald-400"
                  : ship.fuel > 30
                    ? "text-yellow-400"
                    : "text-red-400"
              }
            >
              {ship.fuel.toFixed(0)}
            </span>
            <span className="text-xl text-gray-500">%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full ${getFuelColor()} transition-all duration-300`}
              style={{ width: `${ship.fuel}%` }}
            />
          </div>
        </div>

        <div
          className={`p-1.5 rounded text-[10px] border-l-4 ${
            fuelDecision.status === "CONTINUE"
              ? "bg-emerald-500/20 border-emerald-400"
              : fuelDecision.status === "RETURN"
                ? "bg-yellow-500/20 border-yellow-500"
                : "bg-red-500/20 border-red-500"
          }`}
        >
          <strong className="block mb-0.5">{fuelDecision.status}</strong>
          <p className="text-[9px]">{fuelDecision.recommendedAction}</p>
        </div>

        <div className="mt-1.5 text-[9px] text-gray-400 space-y-0.5">
          <div className="flex items-center gap-1">
            {fuelDecision.canReachDestination ? (
              <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
            ) : (
              <XCircle className="w-2.5 h-2.5 text-red-400" />
            )}
            <span>Reach destination</span>
          </div>
          <div className="flex items-center gap-1">
            {fuelDecision.canReturnToStart ? (
              <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
            ) : (
              <XCircle className="w-2.5 h-2.5 text-red-400" />
            )}
            <span>Return to base</span>
          </div>
        </div>
      </div>

      <h3 className="text-emerald-400 mb-2 text-[10px] font-bold flex items-center gap-1.5">
        <BarChart3 className="w-3 h-3" />
        Current Metrics
      </h3>
      <div className="bg-black/30 p-2 mb-2 rounded-lg text-[10px] space-y-0.5">
        <div>
          Position:{" "}
          <span className="text-emerald-400 font-bold">
            {ship.lat.toFixed(4)}, {ship.lon.toFixed(4)}
          </span>
        </div>
        <div>
          Speed:{" "}
          <span className="text-emerald-400 font-bold">{ship.speed} knots</span>
        </div>
        <div>
          Heading:{" "}
          <span className="text-emerald-400 font-bold">
            {Math.round(ship.heading)}°
          </span>
        </div>
        <div>
          Distance:{" "}
          <span className="text-emerald-400 font-bold">
            {ship.distanceToDestination.toFixed(1)} km
          </span>
        </div>
        <div>
          ETA:{" "}
          <span className="text-emerald-400 font-bold">{ship.eta} min</span>
        </div>
      </div>

      <h3 className="text-emerald-400 mb-2 text-[10px] font-bold flex items-center gap-1.5">
        <Radar className="w-3 h-3" />
        Drone Detections
      </h3>
      <div className="bg-black/30 p-2 rounded-lg max-h-52 overflow-y-auto">
        {mlDetections.length === 0 ? (
          <div className="p-1.5 bg-emerald-500/20 border-l-4 border-emerald-400 rounded text-[10px]">
            No objects in detection range
          </div>
        ) : (
          mlDetections.map((detection, i) => (
            <div
              key={i}
              className={`mb-1.5 p-1.5 rounded text-[10px] border-l-4 ${
                detection.isThreat
                  ? "bg-red-500/20 border-red-500"
                  : "bg-emerald-500/20 border-emerald-400"
              }`}
            >
              <div className="flex items-center gap-1">
                <strong>{detection.name}</strong>
                {detection.correct ? (
                  <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
                ) : (
                  <XCircle className="w-2.5 h-2.5 text-red-400" />
                )}
              </div>
              <span className="text-[9px]">
                {detection.actualThreat ? "ACTUAL THREAT" : "BENIGN"} |{" "}
                {detection.mlResult} ({(detection.confidence * 100).toFixed(1)}
                %)
                <br />
                Distance: {Math.floor(detection.distance * 10)}m
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
