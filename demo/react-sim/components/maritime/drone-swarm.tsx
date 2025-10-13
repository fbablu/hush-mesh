"use client";

import type { DroneState, ShipState } from "@/types/maritime";

interface DroneSwarmProps {
  drones: DroneState[];
  ship: ShipState;
  lastDroneFailure: number | null;
}

export function DroneSwarm({
  drones,
  ship,
  lastDroneFailure,
}: DroneSwarmProps) {
  return (
    <>
      {/* Drone mesh connections */}
      <svg
        className="absolute inset-0 pointer-events-none z-15"
        style={{ width: "100%", height: "100%" }}
      >
        {/* Tether lines from ship to each active drone */}
        {drones
          .filter((d) => d.active)
          .map((drone) => {
            const x1 = Number.isFinite(ship.x) ? ship.x : 0;
            const y1 = Number.isFinite(ship.y) ? ship.y : 0;
            const x2 = Number.isFinite(drone.x) ? drone.x : x1;
            const y2 = Number.isFinite(drone.y) ? drone.y : y1;

            return (
              <line
                key={`tether-${drone.id}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={drone.detectedThreat ? "#ff4444" : "#00ff88"}
                strokeWidth="2"
                strokeDasharray="4 2"
                opacity="0.6"
                className="transition-all duration-500 ease-out"
              />
            );
          })}

        {drones
          .filter((d) => d.active)
          .map((drone) => {
            const droneX = Number.isFinite(drone.x) ? drone.x : ship.x;
            const droneY = Number.isFinite(drone.y) ? drone.y : ship.y;

            const startAngle =
              ((drone.coverageAngle - drone.coverageRange / 2) * Math.PI) / 180;
            const endAngle =
              ((drone.coverageAngle + drone.coverageRange / 2) * Math.PI) / 180;
            const radius = 120;

            const x1 = droneX + Math.cos(startAngle) * radius;
            const y1 = droneY + Math.sin(startAngle) * radius;
            const x2 = droneX + Math.cos(endAngle) * radius;
            const y2 = droneY + Math.sin(endAngle) * radius;

            const largeArcFlag = drone.coverageRange > 180 ? 1 : 0;

            return (
              <path
                key={`coverage-${drone.id}`}
                d={`M ${droneX} ${droneY} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2} Z`}
                fill={drone.detectedThreat ? "#ff444420" : "#4ecdc420"}
                stroke={drone.detectedThreat ? "#ff4444" : "#4ecdc4"}
                strokeWidth="0.5"
                opacity="0.3"
                className="transition-all duration-500 ease-out"
              />
            );
          })}

        {/* Mesh connections between adjacent active drones */}
        {drones
          .filter((d) => d.active)
          .map((drone1, i, activeDrones) => {
            const nextDrone = activeDrones[(i + 1) % activeDrones.length];

            const x1 = Number.isFinite(drone1.x) ? drone1.x : ship.x;
            const y1 = Number.isFinite(drone1.y) ? drone1.y : ship.y;
            const x2 = Number.isFinite(nextDrone.x) ? nextDrone.x : ship.x;
            const y2 = Number.isFinite(nextDrone.y) ? nextDrone.y : ship.y;

            return (
              <line
                key={`mesh-${drone1.id}-${nextDrone.id}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#4ecdc4"
                strokeWidth="1"
                opacity="0.4"
                className="transition-all duration-500 ease-out"
              />
            );
          })}
      </svg>

      {/* Individual drones */}
      {drones.map((drone) => {
        const droneX = Number.isFinite(drone.x) ? drone.x : ship.x;
        const droneY = Number.isFinite(drone.y) ? drone.y : ship.y;
        const isRecentlyFailed = lastDroneFailure === drone.id;

        return (
          <div
            key={drone.id}
            className={`absolute w-4 h-4 rounded-full z-15 ${
              drone.active
                ? "opacity-100 transition-all duration-500 ease-out"
                : "opacity-20"
            }`}
            style={{
              left: `${droneX}px`,
              top: `${droneY}px`,
              transform: "translate(-50%, -50%)",
              backgroundColor: !drone.active
                ? "#666"
                : drone.detectedThreat
                  ? "#ff4444"
                  : "#4ecdc4",
              boxShadow: drone.active
                ? `0 0 ${drone.detectedThreat ? "12" : "8"}px ${drone.detectedThreat ? "#ff4444" : "#4ecdc4"}`
                : "none",
              animation:
                drone.detectedThreat && drone.active
                  ? "pulse 1s infinite"
                  : "none",
            }}
            title={`Drone ${drone.id + 1}${!drone.active ? " - OFFLINE" : drone.detectedThreat ? " - THREAT DETECTED" : ""}`}
          >
            {isRecentlyFailed && (
              <div className="absolute inset-0 rounded-full bg-red-500 animate-ping" />
            )}

            <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[9px] whitespace-nowrap opacity-70 font-mono">
              D{drone.id + 1}
              {!drone.active && " ✗"}
            </div>
          </div>
        );
      })}
    </>
  );
}
