"use client";

import type { DroneState, ShipState } from "@/types/maritime";

interface FogOfWarProps {
  drones: DroneState[];
  ship: ShipState;
  width: number;
  height: number;
}

export function FogOfWar({ drones, ship, width, height }: FogOfWarProps) {
  // Create SVG mask that reveals areas within drone coverage
  const maskId = "fog-mask";

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      style={{ width: "100%", height: "100%", zIndex: 3 }}
    >
      <defs>
        <mask id={maskId}>
          {/* Start with everything hidden (black) */}
          <rect x="0" y="0" width={width} height={height} fill="black" />

          {/* Reveal areas around active drones (white) */}
          {drones
            .filter((drone) => drone.active)
            .map((drone) => (
              <circle
                key={`mask-${drone.id}`}
                cx={drone.x}
                cy={drone.y}
                r={120}
                fill="white"
                opacity={0.9}
              />
            ))}

          {/* Reveal area around ship */}
          <circle cx={ship.x} cy={ship.y} r={60} fill="white" opacity={0.9} />
        </mask>

        {/* Radial gradient for fog effect */}
        <radialGradient id="fog-gradient">
          <stop offset="0%" stopColor="rgba(10, 25, 47, 0.85)" />
          <stop offset="100%" stopColor="rgba(10, 25, 47, 0.95)" />
        </radialGradient>
      </defs>

      {/* Apply fog overlay with mask */}
      <rect
        x="0"
        y="0"
        width={width}
        height={height}
        fill="url(#fog-gradient)"
        mask={`url(#${maskId})`}
      />

      {/* Add subtle scan line effect for explored areas */}
      {drones
        .filter((drone) => drone.active)
        .map((drone) => (
          <circle
            key={`scan-${drone.id}`}
            cx={drone.x}
            cy={drone.y}
            r={120}
            fill="none"
            stroke="rgba(0, 255, 136, 0.1)"
            strokeWidth={2}
            className="animate-pulse"
          />
        ))}
    </svg>
  );
}
