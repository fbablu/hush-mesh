"use client";

import { ShipIcon } from "lucide-react";
import type { ShipState } from "@/types/maritime";

interface ShipProps {
  ship: ShipState;
}

export function Ship({ ship }: ShipProps) {
  return (
    <div
      className="absolute w-6 h-6 bg-emerald-400 rounded-full shadow-[0_0_20px_#00ff88] z-20 transition-all duration-300 ease-out flex items-center justify-center"
      style={{
        left: `${ship.x}px`,
        top: `${ship.y}px`,
        transform: `translate(-50%, -50%) rotate(${ship.heading}deg)`,
      }}
    >
      <ShipIcon className="w-3 h-3 text-emerald-950" />
    </div>
  );
}
