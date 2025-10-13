"use client";

import {
  Ship,
  Bomb,
  Telescope,
  Trash2,
  Fish,
  Package,
  Microscope,
  AlertTriangle,
} from "lucide-react";
import type { ThreatState } from "@/types/maritime";

interface ThreatObjectProps {
  threat: ThreatState;
  index: number;
}

export function ThreatObject({ threat, index }: ThreatObjectProps) {
  const getIcon = () => {
    switch (threat.type) {
      case "small_fast_craft":
        return <Ship className="w-3 h-3" />;
      case "floating_mine_like_object":
        return <Bomb className="w-3 h-3" />;
      case "submarine_periscope":
        return <Telescope className="w-3 h-3" />;
      case "debris_field":
        return <Trash2 className="w-3 h-3" />;
      case "fishing_vessel":
        return <Fish className="w-3 h-3" />;
      case "cargo_ship":
        return <Package className="w-3 h-3" />;
      case "research_vessel":
        return <Microscope className="w-3 h-3" />;
      default:
        return <AlertTriangle className="w-3 h-3" />;
    }
  };

  const getClassName = () => {
    if (!threat.detected) return "hidden";
    if (threat.mlClassification?.threat_detected) return "object threat";
    if (threat.mlClassification) return "object benign";
    return "object unknown";
  };

  return (
    <>
      {/* Threat zone */}
      {threat.detected && threat.mlClassification?.threat_detected && (
        <div
          className="absolute border-2 border-dashed border-red-500 rounded-full bg-red-500/10 z-5 transition-all duration-300"
          style={{
            left: `${threat.x - threat.radius}px`,
            top: `${threat.y - threat.radius}px`,
            width: `${threat.radius * 2}px`,
            height: `${threat.radius * 2}px`,
          }}
        />
      )}

      {/* Threat object */}
      <div
        className={`absolute w-4 h-4 rounded-full z-10 transition-all duration-300 ease-out flex items-center justify-center text-sm ${
          threat.mlClassification?.threat_detected
            ? "bg-red-500 shadow-[0_0_15px_#ff4444] text-white"
            : threat.mlClassification
              ? "bg-cyan-500 shadow-[0_0_15px_#4ecdc4] text-white"
              : "bg-yellow-500 shadow-[0_0_15px_#ffaa00] text-white"
        } ${!threat.detected ? "hidden" : ""}`}
        style={{
          left: `${threat.x}px`,
          top: `${threat.y}px`,
          transform: "translate(-50%, -50%)",
        }}
        title={threat.name}
      >
        {getIcon()}
      </div>
    </>
  );
}
