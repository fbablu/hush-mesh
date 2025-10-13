"use client";

import { AlertTriangle, Radio, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";

interface EmergencyBeaconAlertProps {
  onClose: () => void;
  onUpload?: () => void;
}

export function EmergencyBeaconAlert({
  onClose,
  onUpload,
}: EmergencyBeaconAlertProps) {
  const handleUpload = () => {
    if (onUpload) onUpload();
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 animate-in fade-in">
      <div className="bg-black border-2 border-red-500 rounded-sm shadow-2xl shadow-red-500/50 w-[600px] font-mono">
        {/* Terminal header bar */}
        <div className="bg-red-950 border-b-2 border-red-500 px-4 py-2 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-red-400" />
          <span className="text-red-400 text-sm font-bold">
            EMERGENCY_BEACON.SYS
          </span>
          <div className="ml-auto flex gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
          </div>
        </div>

        {/* Terminal content */}
        <div className="p-6 space-y-4">
          {/* Large logo/icon display */}
          <div className="flex justify-center mb-6">
            <div className="relative">
              <Radio
                className="w-32 h-32 text-red-500 animate-pulse"
                strokeWidth={1.5}
              />
              <AlertTriangle className="w-16 h-16 text-yellow-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
            </div>
          </div>

          {/* Terminal output */}
          <div className="space-y-1 text-red-400">
            <p className="text-xl font-bold text-center mb-4">
              ╔═══════════════════════════════════════╗
            </p>
            <p className="text-2xl font-bold text-center text-red-500 animate-pulse">
              TERMINAL ALERT
            </p>
            <p className="text-xl font-bold text-center mb-4">
              ╚═══════════════════════════════════════╝
            </p>
          </div>

          <div className="bg-red-950/30 border border-red-900 p-4 space-y-2 text-sm">
            <p className="text-red-400">
              &gt; EMERGENCY BEACON:{" "}
              <span className="text-red-500 font-bold">DEPLOYED</span>
            </p>
            <p className="text-red-400">
              &gt; HOMEBASE STATUS:{" "}
              <span className="text-yellow-400 font-bold">NOTIFIED</span>
            </p>
            <p className="text-red-400">
              &gt; FUEL LEVEL:{" "}
              <span className="text-red-500 font-bold animate-pulse">
                CRITICAL
              </span>
            </p>
            <p className="text-red-400">
              &gt; NAVIGATION:{" "}
              <span className="text-red-500 font-bold">IMPOSSIBLE</span>
            </p>
            <p className="text-yellow-400 mt-3">
              &gt; RESCUE OPS:{" "}
              <span className="text-green-400 font-bold">INITIATED</span>
            </p>
          </div>

          <Button
            onClick={handleUpload}
            className="w-full bg-red-600 hover:bg-red-700 font-mono text-sm border border-red-400"
          >
            [ upload observed data to cloud ]
          </Button>
        </div>
      </div>
    </div>
  );
}
