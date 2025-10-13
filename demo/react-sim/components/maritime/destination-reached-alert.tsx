"use client";

import { CheckCircle2, MapPin, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";

interface DestinationReachedAlertProps {
  onClose: () => void;
  onUpload?: () => void;
}

export function DestinationReachedAlert({
  onClose,
  onUpload,
}: DestinationReachedAlertProps) {
  const handleUpload = () => {
    if (onUpload) onUpload();
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 animate-in fade-in">
      <div className="bg-black border-2 border-green-500 rounded-sm shadow-2xl shadow-green-500/50 w-[600px] font-mono">
        {/* Terminal header bar */}
        <div className="bg-green-950 border-b-2 border-green-500 px-4 py-2 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-green-400" />
          <span className="text-green-400 text-sm font-bold">
            NAVIGATION_COMPLETE.SYS
          </span>
          <div className="ml-auto flex gap-1">
            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
          </div>
        </div>

        {/* Terminal content */}
        <div className="p-6 space-y-4">
          {/* Large logo/icon display */}
          <div className="flex justify-center mb-6">
            <div className="relative">
              <MapPin className="w-32 h-32 text-green-500" strokeWidth={1.5} />
              <CheckCircle2 className="w-16 h-16 text-green-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
            </div>
          </div>

          {/* Terminal output */}
          <div className="space-y-1 text-green-400">
            <p className="text-xl font-bold text-center mb-4">
              ╔═══════════════════════════════════════╗
            </p>
            <p className="text-2xl font-bold text-center text-green-500">
              DESTINATION REACHED
            </p>
            <p className="text-xl font-bold text-center mb-4">
              ╚═══════════════════════════════════════╝
            </p>
          </div>

          <div className="bg-green-950/30 border border-green-900 p-4 space-y-2 text-sm">
            <p className="text-green-400">
              &gt; DESTINATION:{" "}
              <span className="text-green-500 font-bold">POINT B</span>
            </p>
            <p className="text-green-400">
              &gt; ARRIVAL STATUS:{" "}
              <span className="text-green-500 font-bold">CONFIRMED</span>
            </p>
            <p className="text-green-400">
              &gt; DRONE MESH:{" "}
              <span className="text-green-500 font-bold">ALL ACTIVE</span>
            </p>
            <p className="text-green-400">
              &gt; NAVIGATION:{" "}
              <span className="text-green-500 font-bold">COMPLETE</span>
            </p>
            <p className="text-yellow-400 mt-3">
              &gt; MISSION STATUS:{" "}
              <span className="text-green-400 font-bold">SUCCESS</span>
            </p>
          </div>

          <Button
            onClick={handleUpload}
            className="w-full bg-green-600 hover:bg-green-700 font-mono text-sm border border-green-400"
          >
            [ upload observed data to cloud ]
          </Button>
        </div>
      </div>
    </div>
  );
}
