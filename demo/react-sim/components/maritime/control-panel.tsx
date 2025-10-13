"use client";

import { Button } from "@/components/ui/button";
import type { MaritimeSimulation } from "@/types/maritime";
import { Play, Pause, Square, RotateCcw, Brain, Zap } from "lucide-react";

interface ControlPanelProps {
  simulation: MaritimeSimulation;
}

export function ControlPanel({ simulation }: ControlPanelProps) {
  const {
    running,
    speed,
    start,
    pause,
    stop,
    reset,
    recalculate,
    setSpeed,
    triggerDroneFailure,
  } = simulation;

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 bg-black/80 px-4 py-2.5 rounded-xl flex items-center gap-2 z-30">
      <Button
        onClick={start}
        disabled={running}
        size="sm"
        className="bg-emerald-500 hover:bg-emerald-600 text-black"
      >
        <Play className="w-3 h-3 mr-1" />
        Start
      </Button>
      <Button onClick={pause} disabled={!running} size="sm" variant="secondary">
        <Pause className="w-3 h-3 mr-1" />
        Pause
      </Button>
      <Button
        onClick={stop}
        disabled={!running}
        size="sm"
        variant="destructive"
      >
        <Square className="w-3 h-3 mr-1" />
        Stop
      </Button>
      <Button onClick={reset} size="sm" variant="secondary">
        <RotateCcw className="w-3 h-3 mr-1" />
        Reset
      </Button>
      <Button onClick={recalculate} size="sm" variant="secondary">
        <Brain className="w-3 h-3 mr-1" />
        Recalculate
      </Button>

      <Button
        onClick={triggerDroneFailure}
        size="sm"
        variant="outline"
        className="border-yellow-500/50 text-yellow-400 hover:bg-yellow-500/20 bg-transparent"
        title="Manually trigger drone failure to demonstrate self-healing"
      >
        <Zap className="w-3 h-3 mr-1" />
        Fail Drone
      </Button>

      <div className="h-5 w-px bg-white/20 mx-1" />

      <span className="text-[10px] text-gray-400">Speed:</span>
      {[0.5, 1, 2, 3].map((s) => (
        <Button
          key={s}
          onClick={() => setSpeed(s)}
          variant={speed === s ? "default" : "secondary"}
          size="sm"
          className="w-10 h-7 text-xs"
        >
          {s}x
        </Button>
      ))}
    </div>
  );
}
