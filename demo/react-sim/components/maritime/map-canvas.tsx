"use client";

import type React from "react";

import { useRef, useEffect, useState } from "react";
import { Ship } from "./ship";
import { DroneSwarm } from "./drone-swarm";
import { ThreatObject } from "./threat-object";
import { RouteDisplay } from "./route-display";
import { FogOfWar } from "./fog-of-war";
import { Button } from "@/components/ui/button";
import { Home, MapPin } from "lucide-react";
import type { MaritimeSimulation } from "@/types/maritime";

interface MapCanvasProps {
  simulation: MaritimeSimulation;
}

export function MapCanvas({ simulation }: MapCanvasProps) {
  const {
    ship,
    drones,
    threats,
    routes,
    selectedRoute,
    scenario,
    lastDroneFailure,
  } = simulation;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.save();
      ctx.translate(pan.x, pan.y);
      ctx.scale(zoom, zoom);

      const gridSize = zoom > 1.5 ? 20 : zoom > 0.8 ? 40 : 60;

      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.lineWidth = 1 / zoom;

      for (let x = 0; x < canvas.width / zoom; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height / zoom);
        ctx.stroke();
      }

      for (let y = 0; y < canvas.height / zoom; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width / zoom, y);
        ctx.stroke();
      }

      ctx.restore();
    };

    draw();
  }, [zoom, pan]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((prev) => Math.max(0.5, Math.min(3, prev * delta)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleHomeClick = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const nauticalMilesPerPixel = 0.1 / zoom;
  const scaleBarPixels = 100;
  const scaleBarNM = (scaleBarPixels * nauticalMilesPerPixel).toFixed(1);

  return (
    <div
      className="relative bg-gradient-to-br from-[#16213e] to-[#0f3460] overflow-hidden cursor-move"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <div className="absolute top-4 left-4 z-20">
        <Button
          onClick={handleHomeClick}
          size="sm"
          variant="ghost"
          className="bg-black/60 hover:bg-black/80 backdrop-blur-sm"
        >
          <Home className="w-4 h-4" />
        </Button>
      </div>

      <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-md z-20">
        <div className="flex items-center gap-2">
          <div
            className="h-0.5 bg-white/60"
            style={{ width: `${scaleBarPixels}px` }}
          />
          <span className="text-xs font-mono text-white/80">
            {scaleBarNM} NM
          </span>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        className="absolute inset-0 pointer-events-none"
      />

      <div
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: "0 0",
          width: "800px",
          height: "600px",
        }}
      >
        <FogOfWar drones={drones} ship={ship} width={800} height={600} />

        {/* Route Paths */}
        <RouteDisplay
          routes={routes}
          selectedRoute={selectedRoute}
          ship={ship}
        />

        {/* Threats */}
        {threats.map((threat, i) => (
          <ThreatObject key={i} threat={threat} index={i} />
        ))}

        {/* Drone Swarm */}
        <DroneSwarm
          drones={drones}
          ship={ship}
          lastDroneFailure={lastDroneFailure}
        />

        {/* Main Ship */}
        <Ship ship={ship} />

        {/* Destination - Replaced emoji with MapPin icon */}
        <div
          className="absolute w-4 h-4 bg-red-500 rounded-full shadow-[0_0_15px_#ff6b6b] z-10 flex items-center justify-center text-xs"
          style={{
            left: `${ship.destination.x}px`,
            top: `${ship.destination.y}px`,
            transform: "translate(-50%, -50%)",
          }}
        >
          <MapPin className="w-3 h-3 text-white" />
        </div>
      </div>
    </div>
  );
}
