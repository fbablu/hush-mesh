"use client";

import type { RouteData, ShipState } from "@/types/maritime";

interface RouteDisplayProps {
  routes: Record<string, RouteData>;
  selectedRoute: string;
  ship: ShipState;
}

// Helper function to create smooth curves through waypoints using Catmull-Rom splines
function createSmoothPath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return "";
  if (points.length === 2)
    return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;

  let path = `M ${points[0].x} ${points[0].y}`;

  // Use quadratic bezier curves for smooth transitions
  for (let i = 0; i < points.length - 1; i++) {
    const current = points[i];
    const next = points[i + 1];

    if (i === points.length - 2) {
      // Last segment - direct line
      path += ` L ${next.x} ${next.y}`;
    } else {
      // Create smooth curve using control point
      const nextNext = points[i + 2];
      const controlX = next.x;
      const controlY = next.y;
      const endX = (next.x + nextNext.x) / 2;
      const endY = (next.y + nextNext.y) / 2;

      path += ` Q ${controlX} ${controlY}, ${endX} ${endY}`;
    }
  }

  return path;
}

export function RouteDisplay({
  routes,
  selectedRoute,
  ship,
}: RouteDisplayProps) {
  const renderRoute = (routeType: string, route: RouteData) => {
    if (!route?.waypoints?.length) return null;

    const isSelected = routeType === selectedRoute;
    const colors = {
      optimal: { stroke: "#00ff88", shadow: "rgba(0,255,136,0.8)" },
      fastest: { stroke: "#ff6b6b", shadow: "rgba(255,107,107,0.6)" },
      safest: { stroke: "#4ecdc4", shadow: "rgba(78,205,196,0.6)" },
      return: { stroke: "#ffa500", shadow: "rgba(255,165,0,0.6)" },
    };

    const color = colors[routeType as keyof typeof colors] || colors.optimal;

    // Create smooth path through ship position and waypoints
    const allPoints = [{ x: ship.x, y: ship.y }, ...route.waypoints];
    const smoothPath = createSmoothPath(allPoints);

    return (
      <svg
        key={routeType}
        className="absolute inset-0 pointer-events-none"
        style={{ width: "100%", height: "100%", zIndex: isSelected ? 6 : 4 }}
      >
        <path
          d={smoothPath}
          stroke={color.stroke}
          strokeWidth={isSelected ? 3 : 2}
          fill="none"
          opacity={isSelected ? 1 : 0.3}
          filter={isSelected ? `drop-shadow(0 0 8px ${color.shadow})` : "none"}
          strokeLinecap="round"
          strokeLinejoin="round"
          className="transition-all duration-300"
        />
      </svg>
    );
  };

  return (
    <>
      {Object.entries(routes).map(([type, route]) => renderRoute(type, route))}
    </>
  );
}
