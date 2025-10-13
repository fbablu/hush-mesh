"use client";
import { MapCanvas } from "./map-canvas";
import { StatusSidebar } from "./status-sidebar";
import { RoutesSidebar } from "./routes-sidebar";
import { ControlPanel } from "./control-panel";
import { EmergencyBeaconAlert } from "./emergency-beacon-alert";
import { DestinationReachedAlert } from "./destination-reached-alert";
import { useMaritimeSimulation } from "@/hooks/use-maritime-simulation";
import { exportDroneObservations } from "@/lib/maritime/data-export";
import { useState, useEffect } from "react";

export function MaritimeDashboard() {
  const simulation = useMaritimeSimulation();
  const [showBeaconAlert, setShowBeaconAlert] = useState(false);
  const [showDestinationAlert, setShowDestinationAlert] = useState(false);

  useEffect(() => {
    if (simulation.emergencyBeacon && !showBeaconAlert) {
      setShowBeaconAlert(true);
    }
  }, [simulation.emergencyBeacon, showBeaconAlert]);

  useEffect(() => {
    if (simulation.destinationReached && !showDestinationAlert) {
      setShowDestinationAlert(true);
    }
  }, [simulation.destinationReached, showDestinationAlert]);

  const handleBeaconUpload = () => {
    exportDroneObservations(simulation.drones, "EMERGENCY_BEACON");
  };

  const handleDestinationUpload = () => {
    exportDroneObservations(simulation.drones, "MISSION_SUCCESS");
  };

  return (
    <div className="grid grid-cols-[240px_1fr_280px] h-screen bg-[#0a0e1a] text-white overflow-hidden">
      <StatusSidebar simulation={simulation} />

      <MapCanvas simulation={simulation} />

      <RoutesSidebar simulation={simulation} />

      <ControlPanel simulation={simulation} />

      {showBeaconAlert && (
        <EmergencyBeaconAlert
          onClose={() => setShowBeaconAlert(false)}
          onUpload={handleBeaconUpload}
        />
      )}
      {showDestinationAlert && (
        <DestinationReachedAlert
          onClose={() => setShowDestinationAlert(false)}
          onUpload={handleDestinationUpload}
        />
      )}
    </div>
  );
}
