"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import type {
  ShipState,
  DroneState,
  ThreatState,
  RouteData,
  FuelDecision,
  MaritimeSimulation,
} from "@/types/maritime";

import {
  DRONE_COUNT,
  DOCKING_ZONE_RADIUS,
  DRONE_FAILURE_CHECK_INTERVAL,
  DRONE_FAILURE_CHANCE,
  MIN_ACTIVE_DRONES,
  ROUTE_RECALC_THRESHOLD,
  ROUTE_RECALC_INTERVAL,
  START_POSITION,
  DESTINATION_POSITION,
} from "@/lib/maritime/constants";
import { pixelToLatLon } from "@/lib/maritime/coordinates";
import {
  initializeDrones,
  reorganizeDrones,
} from "@/lib/maritime/drone-manager";
import { generateRandomThreats } from "@/lib/maritime/threat-generator";
import { updateThreats } from "@/lib/maritime/threat-manager";
import { calculateRoutes } from "@/lib/maritime/route-calculator";
import { calculateFuelDecision } from "@/lib/maritime/fuel-calculator";
import { exportDroneObservations } from "@/lib/maritime/data-export";

export function useMaritimeSimulation(): MaritimeSimulation {
  const initialLatLon = pixelToLatLon(START_POSITION.x, START_POSITION.y);

  // ============================================
  // STATE MANAGEMENT
  // ============================================

  const [ship, setShip] = useState<ShipState>({
    x: START_POSITION.x,
    y: START_POSITION.y,
    lat: initialLatLon.lat,
    lon: initialLatLon.lon,
    heading: 90,
    speed: 12,
    destination: DESTINATION_POSITION,
    distanceToDestination: 0,
    eta: 0,
    fuel: 100,
    fuelConsumptionRate: 0.015,
  });

  const [drones, setDrones] = useState<DroneState[]>([]);
  const [threats, setThreats] = useState<ThreatState[]>([]);
  const [routes, setRoutes] = useState<Record<string, RouteData>>({});
  const [selectedRoute, setSelectedRoute] = useState("optimal");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [emergencyBeacon, setEmergencyBeacon] = useState(false);
  const [destinationReached, setDestinationReached] = useState(false);
  const [missionLog, setMissionLog] = useState<string[]>([
    "[00:00] Drone mesh navigation initialized",
    "[00:00] 6 drones deployed in protective formation",
    "[00:00] ML threat detection active",
  ]);
  const [fuelDecision, setFuelDecision] = useState<FuelDecision>({
    status: "CONTINUE",
    message: "",
    canReachDestination: true,
    canReturnToStart: true,
    recommendedAction: "",
  });
  const [lastDroneFailure, setLastDroneFailure] = useState<number | null>(null);

  const animationRef = useRef<number | undefined>(undefined);
  const stepRef = useRef(0);
  const lastRecalcPositionRef = useRef({ x: ship.x, y: ship.y });
  const droneFailureTimerRef = useRef<number>(0);
  const fuelCheckIntervalRef = useRef<number>(0); // Add fuel check interval to reduce calculations

  // ============================================
  // INITIALIZATION
  // ============================================

  useEffect(() => {
    setThreats(generateRandomThreats());
  }, []);

  useEffect(() => {
    setDrones(
      initializeDrones(START_POSITION.x, START_POSITION.y, DRONE_COUNT),
    );
  }, []);

  // ============================================
  // UTILITY FUNCTIONS
  // ============================================

  const addLog = useCallback((message: string) => {
    const time = new Date().toLocaleTimeString().slice(0, 5);
    setMissionLog((prev) => [`[${time}] ${message}`, ...prev].slice(0, 15));
  }, []);

  const recalculateRoutes = useCallback(() => {
    const newRoutes = calculateRoutes(ship, threats, START_POSITION);
    setRoutes(newRoutes);
  }, [ship, threats]);

  // ============================================
  // DRONE MANAGEMENT
  // ============================================

  const checkDroneFailures = useCallback(() => {
    droneFailureTimerRef.current++;

    if (droneFailureTimerRef.current % DRONE_FAILURE_CHECK_INTERVAL === 0) {
      const activeDrones = drones.filter((d) => d.active);

      if (activeDrones.length > MIN_ACTIVE_DRONES) {
        const failureChance = Math.random();

        if (failureChance < DRONE_FAILURE_CHANCE) {
          const numToFail =
            failureChance < 0.15 ? 1 : failureChance < 0.22 ? 2 : 3;
          const dronesCanFail = Math.min(
            numToFail,
            activeDrones.length - MIN_ACTIVE_DRONES,
          );

          const failedDroneIds: number[] = [];

          for (let i = 0; i < dronesCanFail; i++) {
            const remainingActive = activeDrones.filter(
              (d) => !failedDroneIds.includes(d.id),
            );
            if (remainingActive.length <= MIN_ACTIVE_DRONES) break;

            const randomDrone =
              remainingActive[
                Math.floor(Math.random() * remainingActive.length)
              ];
            failedDroneIds.push(randomDrone.id);
          }

          if (failedDroneIds.length > 0) {
            setDrones((prevDrones) => {
              const updatedDrones = prevDrones.map((d) =>
                failedDroneIds.includes(d.id)
                  ? { ...d, active: false, detectedThreat: false }
                  : d,
              );

              const reorganized = reorganizeDrones(updatedDrones, ship);

              if (failedDroneIds.length === 1) {
                addLog(
                  `⚠️ DRONE ${failedDroneIds[0] + 1} FAILURE - Self-healing initiated`,
                );
              } else {
                addLog(
                  `⚠️ CRITICAL: ${failedDroneIds.length} DRONES FAILED - Emergency reorganization`,
                );
              }

              addLog(
                `✓ Mesh reconfigured: ${reorganized.filter((d) => d.active).length} drones active`,
              );

              setLastDroneFailure(failedDroneIds[0]);
              setTimeout(() => setLastDroneFailure(null), 2000);

              return reorganized;
            });
          }
        }
      }
    }
  }, [drones, addLog, ship]);

  const updateDronePositions = useCallback(() => {
    setDrones((prevDrones) => {
      return prevDrones.map((drone) => {
        if (!drone.active) return drone;

        const newX = ship.x + drone.relativeX;
        const newY = ship.y + drone.relativeY;
        const droneLatLon = pixelToLatLon(newX, newY);

        const shouldRecordObservation = stepRef.current % 10 === 0;

        let nearestThreat: ThreatState | null = null;
        let minDistance = Number.POSITIVE_INFINITY;

        // Only check for threats if we're recording this frame
        if (shouldRecordObservation) {
          threats.forEach((threat) => {
            if (!threat.detected || !threat.mlClassification?.threat_detected)
              return;

            const dx = threat.x - newX;
            const dy = threat.y - newY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const threatAngle =
              ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360;

            const angleDiff = Math.abs(threatAngle - drone.coverageAngle);
            const normalizedDiff = Math.min(angleDiff, 360 - angleDiff);

            if (normalizedDiff <= drone.coverageRange / 2 && distance < 150) {
              if (distance < minDistance) {
                minDistance = distance;
                nearestThreat = threat;
              }
            }
          });
        }

        if (nearestThreat !== null && shouldRecordObservation) {
          const threat: ThreatState = nearestThreat;
          const threatLatLon = pixelToLatLon(threat.x, threat.y);
          const threatAngle = Math.atan2(threat.y - ship.y, threat.x - ship.x);

          const lastObs = drone.observations[drone.observations.length - 1];
          const isDuplicate =
            lastObs &&
            lastObs.objectName === threat.name &&
            Date.now() - lastObs.timestamp < 2000;

          const newObservations = isDuplicate
            ? drone.observations
            : [
                ...drone.observations,
                {
                  timestamp: Date.now(),
                  droneId: drone.id,
                  droneLat: droneLatLon.lat,
                  droneLon: droneLatLon.lon,
                  objectType: threat.type,
                  objectName: threat.name,
                  objectLat: threatLatLon.lat,
                  objectLon: threatLatLon.lon,
                  distance: minDistance * 0.01,
                  isThreat: threat.isThreat,
                  confidence: threat.mlClassification?.confidence || 0,
                },
              ];

          return {
            ...drone,
            x: newX,
            y: newY,
            lat: droneLatLon.lat,
            lon: droneLatLon.lon,
            detectedThreat: true,
            threatDirection: threatAngle,
            observations: newObservations,
          };
        }

        return {
          ...drone,
          x: newX,
          y: newY,
          lat: droneLatLon.lat,
          lon: droneLatLon.lon,
          detectedThreat: shouldRecordObservation
            ? false
            : drone.detectedThreat,
          threatDirection: shouldRecordObservation
            ? null
            : drone.threatDirection,
        };
      });
    });
  }, [threats, ship]);

  const triggerDroneFailure = useCallback(() => {
    const activeDrones = drones.filter((d) => d.active);

    if (activeDrones.length <= MIN_ACTIVE_DRONES) {
      addLog(
        `⚠️ Cannot fail more drones - minimum ${MIN_ACTIVE_DRONES} required`,
      );
      return;
    }

    const numToFail = Math.random() < 0.5 ? 1 : 2;
    const dronesCanFail = Math.min(
      numToFail,
      activeDrones.length - MIN_ACTIVE_DRONES,
    );
    const failedDroneIds: number[] = [];

    for (let i = 0; i < dronesCanFail; i++) {
      const remainingActive = activeDrones.filter(
        (d) => !failedDroneIds.includes(d.id),
      );
      if (remainingActive.length <= MIN_ACTIVE_DRONES) break;

      const randomDrone =
        remainingActive[Math.floor(Math.random() * remainingActive.length)];
      failedDroneIds.push(randomDrone.id);
    }

    setDrones((prevDrones) => {
      const updatedDrones = prevDrones.map((d) =>
        failedDroneIds.includes(d.id)
          ? { ...d, active: false, detectedThreat: false }
          : d,
      );

      const reorganized = reorganizeDrones(updatedDrones, ship);

      addLog(
        `🔧 MANUAL FAILURE: Drone${failedDroneIds.length > 1 ? "s" : ""} ${failedDroneIds.map((id) => id + 1).join(", ")} disabled`,
      );
      addLog(
        `✓ Self-healing complete: ${reorganized.filter((d) => d.active).length}/${DRONE_COUNT} drones operational`,
      );

      setLastDroneFailure(failedDroneIds[0]);
      setTimeout(() => setLastDroneFailure(null), 2000);

      return reorganized;
    });
  }, [drones, addLog, ship]);

  // ============================================
  // SHIP MOVEMENT
  // ============================================

  const moveShip = useCallback(() => {
    if (!routes[selectedRoute]?.waypoints?.length) return;

    const waypoints = routes[selectedRoute].waypoints;

    const distToDest = Math.sqrt(
      (DESTINATION_POSITION.x - ship.x) ** 2 +
        (DESTINATION_POSITION.y - ship.y) ** 2,
    );
    if (distToDest < DOCKING_ZONE_RADIUS && !destinationReached) {
      setDestinationReached(true);
      addLog("TARGET REACHED - Entering docking zone");
      addLog("Mission successful - All systems nominal");
      setRunning(false);
      setTimeout(() => {
        exportDroneObservations(drones, "MISSION_SUCCESS");
        addLog("Drone observation data uploaded to cloud");
      }, 500);
      return;
    }

    if (waypoints.length === 0) return;

    const target = waypoints[0];

    let dx = target.x - ship.x;
    let dy = target.y - ship.y;
    const distanceToTarget = Math.sqrt(dx * dx + dy * dy);

    const directions = [
      { angle: 0, weight: 1.0 },
      { angle: Math.PI / 8, weight: 0.9 },
      { angle: -Math.PI / 8, weight: 0.9 },
      { angle: Math.PI / 4, weight: 0.7 },
      { angle: -Math.PI / 4, weight: 0.7 },
      { angle: Math.PI / 3, weight: 0.5 },
      { angle: -Math.PI / 3, weight: 0.5 },
    ];

    let bestDirection = { dx, dy, risk: Number.POSITIVE_INFINITY };
    const currentAngle = Math.atan2(dy, dx);

    directions.forEach(({ angle, weight }) => {
      const testAngle = currentAngle + angle;
      const testDx = Math.cos(testAngle);
      const testDy = Math.sin(testAngle);

      let threatRisk = 0;
      drones.forEach((drone) => {
        if (drone.detectedThreat && drone.threatDirection !== null) {
          const threatAngle = drone.threatDirection;
          const angleDiff = Math.abs(testAngle - threatAngle);
          const normalizedDiff = Math.min(angleDiff, Math.PI * 2 - angleDiff);

          if (normalizedDiff < Math.PI / 3) {
            threatRisk += (Math.PI / 3 - normalizedDiff) * 0.5;
          }
        }
      });

      const forwardProgress = testDx * dx + testDy * dy;
      const progressBonus =
        forwardProgress > 0 ? -forwardProgress * 2.0 : forwardProgress * 4.0;

      const deviationPenalty = Math.abs(angle) * 0.2;

      const totalRisk = threatRisk + deviationPenalty / weight + progressBonus;

      if (totalRisk < bestDirection.risk) {
        bestDirection = { dx: testDx, dy: testDy, risk: totalRisk };
      }
    });

    const momentum = 0.3;
    const previousAngle = (ship.heading * Math.PI) / 180;
    const previousDx = Math.cos(previousAngle);
    const previousDy = Math.sin(previousAngle);

    dx = bestDirection.dx * (1 - momentum) + previousDx * momentum;
    dy = bestDirection.dy * (1 - momentum) + previousDy * momentum;

    const magnitude = Math.sqrt(dx * dx + dy * dy);
    dx = (dx / magnitude) * distanceToTarget;
    dy = (dy / magnitude) * distanceToTarget;

    const adjustedDistance = Math.sqrt(dx * dx + dy * dy);

    const deviationFromLastRecalc = Math.sqrt(
      (ship.x - lastRecalcPositionRef.current.x) ** 2 +
        (ship.y - lastRecalcPositionRef.current.y) ** 2,
    );

    if (deviationFromLastRecalc > ROUTE_RECALC_THRESHOLD) {
      lastRecalcPositionRef.current = { x: ship.x, y: ship.y };
      recalculateRoutes();
      const currentLatLon = pixelToLatLon(ship.x, ship.y);
      addLog(
        `Route adjusted at ${currentLatLon.lat.toFixed(4)}, ${currentLatLon.lon.toFixed(4)} - navigating least-risk path`,
      );
    }

    if (distanceToTarget < 15) {
      waypoints.shift();
      setRoutes((prev) => ({
        ...prev,
        [selectedRoute]: {
          ...prev[selectedRoute],
          waypoints,
        },
      }));
    } else {
      const moveSpeed = 0.5 * speed;
      const newX = ship.x + (dx / adjustedDistance) * moveSpeed;
      const newY = ship.y + (dy / adjustedDistance) * moveSpeed;
      const newHeading = Math.atan2(dy, dx) * (180 / Math.PI);

      const newLatLon = pixelToLatLon(newX, newY);

      const pixelsMoved = Math.sqrt(
        (newX - ship.x) ** 2 + (newY - ship.y) ** 2,
      );
      const kmMoved = pixelsMoved * 0.01;
      const actualSpeed = Math.round(kmMoved * 60 * 1.852);

      const threatCount = drones.filter((d) => d.detectedThreat).length;
      const fuelConsumption =
        ship.fuelConsumptionRate * speed * (1 + threatCount * 0.05);

      const distanceKm =
        Math.sqrt(
          (ship.destination.x - newX) ** 2 + (ship.destination.y - newY) ** 2,
        ) * 0.01;

      setShip((prev) => ({
        ...prev,
        x: newX,
        y: newY,
        lat: newLatLon.lat,
        lon: newLatLon.lon,
        heading: (newHeading + 360) % 360,
        speed: actualSpeed,
        distanceToDestination: distanceKm,
        eta: Math.floor((distanceKm / (actualSpeed / 60)) * 60),
        fuel: Math.max(0, prev.fuel - fuelConsumption),
      }));

      setDrones((prevDrones) =>
        prevDrones.map((drone) => {
          const droneX = newX + drone.relativeX;
          const droneY = newY + drone.relativeY;
          const droneLatLon = pixelToLatLon(droneX, droneY);

          return {
            ...drone,
            x: droneX,
            y: droneY,
            lat: droneLatLon.lat,
            lon: droneLatLon.lon,
          };
        }),
      );
    }
  }, [
    ship,
    drones,
    routes,
    selectedRoute,
    speed,
    addLog,
    destinationReached,
    recalculateRoutes,
  ]);

  // ============================================
  // ANIMATION LOOP
  // ============================================

  useEffect(() => {
    if (!running) return;

    const animate = () => {
      stepRef.current++;
      updateDronePositions();
      checkDroneFailures();

      setThreats((prevThreats) =>
        updateThreats(prevThreats, drones, speed, addLog),
      );

      moveShip();

      fuelCheckIntervalRef.current++;
      if (
        fuelCheckIntervalRef.current > 10 &&
        fuelCheckIntervalRef.current % 30 === 0
      ) {
        const decision = calculateFuelDecision(
          ship,
          routes,
          START_POSITION,
          DESTINATION_POSITION,
        );
        setFuelDecision(decision);

        if (decision.status === "EMERGENCY" && !emergencyBeacon) {
          setEmergencyBeacon(true);
          setRunning(false);
          addLog("EMERGENCY BEACON DEPLOYED - HOMEBASE NOTIFIED");
          addLog("TERMINAL ALERT: Ship cannot reach destination or return");
          setTimeout(() => {
            exportDroneObservations(drones, "EMERGENCY_BEACON");
            addLog("Emergency observation data uploaded to cloud");
          }, 500);
        }

        if (decision.status === "RETURN" && selectedRoute !== "return") {
          addLog("AUTO-RETURN INITIATED - Insufficient fuel for destination");
          setShip((prev) => ({
            ...prev,
            destination: START_POSITION,
          }));
          setSelectedRoute("return");
        }
      }

      if (stepRef.current % ROUTE_RECALC_INTERVAL === 0) {
        recalculateRoutes();
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [
    running,
    updateDronePositions,
    checkDroneFailures,
    drones,
    speed,
    addLog,
    moveShip,
    ship,
    routes,
    emergencyBeacon,
    selectedRoute,
    recalculateRoutes,
  ]);

  // ============================================
  // CONTROL FUNCTIONS
  // ============================================

  const start = useCallback(() => {
    setRunning(true);
    addLog("Drone mesh navigation started");
  }, [addLog]);

  const pause = useCallback(() => {
    setRunning(false);
    addLog("Navigation paused");
  }, [addLog]);

  const stop = useCallback(() => {
    setRunning(false);
    addLog("Navigation stopped");
  }, [addLog]);

  const reset = useCallback(() => {
    setRunning(false);
    stepRef.current = 0;
    droneFailureTimerRef.current = 0;
    fuelCheckIntervalRef.current = 0; // Reset fuel check interval
    setEmergencyBeacon(false);
    setDestinationReached(false);
    lastRecalcPositionRef.current = {
      x: START_POSITION.x,
      y: START_POSITION.y,
    };

    setDrones(
      initializeDrones(START_POSITION.x, START_POSITION.y, DRONE_COUNT),
    );

    const resetLatLon = pixelToLatLon(START_POSITION.x, START_POSITION.y);

    setShip({
      x: START_POSITION.x,
      y: START_POSITION.y,
      lat: resetLatLon.lat,
      lon: resetLatLon.lon,
      heading: 90,
      speed: 12,
      destination: DESTINATION_POSITION,
      distanceToDestination: 0,
      eta: 0,
      fuel: 100,
      fuelConsumptionRate: 0.015,
    });
    setThreats(generateRandomThreats());
    setMissionLog([
      "[00:00] Drone mesh navigation initialized",
      "[00:00] 6 drones deployed in protective formation",
      "[00:00] ML threat detection active",
    ]);
    recalculateRoutes();
  }, [recalculateRoutes]);

  const recalculate = useCallback(() => {
    recalculateRoutes();
    addLog("Routes recalculated");
  }, [recalculateRoutes, addLog]);

  // ============================================
  // COMPUTED VALUES
  // ============================================

  const mlDetections = threats
    .filter((t) => t.detected)
    .map((t) => ({
      name: t.name,
      isThreat: t.mlClassification?.threat_detected || false,
      actualThreat: t.isThreat,
      mlResult: t.mlClassification?.threat_detected
        ? "ML: THREAT"
        : "ML: BENIGN",
      confidence: t.mlClassification?.confidence || 0,
      correct: t.mlClassification?.threat_detected === t.isThreat,
      distance: Math.sqrt((ship.x - t.x) ** 2 + (ship.y - t.y) ** 2),
    }));

  const threatCount = threats.filter(
    (t) => t.mlClassification?.threat_detected,
  ).length;
  const routeAnalysis = {
    summary: `Fastest: fastest | Safest: safest | Best Score: optimal`,
    mlAnalysis: `Drone Mesh: ${threatCount} threats detected`,
    recommendation:
      threatCount === 0
        ? "No threats - Fastest route recommended"
        : threatCount >= 3
          ? "Multiple threats - Safest route recommended"
          : "Some threats - Optimal route recommended",
  };

  // ============================================
  // RETURN API
  // ============================================

  return {
    ship,
    drones,
    threats,
    routes,
    selectedRoute,
    setSelectedRoute,
    running,
    speed,
    start,
    pause,
    stop,
    reset,
    recalculate,
    setSpeed,
    scenario: {
      title: "Drone Mesh Navigation System",
      description: `${drones.filter((d) => d.active && d.detectedThreat).length} drones detecting threats | ${drones.filter((d) => d.active).length}/${DRONE_COUNT} drones active | Following ${selectedRoute} route`,
    },
    mlDetections,
    routeAnalysis,
    missionLog,
    fuelDecision,
    emergencyBeacon,
    destinationReached,
    triggerDroneFailure,
    lastDroneFailure,
  };
}
