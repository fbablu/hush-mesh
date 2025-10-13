export interface ThreatState {
  x: number;
  y: number;
  lat: number;
  lon: number;
  vx: number;
  vy: number;
  type: string;
  radius: number;
  name: string;
  speed: number;
  behavior: string;
  isThreat: boolean;
  detected: boolean;
  mlClassification?: {
    threat_detected: boolean;
    confidence: number;
    threat_type: string;
  };
  patrolCenter?: { x: number; y: number };
  patrolRadius?: number;
  patrolAngle?: number;
  transitTarget?: { x: number; y: number };
}

export interface DroneState {
  id: number;
  x: number;
  y: number;
  lat: number; // Added lat/lon tracking for drones
  lon: number;
  relativeX: number; // Position relative to ship center
  relativeY: number; // Position relative to ship center
  active: boolean;
  detectedThreat: boolean;
  threatDirection: number | null;
  coverageAngle: number; // Angle this drone is responsible for monitoring
  coverageRange: number; // Range in degrees this drone covers
  observations: DroneObservation[]; // Added observations tracking
}

export interface DroneObservation {
  timestamp: number;
  droneId: number;
  droneLat: number;
  droneLon: number;
  objectType: string;
  objectName: string;
  objectLat: number;
  objectLon: number;
  distance: number;
  isThreat: boolean;
  confidence: number;
}

export interface ShipState {
  x: number;
  y: number;
  lat: number;
  lon: number;
  heading: number;
  speed: number;
  destination: { x: number; y: number };
  distanceToDestination: number;
  eta: number;
  fuel: number;
  fuelConsumptionRate: number;
}

export interface RouteData {
  waypoints: { x: number; y: number }[];
  metrics: {
    distance: number;
    threatExposure: number;
    score: number;
  };
}

export interface FuelDecision {
  status: string;
  message: string;
  canReachDestination: boolean;
  canReturnToStart: boolean;
  recommendedAction: string;
}

export interface MaritimeSimulation {
  ship: ShipState;
  drones: DroneState[];
  threats: ThreatState[];
  routes: Record<string, RouteData>;
  selectedRoute: string;
  setSelectedRoute: (route: string) => void;
  running: boolean;
  speed: number;
  start: () => void;
  pause: () => void;
  stop: () => void;
  reset: () => void;
  recalculate: () => void;
  setSpeed: (speed: number) => void;
  scenario: {
    title: string;
    description: string;
  };
  mlDetections: Array<{
    name: string;
    isThreat: boolean;
    actualThreat: boolean;
    mlResult: string;
    confidence: number;
    correct: boolean;
    distance: number;
  }>;
  routeAnalysis: {
    summary: string;
    mlAnalysis: string;
    recommendation: string;
  };
  missionLog: string[];
  fuelDecision: FuelDecision;
  emergencyBeacon: boolean;
  destinationReached: boolean;
  triggerDroneFailure: () => void;
  lastDroneFailure: number | null;
}
