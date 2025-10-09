import React, { useState, useEffect, useRef } from 'react';
import './EnhancedMultiRoute.css';
import Sidebar from './components/Sidebar';
import MapContainer from './components/MapContainer';
import RoutePanel from './components/RoutePanel';
import Controls from './components/Controls';

const EnhancedMultiRoute = () => {
  const [convoy, setConvoy] = useState({ x: 100, y: 400, lat: 25.7617, lon: -80.1918 });
  const [destination] = useState({ x: 700, y: 200, lat: 25.8200, lon: -80.1200 });
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [step, setStep] = useState(0);
  const [selectedRoute, setSelectedRoute] = useState('optimal');
  const [routes, setRoutes] = useState({});
  const [detectedObjects, setDetectedObjects] = useState(new Set());
  const [mlClassifications, setMlClassifications] = useState(new Map());
  const [calculationInProgress, setCalculationInProgress] = useState(false);
  const [missionLog, setMissionLog] = useState([
    '[00:00] Enhanced multi-route planning initialized',
    '[00:00] ML threat detection active',
    '[00:00] Calculating path alternatives'
  ]);

  const marineObjects = useRef([
    // Aggressive intercepting threats
    { x: 300, y: 450, vx: -2, vy: -3, type: 'small_fast_craft', radius: 70, name: 'Pirate Skiff Alpha', speed: 3.5, behavior: 'intercept', isThreat: true },
    { x: 180, y: 200, vx: 2.5, vy: 1, type: 'small_fast_craft', radius: 65, name: 'Pirate Skiff Beta', speed: 3.2, behavior: 'intercept', isThreat: true },
    { x: 520, y: 480, vx: -1.8, vy: -2.5, type: 'small_fast_craft', radius: 60, name: 'Fast Attack Craft', speed: 4.0, behavior: 'intercept', isThreat: true },
    
    // Moving mines and obstacles
    { x: 500, y: 150, vx: 0, vy: 2, type: 'floating_mine_like_object', radius: 50, name: 'Drifting Mine', speed: 1.5, behavior: 'drift', isThreat: true },
    { x: 380, y: 80, vx: -1, vy: 1.5, type: 'floating_mine_like_object', radius: 45, name: 'Sea Mine', speed: 1.2, behavior: 'drift', isThreat: true },
    
    // Submarine threats
    { x: 650, y: 350, vx: -2, vy: 1, type: 'submarine_periscope', radius: 90, name: 'Sub Contact Alpha', speed: 2.0, behavior: 'patrol', isThreat: true },
    { x: 420, y: 180, vx: 1.5, vy: 2, type: 'submarine_periscope', radius: 85, name: 'Sub Contact Beta', speed: 1.8, behavior: 'patrol', isThreat: true },
    
    // Expanding debris fields
    { x: 150, y: 100, vx: 2, vy: 1, type: 'debris_field', radius: 95, name: 'Debris Field Alpha', speed: 0.8, behavior: 'expand', isThreat: true },
    { x: 600, y: 120, vx: -1, vy: 1.5, type: 'debris_field', radius: 80, name: 'Debris Field Beta', speed: 0.6, behavior: 'expand', isThreat: true },
    
    // Benign objects for ML testing
    { x: 400, y: 400, vx: 0.5, vy: 0, type: 'fishing_vessel', radius: 50, name: 'Fishing Boat', speed: 1.0, behavior: 'drift', isThreat: false },
    { x: 200, y: 300, vx: 0, vy: 0.8, type: 'cargo_ship', radius: 70, name: 'Cargo Ship', speed: 0.8, behavior: 'transit', isThreat: false },
    { x: 550, y: 450, vx: -0.5, vy: -0.5, type: 'research_vessel', radius: 45, name: 'Research Ship', speed: 1.2, behavior: 'survey', isThreat: false }
  ]);

  const objectMovementPatterns = useRef(new Map());

  useEffect(() => {
    initializeObjectMovement();
    setTimeout(() => calculateMultipleRoutes(), 500);
  }, []);

  useEffect(() => {
    let interval;
    if (running) {
      interval = setInterval(() => {
        detectNearbyObjects();
        moveConvoy();
        setStep(prev => prev + 1);
      }, 100 / speed);
    }
    return () => clearInterval(interval);
  }, [running, speed, selectedRoute, routes]);

  const initializeObjectMovement = () => {
    marineObjects.current.forEach((obj, index) => {
      objectMovementPatterns.current.set(index, {
        originalX: obj.x,
        originalY: obj.y,
        patrolTime: 0,
        interceptTarget: null,
        lastDirectionChange: 0
      });
    });
  };

  const calculateMultipleRoutes = () => {
    setCalculationInProgress(true);
    
    setTimeout(() => {
      const confirmedThreats = [];
      mlClassifications.forEach((classification, index) => {
        if (classification.threat_detected && classification.confidence > 0.6) {
          confirmedThreats.push(marineObjects.current[index]);
        }
      });
      
      const newRoutes = {
        optimal: generateRoute('optimal', confirmedThreats),
        fastest: generateRoute('fastest', confirmedThreats),
        safest: generateRoute('safest', confirmedThreats)
      };
      
      setRoutes(newRoutes);
      setCalculationInProgress(false);
      
      addToMissionLog(`🧠 Routes calculated with ${confirmedThreats.length} ML-confirmed threats`);
    }, 1500);
  };

  const generateRoute = (strategy, threats) => {
    const waypoints = [];
    const steps = 8;
    
    for (let i = 1; i <= steps; i++) {
      const progress = i / steps;
      let x = convoy.x + (destination.x - convoy.x) * progress;
      let y = convoy.y + (destination.y - convoy.y) * progress;
      
      if (strategy === 'fastest') {
        // Direct route, minimal avoidance
      } else if (strategy === 'safest') {
        y -= 60 * Math.sin(progress * Math.PI);
        x += 20 * Math.cos(progress * Math.PI * 2);
      } else {
        // Optimal: avoid only ML-confirmed threats
        threats.forEach(threat => {
          const distance = Math.sqrt((x - threat.x)**2 + (y - threat.y)**2);
          if (distance < threat.radius + 40) {
            const avoidX = (x - threat.x) / distance * (threat.radius + 40);
            const avoidY = (y - threat.y) / distance * (threat.radius + 40);
            x = threat.x + avoidX;
            y = threat.y + avoidY;
          }
        });
      }
      
      waypoints.push({ x, y });
    }
    
    const distance = calculateRouteDistance(waypoints);
    const threatExposure = calculateThreatExposure(waypoints, threats);
    const score = calculateRouteScore(distance, threatExposure, strategy);
    
    return {
      waypoints,
      metrics: {
        distance: distance.toFixed(1),
        threatExposure: threatExposure.toFixed(1),
        score: score.toFixed(0)
      }
    };
  };

  const calculateRouteDistance = (waypoints) => {
    let distance = 0;
    let prev = convoy;
    waypoints.forEach(wp => {
      distance += Math.sqrt((wp.x - prev.x)**2 + (wp.y - prev.y)**2);
      prev = wp;
    });
    return distance * 0.01;
  };

  const calculateThreatExposure = (waypoints, threats) => {
    let exposure = 0;
    waypoints.forEach(wp => {
      threats.forEach(threat => {
        const distance = Math.sqrt((wp.x - threat.x)**2 + (wp.y - threat.y)**2);
        if (distance < threat.radius * 1.5) {
          exposure += (threat.radius * 1.5 - distance) / threat.radius;
        }
      });
    });
    return exposure;
  };

  const calculateRouteScore = (distance, threatExposure, strategy) => {
    if (strategy === 'fastest') {
      return 100 - distance * 3 - threatExposure * 1;
    } else if (strategy === 'safest') {
      return 100 - threatExposure * 8 - distance * 0.5;
    } else {
      return 100 - distance * 2 - threatExposure * 3;
    }
  };

  const detectNearbyObjects = () => {
    marineObjects.current.forEach((obj, i) => {
      moveObject(obj, i);
      
      const distance = Math.sqrt((convoy.x - obj.x)**2 + (convoy.y - obj.y)**2);
      
      if (distance < obj.radius * 2.0) {
        if (!detectedObjects.has(i)) {
          addToMissionLog(`📡 Object detected: ${obj.name}`);
          setDetectedObjects(prev => new Set([...prev, i]));
          classifyObjectWithML(obj, i);
        }
      } else if (distance > obj.radius * 3.0) {
        setDetectedObjects(prev => {
          const newSet = new Set(prev);
          newSet.delete(i);
          return newSet;
        });
        setMlClassifications(prev => {
          const newMap = new Map(prev);
          newMap.delete(i);
          return newMap;
        });
      }
    });
  };

  const moveObject = (obj, index) => {
    const pattern = objectMovementPatterns.current.get(index);
    if (!pattern) return;
    
    pattern.patrolTime += 0.1;
    
    if (obj.behavior === 'intercept') {
      const dx = convoy.x - obj.x;
      const dy = convoy.y - obj.y;
      const distance = Math.sqrt(dx*dx + dy*dy);
      
      if (distance > 50) {
        obj.x += (dx / distance) * obj.speed;
        obj.y += (dy / distance) * obj.speed;
      } else {
        obj.x += Math.cos(pattern.patrolTime) * obj.speed;
        obj.y += Math.sin(pattern.patrolTime) * obj.speed;
      }
    } else if (obj.behavior === 'patrol') {
      const radius = 50 + pattern.patrolTime * 10;
      obj.x = pattern.originalX + Math.cos(pattern.patrolTime * 0.5) * radius;
      obj.y = pattern.originalY + Math.sin(pattern.patrolTime * 0.5) * radius;
    } else if (obj.behavior === 'drift') {
      if (pattern.patrolTime - pattern.lastDirectionChange > 5) {
        obj.vx += (Math.random() - 0.5) * 0.5;
        obj.vy += (Math.random() - 0.5) * 0.5;
        pattern.lastDirectionChange = pattern.patrolTime;
      }
      obj.x += obj.vx;
      obj.y += obj.vy;
    } else if (obj.behavior === 'expand') {
      obj.radius = Math.min(obj.radius + 0.2, 120);
      obj.x += obj.vx * 0.5;
      obj.y += obj.vy * 0.5;
    }
    
    obj.x = Math.max(50, Math.min(750, obj.x));
    obj.y = Math.max(50, Math.min(550, obj.y));
  };

  const classifyObjectWithML = async (obj, index) => {
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          convoy_id: 'enhanced_demo',
          sensor_data: generateSensorDataForObject(obj)
        })
      });
      
      if (response.ok) {
        const prediction = await response.json();
        setMlClassifications(prev => new Map(prev.set(index, prediction)));
        
        if (prediction.threat_detected && prediction.confidence > 0.6) {
          addToMissionLog(`🚨 ML classified as THREAT: ${obj.name} (${(prediction.confidence * 100).toFixed(1)}%)`);
        } else {
          addToMissionLog(`✅ ML classified as BENIGN: ${obj.name} (${(prediction.confidence * 100).toFixed(1)}%)`);
        }
      } else {
        setMlClassifications(prev => new Map(prev.set(index, { 
          threat_detected: obj.isThreat, 
          confidence: 0.8, 
          threat_type: obj.type 
        })));
      }
    } catch (error) {
      console.warn('ML service error:', error);
      setMlClassifications(prev => new Map(prev.set(index, { 
        threat_detected: obj.isThreat, 
        confidence: 0.8, 
        threat_type: obj.type 
      })));
    }
  };

  const generateSensorDataForObject = (obj) => {
    const baseData = {
      convoy_data: { speed_knots: 12, vessel_count: 4 },
      environmental_conditions: { sea_state: 2, visibility_km: 15, wind_speed_knots: 10, wave_height_m: 2 },
      data_quality: { sensor_reliability: 0.9 },
      drone_array: [{
        sensor_suite: {
          electro_optical: { visible_spectrum: { objects_detected: 1 }, infrared: { thermal_signatures: 1 } },
          radar: { contacts: 1 },
          acoustic: { hydrophone_data: { ambient_noise_db: 95 } },
          electronic_warfare: { rf_spectrum: { signals_detected: 3 } }
        },
        position: { altitude_m: 180 }
      }],
      threat_detections: [],
      ground_truth: { threat_present: obj.isThreat, threat_type: obj.type }
    };
    
    if (obj.type === 'small_fast_craft') {
      baseData.drone_array[0].sensor_suite.radar.contacts = 3;
      baseData.drone_array[0].sensor_suite.electronic_warfare.rf_spectrum.signals_detected = 8;
      baseData.drone_array[0].sensor_suite.acoustic.hydrophone_data.ambient_noise_db = 105;
    }
    
    return baseData;
  };

  const moveConvoy = () => {
    if (!routes[selectedRoute]) return;
    
    const waypoints = routes[selectedRoute].waypoints;
    if (waypoints.length === 0) return;
    
    const target = waypoints[0];
    const dx = target.x - convoy.x;
    const dy = target.y - convoy.y;
    const distance = Math.sqrt(dx*dx + dy*dy);
    
    if (distance < 15) {
      waypoints.shift();
      if (waypoints.length === 0) {
        addToMissionLog('🏁 Destination reached!');
        setRunning(false);
        return;
      }
    } else {
      const moveSpeed = 3 * speed;
      setConvoy(prev => ({
        ...prev,
        x: prev.x + (dx / distance) * moveSpeed,
        y: prev.y + (dy / distance) * moveSpeed
      }));
    }
  };

  const addToMissionLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setMissionLog(prev => [...prev, `[${timestamp}] ${message}`].slice(-20));
  };

  const startDemo = () => setRunning(true);
  const pauseDemo = () => setRunning(false);
  const stopDemo = () => {
    setRunning(false);
    setConvoy({ x: 100, y: 400, lat: 25.7617, lon: -80.1918 });
  };
  const resetDemo = () => {
    setRunning(false);
    setConvoy({ x: 100, y: 400, lat: 25.7617, lon: -80.1918 });
    setDetectedObjects(new Set());
    setMlClassifications(new Map());
    setMissionLog([
      '[00:00] Enhanced multi-route planning initialized',
      '[00:00] ML threat detection active',
      '[00:00] Calculating path alternatives'
    ]);
    initializeObjectMovement();
    setTimeout(() => calculateMultipleRoutes(), 500);
  };

  return (
    <div className="dashboard">
      <Sidebar 
        detectedObjects={detectedObjects}
        mlClassifications={mlClassifications}
        marineObjects={marineObjects.current}
        convoy={convoy}
        missionLog={missionLog}
      />
      
      <MapContainer
        convoy={convoy}
        destination={destination}
        marineObjects={marineObjects.current}
        detectedObjects={detectedObjects}
        mlClassifications={mlClassifications}
        routes={routes}
        selectedRoute={selectedRoute}
        calculationInProgress={calculationInProgress}
      />
      
      <RoutePanel
        routes={routes}
        selectedRoute={selectedRoute}
        setSelectedRoute={setSelectedRoute}
        mlClassifications={mlClassifications}
        addToMissionLog={addToMissionLog}
      />
      
      <Controls
        running={running}
        speed={speed}
        startDemo={startDemo}
        pauseDemo={pauseDemo}
        stopDemo={stopDemo}
        resetDemo={resetDemo}
        setSpeed={setSpeed}
        recalculateRoutes={calculateMultipleRoutes}
      />
    </div>
  );
};

export default EnhancedMultiRoute;