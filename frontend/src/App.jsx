import React, { useState, useEffect, useRef } from 'react';

const App = () => {
  const [convoyData, setConvoyData] = useState({ vehicles: [], drones: [] });
  const [obstacles, setObstacles] = useState([]);
  const [threats, setThreats] = useState([]);
  const [detections, setDetections] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [dragMode, setDragMode] = useState(null); // 'obstacle' or 'threat'
  const wsRef = useRef(null);
  const mapRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws/telemetry');
    
    wsRef.current.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };
    
    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'convoy_update':
          setConvoyData(message.data);
          if (message.data.detections) {
            setDetections(message.data.detections);
          }
          break;
        case 'obstacles_update':
          setObstacles(message.data);
          break;
        case 'threats_update':
          setThreats(message.data);
          break;
        case 'mission_reset':
          setObstacles([]);
          setThreats([]);
          setDetections([]);
          setConvoyData({ vehicles: [], drones: [] });
          break;
      }
    };
    
    wsRef.current.onclose = () => {
      setIsConnected(false);
      setTimeout(connectWebSocket, 3000);
    };
  };

  const startMission = async () => {
    try {
      await fetch('http://localhost:8000/api/mission/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: 'convoy', vehicles: 3, drones: 6 })
      });
    } catch (error) {
      console.error('Failed to start mission:', error);
    }
  };

  const handleMapClick = async (event) => {
    if (!dragMode) return;
    
    const rect = mapRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    try {
      if (dragMode === 'obstacle') {
        await fetch('http://localhost:8000/api/add-obstacle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ x, y, type: 'physical' })
        });
      } else if (dragMode === 'threat') {
        await fetch('http://localhost:8000/api/add-threat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ x, y, type: 'ambush' })
        });
      }
    } catch (error) {
      console.error('Failed to add item:', error);
    }
  };

  const handleDroneClick = async (drone) => {
    if (!drone.active) return;
    
    try {
      await fetch('http://localhost:8000/api/eliminate-drone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drone_id: drone.id })
      });
    } catch (error) {
      console.error('Failed to eliminate drone:', error);
    }
  };

  const resetMission = async () => {
    try {
      await fetch('http://localhost:8000/api/reset', {
        method: 'POST'
      });
      setObstacles([]);
      setThreats([]);
      setDetections([]);
      setConvoyData({ vehicles: [], drones: [] });
    } catch (error) {
      console.error('Failed to reset mission:', error);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-yellow-400">ACPS - Convoy Protection</h1>
          <div className="flex items-center gap-4">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        {/* Controls */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={startMission}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
          >
            Start Mission
          </button>
          <button
            onClick={resetMission}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded"
          >
            Reset
          </button>
          <button
            onClick={() => setDragMode(dragMode === 'obstacle' ? null : 'obstacle')}
            className={`px-4 py-2 rounded ${dragMode === 'obstacle' ? 'bg-orange-600' : 'bg-gray-600 hover:bg-gray-700'}`}
          >
            {dragMode === 'obstacle' ? 'Stop Adding' : 'Add Obstacles'}
          </button>
          <button
            onClick={() => setDragMode(dragMode === 'threat' ? null : 'threat')}
            className={`px-4 py-2 rounded ${dragMode === 'threat' ? 'bg-red-600' : 'bg-gray-600 hover:bg-gray-700'}`}
          >
            {dragMode === 'threat' ? 'Stop Adding' : 'Add Threats'}
          </button>
        </div>

        {/* Map */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
          <div
            ref={mapRef}
            onClick={handleMapClick}
            className={`relative bg-gray-800 border-2 border-gray-600 ${dragMode ? 'cursor-crosshair' : 'cursor-default'}`}
            style={{ width: '800px', height: '600px' }}
          >
            {/* Vehicles */}
            {convoyData.vehicles.map((vehicle) => (
              <div
                key={vehicle.id}
                className="absolute w-6 h-4 bg-blue-500 border border-blue-300 rounded transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center text-xs font-bold"
                style={{ left: vehicle.x, top: vehicle.y }}
                title={`${vehicle.id} - ${vehicle.type}`}
              >
                V
              </div>
            ))}

            {/* Drones */}
            {convoyData.drones.map((drone) => {
              const isDetecting = drone.detecting;
              return (
                <div
                  key={drone.id}
                  onClick={() => handleDroneClick(drone)}
                  className={`absolute w-4 h-4 border-2 rounded-full transform -translate-x-1/2 -translate-y-1/2 cursor-pointer flex items-center justify-center text-xs font-bold ${
                    !drone.active 
                      ? 'bg-red-500 border-red-300 opacity-50'
                      : isDetecting
                      ? 'bg-red-600 border-red-400 animate-pulse'
                      : drone.role === 'scout' ? 'bg-green-500 border-green-300' :
                        drone.role === 'escort' ? 'bg-yellow-500 border-yellow-300' :
                        drone.role === 'overwatch' ? 'bg-purple-500 border-purple-300' :
                        'bg-cyan-500 border-cyan-300'
                  }`}
                  style={{ left: drone.x, top: drone.y }}
                  title={`${drone.id} - ${drone.role} - ${drone.active ? (isDetecting ? 'DETECTING!' : 'Active') : 'Eliminated'}`}
                >
                  D
                </div>
              );
            })}

            {/* Obstacles */}
            {obstacles.map((obstacle, index) => (
              <div
                key={obstacle.id || index}
                className="absolute w-6 h-6 bg-orange-600 border border-orange-400 transform -translate-x-1/2 -translate-y-1/2"
                style={{ left: obstacle.x, top: obstacle.y }}
                title={`Obstacle - ${obstacle.type}`}
              />
            ))}

            {/* Threats */}
            {threats.map((threat, index) => (
              <div
                key={threat.id || index}
                className="absolute w-4 h-4 bg-red-600 border border-red-400 rounded-full transform -translate-x-1/2 -translate-y-1/2 animate-pulse"
                style={{ left: threat.x, top: threat.y }}
                title={`Threat - ${threat.type} (${(threat.severity * 100).toFixed(0)}%)`}
              />
            ))}

            {/* Instructions */}
            {dragMode && (
              <div className="absolute top-4 left-4 bg-black bg-opacity-75 p-2 rounded text-sm">
                Click to add {dragMode === 'obstacle' ? 'obstacles' : 'threats'}
              </div>
            )}
          </div>
        </div>

        {/* Status Panel */}
        <div className="mt-6 grid grid-cols-4 gap-4">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-yellow-400 mb-2">Vehicles</h3>
            <div className="space-y-1">
              {convoyData.vehicles.map((vehicle) => (
                <div key={vehicle.id} className="text-sm">
                  {vehicle.id}: {vehicle.type} - Health: {vehicle.health}%
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-yellow-400 mb-2">Drones</h3>
            <div className="space-y-1">
              {convoyData.drones.map((drone) => (
                <div key={drone.id} className={`text-sm ${
                  !drone.active ? 'text-red-400' : 
                  drone.detecting ? 'text-red-300 font-bold' : 'text-white'
                }`}>
                  {drone.id}: {drone.role} - {drone.active ? (drone.detecting ? 'DETECTING!' : 'Active') : 'Eliminated'}
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-yellow-400 mb-2">Detections</h3>
            <div className="space-y-1 text-sm">
              {detections.length === 0 ? (
                <div className="text-gray-400">No threats detected</div>
              ) : (
                detections.map((detection, index) => (
                  <div key={index} className="text-red-300">
                    {detection.drone_id}: {detection.type} ({Math.round(detection.distance)}px)
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-yellow-400 mb-2">Status</h3>
            <div className="space-y-1 text-sm">
              <div>Obstacles: {obstacles.length}</div>
              <div>Threats: {threats.length}</div>
              <div>Active Drones: {convoyData.drones.filter(d => d.active).length}</div>
              <div className={detections.length > 0 ? 'text-red-300 font-bold' : 'text-green-300'}>
                Alert Level: {detections.length > 0 ? 'HIGH' : 'NORMAL'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;