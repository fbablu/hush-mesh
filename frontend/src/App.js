import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import axios from 'axios';
import './App.css';

// Configure Amplify (replace with actual config)
Amplify.configure({
  Auth: {
    region: 'us-east-1',
    userPoolId: 'us-east-1_XXXXXXXXX',
    userPoolWebClientId: 'XXXXXXXXXXXXXXXXXXXXXXXXXX'
  }
});

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

function Dashboard() {
  const [threats, setThreats] = useState([]);
  const [convoyPosition, setConvoyPosition] = useState({ lat: 25.7617, lng: -80.1918 });
  const [plannedRoute, setPlannedRoute] = useState([]);
  const [wsConnection, setWsConnection] = useState(null);
  const [scenario, setScenario] = useState('normal');
  const [pendingPlan, setPendingPlan] = useState(null);

  useEffect(() => {
    // WebSocket connection for real-time updates
    const ws = new WebSocket(`ws://${BACKEND_URL.replace('http://', '')}/ws`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'threat_update') {
        setThreats(data.threats);
      }
    };
    
    setWsConnection(ws);
    
    return () => {
      if (ws) ws.close();
    };
  }, []);

  const startScenario = async (scenarioType) => {
    setScenario(scenarioType);
    
    // Trigger scenario simulation
    try {
      await axios.post(`${BACKEND_URL}/scenario/start`, {
        scenario: scenarioType,
        duration: 300
      });
    } catch (error) {
      console.error('Failed to start scenario:', error);
    }
  };

  const requestReplan = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/plan`, {
        start: { lat: convoyPosition.lat, lon: convoyPosition.lng },
        goal: { lat: 25.8617, lon: -80.0918 }  // Example destination
      });
      
      setPendingPlan(response.data);
    } catch (error) {
      console.error('Failed to request replan:', error);
    }
  };

  const approvePlan = async () => {
    if (!pendingPlan) return;
    
    try {
      await axios.post(`${BACKEND_URL}/approve_plan`, {
        plan_id: pendingPlan.plan_id,
        waypoints: pendingPlan.waypoints
      });
      
      setPlannedRoute(pendingPlan.waypoints);
      setPendingPlan(null);
      
      alert('✅ Route plan approved and sent to convoy');
    } catch (error) {
      console.error('Failed to approve plan:', error);
    }
  };

  const rejectPlan = () => {
    setPendingPlan(null);
    alert('❌ Route plan rejected');
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <div className="header">
        <h1>🚢 Maritime ACPS Dashboard</h1>
        <div className="status">
          <span className="scenario">Scenario: {scenario}</span>
          <span className="threats">Active Threats: {threats.length}</span>
        </div>
      </div>

      <div className="main-content">
        {/* Left Panel - Controls */}
        <div className="left-panel">
          <div className="control-section">
            <h3>Scenario Control</h3>
            <button onClick={() => startScenario('piracy_ambush')}>
              🏴‍☠️ Piracy Ambush
            </button>
            <button onClick={() => startScenario('swarm_interdiction')}>
              🚤 Swarm Attack
            </button>
            <button onClick={() => startScenario('mine_detection')}>
              💣 Mine Field
            </button>
          </div>

          <div className="control-section">
            <h3>Mission Control</h3>
            <button onClick={requestReplan} className="replan-btn">
              🗺️ Request Replan
            </button>
            
            {pendingPlan && (
              <div className="pending-plan">
                <h4>⚠️ HUMAN AUTHORIZATION REQUIRED</h4>
                <p>New route plan generated</p>
                <p>Waypoints: {pendingPlan.waypoints?.length || 0}</p>
                <p>Threats considered: {pendingPlan.threats_considered}</p>
                
                <div className="approval-buttons">
                  <button onClick={approvePlan} className="approve-btn">
                    ✅ APPROVE
                  </button>
                  <button onClick={rejectPlan} className="reject-btn">
                    ❌ REJECT
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Center - Map */}
        <div className="map-container">
          <MapContainer
            center={[convoyPosition.lat, convoyPosition.lng]}
            zoom={12}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; OpenStreetMap contributors'
            />
            
            {/* Convoy Position */}
            <Marker position={[convoyPosition.lat, convoyPosition.lng]}>
              <Popup>
                🚢 Convoy Position<br />
                Speed: 12 knots<br />
                Heading: 090°
              </Popup>
            </Marker>

            {/* Threat Markers */}
            {threats.map((threat, index) => (
              threat.location && (
                <Marker
                  key={index}
                  position={[threat.location.lat, threat.location.lon]}
                >
                  <Popup>
                    ⚠️ {threat.threat_type}<br />
                    Confidence: {(threat.confidence * 100).toFixed(1)}%<br />
                    Device: {threat.device_id}
                  </Popup>
                </Marker>
              )
            ))}

            {/* Planned Route */}
            {plannedRoute.length > 0 && (
              <Polyline
                positions={plannedRoute.map(wp => [wp.lat, wp.lon])}
                color="yellow"
                weight={3}
              />
            )}
          </MapContainer>
        </div>

        {/* Right Panel - Threat Details */}
        <div className="right-panel">
          <h3>🎯 Active Threats</h3>
          <div className="threat-list">
            {threats.map((threat, index) => (
              <div key={index} className="threat-card">
                <div className="threat-type">{threat.threat_type}</div>
                <div className="threat-confidence">
                  {(threat.confidence * 100).toFixed(1)}%
                </div>
                <div className="threat-device">{threat.device_id}</div>
                <div className="threat-time">
                  {new Date(threat.timestamp * 1000).toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>

          <div className="safety-notice">
            <h4>🛡️ DEFENSIVE SYSTEM</h4>
            <p>All engagement decisions require human authorization.</p>
            <p>System provides recommendations only.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <Authenticator>
      {({ signOut, user }) => (
        <div className="App">
          <Dashboard />
          <button onClick={signOut} className="signout-btn">Sign out</button>
        </div>
      )}
    </Authenticator>
  );
}

export default App;