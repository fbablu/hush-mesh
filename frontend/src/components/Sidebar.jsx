import React from 'react';
import MissionLog from './MissionLog';

const Sidebar = ({ detectedObjects, mlClassifications, marineObjects, convoy, missionLog }) => {
  const renderMLPanel = () => {
    if (detectedObjects.size === 0) {
      return (
        <div className="status-item safe-status">
          No objects in detection range
        </div>
      );
    }

    return Array.from(detectedObjects).map(objIndex => {
      const obj = marineObjects[objIndex];
      const classification = mlClassifications.get(objIndex);
      const distance = Math.sqrt((convoy.x - obj.x)**2 + (convoy.y - obj.y)**2);
      
      if (!classification) {
        return (
          <div key={objIndex} className="status-item ml-analyzing">
            <strong>{obj.name}</strong><br />
            ML Status: Analyzing... | Distance: {Math.floor(distance * 10)}m
          </div>
        );
      }

      const actualThreat = obj.isThreat ? 'ACTUAL THREAT' : 'BENIGN';
      const mlResult = classification.threat_detected ? 'ML: THREAT' : 'ML: BENIGN';
      const correct = (classification.threat_detected === obj.isThreat) ? '✅' : '❌';
      
      return (
        <div 
          key={objIndex} 
          className={classification.threat_detected ? 'status-item threat-alert' : 'status-item safe-status'}
        >
          <strong>{obj.name}</strong> {correct}<br />
          {actualThreat} | {mlResult} ({(classification.confidence * 100).toFixed(1)}%)<br />
          Distance: {Math.floor(distance * 10)}m
        </div>
      );
    });
  };

  return (
    <div className="sidebar">
      <h3>🛡️ System Status</h3>
      <div className="status-panel">
        <div className="status-item safe-status" id="system-status">
          <strong>System: OPERATIONAL</strong>
        </div>
        <div className="status-item" id="ml-status">
          ML Model: Active (92.86% accuracy)
        </div>
        <div className="status-item" id="drone-status">
          Drones: 4/4 Online
        </div>
      </div>
      
      <h3>📊 Current Metrics</h3>
      <div className="status-panel metrics">
        <div>Position: <span className="metric-value">{convoy.lat.toFixed(4)}, {convoy.lon.toFixed(4)}</span></div>
        <div>Speed: <span className="metric-value">12 knots</span></div>
        <div>Heading: <span className="metric-value">090°</span></div>
        <div>Distance: <span className="metric-value">8.2 km</span></div>
        <div>ETA: <span className="metric-value">24 min</span></div>
      </div>
      
      <h3>🔍 ML Detections</h3>
      <div className="status-panel" id="ml-panel">
        {renderMLPanel()}
      </div>
      
      <h3>🎯 Mission Log</h3>
      <MissionLog missionLog={missionLog} />
    </div>
  );
};

export default Sidebar;