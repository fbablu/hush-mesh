import React from 'react';

const RoutePanel = ({ routes, selectedRoute, setSelectedRoute, mlClassifications, addToMissionLog }) => {
  
  const handleRouteSelection = (routeType) => {
    setSelectedRoute(routeType);
    addToMissionLog(`📍 Switched to ${routeType} route`);
  };

  const analyzeRoutes = () => {
    if (!routes || Object.keys(routes).length === 0) return null;

    const routeKeys = Object.keys(routes);
    const fastest = routeKeys.reduce((a, b) => 
      parseFloat(routes[a].metrics.distance) < parseFloat(routes[b].metrics.distance) ? a : b);
    const safest = routeKeys.reduce((a, b) => 
      parseFloat(routes[a].metrics.threatExposure) < parseFloat(routes[b].metrics.threatExposure) ? a : b);
    const highestScore = routeKeys.reduce((a, b) => 
      parseFloat(routes[a].metrics.score) > parseFloat(routes[b].metrics.score) ? a : b);
    
    const threatCount = Array.from(mlClassifications.values()).filter(c => c.threat_detected && c.confidence > 0.6).length;
    let recommendation;
    if (threatCount === 0) {
      recommendation = 'No ML-confirmed threats - Fastest route recommended';
    } else if (threatCount >= 3) {
      recommendation = 'Multiple threats confirmed - Safest route recommended';
    } else {
      recommendation = 'Some threats detected - Optimal route recommended';
    }

    return {
      analysis: `Fastest: ${fastest} | Safest: ${safest} | Best Score: ${highestScore}`,
      recommendation,
      threatConsideration: `ML Analysis: ${threatCount} confirmed threats`
    };
  };

  const analysis = analyzeRoutes();

  return (
    <div className="sidebar">
      <h3>🗺️ Route Options</h3>
      <div id="route-options">
        <div 
          className={`route-option optimal ${selectedRoute === 'optimal' ? 'selected' : ''}`}
          onClick={() => handleRouteSelection('optimal')}
        >
          <strong>🎯 Optimal Route</strong>
          <div className="route-metrics">
            <span className="metric">Distance: <span>{routes.optimal?.metrics.distance || '--'}km</span></span>
            <span className="metric">Risk: <span>{routes.optimal?.metrics.threatExposure || '--'}</span></span>
            <span className="metric">Score: <span>{routes.optimal?.metrics.score || '--'}</span></span>
          </div>
        </div>
        
        <div 
          className={`route-option fastest ${selectedRoute === 'fastest' ? 'selected' : ''}`}
          onClick={() => handleRouteSelection('fastest')}
        >
          <strong>⚡ Fastest Route</strong>
          <div className="route-metrics">
            <span className="metric">Distance: <span>{routes.fastest?.metrics.distance || '--'}km</span></span>
            <span className="metric">Risk: <span>{routes.fastest?.metrics.threatExposure || '--'}</span></span>
            <span className="metric">Score: <span>{routes.fastest?.metrics.score || '--'}</span></span>
          </div>
        </div>
        
        <div 
          className={`route-option safest ${selectedRoute === 'safest' ? 'selected' : ''}`}
          onClick={() => handleRouteSelection('safest')}
        >
          <strong>🛡️ Safest Route</strong>
          <div className="route-metrics">
            <span className="metric">Distance: <span>{routes.safest?.metrics.distance || '--'}km</span></span>
            <span className="metric">Risk: <span>{routes.safest?.metrics.threatExposure || '--'}</span></span>
            <span className="metric">Score: <span>{routes.safest?.metrics.score || '--'}</span></span>
          </div>
        </div>
      </div>
      
      <h3>📡 Path Analysis</h3>
      <div className="status-panel">
        <div className="status-item">
          {analysis?.analysis || 'Calculating multiple routes...'}
        </div>
        <div className="status-item">
          {analysis?.threatConsideration || 'ML Analysis: Pending'}
        </div>
        <div className="status-item">
          {analysis?.recommendation || 'Recommendation: Analyzing...'}
        </div>
      </div>
    </div>
  );
};

export default RoutePanel;