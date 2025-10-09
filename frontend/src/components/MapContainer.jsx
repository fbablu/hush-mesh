import React from 'react';

const MapContainer = ({ 
  convoy, 
  destination, 
  marineObjects, 
  detectedObjects, 
  mlClassifications, 
  routes, 
  selectedRoute, 
  calculationInProgress 
}) => {
  
  const getObjectIcon = (type) => {
    switch (type) {
      case 'small_fast_craft': return '🚤';
      case 'floating_mine_like_object': return '💣';
      case 'submarine_periscope': return '🔭';
      case 'debris_field': return '🗑️';
      case 'fishing_vessel': return '🎣';
      case 'cargo_ship': return '🚢';
      case 'research_vessel': return '🔬';
      default: return '❓';
    }
  };

  const renderMarineObjects = () => {
    return marineObjects.map((obj, index) => {
      const isDetected = detectedObjects.has(index);
      const classification = mlClassifications.get(index);
      
      if (!isDetected) return null;

      let objectClass = 'object unknown';
      let showThreatZone = false;

      if (classification) {
        if (classification.threat_detected && classification.confidence > 0.6) {
          objectClass = 'object threat';
          showThreatZone = true;
        } else {
          objectClass = 'object benign';
        }
      }

      return (
        <React.Fragment key={index}>
          <div
            className={objectClass}
            style={{
              left: `${obj.x}px`,
              top: `${obj.y}px`,
              fontSize: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            title={obj.name}
          >
            {getObjectIcon(obj.type)}
          </div>
          
          {showThreatZone && (
            <div
              className="threat-zone"
              style={{
                left: `${obj.x - obj.radius}px`,
                top: `${obj.y - obj.radius}px`,
                width: `${obj.radius * 2}px`,
                height: `${obj.radius * 2}px`
              }}
            />
          )}
        </React.Fragment>
      );
    });
  };

  const renderRoutes = () => {
    if (!routes || Object.keys(routes).length === 0) return null;

    return Object.keys(routes).map(strategy => {
      const route = routes[strategy];
      if (!route || !route.waypoints) return null;

      let prevPoint = convoy;
      const opacity = selectedRoute === strategy ? '1' : '0.3';
      const zIndex = selectedRoute === strategy ? '5' : '3';

      return route.waypoints.map((wp, index) => {
        const dx = wp.x - prevPoint.x;
        const dy = wp.y - prevPoint.y;
        const length = Math.sqrt(dx*dx + dy*dy);
        const angle = Math.atan2(dy, dx) * 180 / Math.PI;

        const line = (
          <div
            key={`${strategy}-${index}`}
            className={`path-${strategy}`}
            style={{
              left: `${prevPoint.x}px`,
              top: `${prevPoint.y}px`,
              width: `${length}px`,
              transform: `rotate(${angle}deg)`,
              opacity,
              zIndex
            }}
          />
        );

        prevPoint = wp;
        return line;
      });
    });
  };

  return (
    <div className="map-container">
      <div className="scenario-indicator">
        <h4>Enhanced Multi-Route Planning</h4>
        <p>ML-powered threat detection active...</p>
      </div>
      
      {calculationInProgress && (
        <div className="calculation-status">
          🧠 Analyzing Routes...
        </div>
      )}
      
      <div className="map-canvas">
        <div className="grid-overlay"></div>
        
        {/* Convoy */}
        <div
          className="convoy"
          style={{
            left: `${convoy.x}px`,
            top: `${convoy.y}px`
          }}
        />
        
        {/* Destination */}
        <div
          className="destination"
          style={{
            left: `${destination.x}px`,
            top: `${destination.y}px`,
            fontSize: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          🎯
        </div>
        
        {/* Marine Objects */}
        {renderMarineObjects()}
        
        {/* Routes */}
        {renderRoutes()}
      </div>
    </div>
  );
};

export default MapContainer;