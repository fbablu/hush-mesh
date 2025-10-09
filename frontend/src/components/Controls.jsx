import React from 'react';

const Controls = ({ 
  running, 
  speed, 
  startDemo, 
  pauseDemo, 
  stopDemo, 
  resetDemo, 
  setSpeed, 
  recalculateRoutes 
}) => {
  
  return (
    <div className="controls">
      <button className="btn success" onClick={startDemo}>
        ▶️ Start Demo
      </button>
      <button className="btn" onClick={pauseDemo}>
        ⏸️ Pause
      </button>
      <button className="btn danger" onClick={stopDemo}>
        ⏹️ Stop
      </button>
      <button className="btn" onClick={resetDemo}>
        🔄 Reset
      </button>
      <button className="btn" onClick={recalculateRoutes}>
        🧠 Recalculate
      </button>
      <span style={{ marginLeft: '1rem', color: '#ccc' }}>Speed:</span>
      <button 
        className={`btn ${speed === 1 ? 'success' : ''}`} 
        onClick={() => setSpeed(1)}
      >
        1x
      </button>
      <button 
        className={`btn ${speed === 2 ? 'success' : ''}`} 
        onClick={() => setSpeed(2)}
      >
        2x
      </button>
      <button 
        className={`btn ${speed === 5 ? 'success' : ''}`} 
        onClick={() => setSpeed(5)}
      >
        5x
      </button>
      <button 
        className={`btn ${speed === 10 ? 'success' : ''}`} 
        onClick={() => setSpeed(10)}
      >
        10x
      </button>
    </div>
  );
};

export default Controls;