import React from 'react';

const MissionLog = ({ missionLog }) => {
  return (
    <div className="status-panel" style={{ height: '150px', overflowY: 'auto', fontSize: '0.75rem' }}>
      {missionLog.map((entry, index) => (
        <div key={index}>{entry}</div>
      ))}
    </div>
  );
};

export default MissionLog;