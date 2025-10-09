import React, { useState } from 'react';
import App from './App';
import EnhancedMultiRoute from './EnhancedMultiRoute';
import './App.css';

const AppRouter = () => {
  const [currentView, setCurrentView] = useState('enhanced'); // 'original' or 'enhanced'

  return (
    <div>
      {/* Navigation Header */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        background: 'rgba(0,0,0,0.9)',
        padding: '0.5rem 1rem',
        borderBottom: '1px solid #ffff00',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h2 style={{ color: '#ffff00', margin: 0, fontSize: '1.2rem' }}>
          Maritime ACPS Dashboard
        </h2>
        <div>
          <button
            onClick={() => setCurrentView('original')}
            style={{
              background: currentView === 'original' ? '#ffff00' : '#333',
              color: currentView === 'original' ? '#000' : '#ffff00',
              border: '1px solid #ffff00',
              padding: '0.5rem 1rem',
              marginRight: '0.5rem',
              cursor: 'pointer',
              borderRadius: '4px'
            }}
          >
            Original Dashboard
          </button>
          <button
            onClick={() => setCurrentView('enhanced')}
            style={{
              background: currentView === 'enhanced' ? '#00ff88' : '#333',
              color: currentView === 'enhanced' ? '#000' : '#00ff88',
              border: '1px solid #00ff88',
              padding: '0.5rem 1rem',
              cursor: 'pointer',
              borderRadius: '4px'
            }}
          >
            Enhanced Multi-Route
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ paddingTop: currentView === 'original' ? '60px' : '0' }}>
        {currentView === 'original' ? <App /> : <EnhancedMultiRoute />}
      </div>
    </div>
  );
};

export default AppRouter;