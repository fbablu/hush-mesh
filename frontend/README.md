# Maritime ACPS Dashboard - React Version

This React application provides two dashboard views for the Maritime Autonomous Convoy Protection System (ACPS):

## Features

### Original Dashboard
- Basic convoy protection interface
- Real-time threat detection
- WebSocket connectivity for live updates
- Interactive map with drag-and-drop threat/obstacle placement

### Enhanced Multi-Route Dashboard
- **Advanced ML-powered threat detection**
- **Multi-route planning with 3 strategies:**
  - 🎯 **Optimal Route**: Best balance of speed and safety
  - ⚡ **Fastest Route**: Shortest distance with minimal avoidance
  - 🛡️ **Safest Route**: Maximum threat avoidance
- **Real-time object classification** using ML models
- **Dynamic threat behavior simulation**
- **Interactive route selection**
- **Live mission logging**

## Components Structure

```
src/
├── EnhancedMultiRoute.jsx          # Main enhanced dashboard
├── EnhancedMultiRoute.css          # Styling for enhanced dashboard
├── AppRouter.jsx                   # Navigation between dashboards
├── components/
│   ├── Sidebar.jsx                 # Left panel (system status, metrics, ML detections)
│   ├── MapContainer.jsx            # Central map with convoy, objects, and routes
│   ├── RoutePanel.jsx              # Right panel (route options and analysis)
│   ├── Controls.jsx                # Bottom controls (play/pause/speed)
│   └── MissionLog.jsx              # Mission log component
├── App.jsx                         # Original dashboard
└── App.css                         # Original dashboard styling
```

## Getting Started

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   ```

3. **Access the application:**
   - Open http://localhost:5173
   - Use the navigation buttons to switch between dashboards

## Enhanced Dashboard Features

### ML Threat Detection
- Objects are dynamically detected as the convoy moves
- ML classification determines threat vs benign objects
- Visual indicators show classification accuracy
- Real-time confidence scoring

### Route Planning
- Three simultaneous route calculations
- Dynamic recalculation based on ML-confirmed threats
- Interactive route selection with live switching
- Comprehensive route metrics (distance, risk, score)

### Object Behavior Simulation
- **Intercept**: Aggressive objects that move toward convoy
- **Patrol**: Objects following patrol patterns
- **Drift**: Objects with random movement
- **Expand**: Growing threat zones (debris fields)

### Controls
- ▶️ Start/⏸️ Pause/⏹️ Stop demo
- 🔄 Reset to initial state
- 🧠 Recalculate routes
- Speed controls (1x, 2x, 5x, 10x)

## Styling Approach

The enhanced dashboard maintains the maritime theme with:
- Dark blue/black background simulating ocean
- Bright accent colors for visibility
- Grid overlay for navigation reference
- Glowing effects for active elements
- Smooth animations for object movement

## API Integration

The dashboard is designed to integrate with:
- ML prediction service at `http://localhost:5000/predict`
- WebSocket connections for real-time updates
- RESTful APIs for mission control

## Customization

You can easily customize:
- Object types and behaviors in `marineObjects` array
- Route calculation algorithms in `generateRoute()`
- ML classification logic in `classifyObjectWithML()`
- Visual styling in `EnhancedMultiRoute.css`