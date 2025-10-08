#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
from path_planner import PathPlanner
import json

app = Flask(__name__)
CORS(app)

planner = PathPlanner()

@app.route('/calculate_routes', methods=['POST'])
def calculate_routes():
    """Calculate multiple route alternatives"""
    try:
        data = request.json
        start_pos = (data['start_lat'], data['start_lon'])
        end_pos = (data['end_lat'], data['end_lon'])
        convoy_pos = (data['convoy_lat'], data['convoy_lon'])
        threats = data.get('threats', [])
        
        # Update threat grid
        planner.update_threats(threats, convoy_pos)
        
        # Calculate multiple paths
        routes = planner.find_multiple_paths(start_pos, end_pos, convoy_pos)
        
        return jsonify({
            'status': 'success',
            'routes': routes,
            'threat_count': len(threats)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evasive_maneuver', methods=['POST'])
def evasive_maneuver():
    """Generate immediate evasive maneuver"""
    try:
        data = request.json
        convoy_pos = (data['convoy_lat'], data['convoy_lon'])
        threat_bearing = data['threat_bearing']
        threat_distance = data['threat_distance']
        
        waypoints = planner.generate_evasive_maneuver(convoy_pos, threat_bearing, threat_distance)
        
        return jsonify({
            'status': 'success',
            'evasive_waypoints': waypoints
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)