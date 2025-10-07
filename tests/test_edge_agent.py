import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from edge.agent import EdgeAgent
from edge.yolo_infer import YOLOInference
from edge.camera_adapter import CameraAdapter

class TestEdgeAgent:
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict('os.environ', {
            'DRONE_ID': 'test-drone-01',
            'VEHICLE_ID': 'test-veh-01',
            'USE_MINIMAL': 'true'
        }):
            self.agent = EdgeAgent()
            
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.drone_id == 'test-drone-01'
        assert self.agent.vehicle_id == 'test-veh-01'
        assert self.agent.use_minimal == True
        assert self.agent.running == False
        
    def test_command_processing(self):
        """Test command queue processing"""
        # Add test commands
        commands = [
            {"command": "inspect_aoi", "aoi_id": "test-aoi"},
            {"command": "loiter", "duration_s": 10}
        ]
        
        for cmd in commands:
            self.agent.commands_queue.append(cmd)
            
        assert len(self.agent.commands_queue) == 2
        
    @pytest.mark.asyncio
    async def test_inspect_aoi_command(self):
        """Test AOI inspection command"""
        command = {
            "command": "inspect_aoi",
            "aoi_id": "test-aoi-01",
            "params": {"loiter_time_s": 5}
        }
        
        initial_status = self.agent.status
        
        # Mock the sleep to speed up test
        with patch('asyncio.sleep', new_callable=AsyncMock):
            await self.agent.execute_command(command)
            
        # Status should return to following after inspection
        assert self.agent.status == "following"
        
    @pytest.mark.asyncio
    async def test_loiter_command(self):
        """Test loiter command"""
        command = {
            "command": "loiter",
            "duration_s": 1
        }
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await self.agent.execute_command(command)
            mock_sleep.assert_called_with(1)
            
    def test_detection_processing(self):
        """Test detection data processing"""
        detection = {
            "bbox": [100, 100, 200, 200],
            "class": "person",
            "confidence": 0.85
        }
        
        # Mock MQTT client
        self.agent.mqtt_client = Mock()
        
        # Process detection
        asyncio.run(self.agent.process_detection(detection, 1))
        
        # Should have published to MQTT
        self.agent.mqtt_client.publish.assert_called_once()
        
        # Check published data
        call_args = self.agent.mqtt_client.publish.call_args
        topic = call_args[0][0]
        payload = json.loads(call_args[0][1])
        
        assert topic == f"acps/detections/{self.agent.drone_id}"
        assert payload["class"] == "person"
        assert payload["confidence"] == 0.85
        assert payload["drone_id"] == self.agent.drone_id
        
    def test_immediate_action_trigger(self):
        """Test immediate action triggering for high-threat detections"""
        # High threat detection
        high_threat = {
            "class": "weapon",
            "confidence": 0.9
        }
        
        assert self.agent.should_trigger_immediate_action(high_threat) == True
        
        # Low threat detection
        low_threat = {
            "class": "person", 
            "confidence": 0.5
        }
        
        assert self.agent.should_trigger_immediate_action(low_threat) == False
        
    def test_position_simulation(self):
        """Test position simulation"""
        initial_pos = self.agent.current_position.copy()
        
        # Get simulated position multiple times
        positions = []
        for _ in range(10):
            pos = self.agent.get_current_geo_position()
            positions.append(pos)
            
        # Positions should vary slightly (simulated movement)
        lats = [p["lat"] for p in positions]
        lons = [p["lon"] for p in positions]
        
        # Should have some variation
        assert max(lats) - min(lats) > 0
        assert max(lons) - min(lons) > 0
        
        # But should stay within reasonable bounds
        for pos in positions:
            assert abs(pos["lat"] - initial_pos["lat"]) < 0.001
            assert abs(pos["lon"] - initial_pos["lon"]) < 0.001
            
    def test_synthetic_detection_generation(self):
        """Test synthetic detection generation"""
        # Normal priority detection
        detection = self.agent.generate_synthetic_detection(high_priority=False)
        
        assert "bbox" in detection
        assert "class" in detection
        assert "confidence" in detection
        assert "priority" in detection
        assert detection["class"] in ["person", "car", "truck", "motorcycle"]
        assert 0.6 <= detection["confidence"] <= 0.95
        
        # High priority detection
        high_pri_detection = self.agent.generate_synthetic_detection(high_priority=True)
        
        assert high_pri_detection["class"] in ["person", "weapon", "fire"]
        assert high_pri_detection["priority"] >= 7

class TestYOLOInference:
    def setup_method(self):
        """Setup test fixtures"""
        self.yolo = YOLOInference(use_minimal=True)
        
    def test_initialization(self):
        """Test YOLO initialization"""
        assert self.yolo.use_minimal == True
        assert self.yolo.device == "cpu"
        assert self.yolo.confidence_threshold == 0.5
        
    def test_priority_calculation(self):
        """Test detection priority calculation"""
        # High priority class with high confidence
        priority = self.yolo.calculate_priority("person", 0.9)
        assert priority >= 8
        
        # Low priority class with low confidence
        priority = self.yolo.calculate_priority("bird", 0.6)
        assert priority <= 6
        
    @pytest.mark.asyncio
    async def test_synthetic_detection(self):
        """Test synthetic detection generation"""
        import numpy as np
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Get detections (should use synthetic mode)
        detections = await self.yolo.detect(frame)
        
        # Should return list of detections
        assert isinstance(detections, list)
        
        # Each detection should have required fields
        for detection in detections:
            assert "bbox" in detection
            assert "class" in detection
            assert "confidence" in detection
            assert "priority" in detection
            assert len(detection["bbox"]) == 4

class TestCameraAdapter:
    def setup_method(self):
        """Setup test fixtures"""
        self.camera = CameraAdapter("simulator", use_minimal=True)
        
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test camera initialization"""
        await self.camera.initialize()
        assert self.camera.source == "simulator"
        
    @pytest.mark.asyncio
    async def test_frame_capture(self):
        """Test frame capture"""
        await self.camera.initialize()
        
        frame = await self.camera.capture_frame()
        
        # Should return a valid frame
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == 'uint8'
        
    def test_synthetic_frame_generation(self):
        """Test synthetic frame generation"""
        frame = self.camera.generate_synthetic_frame()
        
        # Should be correct dimensions
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == 'uint8'
        
        # Should have road markings (white line in center)
        center_col = frame.shape[1] // 2
        white_pixels = frame[:, center_col-2:center_col+2, :]
        
        # Should have some white pixels from road marking
        assert (white_pixels > 200).any()
        
    def test_base64_conversion(self):
        """Test frame to base64 conversion"""
        frame = self.camera.generate_synthetic_frame()
        b64_string = self.camera.frame_to_base64(frame)
        
        # Should be valid base64 string
        assert isinstance(b64_string, str)
        assert len(b64_string) > 0
        
        # Should be decodable
        import base64
        try:
            decoded = base64.b64decode(b64_string)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Invalid base64 encoding")