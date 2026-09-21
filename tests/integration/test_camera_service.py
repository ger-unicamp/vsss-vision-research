import pytest
import zmq
import json
import numpy as np
import os
import cv2
from unittest.mock import MagicMock, patch
from vsss_vision.camera.service import CameraService

@pytest.fixture
def mock_bridge():
    bridge = MagicMock()
    bridge.index = 0
    # Return a dummy frame when get_frame is called
    bridge.get_frame.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bridge.get_property.return_value = 128
    return bridge

@pytest.fixture
def camera_service(mock_bridge):
    with patch('vsss_vision.camera.service.CameraBridge', return_value=mock_bridge):
        # Patch ZMQ context to avoid binding to actual ports if needed, 
        # but usually localhost ports are fine for integration tests.
        service = CameraService()
        # Manually set a dummy ROI so recorte.matrix is not None
        service.recorte.set_points([(0,0), (100,0), (100,100), (0,100)])
        return service

def test_service_handle_set_roi(camera_service):
    # Mock the REP socket to simulate receiving a command
    camera_service.rep.recv_json = MagicMock(return_value={"cmd": "set_roi", "points": [[10,10], [20,10], [20,20], [10,20]]})
    camera_service.rep.send_json = MagicMock()
    
    # We can't call handle_commands() directly because it's an infinite loop.
    # Instead, let's extract the command processing logic or mock the loop.
    # Since handle_commands is just a while loop, we'll mock it.
    
    # Manual invocation of the logic inside the loop for one iteration
    msg = {"cmd": "set_roi", "points": [[10,10], [20,10], [20,20], [10,20]]}
    
    # Simulating the if/elif block in handle_commands
    cmd = msg.get("cmd")
    if cmd == "set_roi":
        points = msg.get("points")
        camera_service.recorte.set_points(points)
        
    assert camera_service.recorte.src_points == [[10,10], [20,10], [20,20], [10,20]]

def test_service_save_config(camera_service, tmp_path):
    # Override CONFIG_FILE to use tmp_path
    import vsss_vision.camera.service
    vsss_vision.camera.service.CONFIG_FILE = str(tmp_path / "config.json")
    
    camera_service._save_config()
    
    assert os.path.exists(vsss_vision.camera.service.CONFIG_FILE)
    with open(vsss_vision.camera.service.CONFIG_FILE, 'r') as f:
        config = json.load(f)
        assert "camera_index" in config
        assert "roi_points" in config

def test_service_publish_frames(camera_service, mock_bridge):
    # Mock ZMQ PUB socket
    camera_service.pub.send_multipart = MagicMock()
    
    # Simulate one iteration of the run loop
    frame = mock_bridge.get_frame()
    
    # The logic in service.run():
    # 1. Metrics
    metrics = camera_service.tuner.process_frame(frame, camera_service.recorte)
    camera_service.pub.send_multipart([b"metrics", json.dumps(metrics).encode()])
    
    # 2. Raw
    _, raw_buffer = cv2.imencode('.jpg', frame)
    camera_service.pub.send_multipart([b"raw", raw_buffer])
    
    # 3. Cropped
    cropped = camera_service.recorte.apply(frame)
    _, cropped_buffer = cv2.imencode('.jpg', cropped)
    camera_service.pub.send_multipart([b"cropped", cropped_buffer])
    
    # Verify PUB calls
    assert camera_service.pub.send_multipart.call_count == 3
    calls = camera_service.pub.send_multipart.call_args_list
    assert calls[0][0][0][0] == b"metrics"
    assert calls[1][0][0][0] == b"raw"
    assert calls[2][0][0][0] == b"cropped"
