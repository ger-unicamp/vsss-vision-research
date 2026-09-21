import pytest
import cv2
import numpy as np
import os
from unittest.mock import MagicMock
from vsss_vision.camera.tuning import CameraTuner

@pytest.fixture
def field_image():
    image_path = "Picture 2026-05-09 12-46-10.png"
    if not os.path.exists(image_path):
        return np.full((1080, 1920, 3), 127, dtype=np.uint8)
    return cv2.imread(image_path)

@pytest.fixture
def mock_bridge():
    bridge = MagicMock()
    # Remove side_effect to avoid conflicts with return_value in tests
    bridge.get_property.return_value = 128
    return bridge

@pytest.fixture
def mock_pub():
    return MagicMock()

@pytest.fixture
def tuner(mock_bridge, mock_pub):
    return CameraTuner(mock_bridge, mock_pub)

def test_tuner_start(tuner, mock_bridge):
    res = tuner.start()
    assert res["status"] == "tuning_started"
    assert tuner.active is True
    assert tuner.phase == "focus"
    # Check if neutral values were set
    mock_bridge.set_property.assert_any_call(cv2.CAP_PROP_BRIGHTNESS, 128)
    mock_bridge.set_property.assert_any_call(cv2.CAP_PROP_CONTRAST, 128)
    mock_bridge.set_property.assert_any_call(cv2.CAP_PROP_SATURATION, 128)

def test_tuner_exposure_too_dark(tuner, mock_bridge, field_image):
    tuner.start()
    tuner.phase = "exposure"
    
    # Simulate dark image: multiply by 0.5
    dark_image = (field_image * 0.5).astype(np.uint8)
    
    # Process frame
    metrics = tuner.process_frame(dark_image, recorte=None)
    
    # Mean should be low
    assert metrics["mean"] < tuner.TARGET_MEAN - tuner.MEAN_TOLERANCE
    # Bridge should be called to INCREASE exposure (curr_exp + 1)
    # We need to make sure get_property returns something consistent
    mock_bridge.get_property.return_value = -10
    
    tuner.process_frame(dark_image, recorte=None)
    mock_bridge.set_property.assert_called_with(cv2.CAP_PROP_EXPOSURE, -9)

def test_tuner_exposure_too_bright(tuner, mock_bridge, field_image):
    tuner.start()
    tuner.phase = "exposure"
    
    # Simulate bright image: add 100
    bright_image = np.clip(field_image.astype(np.int16) + 100, 0, 255).astype(np.uint8)
    
    mock_bridge.get_property.return_value = -10
    tuner.process_frame(bright_image, recorte=None)
    
    # Bridge should be called to DECREASE exposure (curr_exp - 1)
    mock_bridge.set_property.assert_called_with(cv2.CAP_PROP_EXPOSURE, -11)

def test_tuner_contrast_too_low(tuner, mock_bridge):
    tuner.start()
    tuner.phase = "contrast"
    
    # Low contrast image (greyish)
    low_contrast_image = np.full((480, 640, 3), 128, dtype=np.uint8)
    
    mock_bridge.get_property.return_value = 128
    tuner.process_frame(low_contrast_image, recorte=None)
    
    # Should increase contrast
    mock_bridge.set_property.assert_called_with(cv2.CAP_PROP_CONTRAST, 129)

def test_tuner_stability_transition(tuner, mock_bridge):
    tuner.start()
    tuner.phase = "exposure"
    
    # Image exactly in the target range
    target_image = np.full((480, 640, 3), 127, dtype=np.uint8)
    
    # Process for STABILITY_THRESHOLD frames
    for _ in range(tuner.STABILITY_THRESHOLD):
        tuner.process_frame(target_image, recorte=None)
        
    # Should transition to contrast phase
    assert tuner.phase == "contrast"

def test_tuner_timeout(tuner, field_image):
    tuner.start()
    tuner.start_time -= (tuner.MAX_TUNING_TIME + 1) # Simulate timeout
    
    tuner.process_frame(field_image, recorte=None)
    
    assert tuner.active is False
    assert tuner.phase is None
