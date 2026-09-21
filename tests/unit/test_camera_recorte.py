import pytest
import cv2
import numpy as np
import os
from vsss_vision.camera.recorte import Recorte

@pytest.fixture
def field_image():
    image_path = "Picture 2026-05-09 12-46-10.png"
    if not os.path.exists(image_path):
        # Fallback to a dummy image if the file is missing in some environment
        return np.zeros((1080, 1920, 3), dtype=np.uint8)
    return cv2.imread(image_path)

@pytest.fixture
def sample_roi():
    # Typical 4 points for a rectangular field perspective
    return [
        (100, 200), (1700, 200), 
        (1900, 1000), (100, 1000)
    ]

def test_recorte_initialization(sample_roi):
    dst_res = (640, 480)
    recorte = Recorte(src_points=sample_roi, dst_resolution=dst_res)
    assert recorte.src_points == sample_roi
    assert recorte.dst_resolution == dst_res
    assert recorte.matrix is not None

def test_recorte_apply_resolution(field_image, sample_roi):
    dst_res = (640, 480)
    recorte = Recorte(src_points=sample_roi, dst_resolution=dst_res)
    cropped = recorte.apply(field_image)
    
    assert cropped.shape[0] == dst_res[1] # height
    assert cropped.shape[1] == dst_res[0] # width

def test_recorte_set_points_valid(sample_roi):
    recorte = Recorte(src_points=[], dst_resolution=(640, 480))
    success = recorte.set_points(sample_roi)
    assert success is True
    assert recorte.src_points == sample_roi
    assert recorte.matrix is not None

def test_recorte_set_points_invalid():
    recorte = Recorte(src_points=[], dst_resolution=(640, 480))
    # Only 3 points instead of 4
    success = recorte.set_points([(0,0), (1,1), (2,2)])
    assert success is False
    assert recorte.matrix is None

def test_recorte_reset(sample_roi):
    recorte = Recorte(src_points=sample_roi, dst_resolution=(640, 480))
    recorte.reset()
    assert recorte.src_points == []
    assert recorte.matrix is None
