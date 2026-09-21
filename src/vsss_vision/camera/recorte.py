import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Recorte:
    def __init__(self, src_points=None, dst_resolution=(640, 480)):
        self.src_points = []
        self.dst_resolution = dst_resolution
        self.matrix = None
        
        if src_points:
            self.set_points(src_points)

    def set_points(self, points):
        if len(points) != 4:
            logger.warning("Recorte requires exactly 4 points")
            return False
        
        self.src_points = points
        self._update_matrix()
        return True

    def _update_matrix(self):
        src = np.float32(self.src_points)
        dst = np.float32([
            [0, 0],
            [self.dst_resolution[0], 0],
            [self.dst_resolution[0], self.dst_resolution[1]],
            [0, self.dst_resolution[1]]
        ])
        self.matrix = cv2.getPerspectiveTransform(src, dst)

    def apply(self, image):
        if self.matrix is None:
            return image
        
        return cv2.warpPerspective(image, self.matrix, self.dst_resolution)

    def reset(self):
        self.src_points = []
        self.matrix = None
