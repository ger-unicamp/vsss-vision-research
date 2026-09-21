import cv2
import logging

logger = logging.getLogger(__name__)

class CameraBridge:
    def __init__(self, preferred_index=None):
        self.cap = None
        self.index = self._find_best_camera(preferred_index)
        self._open_camera()

    def _find_best_camera(self, preferred_index):
        if preferred_index is not None:
            # Try the preferred index first
            if self._is_camera_available(preferred_index):
                return preferred_index

        best_index = None
        max_fps = 0

        # Test first 5 indices to find the best non-integrated camera
        # Integrated camera is usually index 0
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"Camera {i} found with FPS: {fps}")
                if fps > max_fps:
                    max_fps = fps
                    best_index = i
                cap.release()

        if best_index is None:
            logger.warning("No external camera found, falling back to index 0")
            return 0
        
        return best_index

    def _is_camera_available(self, index):
        cap = cv2.VideoCapture(index)
        available = cap.isOpened()
        cap.release()
        return available

    def _open_camera(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.index}")
        logger.info(f"Camera opened successfully at index {self.index}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame from camera")
            return None
        return frame

    def set_property(self, prop_id, value):
        return self.cap.set(prop_id, value)

    def get_property(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        if self.cap:
            self.cap.release()
