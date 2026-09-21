import cv2
import zmq
import numpy as np
import logging
import json
import socket
import subprocess
import time
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera_configurator")

# ZMQ and CV2 properties
REP_ADDR = "tcp://localhost:5556"
PUB_ADDR = "tcp://localhost:5555"

PROPS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "focus": cv2.CAP_PROP_FOCUS
}

class CameraConfigurator:
    def __init__(self):
        self.service_process = None
        self._ensure_service_running()
        
        self.ctx = zmq.Context()
        self.rep = self.ctx.socket(zmq.REQ)
        self.rep.connect(REP_ADDR)
        
        self.pub = self.ctx.socket(zmq.SUB)
        self.pub.connect(PUB_ADDR)
        self.pub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.metrics = {"mean": 0, "std": 0, "phase": "None"}
        self.points = []

        self.window_raw = "Camera RAW"
        self.window_cropped = "Camera Cropped"
        
        cv2.namedWindow(self.window_raw)
        cv2.namedWindow(self.window_cropped)
        
        # Create trackbars
        self.last_settings = {}
        for name in PROPS.keys():
            cv2.createTrackbar(name, self.window_raw, 0, 255, self._on_trackbar)

        cv2.setMouseCallback(self.window_raw, self._on_mouse)
        
        # Initialize sliders from service
        self._sync_sliders()

    def _ensure_service_running(self):
        if self._check_port(5556):
            logger.info("Camera service is already running.")
            return

        logger.info("Camera service not found. Starting in background...")
        try:
            # Use uv run to start the service as a module
            self.service_process = subprocess.Popen(
                [sys.executable, "-m", "vsss_vision.camera.service"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give the service a few seconds to initialize
            time.sleep(3)
            if not self._check_port(5556):
                logger.error("Failed to start camera service automatically.")
        except Exception as e:
            logger.error(f"Error starting camera service: {e}")

    def _check_port(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _send_cmd(self, cmd, data=None):
        msg = {"cmd": cmd}
        if data:
            msg.update(data)
        self.rep.send_json(msg)
        return self.rep.recv_json()

    def _on_trackbar(self, val):
        # We don't know which trackbar triggered this, so we'll just 
        # let the loop handle reading the current trackbar values 
        # and sending them to the service if they changed.
        pass

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                if len(self.points) == 4:
                    self._send_cmd("set_roi", {"points": self.points})
                logger.info(f"Point {len(self.points)} set: ({x}, {y})")

    def _sync_sliders(self):
        resp = self._send_cmd("get_status")
        if resp.get("status") == "ok":
            settings = resp.get("settings", {})
            for name, prop_id in PROPS.items():
                val = settings.get(name, 0)
                cv2.setTrackbarPos(name, self.window_raw, int(val))
            
            roi = resp.get("roi")
            if roi:
                self.points = list(roi)

    def run(self):
        try:
            while True:
                # 1. Get frames and settings from PUB
                try:
                    while True: # Drain the queue to get the latest data
                        topic, msg = self.pub.recv_multipart(flags=zmq.NOBLOCK)
                        
                        if topic == b"settings":
                            settings = json.loads(msg.decode())
                            for name, val in settings.items():
                                if name in PROPS:
                                    cv2.setTrackbarPos(name, self.window_raw, int(val))
                            self.last_settings.update(settings)
                            logger.info(f"Sliders updated from service: {settings}")
                        
                        elif topic == b"metrics":
                            self.metrics = json.loads(msg.decode())
                        
                        elif topic in [b"raw", b"cropped"]:
                            nparr = np.frombuffer(msg, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                if topic == b"raw":
                                    # Draw points and lines on RAW
                                    for i, p in enumerate(self.points):
                                        cv2.circle(frame, p, 5, (0, 255, 0), -1)
                                        cv2.putText(frame, str(i+1), (p[0]+10, p[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    if len(self.points) > 1:
                                        for i in range(len(self.points)):
                                            cv2.line(frame, self.points[i], self.points[(i+1)%len(self.points)], (0, 255, 0), 2)
                                    
                                    # Add legends
                                    cv2.putText(frame, "[Q] Sair | [S] Salvar | [R] Resetar ROI | [A] Auto-Tuning", 
                                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    cv2.imshow(self.window_raw, frame)
                                
                                elif topic == b"cropped":
                                    # Add legend and metrics
                                    cv2.putText(frame, "Imagem Recortada", 
                                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    metrics_text = f"Mean: {self.metrics['mean']:.1f} | Std: {self.metrics['std']:.1f} | Phase: {self.metrics['phase']}"
                                    cv2.putText(frame, metrics_text, 
                                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                                    
                                    cv2.imshow(self.window_cropped, frame)
                except zmq.Again:
                    pass

                # 2. Update hardware settings from sliders
                for name, prop_id in PROPS.items():
                    val = cv2.getTrackbarPos(name, self.window_raw)
                    if self.last_settings.get(name) != val:
                        self._send_cmd("set_property", {"prop": prop_id, "val": val})
                        self.last_settings[name] = val

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.points = []
                    self._send_cmd("reset_roi")
                elif key == ord('s'):
                    self._send_cmd("save")
                elif key == ord('a'):
                    resp = self._send_cmd("auto_config")
                    if resp.get("status") == "ok" and "settings" in resp:
                        settings = resp["settings"]
                        for name, val in settings.items():
                            if name in PROPS:
                                cv2.setTrackbarPos(name, self.window_raw, int(val))

        finally:
            if self.service_process:
                logger.info("Terminating camera service started by configurator...")
                self.service_process.terminate()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    configurator = CameraConfigurator()
    configurator.run()
