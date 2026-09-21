import json
import os
import cv2
import zmq
import numpy as np
import logging
import threading
import time
from vsss_vision.camera.camera_bridge import CameraBridge
from vsss_vision.camera.recorte import Recorte
from vsss_vision.camera.tuning import CameraTuner

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger("camera_service")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

class CameraService:
    def __init__(self):
        self.config = self._load_config()
        self.bridge = CameraBridge(self.config.get("camera_index"))
        
        # Initialize hardware settings from config
        settings = self.config.get("settings", {})
        self.bridge.set_property(cv2.CAP_PROP_BRIGHTNESS, settings.get("brightness", 128))
        self.bridge.set_property(cv2.CAP_PROP_CONTRAST, settings.get("contrast", 128))
        self.bridge.set_property(cv2.CAP_PROP_SATURATION, settings.get("saturation", 128))
        self.bridge.set_property(cv2.CAP_PROP_EXPOSURE, settings.get("exposure", -4))
        self.bridge.set_property(cv2.CAP_PROP_FOCUS, settings.get("focus", 0))

        self.recorte = Recorte(
            src_points=[(p["x"], p["y"]) for p in self.config.get("roi_points", [])],
            dst_resolution=(
                self.config.get("output_resolution", {}).get("width", 640),
                self.config.get("output_resolution", {}).get("height", 480)
            )
        )

        # ZMQ Setup
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind("tcp://*:5555")
        
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind("tcp://*:5556")

        self.tuner = CameraTuner(self.bridge, self.pub)
        self.running = True

    def _load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _save_config(self):
        config = {
            "camera_index": self.bridge.index,
            "roi_points": [{"x": int(p[0]), "y": int(p[1])} for p in self.recorte.src_points] if self.recorte.src_points else [],
            "output_resolution": {"width": self.recorte.dst_resolution[0], "height": self.recorte.dst_resolution[1]},
            "settings": {
                "brightness": self.bridge.get_property(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.bridge.get_property(cv2.CAP_PROP_CONTRAST),
                "saturation": self.bridge.get_property(cv2.CAP_PROP_SATURATION),
                "exposure": self.bridge.get_property(cv2.CAP_PROP_EXPOSURE),
                "focus": self.bridge.get_property(cv2.CAP_PROP_FOCUS),
            }
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved")

    def auto_configure(self, frame=None):
        return self.tuner.start()

    def _process_tuning(self, frame):
        self.tuner.process_frame(frame, self.recorte)

    def handle_commands(self):
        while self.running:
            try:
                msg = self.rep.recv_json()
                cmd = msg.get("cmd")
                
                response = {"status": "ok"}
                
                if cmd == "set_roi":
                    points = msg.get("points")
                    if self.recorte.set_points(points):
                        response["status"] = "success"
                    else:
                        response["status"] = "error"
                
                elif cmd == "set_property":
                    prop = msg.get("prop")
                    val = msg.get("val")
                    self.bridge.set_property(prop, val)
                
                elif cmd == "save":
                    self._save_config()
                
                elif cmd == "reset_roi":
                    self.recorte.reset()
                
                elif cmd == "auto_config":
                    response = self.auto_configure()
                
                elif cmd == "get_status":
                    response["settings"] = {
                        "brightness": self.bridge.get_property(cv2.CAP_PROP_BRIGHTNESS),
                        "contrast": self.bridge.get_property(cv2.CAP_PROP_CONTRAST),
                        "saturation": self.bridge.get_property(cv2.CAP_PROP_SATURATION),
                        "exposure": self.bridge.get_property(cv2.CAP_PROP_EXPOSURE),
                        "focus": self.bridge.get_property(cv2.CAP_PROP_FOCUS),
                    }
                    response["roi"] = self.recorte.src_points

                self.rep.send_json(response)
            except Exception as e:
                logger.error(f"Error handling command: {e}")
                try:
                    self.rep.send_json({"status": "error", "message": str(e)})
                except:
                    pass

    def run(self):
        # Start command handler thread
        cmd_thread = threading.Thread(target=self.handle_commands, daemon=True)
        cmd_thread.start()

        logger.info("Camera service started")
        try:
            while self.running:
                frame = self.bridge.get_frame()
                if frame is None:
                    continue
                
                # Process tuning and get metrics
                metrics = self.tuner.process_frame(frame, self.recorte)
                if metrics:
                    self.pub.send_multipart([b"metrics", json.dumps(metrics).encode()])
                
                # 1. Publish RAW frame
                _, raw_buffer = cv2.imencode('.jpg', frame)
                self.pub.send_multipart([b"raw", raw_buffer])
                
                # 2. Process and publish cropped frame
                cropped = self.recorte.apply(frame)
                _, cropped_buffer = cv2.imencode('.jpg', cropped)
                self.pub.send_multipart([b"cropped", cropped_buffer])
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.bridge.release()
            self.ctx.destroy()

if __name__ == "__main__":
    service = CameraService()
    service.run()
