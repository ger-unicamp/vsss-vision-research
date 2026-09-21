import cv2
import numpy as np
import logging
import time
import json

logger = logging.getLogger("camera_tuner")

class CameraTuner:
    def __init__(self, bridge, pub):
        self.bridge = bridge
        self.pub = pub
        
        self.active = False
        self.phase = None  # "focus", "exposure", "contrast"
        self.start_time = 0
        self.stability_counter = 0
        
        self.STABILITY_THRESHOLD = 30
        self.MAX_TUNING_TIME = 15.0
        
        # Target values
        self.TARGET_MEAN = 127.5
        self.MEAN_TOLERANCE = 5.5 # 122 to 133
        self.TARGET_STD = 55
        self.STD_TOLERANCE = 5 # 50 to 60

    def start(self):
        logger.info("Starting sequential auto-tuning: Focus -> Exposure -> Contrast")
        
        # 1. Reset to neutral values for determinism
        self.bridge.set_property(cv2.CAP_PROP_BRIGHTNESS, 128)
        self.bridge.set_property(cv2.CAP_PROP_CONTRAST, 128)
        self.bridge.set_property(cv2.CAP_PROP_SATURATION, 128)
        
        # 2. Hardware Setup
        # Focus: Auto
        self.bridge.set_property(cv2.CAP_PROP_AUTOFOCUS, 1)
        # Exposure: Manual (MUST be manual for manual value adjustments to work)
        self.bridge.set_property(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        self.active = True
        self.phase = "focus"
        self.start_time = time.time()
        self.stability_counter = 0
        
        return {"status": "tuning_started"}

    def process_frame(self, frame, recorte):
        # Analysis frame (use cropped if available)
        analysis_frame = frame
        if recorte and recorte.matrix is not None:
            analysis_frame = recorte.apply(frame)
            
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        std = np.std(gray)
        
        metrics = {"mean": mean, "std": std, "phase": self.phase}
        
        if not self.active:
            return metrics

        elapsed = time.time() - self.start_time
        if elapsed > self.MAX_TUNING_TIME:
            logger.warning("Auto-tuning timed out. Locking current settings.")
            self._finalize()
            return metrics
        
        # We already calculated mean and std above
        # (Removed duplication of calculation)
        
        if self.phase == "focus":
            # Hardware is handling auto-focus.
            if elapsed > 2.0:
                logger.info("Focus phase complete. Moving to exposure.")
                self.phase = "exposure"
                self.stability_counter = 0
                metrics["phase"] = "exposure"
                
        elif self.phase == "exposure":
            curr_exp = self.bridge.get_property(cv2.CAP_PROP_EXPOSURE)
            
            changed = False
            if mean < (self.TARGET_MEAN - self.MEAN_TOLERANCE):
                # Image too dark -> increase exposure
                self.bridge.set_property(cv2.CAP_PROP_EXPOSURE, curr_exp + 1)
                changed = True
            elif mean > (self.TARGET_MEAN + self.MEAN_TOLERANCE):
                # Image too bright -> decrease exposure
                self.bridge.set_property(cv2.CAP_PROP_EXPOSURE, curr_exp - 1)
                changed = True
            
            if changed:
                logger.debug(f"Exposure tuning: mean={mean:.1f}, exp={curr_exp} -> {'+' if mean < (self.TARGET_MEAN - self.MEAN_TOLERANCE) else '-'}")
                self.stability_counter = 0
            else:
                self.stability_counter += 1
                
            if self.stability_counter >= self.STABILITY_THRESHOLD:
                logger.info("Exposure stabilized. Moving to contrast.")
                self.phase = "contrast"
                self.stability_counter = 0
                metrics["phase"] = "contrast"
                
        elif self.phase == "contrast":
            curr_contrast = self.bridge.get_property(cv2.CAP_PROP_CONTRAST)
            
            changed = False
            if std < (self.TARGET_STD - self.STD_TOLERANCE):
                self.bridge.set_property(cv2.CAP_PROP_CONTRAST, curr_contrast + 1)
                changed = True
            elif std > (self.TARGET_STD + self.STD_TOLERANCE):
                self.bridge.set_property(cv2.CAP_PROP_CONTRAST, curr_contrast - 1)
                changed = True
            
            if not changed:
                self.stability_counter += 1
            else:
                self.stability_counter = 0
                
            if self.stability_counter >= self.STABILITY_THRESHOLD:
                logger.info("Contrast stabilized. Tuning complete.")
                self._finalize()
        
        return metrics

    def _finalize(self):
        # Switch back to manual to lock everything
        self.bridge.set_property(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.bridge.set_property(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        self.active = False
        self.phase = None
        
        final_settings = {
            "brightness": self.bridge.get_property(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.bridge.get_property(cv2.CAP_PROP_CONTRAST),
            "exposure": self.bridge.get_property(cv2.CAP_PROP_EXPOSURE),
            "focus": self.bridge.get_property(cv2.CAP_PROP_FOCUS),
            "saturation": self.bridge.get_property(cv2.CAP_PROP_SATURATION),
        }
        self.pub.send_multipart([b"settings", json.dumps(final_settings).encode()])
        logger.info(f"Auto-configuration complete: {final_settings}")
