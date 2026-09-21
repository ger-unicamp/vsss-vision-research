"""Calibrador interativo das faixas de cor do `ColorDetector`.

Não abre a câmera: consome o frame já retificado publicado por
`vsss_vision.camera.service` via ZMQ (`make camera-service`), a mesma
entrada que o detector recebe — calibrar sobre uma imagem diferente da
que será processada é uma fonte de erro silenciosa.

Relevante para a proposta P1: o **esforço de calibração** é uma variável
medida, não um detalhe de operação. Ao avaliar o protocolo de
recalibração por condição, cronometrar esta sessão e anotar o tempo junto
do resultado; ao avaliar o protocolo de calibração única, usar as faixas
salvas na condição de referência sem reabrir esta ferramenta.

Uso: `make calibrate-colors` (ou `uv run python -m tools.vision_calibrator`).

Teclas:
  1..N   seleciona o perfil de cor a editar (own, opponent, ball, marker_0..)
  s      salva todos os perfis em src/vsss_vision/vision/vision_config.json
  q/Esc  sai sem salvar (edições não salvas com 's' são perdidas)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import zmq

from vsss_vision.vision.config import ColorRange, VisionConfig, load_config, save_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision_calibrator")

CAMERA_SUB_ADDRESS = "tcp://localhost:5555"
CROPPED_TOPIC = b"cropped"
WINDOW_NAME = "Calibrador de Visao"

_TRACKBARS = ("H Min", "H Max", "S Min", "S Max", "V Min", "V Max")
_H_MAX, _SV_MAX = 179, 255


class VisionCalibrator:
    """Editor interativo de faixas HSV, salvo em um único `VisionConfig`."""

    def __init__(self) -> None:
        self.config = load_config()
        self.profiles = self._profile_names()
        self.active_profile = self.profiles[0]

        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(CAMERA_SUB_ADDRESS)
        self.sub.setsockopt(zmq.SUBSCRIBE, CROPPED_TOPIC)

        cv2.namedWindow(WINDOW_NAME)
        for name, max_val in zip(_TRACKBARS, (_H_MAX, _H_MAX, _SV_MAX, _SV_MAX, _SV_MAX, _SV_MAX)):
            cv2.createTrackbar(name, WINDOW_NAME, 0, max_val, lambda _val: None)
        self._sync_trackbars_from_profile()

    def _profile_names(self) -> list[str]:
        names = ["own", "opponent", "ball"]
        names.extend(f"marker_{i}" for i in range(self.config.robots_per_team))
        return names

    def _get_range(self, profile: str) -> ColorRange:
        if profile == "own":
            return self.config.own_color
        if profile == "opponent":
            return self.config.opponent_color
        if profile == "ball":
            return self.config.ball_color
        index = int(profile.split("_")[1])
        return self.config.markers[index]

    def _set_range(self, profile: str, color_range: ColorRange) -> None:
        if profile == "own":
            self.config.own_color = color_range
        elif profile == "opponent":
            self.config.opponent_color = color_range
        elif profile == "ball":
            self.config.ball_color = color_range
        else:
            index = int(profile.split("_")[1])
            self.config.markers[index] = color_range

    def _sync_trackbars_from_profile(self) -> None:
        color_range = self._get_range(self.active_profile)
        values = (
            color_range.lower[0], color_range.upper[0],
            color_range.lower[1], color_range.upper[1],
            color_range.lower[2], color_range.upper[2],
        )
        for name, value in zip(_TRACKBARS, values):
            cv2.setTrackbarPos(name, WINDOW_NAME, value)

    def _read_range_from_trackbars(self) -> ColorRange:
        h_min, h_max, s_min, s_max, v_min, v_max = (
            cv2.getTrackbarPos(name, WINDOW_NAME) for name in _TRACKBARS
        )
        return ColorRange(lower=(h_min, s_min, v_min), upper=(h_max, s_max, v_max))

    def _select_profile(self, profile: str) -> None:
        # Salva as edições do perfil atual antes de trocar, senão elas se
        # perdem ao reposicionar os trackbars para o próximo perfil.
        self._set_range(self.active_profile, self._read_range_from_trackbars())
        self.active_profile = profile
        self._sync_trackbars_from_profile()
        logger.info("Perfil ativo: %s", profile)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        legend = f"Perfil: {self.active_profile} | [1-{len(self.profiles)}] trocar | [S] salvar | [Q] sair"
        cv2.putText(frame, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return frame

    def run(self) -> None:
        try:
            while True:
                frame = self._poll_latest_frame()
                if frame is not None:
                    self._set_range(self.active_profile, self._read_range_from_trackbars())
                    color_range = self._get_range(self.active_profile)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array(color_range.lower), np.array(color_range.upper))

                    cv2.imshow(WINDOW_NAME, self._draw_overlay(frame.copy()))
                    cv2.imshow(f"{WINDOW_NAME} - Mascara", mask)

                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    save_config(self.config)
                    logger.info("Configuração salva em src/vsss_vision/vision/vision_config.json")
                elif ord("1") <= key < ord("1") + len(self.profiles):
                    self._select_profile(self.profiles[key - ord("1")])
        finally:
            cv2.destroyAllWindows()
            self.sub.close()
            self.ctx.destroy()

    def _poll_latest_frame(self) -> np.ndarray | None:
        """Drena o SUB para pegar o frame mais recente (descarta atraso de fila)."""

        frame = None
        try:
            while True:
                _topic, payload = self.sub.recv_multipart(flags=zmq.NOBLOCK)
                array = np.frombuffer(payload, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        except zmq.Again:
            pass
        return frame


if __name__ == "__main__":
    VisionCalibrator().run()
