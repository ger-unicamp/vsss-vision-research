"""Suavização (EMA) e persistência de estado entre frames.

Único ponto do módulo com estado — cada `Tracker` guarda a última posição
e ângulo suavizados de cada entidade (robô próprio, adversário, bola) e
os mantém disponíveis por um tempo mesmo quando a detecção falha num
frame (flicker de 1-2 frames não deve fazer a estratégia perder o robô).

Corrige um bug conhecido da versão antiga: `core/rastreador.py` suavizava
o ângulo linearmente (`alpha*prev + (1-alpha)*novo`), o que quebra perto
da transição 359°→0° (produz um salto de quase 360° em vez de um passo
pequeno). Aqui o ângulo é tratado como direção circular: decompõe-se em
vetor unitário `(cos θ, sin θ)`, suaviza-se cada componente com EMA
linear normal, e o ângulo final é `atan2` do vetor resultante — isso é
uma média circular correta.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from vsss_vision.vision.detectors.base import DetectedBall, DetectedRobot, DetectionResult

_BALL_KEY = "ball"


@dataclass
class _EntityState:
    x: float = 0.0
    y: float = 0.0
    angle_vx: float = 1.0
    angle_vy: float = 0.0
    has_angle: bool = False
    last_seen: float = field(default_factory=time.monotonic)


class Tracker:
    """Suaviza posição/ângulo por entidade e expira entidades não vistas."""

    def __init__(
        self,
        alpha_pos: float = 0.5,
        alpha_angle: float = 0.4,
        stale_timeout_s: float = 1.0,
    ) -> None:
        self._alpha_pos = alpha_pos
        self._alpha_angle = alpha_angle
        self._stale_timeout_s = stale_timeout_s
        self._states: dict[str, _EntityState] = {}

    def update(self, detection: DetectionResult) -> DetectionResult:
        """Suaviza a detecção bruta de um frame usando o histórico interno.

        Entidades vistas neste frame têm sua posição/ângulo atualizados via
        EMA. Entidades não vistas mas ainda dentro de `stale_timeout_s`
        continuam aparecendo no resultado com o último estado conhecido.
        Entidades expiradas são descartadas do estado interno (purge).
        """

        now = time.monotonic()
        seen_keys: set[str] = set()

        smoothed_robots: list[DetectedRobot] = []
        for robot in detection.robots:
            key = f"{robot.team}:{robot.robot_id}"
            seen_keys.add(key)
            x, y = self._ema_position(key, robot.x, robot.y, now)
            theta = self._ema_angle(key, robot.theta)
            smoothed_robots.append(DetectedRobot(robot_id=robot.robot_id, team=robot.team, x=x, y=y, theta=theta))

        smoothed_ball: DetectedBall | None = None
        if detection.ball is not None:
            seen_keys.add(_BALL_KEY)
            x, y = self._ema_position(_BALL_KEY, detection.ball.x, detection.ball.y, now)
            smoothed_ball = DetectedBall(x=x, y=y)

        self._append_stale_entities(seen_keys, now, smoothed_robots)
        if smoothed_ball is None:
            smoothed_ball = self._stale_ball(seen_keys, now)

        self._purge_stale(now)

        return DetectionResult(robots=smoothed_robots, ball=smoothed_ball, frame_shape=detection.frame_shape)

    def reset(self) -> None:
        """Limpa todo o estado (usado em testes / reinício de partida)."""

        self._states.clear()

    def _ema_position(self, key: str, x: float, y: float, now: float) -> tuple[float, float]:
        state = self._states.get(key)
        if state is None:
            state = _EntityState(x=x, y=y, last_seen=now)
            self._states[key] = state
            return x, y

        alpha = self._alpha_pos
        state.x = alpha * state.x + (1 - alpha) * x
        state.y = alpha * state.y + (1 - alpha) * y
        state.last_seen = now
        return state.x, state.y

    def _ema_angle(self, key: str, theta: float | None) -> float | None:
        # Nada a suavizar sem observação nova (ex.: adversário, que nunca
        # tem ângulo). Robôs próprios que perdem o marcador nem chegam
        # aqui — ficam de fora de detection.robots e são recuperados via
        # _append_stale_entities, que já usa o último ângulo suavizado.
        if theta is None:
            return None

        # _ema_position já roda antes de _ema_angle para a mesma chave em
        # update(), então o estado sempre existe neste ponto.
        state = self._states[key]
        vx, vy = math.cos(theta), math.sin(theta)
        if not state.has_angle:
            state.angle_vx, state.angle_vy, state.has_angle = vx, vy, True
        else:
            alpha = self._alpha_angle
            state.angle_vx = alpha * state.angle_vx + (1 - alpha) * vx
            state.angle_vy = alpha * state.angle_vy + (1 - alpha) * vy

        return math.atan2(state.angle_vy, state.angle_vx)

    def _append_stale_entities(
        self, seen_keys: set[str], now: float, out_robots: list[DetectedRobot]
    ) -> None:
        for key, state in self._states.items():
            if key in seen_keys or key == _BALL_KEY:
                continue
            if now - state.last_seen > self._stale_timeout_s:
                continue
            team, robot_id = key.split(":")
            theta = math.atan2(state.angle_vy, state.angle_vx) if state.has_angle else None
            out_robots.append(DetectedRobot(robot_id=int(robot_id), team=team, x=state.x, y=state.y, theta=theta))

    def _stale_ball(self, seen_keys: set[str], now: float) -> DetectedBall | None:
        if _BALL_KEY in seen_keys:
            return None
        state = self._states.get(_BALL_KEY)
        if state is None or now - state.last_seen > self._stale_timeout_s:
            return None
        return DetectedBall(x=state.x, y=state.y)

    def _purge_stale(self, now: float) -> None:
        expired = [key for key, state in self._states.items() if now - state.last_seen > self._stale_timeout_s]
        for key in expired:
            del self._states[key]
