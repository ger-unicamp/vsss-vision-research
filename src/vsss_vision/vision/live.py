"""Pipeline ao vivo: câmera -> detector -> tracker -> estado do campo, cronometrado.

Existe por causa da proposta P2. O trabalho de referência processa apenas
arquivos, offline, e por isso não consegue dizer nada sobre latência em
jogo. Aqui o mesmo pipeline avaliado offline em `benchmark/runner.py` roda
sobre o feed real da câmera, com a **mesma** instrumentação de latência —
a comparação offline/online é, portanto, honesta.

Entrada: socket PUB de `vsss_vision.camera.service` (ZMQ, porta 5555),
tópico `cropped` (JPEG já retificado).

Saída: `FieldState` (metros). O transporte é escolhido por `--sink`:
  `none`  — nada; só mede o custo do pipeline de visão em si.
  `log`   — imprime o estado, para inspeção manual.
  `zmq`   — republica como JSON num PUB próprio, para que um consumidor
            (estratégia, gravador, visualizador) se conecte sem que este
            módulo precise conhecer o protocolo de ninguém.

O repositório de competição publica um protobuf `Environment` por UDP
multicast neste ponto; aqui isso foi deixado de fora de propósito (ver
`geometry.py`). Fechar o laço com um robô real — necessário para medir o
efeito da latência sobre o comportamento em jogo — é acrescentar um sink,
não mudar o pipeline.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from typing import Callable, Optional

import cv2
import numpy as np
import zmq

from vsss_vision.benchmark.latency import LatencyRecorder
from vsss_vision.vision.config import VisionConfig, load_config
from vsss_vision.vision.detectors import get_detector
from vsss_vision.vision.geometry import FieldState, to_field_state
from vsss_vision.vision.tracker import Tracker

logger = logging.getLogger("vsss_vision.live")

CROPPED_TOPIC = b"cropped"
STATE_TOPIC = b"field_state"
DEFAULT_STATE_PORT = 5557
_POLL_TIMEOUT_MS = 200

Sink = Callable[[FieldState], None]


def null_sink(_state: FieldState) -> None:
    """Descarta o resultado — isola o custo do pipeline do custo do transporte."""


def log_sink(state: FieldState) -> None:
    ball = f"({state.ball.x_m:+.3f}, {state.ball.y_m:+.3f})" if state.ball else "nao vista"
    robots = " ".join(
        f"{robot.team[0]}{robot.robot_id}=({robot.x_m:+.3f},{robot.y_m:+.3f})" for robot in state.robots
    )
    logger.info("bola=%s robos: %s", ball, robots or "nenhum")


def zmq_state_sink(port: int = DEFAULT_STATE_PORT) -> Sink:
    """Republica o estado como JSON num socket PUB próprio."""

    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{port}")
    logger.info("Estado do campo publicado em tcp://*:%d (topico %s)", port, STATE_TOPIC.decode())

    def sink(state: FieldState) -> None:
        payload = json.dumps(asdict(state)).encode("utf-8")
        publisher.send_multipart([STATE_TOPIC, payload])

    return sink


def build_sink(name: str, state_port: int = DEFAULT_STATE_PORT) -> Sink:
    if name == "none":
        return null_sink
    if name == "log":
        return log_sink
    if name == "zmq":
        return zmq_state_sink(state_port)
    raise ValueError(f"Sink desconhecido: {name!r} (use none, log ou zmq)")


def decode_frame(payload: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def run(
    config: Optional[VisionConfig] = None,
    detector_name: Optional[str] = None,
    sink: Sink | str = "none",
    shutdown_event: Optional[threading.Event] = None,
    report_every: int = 300,
    warmup_frames: int = 30,
) -> LatencyRecorder:
    """Roda o pipeline até `shutdown_event` ser sinalizado; devolve a latência medida.

    `warmup_frames` descarta o início da medição: os primeiros frames
    incluem alocação de buffers e carga de modelo, que não representam o
    regime permanente do laço de controle.
    """

    config = config or load_config()
    shutdown_event = shutdown_event or threading.Event()
    publish = build_sink(sink) if isinstance(sink, str) else sink

    detector = get_detector(config, detector_name)
    tracker = Tracker(
        alpha_pos=config.alpha_pos,
        alpha_angle=config.alpha_angle,
        stale_timeout_s=config.stale_timeout_s,
    )
    recorder = LatencyRecorder(warmup_frames=warmup_frames)
    detector.warmup()

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(config.camera_sub_address)
    subscriber.setsockopt(zmq.SUBSCRIBE, CROPPED_TOPIC)
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)

    logger.info(
        "Pipeline ao vivo iniciado (detector=%s, camera=%s)",
        detector_name or config.detector,
        config.camera_sub_address,
    )

    try:
        while not shutdown_event.is_set():
            if subscriber not in dict(poller.poll(timeout=_POLL_TIMEOUT_MS)):
                continue
            topic, payload = subscriber.recv_multipart()
            if topic != CROPPED_TOPIC:
                continue

            with recorder.stage("total"):
                with recorder.stage("decode"):
                    frame = decode_frame(payload)
                if frame is None:
                    continue
                with recorder.stage("detect"):
                    detection = detector.detect(frame)
                with recorder.stage("track"):
                    detection = tracker.update(detection)
                with recorder.stage("convert"):
                    state = to_field_state(detection, config)
                with recorder.stage("publish"):
                    publish(state)

            for stage, duration_ms in detector.profile().items():
                recorder.record(stage, int(duration_ms * 1_000_000))
            recorder.end_frame()
            if report_every and recorder.frames and recorder.frames % report_every == 0:
                _log_latency(recorder)
    finally:
        subscriber.close()
        context.destroy()
        _log_latency(recorder)
        logger.info("Pipeline ao vivo finalizado")

    return recorder


def _log_latency(recorder: LatencyRecorder) -> None:
    total = recorder.stats("total")
    if total is None:
        return
    logger.info(
        "latencia total: media=%.2f ms p95=%.2f ms p99=%.2f ms (%d frames, %.1f FPS)",
        total.mean_ms, total.p95_ms, total.p99_ms, total.samples, recorder.throughput_fps(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    run(sink="log")
