"""Implementações plugáveis de `Detector` e a fábrica que as seleciona.

O contrato está em `base.py`: entrada é um frame retificado (BGR), saída é
`DetectionResult` em pixel. Todo o resto do pipeline (suavização,
conversão para metros, métricas, latência) é comum às implementações —
é isso que permite comparar abordagens sem que a comparação seja
contaminada por diferenças de infraestrutura entre elas.

  `ColorDetector` — segmentação HSV + contornos. Linha de base clássica.
  `YoloDetector`  — detecção por rede neural. Ainda não implementado.
"""
from __future__ import annotations

from vsss_vision.vision.config import VisionConfig
from vsss_vision.vision.detectors.base import DetectedBall, DetectedRobot, Detector, DetectionResult
from vsss_vision.vision.detectors.color import ColorDetector
from vsss_vision.vision.detectors.yolo import YoloDetector

__all__ = [
    "ColorDetector",
    "DetectedBall",
    "DetectedRobot",
    "DetectionResult",
    "Detector",
    "YoloDetector",
    "available_detectors",
    "get_detector",
]

_REGISTRY: dict[str, type[Detector]] = {
    "color": ColorDetector,
    "yolo": YoloDetector,
}


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)


def get_detector(config: VisionConfig, name: str | None = None) -> Detector:
    """Instancia o detector pedido (`name`) ou o de `config.detector`."""

    key = name or config.detector
    try:
        detector_class = _REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"Detector desconhecido: {key!r}. Disponíveis: {', '.join(available_detectors())}"
        ) from None
    return detector_class(config)
