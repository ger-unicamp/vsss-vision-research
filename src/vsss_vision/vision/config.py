"""Configuração da visão, persistida num único arquivo JSON.

Tudo o que muda entre montagens (faixas de cor, dimensões do campo,
parâmetros de suavização) vive aqui e é serializável — o que importa num
repositório de pesquisa: a configuração usada é parte do resultado, e
`benchmark/runner.py` grava um recorte dela junto de cada execução.

Em relação ao repositório de competição, os campos de saída
(`output_host`/`output_port`, endereço multicast da estratégia) foram
removidos: aqui o pipeline termina em `FieldState`, não num pacote de
rede (ver `geometry.py`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

CONFIG_PATH = Path(__file__).parent / "vision_config.json"


@dataclass
class ColorRange:
    """Faixa HSV (H: 0-179, S/V: 0-255, convenção do OpenCV)."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class VisionConfig:
    """Configuração completa do pipeline de visão."""

    detector: Literal["color", "yolo"] = "color"
    team_color: Literal["yellow", "blue"] = "yellow"
    robots_per_team: int = 3

    # Entrada do modo ao vivo: ZMQ SUB no serviço de câmera, tópico "cropped".
    camera_sub_address: str = "tcp://localhost:5555"

    # Dimensões físicas do campo VSSS oficial (regra da categoria), em metros.
    # É esta escala que converte pixel em centímetro — se o campo medido em
    # bancada divergir, corrigir aqui antes de reportar qualquer erro.
    field_length_m: float = 1.50
    field_width_m: float = 1.30

    # Suavização temporal (EMA) — ver tracker.py.
    alpha_pos: float = 0.5
    alpha_angle: float = 0.4
    stale_timeout_s: float = 1.0

    # Área mínima de contorno (px^2) para considerar uma detecção válida.
    min_area_robot: float = 100.0
    min_area_ball: float = 50.0
    min_area_marker: float = 20.0
    # Distância máxima (px) entre um marcador e o corpo mais próximo para
    # associá-los ao mesmo robô — evita casar o marcador de um robô com o
    # corpo de outro quando estão espalhados pelo campo.
    max_marker_body_distance_px: float = 80.0

    # Caminho do modelo do detector por rede neural (`detector: "yolo"`).
    # Fora do git por tamanho — ver .gitignore.
    yolo_model_path: str = "models/best.pt"
    yolo_confidence: float = 0.25

    # Pontos de partida. TODOS devem ser recalibrados por câmera e por
    # condição de iluminação com tools/vision_calibrator.py: o custo dessa
    # recalibração é, ele próprio, uma das variáveis medidas na proposta P1.
    own_color: ColorRange = field(default_factory=lambda: ColorRange((90, 35, 0), (125, 255, 255)))
    opponent_color: ColorRange = field(default_factory=lambda: ColorRange((0, 120, 60), (10, 255, 255)))
    ball_color: ColorRange = field(default_factory=lambda: ColorRange((0, 40, 200), (35, 255, 225)))
    markers: list[ColorRange] = field(default_factory=lambda: [
        ColorRange((70, 50, 185), (100, 110, 230)),
        ColorRange((160, 55, 210), (180, 97, 255)),
        ColorRange((27, 16, 230), (39, 69, 255)),
    ])


def default_config() -> VisionConfig:
    """Configuração padrão (sem nenhuma calibração salva ainda)."""

    return VisionConfig()


def _config_to_dict(config: VisionConfig) -> dict[str, Any]:
    return asdict(config)


def _dict_to_config(data: dict[str, Any]) -> VisionConfig:
    color_fields = ("own_color", "opponent_color", "ball_color")
    kwargs = dict(data)
    for name in color_fields:
        if name in kwargs and kwargs[name] is not None:
            kwargs[name] = ColorRange(**kwargs[name])
    if "markers" in kwargs:
        kwargs["markers"] = [ColorRange(**marker) for marker in kwargs["markers"]]
    return VisionConfig(**kwargs)


def load_config(path: Path = CONFIG_PATH) -> VisionConfig:
    """Carrega `vision_config.json`; se ausente ou inválido, cria com defaults."""

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _dict_to_config(data)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError):
        config = default_config()
        save_config(config, path)
        return config


def save_config(config: VisionConfig, path: Path = CONFIG_PATH) -> None:
    """Persiste a configuração completa num único arquivo JSON."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_config_to_dict(config), handle, indent=2, ensure_ascii=False)
