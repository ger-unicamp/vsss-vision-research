"""Carregamento de cenas anotadas (imagem ou vídeo + ground truth JSON).

Esquema documentado em `datasets/vision/README.md`. Duas decisões
importantes em relação ao esquema herdado do repositório de competição:

1. **Unidades explícitas.** Cada bloco `vision` declara `units` (`"m"` ou
   `"px"`). Ground truth em pixel só é comparável dentro da mesma
   montagem de câmera e resolução; anotar em metros é o que permite
   reportar erro em centímetros e comparar com outros trabalhos. Quando
   `units == "px"`, a conversão usa a mesma escala do pipeline
   (`geometry.pixel_to_meters`), e o resultado herda a incerteza do ROI.
2. **Bloco `conditions`.** Iluminância medida (lux), tipo de luz e
   uniformidade. Sem isso não é possível agrupar resultados por condição
   de iluminação, que é a variável independente da proposta P1.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Optional

import cv2
import numpy as np

from vsss_vision.vision.geometry import FieldBall, FieldRobot, FieldState, pixel_to_meters

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


@dataclass(frozen=True)
class Conditions:
    """Condições de captura da cena — variáveis independentes dos experimentos."""

    illuminance_lux: Optional[float] = None
    light_type: Optional[Literal["cold", "warm", "mixed", "natural"]] = None
    uniform: Optional[bool] = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict | None) -> "Conditions":
        data = data or {}
        return cls(
            illuminance_lux=data.get("illuminance_lux"),
            light_type=data.get("light_type"),
            uniform=data.get("uniform"),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class Scene:
    """Um frame anotado: a imagem em si + o estado verdadeiro do campo."""

    scene_id: str
    frame: np.ndarray
    ground_truth: FieldState
    conditions: Conditions
    frame_index: int = 0
    source: Optional[Path] = None


@dataclass
class Dataset:
    """Coleção de cenas anotadas sob um diretório."""

    root: Path
    field_length_m: float = 1.50
    field_width_m: float = 1.30
    scenes: list[Scene] = field(default_factory=list)

    def __iter__(self) -> Iterator[Scene]:
        return iter(self.scenes)

    def __len__(self) -> int:
        return len(self.scenes)

    def by_illuminance(self) -> dict[Optional[float], list[Scene]]:
        """Agrupa cenas por iluminância medida — eixo X dos gráficos da proposta P1."""

        grouped: dict[Optional[float], list[Scene]] = {}
        for scene in self.scenes:
            grouped.setdefault(scene.conditions.illuminance_lux, []).append(scene)
        return grouped


def load_dataset(
    root: Path | str,
    field_length_m: float = 1.50,
    field_width_m: float = 1.30,
) -> Dataset:
    """Carrega todas as cenas anotadas sob `root`.

    Um arquivo de mídia sem JSON correspondente é **ignorado em
    silêncio**: sem ground truth não há nada a medir, e falhar aqui
    impediria manter mídia bruta ainda não anotada junto do dataset.
    """

    root = Path(root)
    dataset = Dataset(root=root, field_length_m=field_length_m, field_width_m=field_width_m)

    for media_path in sorted(root.rglob("*")):
        if media_path.suffix.lower() not in _IMAGE_SUFFIXES + _VIDEO_SUFFIXES:
            continue
        annotation_path = media_path.with_suffix(".json")
        if not annotation_path.exists():
            continue

        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        conditions = Conditions.from_dict(annotation.get("conditions"))

        if media_path.suffix.lower() in _IMAGE_SUFFIXES:
            dataset.scenes.extend(
                _load_image_scene(media_path, annotation, conditions, field_length_m, field_width_m)
            )
        else:
            dataset.scenes.extend(
                _load_video_scenes(media_path, annotation, conditions, field_length_m, field_width_m)
            )

    return dataset


def _load_image_scene(
    path: Path, annotation: dict, conditions: Conditions, field_length_m: float, field_width_m: float
) -> list[Scene]:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return []
    vision_block = annotation.get("vision")
    if not vision_block:
        return []
    ground_truth = parse_ground_truth(vision_block, frame.shape[:2], field_length_m, field_width_m)
    return [Scene(scene_id=path.stem, frame=frame, ground_truth=ground_truth, conditions=conditions, source=path)]


def _load_video_scenes(
    path: Path, annotation: dict, conditions: Conditions, field_length_m: float, field_width_m: float
) -> list[Scene]:
    """Extrai só os frames que têm anotação (`vision.frames`), não o vídeo inteiro."""

    frames_block = (annotation.get("vision") or {}).get("frames")
    if not frames_block:
        return []

    units = (annotation.get("vision") or {}).get("units", "m")
    wanted = {int(index): block for index, block in frames_block.items()}

    capture = cv2.VideoCapture(str(path))
    scenes: list[Scene] = []
    try:
        for frame_index in sorted(wanted):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            block = dict(wanted[frame_index])
            block.setdefault("units", units)
            ground_truth = parse_ground_truth(block, frame.shape[:2], field_length_m, field_width_m)
            scenes.append(
                Scene(
                    scene_id=f"{path.stem}#{frame_index}",
                    frame=frame,
                    ground_truth=ground_truth,
                    conditions=conditions,
                    frame_index=frame_index,
                    source=path,
                )
            )
    finally:
        capture.release()
    return scenes


def parse_ground_truth(
    block: dict,
    frame_shape: tuple[int, int],
    field_length_m: float,
    field_width_m: float,
) -> FieldState:
    """Converte um bloco `vision` anotado em `FieldState` (sempre em metros)."""

    units = block.get("units", "m")
    if units not in ("m", "px"):
        raise ValueError(f"units desconhecida no ground truth: {units!r} (esperado 'm' ou 'px')")

    def to_meters(x: float, y: float) -> tuple[float, float]:
        if units == "m":
            return float(x), float(y)
        return pixel_to_meters(float(x), float(y), frame_shape, field_length_m, field_width_m)

    robots: list[FieldRobot] = []
    for entry in block.get("robots", []):
        x_m, y_m = to_meters(entry["x"], entry["y"])
        theta = entry.get("theta_deg")
        theta_rad = math.radians(float(theta)) if theta is not None else entry.get("theta")
        robots.append(
            FieldRobot(
                robot_id=int(entry["id"]),
                team=entry.get("team", "own"),
                x_m=x_m,
                y_m=y_m,
                theta_rad=float(theta_rad) if theta_rad is not None else None,
            )
        )

    ball = None
    if block.get("ball") is not None:
        x_m, y_m = to_meters(block["ball"]["x"], block["ball"]["y"])
        ball = FieldBall(x_m=x_m, y_m=y_m)

    return FieldState(robots=robots, ball=ball)
