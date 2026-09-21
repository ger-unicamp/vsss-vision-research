"""Exportação do dataset anotado para o formato YOLO (Ultralytics).

Motivação: as anotações deste repositório guardam **centros em metros e
ângulos**, não caixas em pixel. Isso é proposital — é o que as métricas em
escala real exigem, e anotar um centro é muito mais barato que desenhar
uma caixa. As caixas de treino são **derivadas** daí, usando as dimensões
físicas que as regras do VSSS fixam:

  robô  7,5 x 7,5 x 7,5 cm  (dimensão máxima por regra)
  bola  42,7 mm de diâmetro (bola de golfe)

Como o frame é retificado e cobre o campo inteiro, a escala px/m é
conhecida, então a caixa sai direto do tamanho físico. Consequência boa:
as caixas ficam consistentes entre todas as amostras, sem a variância de
anotador que caixas desenhadas à mão introduzem.

Divisão treino/validação/teste
------------------------------
A divisão é **por arquivo de origem**, nunca por frame. Frames vizinhos de
um mesmo vídeo são quase idênticos: dividir aleatoriamente coloca
quase-duplicatas dos dois lados e a métrica de teste mede memorização, não
generalização. É um risco concreto no trabalho de referência, que formou
653 imagens a partir de vídeos de um único evento e reportou mAP50-95 de
0,99 — número difícil de sustentar sem separação por origem. Ver
`docs/research/methodology.md`.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import cv2

from vsss_vision.benchmark.dataset import Dataset, Scene
from vsss_vision.vision.config import VisionConfig
from vsss_vision.vision.geometry import meters_to_pixel, pixels_per_meter

# Dimensões físicas fixadas pelas regras da categoria, em metros.
ROBOT_SIZE_M = 0.075
BALL_DIAMETER_M = 0.0427
# Folga na caixa do robô: a etiqueta de time e a sombra projetada ficam um
# pouco além do corpo, e uma caixa justa demais corta feature útil.
DEFAULT_BOX_MARGIN = 1.15

Task = Literal["detect", "pose"]


@dataclass
class ExportStats:
    images: dict[str, int]
    labels: dict[str, int]
    skipped_without_orientation: int = 0

    def as_text(self) -> str:
        lines = [f"  {split}: {count} imagens, {self.labels[split]} objetos" for split, count in self.images.items()]
        if self.skipped_without_orientation:
            lines.append(
                f"  {self.skipped_without_orientation} robos sem orientacao anotada "
                "exportados sem keypoints (visibilidade 0)"
            )
        return "\n".join(lines)


def class_names(config: VisionConfig) -> list[str]:
    """Nomes das classes, na ordem dos índices, a partir de `yolo_class_map`.

    O mapa da configuração é a fonte única: o modelo treinado e o detector
    em produção leem o mesmo arquivo, então não há como o treino e a
    inferência discordarem sobre o que é a classe 1.
    """

    return list(config.yolo_class_map)


def class_index(config: VisionConfig, kind: str, color: Optional[str] = None, robot_id: Optional[int] = None) -> Optional[int]:
    """Índice da classe correspondente a um papel semântico, ou None se não houver."""

    for index, (_name, role) in enumerate(config.yolo_class_map.items()):
        if role.get("kind") != kind:
            continue
        if kind == "ball":
            return index
        if role.get("color") != color:
            continue
        mapped_id = role.get("robot_id")
        if mapped_id is None or mapped_id == robot_id:
            return index
    return None


def team_color_of(team: str, config: VisionConfig) -> str:
    """`own`/`opponent` -> cor real do time, segundo `config.team_color`."""

    other = "blue" if config.team_color == "yellow" else "yellow"
    return config.team_color if team == "own" else other


def split_by_source(
    dataset: Dataset, ratios: tuple[float, float, float] = (0.7, 0.2, 0.1)
) -> dict[str, list[Scene]]:
    """Divide as cenas em treino/validação/teste **agrupando por arquivo de origem**.

    Arquivos inteiros vão para um único split. Com poucos arquivos a
    proporção sai aproximada — e é melhor assim do que exata com vazamento.
    """

    groups: dict[str, list[Scene]] = {}
    for scene in dataset:
        key = str(scene.source) if scene.source else scene.scene_id
        groups.setdefault(key, []).append(scene)

    # Maiores grupos primeiro, cada um para o split mais distante da sua
    # cota — reparte melhor que ordem alfabética quando há poucos arquivos.
    ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    total = sum(len(scenes) for _key, scenes in ordered)
    targets = {"train": ratios[0] * total, "val": ratios[1] * total, "test": ratios[2] * total}
    splits: dict[str, list[Scene]] = {"train": [], "val": [], "test": []}

    for _key, scenes in ordered:
        deficits = {name: targets[name] - len(splits[name]) for name in splits}
        chosen = max(deficits, key=lambda name: deficits[name])
        splits[chosen].extend(scenes)

    return splits


def export_dataset(
    dataset: Dataset,
    output_root: Path | str,
    config: VisionConfig,
    task: Task = "pose",
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    box_margin: float = DEFAULT_BOX_MARGIN,
    clean: bool = False,
) -> ExportStats:
    """Escreve o dataset em formato Ultralytics sob `output_root`."""

    output_root = Path(output_root)
    if clean and output_root.exists():
        shutil.rmtree(output_root)

    splits = split_by_source(dataset, ratios)
    stats = ExportStats(images={}, labels={})

    for split, scenes in splits.items():
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        label_count = 0
        for scene in scenes:
            stem = _safe_stem(scene.scene_id)
            cv2.imwrite(str(image_dir / f"{stem}.png"), scene.frame)
            lines, missing_orientation = _label_lines(scene, config, task, box_margin)
            stats.skipped_without_orientation += missing_orientation
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            label_count += len(lines)

        stats.images[split] = len(scenes)
        stats.labels[split] = label_count

    _write_data_yaml(output_root, config, task)
    return stats


def _label_lines(
    scene: Scene, config: VisionConfig, task: Task, box_margin: float
) -> tuple[list[str], int]:
    height, width = scene.frame.shape[:2]
    frame_shape = (height, width)
    scale_x, scale_y = pixels_per_meter(frame_shape, config.field_length_m, config.field_width_m)

    lines: list[str] = []
    missing_orientation = 0

    for robot in scene.ground_truth.robots:
        color = team_color_of(robot.team, config)
        index = class_index(config, "robot", color=color, robot_id=robot.robot_id)
        if index is None:
            continue

        center = meters_to_pixel(robot.x_m, robot.y_m, frame_shape, config.field_length_m, config.field_width_m)
        box_w = ROBOT_SIZE_M * box_margin * scale_x
        box_h = ROBOT_SIZE_M * box_margin * scale_y
        fields = [str(index), *_normalized_box(center, box_w, box_h, width, height)]

        if task == "pose":
            if robot.theta_rad is None:
                missing_orientation += 1
                # Keypoint com visibilidade 0: o Ultralytics ignora na perda
                # em vez de aprender um ponto errado na origem.
                fields += ["0", "0", "0", "0", "0", "0"]
            else:
                fields += _keypoint_fields(center, robot.theta_rad, scale_x, scale_y, width, height)

        lines.append(" ".join(fields))

    ball = scene.ground_truth.ball
    if ball is not None:
        index = class_index(config, "ball")
        if index is not None:
            center = meters_to_pixel(ball.x_m, ball.y_m, frame_shape, config.field_length_m, config.field_width_m)
            box_w = BALL_DIAMETER_M * box_margin * scale_x
            box_h = BALL_DIAMETER_M * box_margin * scale_y
            fields = [str(index), *_normalized_box(center, box_w, box_h, width, height)]
            if task == "pose":
                # A bola é homogênea: não tem frente nem trás. Os dois
                # keypoints ficam no centro, com visibilidade 0 — anotar um
                # eixo arbitrário, como fez o trabalho de referência,
                # ensina a rede a prever ruído.
                fields += ["0", "0", "0", "0", "0", "0"]
            lines.append(" ".join(fields))

    return lines, missing_orientation


def _keypoint_fields(
    center: tuple[float, float],
    theta_rad: float,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> list[str]:
    """Keypoints `front` e `back`, derivados do centro e do ângulo anotados.

    Meio corpo para cada lado, ao longo do eixo do robô. Y da imagem cresce
    para baixo, daí o sinal invertido em `dy` (mesma convenção de
    `geometry.py`).
    """

    half = ROBOT_SIZE_M / 2
    dx = math.cos(theta_rad) * half * scale_x
    dy = -math.sin(theta_rad) * half * scale_y

    fields: list[str] = []
    for sign in (1, -1):  # front, back
        x = _clamp((center[0] + sign * dx) / width)
        y = _clamp((center[1] + sign * dy) / height)
        fields += [f"{x:.6f}", f"{y:.6f}", "2"]  # 2 = visível
    return fields


def _normalized_box(
    center: tuple[float, float], box_w: float, box_h: float, width: int, height: int
) -> list[str]:
    return [
        f"{_clamp(center[0] / width):.6f}",
        f"{_clamp(center[1] / height):.6f}",
        f"{_clamp(box_w / width):.6f}",
        f"{_clamp(box_h / height):.6f}",
    ]


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _safe_stem(scene_id: str) -> str:
    return scene_id.replace("#", "_frame").replace("/", "_")


def _write_data_yaml(output_root: Path, config: VisionConfig, task: Task) -> Path:
    names = class_names(config)
    lines = [
        "# Gerado por vsss_vision.benchmark.yolo_export — nao editar a mao.",
        "# As classes vem de vision_config.json (yolo_class_map), para que",
        "# treino e inferencia nao possam discordar sobre o que e cada indice.",
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines += [f"  {index}: {name}" for index, name in enumerate(names)]
    if task == "pose":
        lines += [
            "kpt_shape: [2, 3]  # front, back (x, y, visibilidade)",
            "flip_idx: [0, 1]",
        ]

    path = output_root / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_splits(splits: dict[str, Sequence[Scene]]) -> str:
    return "\n".join(
        f"  {split}: {len(scenes)} cenas de {len({str(scene.source) for scene in scenes})} arquivos"
        for split, scenes in splits.items()
    )


def iter_sources(scenes: Iterable[Scene]) -> set[str]:
    return {str(scene.source) for scene in scenes if scene.source}
