"""Execução de um detector sobre um dataset anotado, com métricas e latência.

É o laço central dos experimentos offline: para cada cena anotada, roda
detecção -> suavização -> conversão para metros, compara com o ground
truth e cronometra cada estágio. O resultado é um `RunResult`
serializável, para que uma execução possa ser versionada junto do artigo.

Sobre a suavização no modo offline: o `Tracker` mantém estado entre
frames, o que só faz sentido em sequências temporais (vídeo). Para um
conjunto de imagens independentes ele seria ruído — daí `use_tracker`
ser desligável, e o runner reiniciar o tracker sempre que a cena vem de
um arquivo de origem diferente.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vsss_vision.benchmark.dataset import Dataset, Scene, load_dataset
from vsss_vision.benchmark.latency import LatencyRecorder
from vsss_vision.benchmark.metrics import (
    DEFAULT_MATCH_RADIUS_CM,
    FrameMetrics,
    Summary,
    aggregate,
    evaluate_frame,
)
from vsss_vision.vision.config import VisionConfig, load_config
from vsss_vision.vision.detectors import get_detector
from vsss_vision.vision.geometry import to_field_state
from vsss_vision.vision.tracker import Tracker


@dataclass
class RunResult:
    """Saída completa de uma execução — o registro reprodutível do experimento."""

    detector: str
    dataset_root: str
    summary: Summary
    latency: dict
    per_frame: list[FrameMetrics] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)

    def by_illuminance(self) -> dict[Optional[float], Summary]:
        """Um resumo por nível de iluminância — a tabela principal da proposta P1."""

        grouped: dict[Optional[float], list[FrameMetrics]] = {}
        for frame in self.per_frame:
            grouped.setdefault(frame.illuminance_lux, []).append(frame)
        return {level: aggregate(frames) for level, frames in grouped.items()}

    def as_dict(self) -> dict:
        return {
            "detector": self.detector,
            "dataset_root": self.dataset_root,
            "summary": self.summary.as_dict(),
            "latency": self.latency,
            "by_illuminance": {
                ("unknown" if level is None else level): summary.as_dict()
                for level, summary in self.by_illuminance().items()
            },
            "config": self.config_snapshot,
            "per_frame": [
                {
                    "scene_id": frame.scene_id,
                    "illuminance_lux": frame.illuminance_lux,
                    "position_errors_cm": frame.position_errors_cm,
                    "orientation_errors_deg": frame.orientation_errors_deg,
                    "ball_error_cm": frame.ball_error_cm,
                    "ground_truth": frame.ground_truth_count,
                    "matched": frame.matched_count,
                    "missed": frame.missed_count,
                    "false_positives": frame.false_positive_count,
                    "id_swaps": frame.id_swap_count,
                    "ball_missed": frame.ball_missed,
                }
                for frame in self.per_frame
            ],
        }

    def write_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def run_benchmark(
    dataset: Dataset | Path | str,
    detector_name: Optional[str] = None,
    config: Optional[VisionConfig] = None,
    use_tracker: bool = True,
    match_radius_cm: float = DEFAULT_MATCH_RADIUS_CM,
    warmup_frames: int = 0,
) -> RunResult:
    """Roda um detector sobre todas as cenas anotadas e devolve métricas + latência."""

    config = config or load_config()
    if isinstance(dataset, (str, Path)):
        dataset = load_dataset(dataset, config.field_length_m, config.field_width_m)

    detector = get_detector(config, detector_name)
    tracker = Tracker(
        alpha_pos=config.alpha_pos,
        alpha_angle=config.alpha_angle,
        stale_timeout_s=config.stale_timeout_s,
    )
    recorder = LatencyRecorder(warmup_frames=warmup_frames)

    per_frame: list[FrameMetrics] = []
    previous_source = None

    for scene in dataset:
        # Cada arquivo de origem é uma sequência temporal independente; o
        # estado do tracker não pode vazar de uma para a outra.
        if use_tracker and scene.source != previous_source:
            tracker.reset()
        previous_source = scene.source

        per_frame.append(
            _evaluate_scene(scene, detector, tracker, config, recorder, use_tracker, match_radius_cm)
        )
        recorder.end_frame()

    return RunResult(
        detector=detector_name or config.detector,
        dataset_root=str(dataset.root),
        summary=aggregate(per_frame),
        latency=recorder.as_dict(),
        per_frame=per_frame,
        config_snapshot={
            "detector": detector_name or config.detector,
            "use_tracker": use_tracker,
            "match_radius_cm": match_radius_cm,
            "field_length_m": config.field_length_m,
            "field_width_m": config.field_width_m,
            "alpha_pos": config.alpha_pos,
            "alpha_angle": config.alpha_angle,
        },
    )


def _evaluate_scene(
    scene: Scene,
    detector,
    tracker: Tracker,
    config: VisionConfig,
    recorder: LatencyRecorder,
    use_tracker: bool,
    match_radius_cm: float,
) -> FrameMetrics:
    with recorder.stage("total"):
        with recorder.stage("detect"):
            detection = detector.detect(scene.frame)
        if use_tracker:
            with recorder.stage("track"):
                detection = tracker.update(detection)
        with recorder.stage("convert"):
            prediction = to_field_state(detection, config)

    return evaluate_frame(
        prediction=prediction,
        ground_truth=scene.ground_truth,
        scene_id=scene.scene_id,
        match_radius_cm=match_radius_cm,
        illuminance_lux=scene.conditions.illuminance_lux,
    )


def format_summary(result: RunResult) -> str:
    """Resumo legível no terminal — o JSON continua sendo a saída canônica."""

    summary = result.summary
    lines = [
        f"detector={result.detector}  dataset={result.dataset_root}  frames={summary.frames}",
        f"  erro de posicao     media={summary.position_error_mean_cm:.2f} cm  "
        f"p95={summary.position_error_p95_cm:.2f} cm  max={summary.position_error_max_cm:.2f} cm",
        f"  erro de orientacao  media={summary.orientation_error_mean_deg:.2f} graus  "
        f"p95={summary.orientation_error_p95_deg:.2f} graus",
        f"  erro da bola        media={summary.ball_error_mean_cm:.2f} cm",
        f"  deteccoes perdidas  {summary.miss_rate:.1%}   "
        f"trocas de id {summary.id_swap_rate:.1%}   falsos positivos {summary.false_positive_rate:.1%}",
    ]

    total = next((entry for entry in result.latency.get("stages", []) if entry["stage"] == "total"), None)
    if total:
        lines.append(
            f"  latencia (total)    media={total['mean_ms']:.2f} ms  p95={total['p95_ms']:.2f} ms  "
            f"p99={total['p99_ms']:.2f} ms  -> {result.latency['throughput_fps']:.1f} FPS"
        )

    by_level = result.by_illuminance()
    if len(by_level) > 1:
        lines.append("  por iluminancia:")
        for level in sorted(by_level, key=lambda value: (value is None, value)):
            entry = by_level[level]
            label = "desconhecida" if level is None else f"{level:g} lux"
            lines.append(
                f"    {label:>16}  n={entry.frames:<4} pos={entry.position_error_mean_cm:.2f} cm  "
                f"ori={entry.orientation_error_mean_deg:.2f} graus  perdidas={entry.miss_rate:.1%}"
            )

    return "\n".join(lines)
