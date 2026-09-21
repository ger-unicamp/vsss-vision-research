"""Métricas de erro em escala real, associando predição a ground truth.

As quatro métricas reportadas são exatamente as do desenho experimental
(ver `docs/research/p1-illumination.md`):

  - **erro de posição (cm)** — distância euclidiana no plano do campo;
  - **erro de orientação (graus)** — diferença angular absoluta, calculada
    de forma circular (`geometry.angular_difference`), nunca por subtração
    crua, que reporta ~360 graus na transição 359 graus -> 1 grau;
  - **taxa de detecções perdidas** — entidade anotada sem nenhuma predição
    associada dentro do raio de casamento;
  - **taxa de identificações trocadas** — predição associada à entidade
    certa no espaço, mas com `robot_id` diferente do anotado.

Associação predição <-> ground truth
------------------------------------
Casamento guloso por distância, restrito ao mesmo time e a um raio máximo
(`match_radius_cm`), e **ignorando o `robot_id`** na hora de associar. Isso
é deliberado: se a associação usasse o id, uma troca de identidade
apareceria como uma detecção perdida somada a um falso positivo, e a
métrica de troca de identidade seria sempre zero. Associando por posição e
comparando os ids depois, cada erro é contado como o que de fato é.

O guloso é adequado aqui porque o campo VSSS tem no máximo 3 robôs por
time e o raio de casamento é da ordem do diâmetro do robô (8 cm), então os
candidatos raramente competem; para cenas mais densas, trocar por
atribuição ótima (Hungarian) seria o próximo passo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from vsss_vision.vision.geometry import FieldRobot, FieldState, angular_difference

# Raio padrão de casamento: um robô VSSS cabe num cubo de 7,5 cm, então
# 10 cm aceita o robô certo deslocado por ruído sem alcançar o vizinho.
DEFAULT_MATCH_RADIUS_CM = 10.0


@dataclass
class FrameMetrics:
    """Erros de um único frame. Agregar com `aggregate`."""

    scene_id: str
    position_errors_cm: list[float] = field(default_factory=list)
    orientation_errors_deg: list[float] = field(default_factory=list)
    ball_error_cm: Optional[float] = None
    ground_truth_count: int = 0
    matched_count: int = 0
    missed_count: int = 0
    false_positive_count: int = 0
    id_swap_count: int = 0
    ball_missed: bool = False
    illuminance_lux: Optional[float] = None


@dataclass
class Summary:
    """Agregado de todos os frames de uma execução — o que vai para a tabela do artigo."""

    frames: int = 0
    position_error_mean_cm: float = float("nan")
    position_error_p95_cm: float = float("nan")
    position_error_max_cm: float = float("nan")
    orientation_error_mean_deg: float = float("nan")
    orientation_error_p95_deg: float = float("nan")
    ball_error_mean_cm: float = float("nan")
    miss_rate: float = float("nan")
    id_swap_rate: float = float("nan")
    false_positive_rate: float = float("nan")
    ball_miss_rate: float = float("nan")
    # Precisão/revocação sobre a associação espacial, não sobre IoU de
    # caixas: o que importa para o laço de controle é se existe uma
    # estimativa utilizável na posição certa, não o quanto duas caixas se
    # sobrepõem. Reportar mAP ao lado disso é legítimo, mas mAP não diz em
    # quantos centímetros o robô vai errar.
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "frames": self.frames,
            "position_error_mean_cm": self.position_error_mean_cm,
            "position_error_p95_cm": self.position_error_p95_cm,
            "position_error_max_cm": self.position_error_max_cm,
            "orientation_error_mean_deg": self.orientation_error_mean_deg,
            "orientation_error_p95_deg": self.orientation_error_p95_deg,
            "ball_error_mean_cm": self.ball_error_mean_cm,
            "miss_rate": self.miss_rate,
            "id_swap_rate": self.id_swap_rate,
            "false_positive_rate": self.false_positive_rate,
            "ball_miss_rate": self.ball_miss_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def evaluate_frame(
    prediction: FieldState,
    ground_truth: FieldState,
    scene_id: str = "",
    match_radius_cm: float = DEFAULT_MATCH_RADIUS_CM,
    illuminance_lux: Optional[float] = None,
) -> FrameMetrics:
    """Compara um `FieldState` predito com o anotado."""

    metrics = FrameMetrics(scene_id=scene_id, illuminance_lux=illuminance_lux)
    metrics.ground_truth_count = len(ground_truth.robots)

    unmatched_predictions = list(prediction.robots)
    for truth in ground_truth.robots:
        match = _pop_closest(truth, unmatched_predictions, match_radius_cm)
        if match is None:
            metrics.missed_count += 1
            continue

        metrics.matched_count += 1
        metrics.position_errors_cm.append(_distance_cm(truth, match))
        if truth.robot_id != match.robot_id:
            metrics.id_swap_count += 1
        if truth.theta_rad is not None and match.theta_rad is not None:
            error_rad = abs(angular_difference(match.theta_rad, truth.theta_rad))
            metrics.orientation_errors_deg.append(math.degrees(error_rad))

    metrics.false_positive_count = len(unmatched_predictions)

    if ground_truth.ball is not None:
        if prediction.ball is None:
            metrics.ball_missed = True
        else:
            metrics.ball_error_cm = 100.0 * math.hypot(
                prediction.ball.x_m - ground_truth.ball.x_m,
                prediction.ball.y_m - ground_truth.ball.y_m,
            )

    return metrics


def aggregate(frames: Sequence[FrameMetrics]) -> Summary:
    """Consolida métricas por frame numa linha de resultado."""

    summary = Summary(frames=len(frames))
    if not frames:
        return summary

    position_errors = [error for frame in frames for error in frame.position_errors_cm]
    orientation_errors = [error for frame in frames for error in frame.orientation_errors_deg]
    ball_errors = [frame.ball_error_cm for frame in frames if frame.ball_error_cm is not None]

    total_truth = sum(frame.ground_truth_count for frame in frames)
    total_matched = sum(frame.matched_count for frame in frames)
    total_missed = sum(frame.missed_count for frame in frames)
    total_swaps = sum(frame.id_swap_count for frame in frames)
    total_false_positives = sum(frame.false_positive_count for frame in frames)
    ball_frames = [frame for frame in frames if frame.ball_missed or frame.ball_error_cm is not None]

    if position_errors:
        summary.position_error_mean_cm = _mean(position_errors)
        summary.position_error_p95_cm = percentile(position_errors, 95)
        summary.position_error_max_cm = max(position_errors)
    if orientation_errors:
        summary.orientation_error_mean_deg = _mean(orientation_errors)
        summary.orientation_error_p95_deg = percentile(orientation_errors, 95)
    if ball_errors:
        summary.ball_error_mean_cm = _mean(ball_errors)
    if total_truth:
        summary.miss_rate = total_missed / total_truth
        summary.false_positive_rate = total_false_positives / total_truth
    # Troca de identidade só faz sentido sobre o que foi de fato associado:
    # dividir pelo total anotado misturaria o efeito de detecções perdidas.
    if total_matched:
        summary.id_swap_rate = total_swaps / total_matched
    if total_matched + total_false_positives:
        summary.precision = total_matched / (total_matched + total_false_positives)
    if total_truth:
        summary.recall = total_matched / total_truth
    if summary.precision + summary.recall > 0 and not (
        math.isnan(summary.precision) or math.isnan(summary.recall)
    ):
        summary.f1 = 2 * summary.precision * summary.recall / (summary.precision + summary.recall)
    if ball_frames:
        summary.ball_miss_rate = sum(1 for frame in ball_frames if frame.ball_missed) / len(ball_frames)

    return summary


def percentile(values: Sequence[float], percent: float) -> float:
    """Percentil por interpolação linear. Evita puxar numpy só para isto."""

    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _distance_cm(a: FieldRobot, b: FieldRobot) -> float:
    return 100.0 * math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def _pop_closest(
    truth: FieldRobot, candidates: list[FieldRobot], match_radius_cm: float
) -> Optional[FieldRobot]:
    """Remove e devolve o candidato mais próximo do mesmo time dentro do raio."""

    best_index, best_distance = None, float("inf")
    for index, candidate in enumerate(candidates):
        if candidate.team != truth.team:
            continue
        distance = _distance_cm(truth, candidate)
        if distance < best_distance:
            best_index, best_distance = index, distance

    if best_index is None or best_distance > match_radius_cm:
        return None
    return candidates.pop(best_index)
