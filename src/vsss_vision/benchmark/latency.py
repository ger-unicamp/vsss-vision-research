"""Instrumentação de latência por estágio do pipeline.

Motivação (proposta P2): o trabalho de referência processa apenas offline
e não mede FPS nem latência — logo não diz nada sobre o que acontece
quando o detector entra no laço de controle de um robô real. Aqui a
cronometragem é parte do pipeline, não um script à parte, e é a mesma nos
dois modos (offline sobre dataset e ao vivo sobre a câmera), para que os
números sejam comparáveis entre si.

Usa `time.perf_counter_ns`: monotônico e de alta resolução. `time.time()`
pode andar para trás com ajuste de relógio e tem resolução pior que o
estágio mais rápido que queremos medir.

Estágios convencionados (quem não se aplicar, simplesmente não é gravado):
  `capture`  — leitura do frame na câmera (ou do disco, no modo offline)
  `decode`   — decodificação JPEG do frame recebido por ZMQ
  `detect`   — `Detector.detect`
  `track`    — `Tracker.update`
  `convert`  — pixel -> metro (`geometry.to_field_state`)
  `publish`  — envio da saída (quando houver consumidor)
  `total`    — ponta a ponta, do frame disponível ao resultado pronto
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from vsss_vision.benchmark.metrics import percentile

_NS_PER_MS = 1_000_000.0


@dataclass
class StageStats:
    """Estatísticas de um estágio, em milissegundos."""

    stage: str
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "stage": self.stage,
            "samples": self.samples,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class LatencyRecorder:
    """Acumula durações por estágio e reporta percentis.

    Percentis importam mais que a média num laço de controle: um p99 de
    80 ms com média de 12 ms significa que o robô ocasionalmente age sobre
    um estado velho o bastante para ele já ter saído da posição — efeito
    que a média esconde completamente.
    """

    warmup_frames: int = 0
    _durations_ns: dict[str, list[int]] = field(default_factory=dict)
    _frames_seen: int = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Cronometra um bloco: `with recorder.stage("detect"): ...`"""

        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(name, time.perf_counter_ns() - started)

    def record(self, stage: str, duration_ns: int) -> None:
        """Grava uma duração já medida (em nanossegundos)."""

        if self._frames_seen < self.warmup_frames:
            return
        self._durations_ns.setdefault(stage, []).append(duration_ns)

    def end_frame(self) -> None:
        """Marca o fim de um frame — necessário para descartar os frames de warm-up.

        Os primeiros frames incluem custo único (alocação de buffers do
        OpenCV, JIT/carga de pesos de um modelo, primeiro acesso à câmera)
        que não representa o regime permanente. Chamar ao final de cada
        iteração do laço.
        """

        self._frames_seen += 1

    @property
    def frames(self) -> int:
        """Frames medidos (já descontado o warm-up)."""

        return max(0, self._frames_seen - self.warmup_frames)

    def stats(self, stage: str) -> StageStats | None:
        durations = self._durations_ns.get(stage)
        if not durations:
            return None
        values_ms = [duration / _NS_PER_MS for duration in durations]
        return StageStats(
            stage=stage,
            samples=len(values_ms),
            mean_ms=sum(values_ms) / len(values_ms),
            p50_ms=percentile(values_ms, 50),
            p95_ms=percentile(values_ms, 95),
            p99_ms=percentile(values_ms, 99),
            max_ms=max(values_ms),
        )

    def all_stats(self) -> list[StageStats]:
        stats = [self.stats(stage) for stage in self._durations_ns]
        return [entry for entry in stats if entry is not None]

    def throughput_fps(self, stage: str = "total") -> float:
        """FPS sustentável implícito pelo tempo médio de `stage`.

        É um teto, não uma medida de campo: ignora o tempo em que o
        pipeline fica ocioso esperando o próximo frame da câmera. Para o
        FPS real ao vivo, medir o intervalo entre frames entregues.
        """

        entry = self.stats(stage)
        if entry is None or entry.mean_ms <= 0:
            return float("nan")
        return 1000.0 / entry.mean_ms

    def as_dict(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "warmup_frames": self.warmup_frames,
            "throughput_fps": self.throughput_fps(),
            "stages": [entry.as_dict() for entry in self.all_stats()],
        }
