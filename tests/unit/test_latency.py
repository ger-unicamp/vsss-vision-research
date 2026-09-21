"""Cronometragem por estágio: warm-up, percentis e FPS."""
import math

import pytest

from vsss_vision.benchmark.latency import LatencyRecorder

_MS = 1_000_000  # ns por ms


def test_warmup_frames_are_excluded_from_measurement():
    recorder = LatencyRecorder(warmup_frames=2)

    for duration_ms in (100, 100, 10, 20):  # os dois primeiros são warm-up
        recorder.record("detect", duration_ms * _MS)
        recorder.end_frame()

    stats = recorder.stats("detect")
    assert stats.samples == 2
    assert stats.mean_ms == pytest.approx(15.0)
    assert recorder.frames == 2


def test_percentiles_expose_the_tail_the_mean_hides():
    """2% dos frames a 500 ms: a media fica em ~20 ms, o p99 denuncia o pico."""

    recorder = LatencyRecorder()
    for duration_ms in [10] * 98 + [500] * 2:
        recorder.record("total", duration_ms * _MS)
        recorder.end_frame()

    stats = recorder.stats("total")
    assert stats.mean_ms == pytest.approx(19.8)
    assert stats.p50_ms == pytest.approx(10.0)
    assert stats.p99_ms == pytest.approx(500.0)
    assert stats.max_ms == pytest.approx(500.0)


def test_stage_context_manager_records_a_sample():
    recorder = LatencyRecorder()
    with recorder.stage("convert"):
        sum(range(1000))
    recorder.end_frame()

    assert recorder.stats("convert").samples == 1


def test_throughput_is_derived_from_mean_total():
    recorder = LatencyRecorder()
    for _ in range(10):
        recorder.record("total", 20 * _MS)  # 20 ms -> 50 FPS
        recorder.end_frame()

    assert recorder.throughput_fps() == pytest.approx(50.0)


def test_unmeasured_stage_reports_nothing_instead_of_failing():
    recorder = LatencyRecorder()
    assert recorder.stats("detect") is None
    assert math.isnan(recorder.throughput_fps())
