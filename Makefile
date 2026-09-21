SHELL := /usr/bin/env bash
UV ?= uv
DATASET ?= datasets/vision

.PHONY: sync test bench compare live camera-service calibrate-colors calibrate-camera

sync:
	$(UV) sync --extra dev

test:
	$(UV) run pytest

# --- Experimentos --------------------------------------------------------
# bench: avaliação offline de um detector sobre o dataset anotado.
bench:
	$(UV) run vsss-vision bench --dataset $(DATASET) --out experiments/results/bench.json

# compare: mesma cena, mesma calibração, detectores diferentes — é esta
# comparação, e não o número absoluto de um detector, que sustenta o artigo.
compare:
	$(UV) run vsss-vision compare --dataset $(DATASET)

# live: pipeline sobre a câmera com latência instrumentada (exige camera-service).
live:
	$(UV) run vsss-vision live --sink log

# --- Aquisição e calibração ---------------------------------------------
camera-service:
	$(UV) run python -m vsss_vision.camera.service

calibrate-camera:
	$(UV) run python tools/camera_configurator.py

calibrate-colors:
	$(UV) run python -m tools.vision_calibrator
