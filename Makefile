SHELL := /usr/bin/env bash
UV ?= uv
DATASET ?= datasets/vision

.PHONY: sync test bench compare sweep live camera-service calibrate-colors calibrate-camera \
	export-yolo train-yolo train-yolo-sem-augmentation

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

# sweep: curva precisão x latência (proposta P2). MODELS precisa apontar para
# os pesos de cada variante treinada; o hardware tem de ser o mesmo durante
# toda a varredura, senão a curva mistura dois eixos.
MODELS ?= experiments/training/yolov8n-pose_brightness_640/weights/best.pt
IMGSZ ?= 320 480 640
sweep:
	$(UV) run vsss-vision sweep --models $(MODELS) --imgsz $(IMGSZ) --include-color \
		--csv experiments/results/resumo.csv --markdown experiments/results/tabela.md

# --- Detector por rede neural -------------------------------------------
# Exporta as anotações para o formato Ultralytics (caixas derivadas das
# dimensões físicas; divisão por arquivo de origem, não por frame).
export-yolo:
	$(UV) run python -m tools.export_yolo_dataset --out datasets/yolo --clean

# Dois braços de treino: com e sem augmentation fotométrico. A diferença
# entre eles é um resultado da proposta P1, não uma preferência.
YOLO_MODEL ?= yolov8n-pose.pt
train-yolo:
	$(UV) run python -m tools.train_yolo --data datasets/yolo/data.yaml --model $(YOLO_MODEL) --augment brightness

train-yolo-sem-augmentation:
	$(UV) run python -m tools.train_yolo --data datasets/yolo/data.yaml --model $(YOLO_MODEL) --augment none

# --- Aquisição e calibração ---------------------------------------------
camera-service:
	$(UV) run python -m vsss_vision.camera.service

calibrate-camera:
	$(UV) run python tools/camera_configurator.py

calibrate-colors:
	$(UV) run python -m tools.vision_calibrator
