"""Camada de avaliação: é o que diferencia este repositório do de competição.

O pipeline de jogo só precisa produzir posições. Um experimento precisa
produzir *números comparáveis*: erro em centímetros e graus (não em
pixel), taxa de detecções perdidas, trocas de identidade, e latência por
estágio com percentis. Estes módulos existem para isso.

  `dataset.py`  — carrega cenas anotadas (imagem/vídeo + ground truth JSON).
  `metrics.py`  — associa predição a ground truth e calcula os erros.
  `latency.py`  — cronometragem por estágio do pipeline, com percentis.
  `runner.py`   — roda um detector sobre um dataset e junta métricas + latência.
"""
