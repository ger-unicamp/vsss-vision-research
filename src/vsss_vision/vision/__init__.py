"""Pipeline de visão: frame retificado -> detecção -> suavização -> metros.

  `detectors/` — implementações plugáveis por trás de um contrato comum.
  `tracker.py` — suavização temporal (EMA, com média circular no ângulo).
  `geometry.py`— conversão pixel -> metro e o tipo `FieldState`.
  `live.py`    — o pipeline rodando sobre o feed da câmera, cronometrado.
  `config.py`  — a configuração, serializável junto de cada experimento.

O ponto de extensão é `detectors/base.py`: entra frame retificado (BGR),
sai `DetectionResult` em pixel. Tudo depois disso é comum a qualquer
detector, que é o que torna a comparação entre eles justa.
"""
