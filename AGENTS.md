# Guia rápido para agentes

**Repositório de pesquisa**, não de competição. Derivado de `src/vision/` e
`src/camera/` de [`ger-unicamp/futebol-vsss`](https://github.com/ger-unicamp/futebol-vsss).
Objetivo: produzir números publicáveis comparando detectores de robôs/bola em
VSSS. Ver `README.md` e `docs/research/`.

Python 3.11+, gerenciado com `uv`. **Nunca usar `pip`.**

## Comandos

```bash
make sync                 # uv sync --extra dev
make test                 # uv run pytest
make bench                # avaliacao offline sobre datasets/vision
make compare              # varios detectores, mesmas cenas
make camera-service       # precisa estar rodando para `live` e calibracao
make live                 # pipeline ao vivo, latencia instrumentada
make calibrate-camera     # ROI de 4 pontos
make calibrate-colors     # faixas HSV
```

## Arquitetura

```
camera/      aquisicao + warpPerspective + auto-tuning  ──ZMQ "cropped" (5555)──┐
vision/detectors/  ColorDetector | YoloDetector (stub)  → DetectionResult (px) ←┘
vision/tracker.py  suavizacao EMA (media circular no angulo)
vision/geometry.py px → m, FieldState                   ← ponto unico de escala
benchmark/         dataset + metrics + latency + runner
vision/live.py     mesmo pipeline sobre a camera, sink configuravel
cli.py             vsss-vision bench | compare | live
```

**Ponto de extensão**: `vision/detectors/base.py`. Entra frame retificado (BGR),
sai `DetectionResult` em pixel. Registrar implementações novas em
`vision/detectors/__init__.py::_REGISTRY`.

## Invariantes (não quebrar sem motivo declarado)

- **Métricas em cm e graus, nunca em pixel.** Erro em pixel não é comparável
  entre resoluções nem entre montagens — foi a limitação central do trabalho de
  referência.
- **Ângulo é tratado de forma circular** (`geometry.angular_difference`,
  `tracker._ema_angle`). Subtração crua reporta ~360 graus na virada 359°→1°.
- **Associação predição↔ground truth é por posição, ignorando o `robot_id`**;
  os ids são comparados depois. Associar por id zeraria a métrica de trocas de
  identidade, que é justamente um resultado esperado.
- **Nada de `vssproto`/protobuf aqui.** O pipeline termina em `FieldState`. Para
  fechar o laço com um robô, acrescentar um *sink* em `vision/live.py`.
- **A mesma instrumentação de latência** roda offline e ao vivo. Não duplicar.
- **O pipeline clássico deve estar bem calibrado** em qualquer comparação. Uma
  linha de base fraca invalida o resultado.

## Convenções

- Docstrings e comentários em **português**, explicando o *porquê* — o motivo de
  uma decisão, não o que o código faz.
- Código, nomes e mensagens de erro em português quando já forem assim; não
  traduzir termos técnicos consagrados.
- Toda métrica nova precisa de teste com um caso onde ela falharia se
  implementada de forma ingênua (ver `tests/unit/test_metrics.py`).

## Pegadinhas

- `vsss-vision live` exige `make camera-service` rodando antes.
- `YoloDetector` levanta `NotImplementedError` de propósito; roteiro de
  implementação no topo de `vision/detectors/yolo.py`.
- Dataset: mídia sem JSON irmão é **silenciosamente ignorada**. Se o `bench`
  disser "nenhuma cena anotada", o JSON provavelmente está faltando ou com nome
  diferente da mídia.
- `experiments/results/` e pesos de modelo (`*.pt`, `*.onnx`) estão no
  `.gitignore`.
