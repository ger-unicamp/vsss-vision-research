# Guia rápido para agentes

**Repositório de pesquisa**, não de competição. Derivado de `src/vision/` e
`src/camera/` de [`ger-unicamp/futebol-vsss`](https://github.com/ger-unicamp/futebol-vsss).
Objetivo: produzir números publicáveis comparando detectores de robôs/bola em
VSSS. Ver `README.md`, `docs/research/methodology.md` (o porquê de cada
decisão experimental) e `docs/research/related-work.md`.

Python 3.11+, gerenciado com `uv`. **Nunca usar `pip`.**

## Comandos

```bash
make sync                 # uv sync --extra dev
make test                 # uv run pytest
make bench                # avaliacao offline sobre datasets/vision
make compare              # varios detectores, mesmas cenas
make sweep                # curva precisao x latencia (P2)
make export-yolo          # anotacoes -> formato Ultralytics
make train-yolo           # fine-tuning, braco com augmentation
make train-yolo-sem-augmentation   # braco de controle
make camera-service       # precisa estar rodando para `live` e calibracao
make live                 # pipeline ao vivo, latencia instrumentada
make calibrate-camera     # ROI de 4 pontos
make calibrate-colors     # faixas HSV
```

## Arquitetura

```
camera/      aquisicao + warpPerspective + auto-tuning  ──ZMQ "cropped" (5555)──┐
vision/detectors/  ColorDetector | YoloDetector          → DetectionResult (px) ←┘
vision/identity.py atribui robot_id (marcador na caixa | posicional)
vision/tracker.py  suavizacao EMA (media circular no angulo)
vision/geometry.py px → m, FieldState                   ← ponto unico de escala
benchmark/         dataset + metrics + latency + runner + report + yolo_export
vision/live.py     mesmo pipeline sobre a camera, sink configuravel
cli.py             vsss-vision bench | compare | sweep | live
tools/             calibradores + export_yolo_dataset + train_yolo
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
  `detect` inclui pré-processamento e transferência para GPU; medir só o
  *forward* subestima o que o laço de controle espera.
- **O pipeline clássico deve estar bem calibrado** em qualquer comparação. Uma
  linha de base fraca invalida o resultado.
- **Classes do YOLO são por cor de time**, nunca por robô. As regras mandam a
  cor alternar entre partidas com etiqueta destacável; classes por robô
  prendem o modelo a uma equipe e não detectam adversário.
- **`yolo_class_map` é a fonte única de classes** — alimenta o exportador de
  dataset e o detector. Mapa que não casa com `model.names` levanta erro na
  construção, de propósito: silenciosamente não detectar nada viraria "100%
  de detecções perdidas" reportado como resultado.
- **Divisão de dataset por arquivo de origem**, nunca por frame. Frames
  vizinhos de um vídeo são quase duplicatas.
- **Caixas e keypoints de treino são derivados** das dimensões da regra
  (robô 7,5 cm, bola 42,7 mm), não anotados. Declarar isso ao reportar mAP.
- **Bola não tem orientação.** Keypoints da bola sempre com visibilidade 0.

## Convenções

- Docstrings e comentários em **português**, explicando o *porquê* — o motivo de
  uma decisão, não o que o código faz.
- Código, nomes e mensagens de erro em português quando já forem assim; não
  traduzir termos técnicos consagrados.
- Toda métrica nova precisa de teste com um caso onde ela falharia se
  implementada de forma ingênua (ver `tests/unit/test_metrics.py`).

## Pegadinhas

- `vsss-vision live` exige `make camera-service` rodando antes.
- `YoloDetector` exige `uv sync --extra yolo` e pesos treinados. Para teste,
  aceita um modelo injetado (`YoloDetector(config, model=...)`) — é assim que
  `tests/unit/test_yolo_detector.py` roda sem ultralytics, sem pesos e sem GPU.
- Keypoint ausente no Ultralytics vem como `(0, 0)`: tratar como "sem
  orientação", nunca como um ângulo apontando para o canto da imagem.
- Dataset: mídia sem JSON irmão é **silenciosamente ignorada**. Se o `bench`
  disser "nenhuma cena anotada", o JSON provavelmente está faltando ou com nome
  diferente da mídia.
- `experiments/results/` e pesos de modelo (`*.pt`, `*.onnx`) estão no
  `.gitignore`.
