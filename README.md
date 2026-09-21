# vsss-vision-research

**Repositório de pesquisa.** Avaliação quantitativa de visão computacional para
Very Small Size Soccer (VSSS): segmentação por cor *versus* detecção por redes
neurais, medida em **escala real** (centímetros e graus, não pixels), sob
**condições de iluminação controladas e medidas**, e com **latência
instrumentada** ponta a ponta.

Este repositório **não** é o sistema de jogo do GER. Ele foi derivado do
subsistema de visão de [`ger-unicamp/futebol-vsss`](https://github.com/ger-unicamp/futebol-vsss)
e sua finalidade é produzir resultados publicáveis, não ganhar partidas. As duas
coisas puxam o código em direções diferentes: o sistema de jogo quer o menor
caminho até o pacote de rede da estratégia; um experimento quer instrumentação,
ground truth, reprodutibilidade e comparabilidade. Separar os repositórios evita
que uma finalidade degrade a outra.

---

## A lacuna que motiva o trabalho

O ponto de partida é o TCC de referência direto da área:

> **Marques, A. G. O. (2025).** *Detecção de objetos Very Small Size Soccer
> utilizando redes neurais convolucionais YOLO.* TCC, Ciência da Computação,
> UnB. Orientador: Prof. Marcus Vinicius Lamar.

O trabalho treinou YOLOv8-pose (nano a extra-large) com keypoints de frente e
trás de cada robô — o que permite estimar orientação — e comparou com o *Main
System* da equipe UnBall (Python/OpenCV, segmentação por faixas HSV),
concluindo que o YOLOv8x-pose teve menor erro de posição e de orientação que o
sistema clássico. Dataset e modelo são públicos (`ballnet_dataset` no Roboflow
Universe; `ayssag/BallnetPose` no Hugging Face) e servem de linha de base aqui.

As limitações declaradas ou observáveis nesse trabalho são exatamente as
aberturas que este repositório persegue:

| Limitação no trabalho de referência | Como este repositório ataca |
| :--- | :--- |
| Processamento apenas offline; sem detecção em tempo real, sem medição de FPS ou latência | `benchmark/latency.py` cronometra cada estágio; `vision/live.py` roda o **mesmo** pipeline sobre a câmera real com a mesma instrumentação |
| Métricas em pixels — o próprio autor aponta que não bastam para avaliar erro em escala real | Toda métrica é calculada em `FieldState` (metros), e reportada em cm e graus |
| Sem teste sob variação de iluminação e sem *data augmentation* | Proposta P1: iluminância medida com luxímetro é variável independente e vive no ground truth (`conditions`) |
| Base pequena e pouco diversa: 653 imagens, um único evento (IronCup 2020), camisas de uma única equipe | Dataset próprio anotado com as camisas do GER, versionado junto do esquema |
| Parte da avaliação qualitativa foi visual | Nenhuma métrica deste repositório depende de inspeção visual |
| Não avalia o efeito da visão sobre controle, estratégia ou desempenho em jogo | Proposta P2, ver abaixo |

> Antes de escrever qualquer artigo, fazer uma busca de literatura mais ampla
> (IEEE Xplore, SOL/SBC, anais do RoboCup) sobre visão com *deep learning* nas
> ligas SSL e VSSS, para posicionar cada contribuição. Ver
> [`docs/research/related-work.md`](docs/research/related-work.md).

---

## As duas propostas

### P1 — Robustez à iluminação

Primeira avaliação quantitativa, em escala real, de YOLO e de segmentação por
cor em VSSS sob condições de iluminação **medidas** e controladas, incluindo o
**esforço de calibração** de cada método.

O ponto não é "rede neural ganha". É medir a curva: a que nível de iluminância,
e com que tipo de luz, cada método se degrada — e quanto trabalho humano custa
manter o método clássico competitivo, já que numa competição real a recalibração
acontece entre partidas, sob pressão de tempo.

Protocolo completo em [`docs/research/p1-illumination.md`](docs/research/p1-illumination.md).

### P2 — Latência em jogo real

Curva de compromisso entre precisão e latência ponta a ponta, para o pipeline
clássico e para variantes do YOLO, em hardware fixo — com medição do efeito
sobre o comportamento do robô em jogo.

Sem esse último ponto, o artigo tende a ser visto como incremental: medir FPS é
fácil, mostrar que a latência muda o que o robô faz é a contribuição.

Protocolo completo em [`docs/research/p2-latency.md`](docs/research/p2-latency.md).

**Onde publicar:** SBAI 2027 (SBA — automação inteligente; aceita português) e
SBR 2027 (SBC — visão em robótica e futebol de robôs no escopo; revisão
duplo-cega, apresentação em inglês, presencial). Datas a confirmar.

---

## Métricas

Calculadas em `benchmark/metrics.py`, sempre sobre o campo em metros:

- **Erro de posição (cm)** — distância euclidiana até a posição anotada.
- **Erro de orientação (graus)** — diferença angular **circular**; a subtração
  crua reporta ~360 graus na transição 359°→1°.
- **Taxa de detecções perdidas** — entidade anotada sem predição associada.
- **Taxa de identificações trocadas** — predição na posição certa, `robot_id`
  errado.
- **Latência por estágio (ms)** — média, p50, p95, p99 e máximo. Percentis
  importam mais que a média num laço de controle: p99 de 80 ms com média de
  12 ms significa que o robô ocasionalmente age sobre um estado velho demais.

A associação predição↔ground truth é feita **por posição, ignorando o id**, e os
ids são comparados depois. Se a associação usasse o id, uma troca de identidade
apareceria como "detecção perdida + falso positivo" e a taxa de trocas seria
sempre zero.

---

## Pipeline

```
Câmera (USB, acima do campo)
  └─ camera/            aquisição, auto-tuning, warpPerspective (ROI de 4 pontos)
       └─ frame retificado, vista de cima  ──ZMQ "cropped"──┐
                                                            │
  ┌─────────────────────────────────────────────────────────┘
  └─ vision/detectors/  ColorDetector (HSV) | YoloDetector     → posições em pixel
       └─ vision/tracker.py    suavização EMA, média circular no ângulo
            └─ vision/geometry.py   pixel → metro  → FieldState
                 ├─ benchmark/   compara com ground truth → métricas + latência
                 └─ live.py      sink opcional (log | ZMQ JSON | nenhum)
```

O ponto de extensão é `vision/detectors/base.py`: entra frame retificado (BGR),
sai `DetectionResult` em pixel. Tudo depois disso é comum a qualquer detector —
é o que mantém a comparação livre de diferenças de infraestrutura entre eles.

---

## Uso

Requer Python 3.11+ e [`uv`](https://docs.astral.sh/uv/).

```bash
make sync                 # instala dependências (uv sync --extra dev)
make test                 # suíte de testes

# Avaliação offline sobre o dataset anotado
make bench                          # detector de vision_config.json
uv run vsss-vision bench --detector color --dataset datasets/vision --out out.json

# Comparação lado a lado, mesmas cenas e mesma calibração
make compare

# Pipeline ao vivo, com latência instrumentada (em outro terminal: make camera-service)
make live
uv run vsss-vision live --detector color --sink zmq
```

Calibração:

```bash
make camera-service       # precisa estar rodando para as duas ferramentas abaixo
make calibrate-camera     # ROI de 4 pontos nos cantos físicos do campo
make calibrate-colors     # faixas HSV por perfil (own, opponent, ball, marcadores)
```

O detector por rede neural exige `uv sync --extra yolo` e ainda **não está
implementado** — o roteiro está em `src/vsss_vision/vision/detectors/yolo.py`.

---

## Dataset e ground truth

Cada mídia (`.png`, `.mp4`, …) tem um `.json` irmão com o estado verdadeiro do
campo e as condições de captura. Esquema em
[`datasets/vision/README.md`](datasets/vision/README.md).

Duas exigências que não são negociáveis para os resultados valerem:

1. **Anotar em metros** (`units: "m"`) sempre que possível. Ground truth em pixel
   só é comparável dentro da mesma montagem de câmera e resolução.
2. **Registrar as condições** (`conditions`): iluminância em lux, tipo de luz,
   uniformidade. Sem isso não há como agrupar resultados por condição, que é a
   variável independente da P1.

Mídia bruta ainda não anotada pode ficar no diretório: arquivos sem `.json`
irmão são ignorados pelo carregador.

---

## Reprodutibilidade

- Cada execução do `bench` grava um JSON com as métricas, a latência **e um
  recorte da configuração usada** (detector, escala do campo, parâmetros de
  suavização, raio de casamento). A configuração faz parte do resultado.
- `experiments/results/` está no `.gitignore`: resultados são regeráveis a partir
  do dataset e da configuração. Versionar à mão apenas os que forem citados em um
  artigo, junto do commit exato que os gerou.
- Pesos de modelo (`*.pt`, `*.onnx`) não vão para o git — publicar em release ou
  no Hugging Face e referenciar por URL.

---

## Relação com `futebol-vsss`

Cópia derivada e independente, feita a partir de `src/vision/` e `src/camera/`
do repositório de competição. Os dois evoluem separadamente e **não** se
sincronizam automaticamente; uma melhoria de detecção que valha para o jogo
precisa ser portada à mão.

Uma diferença deliberada: a saída em protobuf `Environment` (vssproto) por UDP
multicast, que lá alimenta a estratégia, foi removida aqui. O pipeline termina
em `FieldState`, em metros — que é o que as métricas exigem. Fechar o laço com
um robô real (necessário para a parte final da P2) é acrescentar um *sink* em
`vision/live.py`, não mudar o pipeline.

---

## Licença

Apache 2.0 — ver [`LICENSE`](LICENSE).
