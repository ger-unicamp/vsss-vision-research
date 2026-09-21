# Proposta P2 — Latência em jogo real

## Contribuição

Curva de compromisso entre **precisão e latência ponta a ponta** para o pipeline
clássico e para variantes do YOLO, em hardware fixo, **com medição do efeito da
latência sobre o comportamento do robô em jogo**.

Essa última parte não é opcional. Medir FPS de variantes de YOLO é fácil e já
foi feito em outros domínios; sem mostrar o efeito no comportamento, o artigo
tende a ser lido como incremental. A contribuição é ligar latência de percepção
a desempenho de tarefa.

## Hipóteses

- **H1.** Existe um joelho na curva precisão × latência: acima de certa latência,
  ganhos de precisão do detector não se traduzem em ganho de desempenho em
  jogo, porque o estado já está velho quando o comando chega ao robô.
- **H2.** Os **percentis altos** (p95, p99) predizem o desempenho em jogo melhor
  que a latência média: um pico ocasional grande causa um erro de ação visível,
  que a média esconde.

## O que é medido

`benchmark/latency.py` cronometra por estágio, com `time.perf_counter_ns`:

| Estágio | O que cobre |
| :--- | :--- |
| `capture` | leitura do frame na câmera |
| `decode` | decodificação JPEG do frame recebido por ZMQ |
| `detect` | `Detector.detect` — para o YOLO, **incluindo** pré-processamento e transferência para a GPU, não só o *forward* |
| `track` | `Tracker.update` |
| `convert` | pixel → metro |
| `publish` | envio ao consumidor |
| `total` | ponta a ponta |

Reportar sempre média, p50, p95, p99 e máximo. Os primeiros frames são
descartados (`--warmup`): incluem alocação de buffers e carga de pesos, que não
representam o regime permanente.

A mesma instrumentação roda nos dois modos — offline (`vsss-vision bench`) e ao
vivo (`vsss-vision live`) — então a comparação entre eles é direta e a diferença
é atribuível ao que de fato muda: a câmera e o transporte.

## Desenho experimental

### Variáveis independentes

| Variável | Níveis |
| :--- | :--- |
| Detector | `color`; YOLO nano / small / medium / large / extra-large |
| Resolução de entrada | pelo menos dois valores por variante |
| Hardware | **fixo** ao longo de todo o experimento; especificar CPU, GPU, memória e versões de driver no artigo |

### Variáveis dependentes

1. **Latência** por estágio (acima) e FPS sustentado.
2. **Precisão** — as mesmas métricas da P1, sobre o mesmo dataset.
3. **Desempenho em jogo** — ver abaixo.

### Medindo o efeito no comportamento

O elo que falta na literatura. Duas abordagens, da mais simples à mais forte:

**(a) Latência injetada, tarefa controlada.** Manter o detector fixo e injetar
atraso artificial no *sink* (por exemplo, 0, 20, 40, 80, 160 ms). Com isso, a
latência vira variável independente isolada, sem confundir com a precisão do
detector. Tarefas com métrica objetiva:

- interceptar a bola em movimento: erro de interceptação em cm, taxa de sucesso;
- percorrer um caminho até um alvo: erro de posicionamento final, oscilação em
  torno da referência, tempo até estabilizar.

**(b) Detectores reais, partidas curtas.** Rodar partidas com cada variante e
medir posse, gols e faltas. Mais realista e muito mais ruidoso — exige muitas
repetições para ter significância. Usar como confirmação de (a), não como
evidência principal.

O caminho (a) é o que sustenta H1 e H2, porque isola a variável. O caminho (b)
mostra que o efeito sobrevive fora do laboratório.

## Requisitos de bancada

- Mesma montagem física da P1 (campo, câmera, iluminação) — manter a iluminação
  **constante** aqui, já que a variável de interesse é a latência.
- Hardware de inferência fixo, documentado.
- Robôs funcionais, com rádio e estratégia, para a parte (a) e (b).
- Para fechar o laço: um *sink* em `vision/live.py` que alimente a estratégia.
  O pipeline termina em `FieldState` (ver `vision/geometry.py`); publicar isso
  no formato que a estratégia consome é um arquivo novo, não uma mudança no
  pipeline.

## Análise

- Gráfico precisão × latência, um ponto por variante e resolução; identificar o
  joelho.
- Correlacionar cada métrica de latência (média, p95, p99) com a métrica de
  desempenho em jogo; reportar qual prediz melhor (H2).
- Reportar o custo de latência de cada estágio separadamente: se `decode`
  domina, a conclusão sobre detectores muda.

## Onde publicar

Mesmos veículos da P1 (SBAI 2027, SBR 2027). Se as duas propostas amadurecerem
juntas, considerar um único artigo mais forte em vez de dois incrementais — a
decisão depende do volume de resultados sólidos.
