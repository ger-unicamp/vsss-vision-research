# Trabalhos relacionados e posicionamento

Levantamento feito em setembro de 2026. Cada item traz o que importa para
o desenho experimental deste repositório, não um resumo geral.

---

## 1. Trabalho de referência direta

**Marques, A. G. O. (2025).** *Detecção de objetos Very Small Size Soccer
utilizando redes neurais convolucionais YOLO.* Monografia (Bacharelado em
Ciência da Computação), Universidade de Brasília. Orientador: Prof. Marcus
Vinicius Lamar. 51 p.
[PDF](https://bdm.unb.br/bitstream/10483/41606/1/2025_AyssaGiovannaDeOliveiraMarques_tcc.pdf) ·
[registro](https://bdm.unb.br/handle/10483/41606) ·
[modelo](https://huggingface.co/ayssag/BallnetPose) ·
[dataset](https://universe.roboflow.com/ayssag/ballnet_dataset)

### O que foi feito

- **Arquitetura:** YOLOv8-pose, variantes nano, small, large e extra-large.
  Busca de épocas de 100 a 1000, em passos de 100; critério de seleção foi
  a **perda dos keypoints**. Melhor resultado: YOLOv8x-pose, 400 épocas,
  **mAP50-95 = 0,99** (sobre OKS).
- **Classes:** `robot0`, `robot1`, `robot2`, `ball` — uma classe **por
  robô**, atrelada às camisas de EVA rosa da UnBall, que distinguem os
  robôs por formas geométricas inscritas. O card do modelo público declara
  as três classes de robô como *blue jersey*.
- **Keypoints:** um segmento `front`–`back` por objeto, anotado no Roboflow
  Annotate (projeto do tipo *Keypoint Detection*). Orientação calculada
  como `θ = arctan((y_front − y_back)/(x_front − x_back))`.
- **Dataset:** `ballnet_dataset` v1 — 653 imagens extraídas de vídeos da
  UnBall na **IRONCup 2020**, divididas em 437 treino / 131 validação /
  85 teste. **Sem *data augmentation*** (a ferramenta oferece; não foi
  usada).
- **Baseline:** *Main System* da UnBall (Python/OpenCV): recorte
  retangular ou homografia, conversão para HSV, `cv2.inRange`, contornos,
  `minEnclosingCircle` para a bola, e para os robôs o menor retângulo
  inscrito no contorno mais análise de forma da camisa para o id e para o
  ângulo.
- **Hardware de treino:** Threadripper 3970X, 2× RTX 3080, 192 GiB RAM.

### Resultados reportados

Erro de orientação médio (graus):

| Robô | Ballnet Pose | Main System |
| :--- | ---: | ---: |
| 0 | 17,19 | 73,34 |
| 1 | 5,16 | 36,67 |
| 2 | 20,05 | 97,40 |

Erro de centro: em **pixels**, com a Ballnet Pose concentrando a maioria
das detecções abaixo de 1 px e o MS chegando a faixas de 11,7–14,5 px e a
dois valores acima de 50 px (duas imagens em que confundiu a bola com o
Robô 2).

### Limitações declaradas pela autora

- Processamento **apenas offline**; o módulo só lê arquivos. A própria
  seção de trabalhos futuros aponta que usar em cenário real exige
  acrescentar detecção em tempo real a partir de webcam.
- As comparações "avaliam as detecções quanto às métricas relativas às
  dimensões das imagens. Esses resultados **não são suficientes para
  avaliar a detecção dos objetos no campo em escala real**".
- Base pequena e pouco diversa; ampliar para outras partidas e **para robôs
  de outras equipes** é sugerido como continuidade.

### Limitações observáveis, não declaradas

Estas foram verificadas na leitura do texto e são o que mais orienta a
metodologia daqui. Não invalidam o trabalho — é um TCC, e o mérito de ter
construído dataset, modelo e comparação públicos continua de pé. Mas
qualquer resultado construído sobre ele precisa corrigi-las.

**(a) A métrica de erro de posição permite cancelamento.** A Equação 5.1 é

```
Erro_centro = |(x − x₀) + (y − y₀)| / 2
```

isto é, o módulo da **soma** dos erros com sinal, dividido por 2 — não a
distância euclidiana. Uma detecção deslocada em (+10, −10) px recebe erro
**zero**. Isso favorece sistematicamente qualquer detector cujos erros em
X e Y sejam anticorrelacionados e torna os números de erro de centro
difíceis de interpretar.

**(b) O erro de orientação não é circular.** A Equação 5.2 é `|θ₀ − θ|`,
sem redução ao intervalo (−π, π]. O sintoma aparece no próprio texto: um
histograma do MS com valores no intervalo **"(172,53; 215,53] graus"**.
Diferença angular verdadeira não passa de 180°. Parte dos 73°–97° médios
atribuídos ao baseline é, portanto, artefato de métrica — erros próximos
da virada 359°→1° contados como quase uma volta inteira.

**(c) A linha de base pode estar subcalibrada.** As tabelas de HSV do MS
trazem, para a segmentação de time, faixas como *Saturation* [0:255] e
*Value* [0:255] — o eixo inteiro. Uma comparação contra um pipeline
clássico mal calibrado não sustenta conclusão sobre os métodos.

**(d) Risco de vazamento entre treino e teste.** 653 imagens extraídas de
vídeos de um único evento, sem menção a divisão por vídeo de origem.
Frames vizinhos são quase idênticos; divisão por frame coloca
quase-duplicatas nos dois lados. mAP50-95 = 0,99 é consistente com esse
efeito.

**(e) Keypoints na bola são ruído de rótulo.** O texto diz que, para a
bola, "foram escolhidas quaisquer semirretas entre duas extremidades da
circunferência". Isso ensina a rede a prever um eixo que não existe.

**(f) O modelo público não detecta adversários.** Classes por robô, treinadas
com as camisas de uma equipe. Num jogo real, metade dos objetos em campo
fica invisível para o modelo.

---

## 2. Visão em ligas de futebol de robôs

- **Zickler, S.; Laue, T.; Birbach, O.; Wongphati, M.; Veloso, M. (2010).**
  *SSL-Vision: The Shared Vision System for the RoboCup Small Size League.*
  RoboCup 2009, LNCS.
  [PDF](https://www.cs.cmu.edu/~mmv/papers/09robocup-sslvision.pdf) ·
  [código](https://github.com/RoboCup-SSL/ssl-vision).
  A SSL abandonou sistemas de visão por equipe porque "a maioria das
  equipes havia convergido para soluções semelhantes, com poucos resultados
  de pesquisa significativos". O VSSS ainda está no estágio anterior: cada
  equipe com seu pipeline HSV. Isso posiciona o trabalho daqui — o valor não
  está em construir mais um sistema, e sim em **medir** o que se ganha ou se
  perde ao trocar o método.
- **Szemenyei, M.; Estivill-Castro, V. (2019).** *ROBO: Robust, Fully Neural
  Object Detection for Robot Soccer.* [arXiv](https://arxiv.org/pdf/1910.10949).
  Detecção neural de bola, robô, trave e cruzamento de linhas; argumenta
  que abordagens neurais toleram variação de iluminação melhor que tabelas
  de cor.
- **Barry, D. et al. (2019).** *xYOLO: A Model For Real-Time Object Detection
  In Humanoid Soccer On Low-End Hardware.*
  [ResearchGate](https://www.researchgate.net/publication/338647215).
  YOLOv3-tiny reduzido com camadas XNOR, para caber em hardware fraco. É o
  precedente direto da pergunta da P2 — só que lá o alvo é embarcado no
  robô, e no VSSS o processamento é externo, o que muda o compromisso.
- **Albani, D. et al. (2017).** *A Deep Learning Approach for Object
  Recognition with NAO Soccer Robots.*
  [Springer](https://link.springer.com/chapter/10.1007/978-3-319-68792-6_33).
  Segmentação adaptativa mais validação por CNN; registra que o sistema de
  visão do NAO "não é muito robusto a variações de iluminação" e que a
  iluminação é um dos fatores mais influentes durante uma partida.
- **Cruz, N.; Lobos-Tsunekawa, K.; Ruiz-del-Solar, J. (2018).** *Playing
  Soccer without Colors in the SPL: A Convolutional Neural Network
  Approach.* [arXiv](https://arxiv.org/pdf/1811.12493).
  A SPL removeu as cores do ambiente e forçou a migração para CNN — a
  trajetória que o VSSS ainda não percorreu.

## 3. Iluminação como variável controlada

- Iluminação em competições de futebol de robôs varia tipicamente entre
  **400 e 1500 lux** — ancora os níveis da P1 em algo defensável, em vez de
  níveis escolhidos arbitrariamente.
- **Lu, H.; Zhang, H.; Yang, S.; Zheng, Z. (2010).** *Illumination Invariant
  Color Model for Object Recognition in Robot Soccer.*
  [Springer](https://link.springer.com/chapter/10.1007/978-3-642-13498-2_89).
- **Sridharan, M.; Stone, P.** *Color Classification and Object Recognition
  for Robot Soccer Under Variable Illumination.*
  [PDF](https://research-repository.griffith.edu.au/server/api/core/bitstreams/284bd8ca-d47c-531e-96c8-877c4e2a9f80/content).
  Trabalhos anteriores atacam o problema **mudando o modelo de cor**; a
  pergunta da P1 é diferente e complementar: quanto custa, em erro e em
  minutos de operador, manter o método clássico competitivo sem trocá-lo.
- Revisões recentes de detecção em baixa luminosidade padronizaram o
  protocolo de avaliar **qualidade de imagem e acurácia de detecção em
  paralelo**, com mAP complementado por precisão, revocação e F1
  ([revisão sistemática](https://link.springer.com/article/10.1007/s42452-025-08051-5)).
  Vale seguir a forma, mas acrescentando as métricas em escala real: mAP não
  diz em quantos centímetros o robô vai errar.

## 4. Latência no laço percepção-ação

- **Behnke, S.; Egorova, A.; Gloye, A.; Rojas, R.; Simon, M. (2003).**
  *Predicting Away Robot Control Latency.* RoboCup 2003 Symposium.
  [PDF](https://people.idsia.ch/~alexander/2003/4/robocup03b.pdf).
  Trata o atraso de controle como problema **imanente** da Small Size
  League e o ataca prevendo a posição futura do robô. Mede o atraso entre
  percepção e ação: **53 ms em média (máx. 74 ms) a 50 fps** e **65 ms em
  média (máx. 99 ms) a 30 fps**. São os números de referência contra os
  quais comparar qualquer pipeline medido aqui.
- **Corke, P.; Good, M.** *Dynamics and system performance of visual
  servoing.* Atraso de realimentação visual como fonte de **instabilidade**,
  com condições de estabilidade dependentes do atraso. É o embasamento
  teórico da hipótese H1 da P2: existe um ponto a partir do qual mais
  precisão não compensa mais atraso.
- Sistemas de *visual servoing* práticos operam entre ~46 ms e ~155 ms
  conforme a carga — a mesma ordem de grandeza da faixa que a P2 vai varrer.

---

## 5. Onde este repositório se encaixa

| Lacuna | Quem já tocou | O que falta, e é o que fazemos |
| :--- | :--- | :--- |
| Detecção neural em VSSS | Marques (2025) | Classes por cor de time (detecta adversário), avaliação em escala real |
| Robustez à iluminação | SPL/NAO, modelos de cor invariantes | Medição em **lux**, em VSSS, com o **esforço de calibração** como variável |
| Latência | Behnke (2003) na SSL; xYOLO no humanoide | Curva precisão × latência em VSSS e **efeito no comportamento em jogo** |
| Protocolo de métrica | — | Distância euclidiana em cm, diferença angular circular, divisão por origem |

Ver [`methodology.md`](methodology.md) para como cada uma dessas decisões
está implementada, e [`p1-illumination.md`](p1-illumination.md) /
[`p2-latency.md`](p2-latency.md) para os protocolos.

## 6. Busca ainda pendente

Feito por busca aberta na web; **falta busca sistemática** em base indexada
antes de submeter:

- **IEEE Xplore** e **SOL/SBC** (anais de SBR, SBAI, CBA, LARS/LARC) com os
  termos "Very Small Size Soccer", "IEEE VSSS", "robot soccer vision",
  "global vision", restrito a 2015–2026.
- **Anais do RoboCup Symposium** e *team description papers* da SSL.
- Verificar especificamente: (1) alguém já mediu robustez à iluminação em
  VSSS/SSL **com iluminância em lux**? (2) existe trabalho ligando latência
  de percepção a desempenho de tarefa em futebol de robôs? (3) há dataset
  público de VSSS além do `ballnet_dataset`?

Se (1) ou (2) já existirem, a contribuição muda de "primeira avaliação"
para "avaliação com X", e é melhor descobrir isso agora do que na revisão.
